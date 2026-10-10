from convolutional_autoencoder import AutoencoderDataSet, ConvolutionalAutoencoder
from common import fit, get_device, make_dataloaders, GaussianNoise, MaskWindows, BaselineShift, SyntheticContamination, Compose, absorption_loss

import yaml
import torch
import optuna
import logging

from functools import partial
from pathlib import Path
from typing import Any
from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('search_autoencoder')


search_configs_path = Path(__file__).with_name('search_configs.yaml')
with open(search_configs_path, 'r', encoding='utf-8') as file:
    configs = yaml.safe_load(file)


train_path = Path(configs['train'])
valid_path = Path(configs['valid'])
test_path = Path(configs['test'])  # não usado na busca; só repassado ao best_configs
output_dir = Path(configs['output_dir'])

# Canais físicos (wavenumber, absorção) + canal de máscara, se houver
OUTPUT_CHANNELS = 2
input_channels = OUTPUT_CHANNELS + int(configs['mask_channel'])


def suggest(trial: optuna.Trial, name: str, spec: Any) -> Any:
    """Sorteia o valor de `name` conforme o `spec` do search_configs.yaml.

    Um dict com `type` vira uma sugestão do Optuna; qualquer outro valor é fixo.
    """
    if not isinstance(spec, dict):
        return spec

    if spec['type'] == 'categorical':
        return trial.suggest_categorical(name, spec['choices'])
    if spec['type'] == 'int':
        return trial.suggest_int(name, spec['low'], spec['high'], step = spec.get('step', 1))
    if spec['type'] == 'float':
        return trial.suggest_float(name, spec['low'], spec['high'], log = spec.get('log', False))

    raise ValueError(f"Tipo de busca desconhecido para '{name}': {spec['type']}")


def build_architecture(num_blocks: int, base_channels: int, kernel_size_max_pooling: int,
                       kernel_size_conv: int, num_layers: int, in_channels: int = 2) -> list[tuple[int, int, int, int, int]]:
    """Monta a lista de blocos do encoder; os canais dobram a cada bloco."""
    architecture = []
    out_channels = base_channels
    for _ in range(num_blocks):
        architecture.append((in_channels, out_channels, kernel_size_max_pooling, kernel_size_conv, num_layers))
        in_channels = out_channels
        out_channels *= 2
    return architecture


def main():

    # mean/std calculados uma vez só no treino e reaproveitados por todos os trials
    stats_dataset = AutoencoderDataSet(data_path = train_path)
    mean, std = stats_dataset.mean, stats_dataset.std
    spectrum_length = stats_dataset[0][0].shape[-1]

    # Validação: estatísticas do treino, sem ruído e sem máscara (igual em todos os trials)
    valid_dataset = AutoencoderDataSet(data_path = valid_path, mean = mean, std = std,
                                      mask_channel = configs['mask_channel'])

    device = get_device()
    logger.info(f"Device: {device}")

    def objective(trial: optuna.Trial) -> float:
        params = {name: suggest(trial, name, spec) for name, spec in configs['search_space'].items()}

        architecture = build_architecture(params['num_blocks'], params['base_channels'],
                                          params['kernel_size_max_pooling'],
                                          params['kernel_size_conv'], params['num_layers'],
                                          in_channels = input_channels)

        # AdaptiveAvgPool1d maior que a saída do encoder só replicaria pontos: config inválida
        encoder_length = spectrum_length // 2 ** params['num_blocks']
        if params['pooling_output_size'] > encoder_length:
            raise optuna.TrialPruned(f"pooling_output_size {params['pooling_output_size']} > "
                                     f"comprimento após o encoder ({encoder_length})")

        # Ruído antes da máscara, para a região apagada ficar zerada de fato
        # Linha de base e contaminação (fixas por estudo) antes do ruído
        augmentations = [BaselineShift(offset_max = configs['baseline_shift']['offset_max'],
                                       slope_max = configs['baseline_shift']['slope_max'],
                                       p = configs['baseline_shift']['p']),
                         SyntheticContamination(**configs['contamination']),
                         GaussianNoise(std_min = params['noise_std_min'], std_max = params['noise_std_max'])]
        if configs['mask_channel']:
            augmentations.append(MaskWindows(min_size = params['mask_min_size'],
                                             max_size = params['mask_max_size'],
                                             n_windows = params['mask_n_windows'],
                                             p = params['mask_p']))

        # Só a augmentation muda por trial; mean/std já vêm prontos, sem recalcular
        train_dataset = AutoencoderDataSet(data_path = train_path, mean = mean, std = std,
                                           augmentation = Compose(augmentations),
                                           mask_channel = configs['mask_channel'])

        train_dataloader, valid_dataloader = make_dataloaders(train_dataset, valid_dataset,
                                                             batch_size = params['batch_size'],
                                                             num_workers = configs['num_workers'],
                                                             device = device)

        model = ConvolutionalAutoencoder(architecture, params['pooling_output_size'], params['latent_space'],
                                         output_channels = OUTPUT_CHANNELS)
        model.to(device)
        otimizador = torch.optim.Adam(params = model.parameters(), lr = params['learning_rate'])
        loss_fn = partial(absorption_loss, mean = float(mean[1]), std = float(std[1]), **configs['loss'])

        logger.info(f"Trial {trial.number}: {params}")

        # Reporta a loss de validação ao Optuna a cada época, para o pruner decidir se continua
        def report(epoch: int, valid_loss: float) -> None:
            trial.report(valid_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        writer = SummaryWriter(f"runs/search/{configs['study_name']}/trial_{trial.number}")
        try:
            return fit(model, train_dataloader, valid_dataloader, loss_fn, otimizador, device,
                       epochs = configs['epochs'], writer = writer, epoch_callback = report, logger = logger,
                       eval_loss_fn = absorption_loss)
        except torch.cuda.OutOfMemoryError:
            # Config grande demais para a GPU: descarta o trial em vez de derrubar a busca
            raise optuna.TrialPruned("CUDA out of memory")
        finally:
            writer.close()
            del model, otimizador
            if device.type == "cuda":
                torch.cuda.empty_cache()

    study = optuna.create_study(study_name = configs['study_name'],
                                storage = configs['storage'],
                                load_if_exists = True,
                                direction = 'minimize',
                                sampler = optuna.samplers.TPESampler(),
                                pruner = optuna.pruners.MedianPruner(**configs['pruner']))
    study.optimize(objective, n_trials = configs['n_trials'])

    best = study.best_trial
    logger.info(f"Melhor trial: {best.number} - valid loss {best.value}")

    # Parâmetros fixos não aparecem em best.params; junta os dois
    params = {name: best.params.get(name, spec) for name, spec in configs['search_space'].items()}
    architecture = build_architecture(params['num_blocks'], params['base_channels'],
                                      params['kernel_size_max_pooling'],
                                      params['kernel_size_conv'], params['num_layers'],
                                      in_channels = input_channels)
    keys = ['in_channels', 'out_channels', 'kernel_size_max_pooling', 'kernel_size_conv', 'num_layers']

    # Mesmo formato do train_configs.yaml (paths e epochs devem ser revistos antes de treinar)
    best_configs = {
        'train': str(train_path),
        'valid': str(valid_path),
        'test': str(test_path),
        'save_model_path': None,
        'batch_size': params['batch_size'],
        'num_workers': configs['num_workers'],
        'learning_rate': params['learning_rate'],
        'epochs': configs['epochs'],
        'noise_std_min': params['noise_std_min'],
        'noise_std_max': params['noise_std_max'],
        'loss': configs['loss'],
        'baseline_shift': configs['baseline_shift'],
        'contamination': configs['contamination'],
        'mask_channel': configs['mask_channel'],
        'mask_p': params['mask_p'],
        'mask_min_size': params['mask_min_size'],
        'mask_max_size': params['mask_max_size'],
        'mask_n_windows': params['mask_n_windows'],
        'latent_space': params['latent_space'],
        'pooling_output_size': params['pooling_output_size'],
        'model_architecture': [dict(zip(keys, block)) for block in architecture],
        'best_valid_loss': best.value,
    }

    output_dir.mkdir(parents = True, exist_ok = True)
    output_path = output_dir / f"best_configs_{configs['study_name']}.yaml"
    with open(output_path, 'w', encoding='utf-8') as file:
        yaml.safe_dump(best_configs, file, sort_keys = False, allow_unicode = True)
    logger.info(f"Melhor config salva em {output_path}")


if __name__ == "__main__":
    main()
