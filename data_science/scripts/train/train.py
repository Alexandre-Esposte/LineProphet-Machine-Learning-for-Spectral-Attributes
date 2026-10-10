from convolutional_autoencoder import AutoencoderDataSet, ConvolutionalAutoencoder
from common import fit, test, get_device, make_dataloaders, GaussianNoise, MaskWindows, BaselineShift, SyntheticContamination, Compose, absorption_loss

import yaml
import torch
import logging

from datetime import datetime
from functools import partial
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train_autoencoder')


train_configs_path = Path(__file__).with_name('train_configs.yaml')
with open(train_configs_path, 'r', encoding='utf-8') as file:
    configs = yaml.safe_load(file)


train_path = Path(configs['train'])
valid_path =  Path(configs['valid'])
test_path =  Path(configs['test'])
model_architecture = configs['model_architecture']
keys = ['in_channels', 'out_channels', 'kernel_size_max_pooling', 'kernel_size_conv', 'num_layers']
architecture = [tuple(block[k] for k in keys) for block in model_architecture]

# Canais físicos (wavenumber, absorção) + canal de máscara, se houver
OUTPUT_CHANNELS = 2
input_channels = OUTPUT_CHANNELS + int(configs['mask_channel'])
if architecture[0][0] != input_channels:
    raise ValueError(f"in_channels do 1º bloco deve ser {input_channels} (mask_channel: {configs['mask_channel']}).")


def main():



    #Instanciando dataset
    # Ruído antes da máscara, para a região apagada ficar zerada de fato
    # Linha de base antes do ruído; ruído antes da máscara
    # Contaminação depois da linha de base e antes do ruído (linhas estranhas também são medidas com ruído)
    augmentations = [BaselineShift(offset_max = configs['baseline_shift']['offset_max'],
                                   slope_max = configs['baseline_shift']['slope_max'],
                                   p = configs['baseline_shift']['p']),
                     SyntheticContamination(**configs['contamination']),
                     GaussianNoise(std_min = configs['noise_std_min'], std_max = configs['noise_std_max'])]
    if configs['mask_channel']:
        augmentations.append(MaskWindows(min_size = configs['mask_min_size'],
                                         max_size = configs['mask_max_size'],
                                         n_windows = configs['mask_n_windows'],
                                         p = configs['mask_p']))

    # mean/std calculados só no treino
    train_dataset = AutoencoderDataSet(data_path = train_path,
                                       augmentation = Compose(augmentations),
                                       mask_channel = configs['mask_channel'])
    # Validação usa as estatísticas do treino, sem ruído e sem máscara (canal todo zero)
    valid_dataset = AutoencoderDataSet(data_path = valid_path,
                                      mean = train_dataset.mean,
                                      std = train_dataset.std,
                                      mask_channel = configs['mask_channel'])

    # Teste: mesmo tratamento da validação, mas só avaliado uma vez, ao fim do treino
    test_dataset = AutoencoderDataSet(data_path = test_path,
                                      mean = train_dataset.mean,
                                      std = train_dataset.std,
                                      mask_channel = configs['mask_channel'])

    device = get_device()

    logger.info(f"Train Path: {train_path}")
    logger.info(f"Valid Path: {valid_path}")
    logger.info(f"Test Path:  {test_path}")
    logger.info(f"Model Architecture: {architecture}")
    logger.info(f"Device: {device}")

    train_dataloader, valid_dataloader = make_dataloaders(train_dataset, valid_dataset,
                                                         batch_size = configs['batch_size'],
                                                         num_workers = configs['num_workers'],
                                                         device = device)
    _, test_dataloader = make_dataloaders(train_dataset, test_dataset,
                                          batch_size = configs['batch_size'],
                                          num_workers = configs['num_workers'],
                                          device = device)

    # Instanciando modelo  
    model = ConvolutionalAutoencoder(architecture, configs['pooling_output_size'], configs['latent_space'], output_channels = OUTPUT_CHANNELS)
    model.to(device)

    # Definindo otimizador
    otimizador = torch.optim.Adam(params = model.parameters(),
                                  lr = configs['learning_rate'])

    # Loss de treino do yaml; validação e teste usam sempre o MSE da absorção
    loss_fn = partial(absorption_loss, mean = float(train_dataset.mean[1]), std = float(train_dataset.std[1]),
                      **configs['loss'])
    eval_loss_fn = absorption_loss

    #Iniciando writer do tensorboard
    # Um subdiretório por execução, para os runs não se misturarem no TensorBoard
    run_name = datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(f'runs/autoencoder/{run_name}')

    #Adicionando arquitetura no tensorboard
    example_input = torch.randn(1, input_channels, 1024).to(device)
    writer.add_graph(model,example_input)

    fit(model, train_dataloader, valid_dataloader, loss_fn, otimizador, device,
        epochs = configs['epochs'], writer = writer, log_histograms = True, logger = logger,
        eval_loss_fn = eval_loss_fn)

    # Teste só depois de todas as épocas, com o melhor modelo da validação (o fit já restaura)
    test_loss = test(dataloader = test_dataloader, model = model, loss_fn = eval_loss_fn, device = device)
    logger.info(f"Test loss (melhor modelo da validação): {test_loss}")
    writer.add_scalar("Loss/test_final", test_loss)

    writer.close()

    torch.save({"state_dict": model.state_dict(),
            "architecture": architecture,
            "pooling_output_size": configs['pooling_output_size'],
            "latent_space": configs['latent_space'],
            "output_channels": OUTPUT_CHANNELS,
            "mask_channel": configs['mask_channel'],
            "mean": train_dataset.mean,
            "std": train_dataset.std,
            "test_loss": test_loss},
           Path(configs['save_model_path']) / f"autoencoder_{run_name}.pt")


if __name__ == "__main__":
    main()
