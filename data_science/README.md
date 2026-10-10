# data_science

Autoencoder convolucional 1D para remover ruído de espectros, treinado com
os datasets que [`data_eng`](../data_eng/README.md) gera em
`../database/{train,valid,test}/`.

## Instalação

Na raiz do monorepo (Python 3.12 + [`uv`](https://docs.astral.sh/uv/)):

```bash
uv sync --package data-science
```

O `torch` vem do índice CUDA 12.6 (configurado no `pyproject.toml` da raiz).

## Estrutura

| Caminho | Descrição |
|---|---|
| `src/convolutional_autoencoder/` | `ConvolutionalAutoencoder` (encoder → vetor latente → decoder) e `AutoencoderDataSet` (lê os `.npz` e normaliza por canal). |
| `src/common/train_and_test.py` | `fit` (loop de épocas, guarda a melhor época, TensorBoard opcional), `train`, `test`, `make_dataloaders`, `get_device`. |
| `src/common/losses.py` | `absorption_loss`: MSE da absorção, com peso opcional nos picos (`alpha`) e penalidade opcional de absorbância negativa (`delta`) para o treino; sem eles é a métrica de validação. |
| `src/common/transforms.py` | Augmentations aplicadas só na entrada: `GaussianNoise`, `BaselineShift`, `MaskWindows`, `Compose`. |
| `scripts/train/` | Treino do modelo (`train.py` + `train_configs.yaml`). |
| `scripts/search/` | Busca de hiperparâmetros com Optuna (`search.py` + `search_configs.yaml`). |
| `nbs/tests.ipynb` | Notebook de testes do modelo. |

`convolutional_autoencoder` e `common` são instalados como pacotes, então
podem ser importados diretamente:

```python
from convolutional_autoencoder import ConvolutionalAutoencoder, AutoencoderDataSet
from common import fit, absorption_loss, GaussianNoise
```

## Dados

Cada amostra é um `.npz` com `wavenumbers` e `absorption`. O dataset monta
`X` (entrada com augmentation) e `y` (alvo limpo), com shape `(canais, N)`:

- canais: número de onda e absorção, mais um canal de máscara opcional (`mask_channel=True`);
- normalização: média e desvio por canal, calculados no treino e reaproveitados em validação e teste.

## Treino

1. Ajuste caminhos, arquitetura, loss e augmentations em `scripts/train/train_configs.yaml`.
2. Rode:

```bash
uv run --package data-science python data_science/scripts/train/train.py
```

O modelo final é o da melhor época na validação. Ele é avaliado uma vez no
teste e salvo em `save_model_path` como `autoencoder_<data-hora>.pt`, junto
com a arquitetura e o `mean`/`std` do treino. As curvas vão para
`runs/autoencoder/`.

## Busca de hiperparâmetros

1. Ajuste os caminhos, o espaço de busca e o estudo em `scripts/search/search_configs.yaml`.
2. Rode:

```bash
uv run --package data-science python data_science/scripts/search/search.py
```

A melhor configuração é salva em `output_dir` como
`best_configs_<study_name>.yaml`, no mesmo formato do `train_configs.yaml`.
Revise caminhos e épocas antes de usá-la no treino. As curvas de cada trial
vão para `runs/search/`.

Para ver as curvas de treino e busca: `tensorboard --logdir runs`.
