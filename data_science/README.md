# data_science

Notebook de **ciência de dados**: a partir dos parâmetros de linha (centro,
sigma, gamma, fwhm, amplitude, altura...) que
[`data_eng`](../data_eng/README.md) gera em `../database/`, treina modelos
de ML para estimar **temperatura** e **pressão** do gás a partir do
espectro (motivação, metodologia e discussão completas no notebook
[`nbs/ML_modeling.ipynb`](nbs/ML_modeling.ipynb)).

Este diretório **não é uma biblioteca**: não tem `src/`, não é instalado
como pacote importável — é um membro do workspace uv só para declarar suas
dependências e rodar o notebook (`[tool.uv] package = false` no
`pyproject.toml`). `nbs/utils.py` é um módulo auxiliar simples, importado
diretamente pelo notebook por estar na mesma pasta (`from utils import *`).
A lógica de simulação/ajuste de espectro vem da biblioteca
[`spectra-hitran`](../packages/spectra_hitran/README.md).

## Instalação

Requer Python 3.12 e [`uv`](https://docs.astral.sh/uv/). A partir da raiz do
monorepo:

```bash
uv sync --package data-science
```

(ou `uv sync` na raiz, que instala o workspace inteiro — ver
[README da raiz](../README.md)).

O notebook também usa `shap`, `lightgbm` e `xgboost`, que **não** estão
fixados em `pyproject.toml` (não eram dependência declarada do projeto
antes desta reorganização) — instale à parte se for rodá-lo:

```bash
uv add --package data-science shap lightgbm xgboost
```

## Estrutura

| Caminho | O quê |
|---|---|
| `nbs/utils.py` | Funções de apoio à modelagem: engenharia de features (`eng_features`), validação cruzada (`crossVal`, `validacaoCruzada`), filtro de qualidade de ajuste (`r2Filter`), intervalo de confiança via bootstrap. |
| `nbs/ML_modeling.ipynb` | Notebook principal: treino/avaliação de modelos de temperatura e pressão, comparação espectros simulados vs. reais. |

## Uso rápido

Rode o notebook com o kernel do workspace a partir da pasta `nbs/` (para
`from utils import *` funcionar) — ex.: `uv run --package data-science
jupyter lab` de dentro de `data_science/nbs/`.
