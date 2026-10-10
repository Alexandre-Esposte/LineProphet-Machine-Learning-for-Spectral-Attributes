# data_eng

Scripts e notebooks de **engenharia de dados espectrais**: usam a biblioteca
[`spectra-hitran`](../packages/spectra_hitran/README.md) para baixar dados
de linha do HITRAN, simular espectros e organizar tudo em datasets
(`../database/`) consumidos pelo pacote irmão
[`data_science`](../data_science/README.md).

Este diretório **não é uma biblioteca**: não tem `src/`, não é instalado
como pacote importável — é um membro do workspace uv só para declarar suas
dependências e rodar scripts/notebooks (`[tool.uv] package = false` no
`pyproject.toml`). Toda a lógica reutilizável vive em `spectra-hitran`.

## Instalação

Requer Python 3.12 e [`uv`](https://docs.astral.sh/uv/). A partir da raiz do
monorepo:

```bash
uv sync --package data-eng
```

(ou `uv sync` na raiz, que instala o workspace inteiro — ver
[README da raiz](../README.md)).

## Estrutura

| Caminho | O quê |
|---|---|
| `scripts/download_spectra.py` | CLI para baixar dados de linha do HITRAN de uma molécula. Rodar manualmente antes do `spectra-generator.py` (que não baixa nada sozinho — assume que os dados já estão em `database/hitran_database/`). |
| `scripts/spectra-generator.py` | **CLI atual** para gerar dataset de espectros simulados (treino/validação): salva o espectro bruto (wavenumbers/absorção) + metadados (temperatura, pressão, caminho ótico, molécula, wavenumber min/max...) em `.npz`, sem separar linhas nem calcular features. Config por constantes no topo do arquivo (não por flags de linha de comando) — edite as constantes e rode. Paraleliza com `multiprocessing.Pool` (`N_PROCESSES`). |
| `nbs/teste_download.ipynb` | Notebook de teste de download, atualizado com a API atual de `spectra-hitran`. |
| `nbs/selfBroadeningHCl.ipynb` | Notebook legado (usa `BroadeningProcessing`, API removida) — pendente de avaliação/limpeza. |

> `nbs/selfBroadeningHCl.ipynb` é anterior à reformulação da lib
> (`spectra-hitran`) e referencia uma API já removida
> (`BroadeningProcessing`) — ver `TODO.md` na raiz. Não corrigir sem pedido
> explícito.

## Uso rápido

`spectra-generator.py` não baixa a molécula — isso é feito à parte (manualmente
ou via `download_spectra.py`) antes de rodar:

```bash
uv run --package data-eng python data_eng/scripts/download_spectra.py \
    --name hcl --isotopes 52,53 --range 5200,5900

uv run --package data-eng python data_eng/scripts/spectra-generator.py
```

Configuração (faixas de pressão/temperatura/caminho ótico, `QTE_TRAIN`/`QTE_VALID`,
molécula, diluente, `N_PROCESSES` etc.) fica nas constantes no topo de
`spectra-generator.py` — edite o arquivo direto pra mudar.

Por código:

```python
from SpectraHitran.SpectraGenerator.spectragenerator import SpectraGenerator
```

`spectra-hitran` é dependência de workspace deste pacote, então fica
importável direto em qualquer script/notebook rodado com `uv run`.
