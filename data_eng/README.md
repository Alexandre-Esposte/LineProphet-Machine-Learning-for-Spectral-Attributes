# data_eng

Scripts para baixar dados do HITRAN e gerar os datasets de espectros simulados
em `../database/`, usados por [`data_science`](../data_science/README.md).
Toda a lógica reutilizável está em [`spectra-hitran`](../packages/spectra_hitran/README.md);
este diretório não é um pacote instalável.

## Instalação

Na raiz do monorepo (Python 3.12 + [`uv`](https://docs.astral.sh/uv/)):

```bash
uv sync --package data-eng
```

## Uso

1. Baixe as moléculas (salvas em `database/hitran_database/`):

```bash
uv run --package data-eng python data_eng/scripts/download_spectra.py --name hcl --isotopes 52,53 --range 5200,5900
```

2. Gere o dataset:

```bash
uv run --package data-eng python data_eng/scripts/spectra-generator.py
```

A configuração do gerador (janelas, faixas de P/T/caminho ótico, `QTE_TOTAL`,
`SPLIT_FRACTIONS`, `N_PROCESSES` etc.) fica nas constantes no topo de
`spectra-generator.py`.

## Saída

Um `.npz` por espectro em `database/<split>/` (`train`, `valid`, `test`), com
todas as janelas misturadas. Cada arquivo contém `wavenumbers` e `absorption`
(float32) e os metadados da simulação (molécula, janela, pressão, temperatura,
caminho ótico, resolução etc.).

## Estrutura

| Caminho | Descrição |
|---|---|
| `scripts/download_spectra.py` | Baixa dados de linha de uma molécula do HITRAN. |
| `scripts/spectra-generator.py` | Gera os espectros simulados em paralelo. |
| `nbs/teste_download.ipynb` | Exemplo de download com a API atual. |
| `nbs/selfBroadeningHCl.ipynb` | Legado: usa API removida (`BroadeningProcessing`). |
