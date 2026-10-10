# spectra-hitran — Documentação

Biblioteca núcleo em cima do HITRAN/HAPI: simula espectros de absorção para
qualquer molécula/isótopo/mistura de gases suportada pelo HITRAN, detecta
linhas espectrais e ajusta perfis de Voigt a elas — usada como biblioteca
Python (`src/SpectraHitran`) pelos projetos `data_eng` e `data_science` do
monorepo.

## Instalação e execução

```bash
uv sync --group test        # instala dependências (núcleo + testes)
uv run pytest tests/ -v     # roda a suíte de testes
python3 scripts/run_tests.py  # idem, com sync + resumo final
```

Python 3.12, gerenciado via `uv` (`pyproject.toml`/`uv.lock`). O pacote em
`src/` é instalado em modo editável pelo próprio `uv sync`.
## Estrutura do repositório

| Caminho | Conteúdo |
|---|---|
| `src/SpectraHitran/` | Biblioteca núcleo: download HITRAN, simulação, detecção de linhas, ajuste Voigt. Ver [docs/src/](src/). |
| `tests/spectra_hitran/` | Testes automatizados de `src/SpectraHitran` (pytest). |
| `scripts/run_tests.py` | Orquestrador: sincroniza o ambiente, roda a suíte, imprime resumo. |
| `../../../database/` | Dados gerados/baixados (git-ignorado, raiz do monorepo), compartilhado entre `data_eng` e `data_science`. |
| `../../../hitran_database/` | Cache local do HAPI com os dados de linha baixados do HITRAN (git-ignorado, recriado sob demanda). |

Os scripts/notebooks que usam esta biblioteca para gerar dados ficam em
[`../../../data_eng/`](../../../data_eng/README.md); a modelagem que consome
esses dados fica em [`../../../data_science/`](../../../data_science/README.md).

## Documentos

- [`src/spectra_generator.md`](src/spectra_generator.md) — download e simulação de espectros (`SpectraGenerator`)
- [`src/spectra_processing.md`](src/spectra_processing.md) — detecção de linhas e utilitários de espectro
- [`src/spectra_fit.md`](src/spectra_fit.md) — ajuste de perfil de Voigt por linha

## Princípio de design

Toda a lógica científica (HITRAN, física do espectro, ajuste matemático)
vive em `src/SpectraHitran` e é testada isoladamente (`tests/`). O pacote é
usado diretamente por script Python, sem interface gráfica.
