# spectra-hitran

Biblioteca núcleo construída sobre o
[HITRAN](https://hitran.org/)/[HAPI](https://hitran.org/hapi/) (HITRAN
Application Programming Interface): simula espectros de absorção para
qualquer molécula, isotopólogo e mistura de gases diluentes suportada pelo
HITRAN, detecta linhas espectrais automaticamente e ajusta perfis de Voigt a
cada linha — por código Python puro.

O projeto nasceu de um estudo focado em HCl (broadening por
pressão/temperatura), mas foi generalizado: hoje não há nenhuma lógica
específica de uma molécula — tudo funciona para qualquer espécie que o
HITRAN cubra, contanto que os dados de linha necessários existam na base.

Este é o único **pacote reutilizável de verdade** do monorepo LineProphet —
ver o [README da raiz](../../README.md). Os outros dois diretórios,
[`data_eng`](../../data_eng/README.md) e
[`data_science`](../../data_science/README.md), não são bibliotecas: são
scripts/notebooks que **chamam** esta biblioteca.

## O que dá pra fazer com isso

- **Baixar** dados de linha do HITRAN para qualquer molécula/isotopólogo (via os IDs globais de isotopólogo do próprio HITRAN), numa faixa espectral arbitrária.
- **Simular** o espectro resultante escolhendo:
  - a **mistura de gases diluentes** (frações de `air`, `self`, `H2`, `CO2`, `He`, `H2O` — os únicos com alargamento de linha publicado pelo HITRAN; cada molécula tem um subconjunto real disponível, e o sistema avisa quando um gás pedido não tem dado medido em vez de falhar silenciosamente);
  - as **condições físicas** (temperatura, pressão, comprimento de caminho ótico);
  - o **perfil de linha** (Voigt, Lorentz, Doppler, Hartmann-Tran, Voigt speed-dependent, ou deixar o HAPI escolher automaticamente);
  - a **grandeza de saída** (absorbância decádica de Beer-Lambert, coeficiente de absorção, seção de choque por linha, transmitância, absorptância ou radiância);
  - e opcionalmente convoluir o resultado com uma fenda instrumental (simulando um espectrômetro real).
- **Detectar linhas** automaticamente (detecção de picos) em qualquer espectro simulado.
- **Ajustar** um perfil de Voigt a cada linha — uma de cada vez ou todas de uma vez — com busca automática (Optuna) de um palpite inicial melhor quando o ajuste direto converge mal.
- Fazer tudo isso **por script Python** (`src/SpectraHitran`).

## Instalação

Requer Python 3.12 e [`uv`](https://docs.astral.sh/uv/). A partir da raiz do
monorepo:

```bash
uv sync --package spectra-hitran --group test
```

(ou `uv sync` na raiz, que instala o workspace inteiro — ver
[README da raiz](../../README.md)).

## Uso rápido

### Por código

```python
from SpectraHitran.SpectraGenerator.spectragenerator import SpectraGenerator
from SpectraHitran.SpectraProcessing.spectraprocessing import detectLines
from SpectraHitran.SpectraFit.spectrafit import multipleLineFit

gen = SpectraGenerator()
gen.downloadMolecule('hcl_52', (52,), (2900, 2910), diluents=('air', 'self'))
gen.simulateSpectra('hcl_52', {'air': 0, 'self': 1}, {'l': 10, 'p': 1, 'T': 296})

lines = detectLines(gen.spectra, {'height': 0.01, 'width': 0, 'distance': 1})
fitted = multipleLineFit(gen.spectra, lines)
print(fitted[['center', 'sigma', 'gamma', 'r2']])
```

Guia completo de cada função em [`docs/src/`](docs/src/).

### Testes

```bash
uv run --package spectra-hitran pytest packages/spectra_hitran/tests -v   # da raiz do monorepo
python3 scripts/run_tests.py                                              # de dentro de packages/spectra_hitran/, com sync automático + resumo final
```

## Estrutura do pacote

| Caminho | O quê |
|---|---|
| `src/SpectraHitran/` | Biblioteca núcleo (download, simulação, detecção de linhas, ajuste). Toda a lógica científica mora aqui. |
| `tests/spectra_hitran/` | Testes automatizados (pytest). |
| `scripts/run_tests.py` | Orquestrador de teste (sync + suíte + resumo). |
| `docs/` | Documentação detalhada de cada módulo — comece por [`docs/README.md`](docs/README.md). |

## Documentação

- [`docs/README.md`](docs/README.md) — índice, instalação, princípios de design.
- [`docs/src/spectra_generator.md`](docs/src/spectra_generator.md) — download e simulação (`SpectraGenerator`).
- [`docs/src/spectra_processing.md`](docs/src/spectra_processing.md) — detecção de linhas.
- [`docs/src/spectra_fit.md`](docs/src/spectra_fit.md) — ajuste de Voigt.

## Para quem for construir em cima disto

O design deliberado é: **toda lógica científica em `src/SpectraHitran`, testável e usável sem GUI nenhuma**; qualquer interface futura (API/CLI/frontend) deveria só chamar essas funções, nunca duplicar cálculo. Os arquivos em `docs/` descrevem contrato de cada função (assinatura, parâmetros, o que valida, o que já foi verificado contra o código-fonte real do HAPI) — comece por ali antes de estender algo. A suíte de testes em `tests/spectra_hitran/` cobre as regressões e decisões não óbvias já corrigidas (motivo documentado em cada teste) — rode-a antes e depois de qualquer mudança.
