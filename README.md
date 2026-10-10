# LineProphet

Monorepo do projeto LineProphet: parâmetros de linhas espectrais via
HITRAN/HAPI, dos dados sintéticos até modelos preditivos de temperatura e
pressão de gases a partir do espectro.

Gerenciado como um **workspace [`uv`](https://docs.astral.sh/uv/)** com um
pacote reutilizável e dois diretórios de projeto que o consomem:

| Diretório | O quê |
|---|---|
| [`packages/spectra_hitran/`](packages/spectra_hitran/README.md) | **A biblioteca.** Download HITRAN, simulação de espectros, detecção de linhas, ajuste de Voigt. |
| [`data_eng/`](data_eng/README.md) | **Engenharia de dados.** Scripts/notebooks que chamam `spectra_hitran` pra baixar, simular e organizar espectros, gravando os datasets em `database/`. |
| [`data_science/`](data_science/README.md) | **Ciência de dados.** Autoencoder convolucional para remover ruído de espectros, com scripts de treino e busca de hiperparâmetros sobre os datasets de `data_eng`. |

`data_eng` é só scripts/notebooks (`[tool.uv] package = false`).
`data_science` instala seus próprios módulos (`convolutional_autoencoder` e
`common`) para uso nos seus scripts. A lógica espectroscópica reutilizável
fica em `spectra_hitran`.

## Instalação

Requer Python 3.12 e [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <este-repositório>
cd LineProphet-Machine-Learning-for-Spectral-Attributes
uv sync   # instala os três membros do workspace (+ dependências) num único .venv na raiz
```

`uv sync` na raiz resolve o workspace inteiro: `spectra-hitran`, `data-eng`
e `data-science` ficam disponíveis no mesmo ambiente, então qualquer
script/notebook rodado com `uv run` (de qualquer lugar do repo) enxerga a
biblioteca (`import SpectraHitran`).

Para trabalhar em só um membro (ex.: instalar as dependências de teste da
lib):

```bash
uv sync --package spectra-hitran --group test
uv sync --package data-eng
uv sync --package data-science
```

## Dados compartilhados

`database/` e `hitran_database/` ficam na raiz do repo (git-ignorados,
recriados sob demanda) e são compartilhados pelos dois projetos — é onde
`data_eng` grava o que gera e de onde `data_science` lê para treinar/avaliar
modelos.

## Estrutura do repositório

```
pyproject.toml            # raiz do workspace uv (sem código próprio)
packages/
  spectra_hitran/          # a biblioteca: download/simulação/detecção/ajuste
data_eng/                  # scripts/notebooks: geração dos datasets a partir da lib
data_science/               # autoencoder: modelo, treino e busca de hiperparâmetros
database/, hitran_database/ # dados gerados/baixados, compartilhados, git-ignorados
```

Detalhes de cada membro nos respectivos READMEs:
[`packages/spectra_hitran/README.md`](packages/spectra_hitran/README.md),
[`data_eng/README.md`](data_eng/README.md) e
[`data_science/README.md`](data_science/README.md).

## Licença

MIT — ver [`LICENSE`](LICENSE).
