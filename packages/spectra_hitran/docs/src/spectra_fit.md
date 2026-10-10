# `SpectraHitran.SpectraFit`

`src/SpectraHitran/SpectraFit/spectrafit.py` — ajuste de perfil de Voigt
(via [`lmfit`](https://lmfit.github.io/lmfit-py/)) a cada linha detectada,
com busca automática via [Optuna](https://optuna.org/) do palpite inicial
quando o ajuste direto converge mal. Só perfil de Voigt por enquanto
(decisão deliberada).

## Constantes

- `_SIGMA_GAMMA_BOUNDS = (1e-6, 5.0)` — limites usados tanto no ajuste
  (`pars['sigma']/['gamma'].set(min=..., max=...)`) quanto no espaço de
  busca do Optuna. Compartilhados de propósito: desalinhados, o Optuna não
  conseguia propor palpite em boa parte da faixa que o ajuste permitia.
- `_FIT_FIELDS` — chaves numéricas retornadas por `resultFit` além de
  `evals`/`npoints`/`r2`.

## `filterLine(spectra, linecenter, lineintensity, intensidade_relativa) -> Tuple[List[float], List[float]]`

Recorta os pontos ao redor de uma linha, andando de cada lado do centro
até a absorção cair abaixo do limiar relativo OU parar de decrescer
(vale/mínimo local) — a segunda condição evita engolir a linha vizinha
quando `lineintensity` é baixa e o limiar absoluto cai no nível de ruído.

- `linecenter`: número de onda aproximado do centro (*snapado* pro ponto
  mais próximo via `np.argmin`, não precisa ser exato).
- `lineintensity`: absorção no centro, usada como referência do limiar.
- `intensidade_relativa`: fração de `lineintensity` abaixo da qual o
  recorte para (ex.: `0.01` = corta a 1% do pico).
- Usado tanto por `singleLineFit`/`multipleLineFit`.

## `resultFit(result: Optional[ModelResult], successful: bool) -> Dict[str, float]`

Extrai os parâmetros do ajuste de Voigt num dicionário plano.

- `successful=True`: `evals`, `npoints`, `r2`, e para cada um de
  `amplitude`/`center`/`sigma`/`gamma`/`fwhm`/`height` — valor e erro
  padrão (sufixo `_stderr`).
- `successful=False`: todas as chaves com `np.nan`, sem tentar ler
  `result` (que pode não ser válido nesse caso).

## `singleLineFit(data, ex, chute_centro, chute_sigma, chute_gamma, vgamma, vsigma) -> Tuple[np.ndarray, Optional[ModelResult], Dict[str, float], bool]`

Ajusta um `VoigtModel` a uma única linha já recortada (tipicamente saída
de `filterLine`).

- `chute_centro`/`chute_sigma`/`chute_gamma`: palpites iniciais.
- `vgamma`/`vsigma`: se `True`, parâmetro livre no ajuste (`vary=True`);
  se `False`, fica fixo no palpite.
- `sigma`/`gamma` limitados a `_SIGMA_GAMMA_BOUNDS`; `amplitude` a `[0, 1]`.
- `guess` e configuração dos parâmetros rodam dentro do `try/except` — se
  `modelo.guess(...)` falhar (ex.: dados vazios), cai no mesmo caminho de
  falha controlada (`successful=False`) em vez de propagar.
- Retorna `(final, result, params, successful)`: `final` é a curva
  ajustada (`data + result.residual`) ou array vazio se falhou.

## `multipleLineFit(spectra, lines, linepercentage=0.01) -> pd.DataFrame`

Ajusta um Voigt a cada linha de `lines`, com fallback via Optuna quando o
R² do ajuste direto é `< 0.99`.

- `lines`: precisa só das colunas `wavenumber`/`absorption` (saída de
  `SpectraProcessing.detectLines`). `temperature`/`pressure` são opcionais
  e só enriquecem o log (`_describeLine`).
- Por linha: `filterLine` recorta a janela, `singleLineFit` tenta com
  palpite fixo (`8e-3` para sigma/gamma); se `r² < 0.99`, `optimize`
  (Optuna, 50 trials) busca um palpite melhor e refaz o ajuste.
- Retorna um `DataFrame` com uma linha por linha espectral, colunas =
  chaves de `resultFit` (não reinclui `wavenumber`/`absorption`).
- Imprime uma linha de log por linha ajustada (sucesso/fallback/falha).

## `objective(trial, xfilterline, yfilterline, center) -> float` / `optimize(xfilterline, yfilterline, center) -> Tuple[Dict, float]`

Função objetivo do Optuna (minimiza `1 - r²`, retorna `1.0` se o ajuste
falhar) e o `study.optimize(..., n_trials=50)` que a usa, buscando
`chute_sigma`/`chute_gamma` dentro de `_SIGMA_GAMMA_BOUNDS`. `optimize`
retorna `(study.best_params, study.best_value)`, usado por
`multipleLineFit` para refazer o `singleLineFit`.

## Exemplo de uso

```python
from SpectraHitran.SpectraFit.spectrafit import filterLine, singleLineFit, multipleLineFit

# uma linha só:
x, y = filterLine(gen.spectra, linecenter=2906.25, lineintensity=0.02, intensidade_relativa=0.01)
_, result, params, ok = singleLineFit(y, x, 2906.25, 8e-3, 8e-3, vgamma=True, vsigma=True)

# todas as linhas de uma tabela detectada:
fitted = multipleLineFit(gen.spectra, lines, linepercentage=0.01)
```
