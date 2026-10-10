import pandas as pd
import numpy as np
from typing import Dict, List, Sequence, Tuple, Optional
from lmfit.models import VoigtModel
from lmfit.model import ModelResult
from functools import partial
import optuna
from optuna.trial import Trial

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Limites físicos usados tanto no ajuste do Voigt quanto na busca do Optuna,
# para que o palpite otimizado sempre caia dentro do que o ajuste permite.
_SIGMA_GAMMA_BOUNDS = (1e-6, 5.0)

_FIT_FIELDS = ('amplitude', 'amplitude_stderr', 'center', 'center_stderr',
              'sigma', 'sigma_stderr', 'gamma', 'gamma_stderr',
              'fwhm', 'fwhm_stderr', 'height', 'height_stderr')


def filterLine(spectra: Dict[str, Optional[np.ndarray]], linecenter: float, lineintensity: float,
               intensidade_relativa: float) -> Tuple[List[float], List[float]]:
    """Recorta os pontos ao redor de uma linha espectral.

    Caminha para cada lado do centro até a absorção cair abaixo do limiar
    relativo OU parar de decrescer (vale/mínimo local) — essa segunda condição
    evita engolir a linha vizinha quando `lineintensity` é baixa e o limiar
    absoluto cai no nível de ruído.

    Args:
        spectra: `{'wavenumbers': array, 'absorption': array}`.
        linecenter: número de onda (cm⁻¹) do centro da linha.
        lineintensity: absorção no centro da linha (usada como referência do limiar).
        intensidade_relativa: fração de `lineintensity` abaixo da qual o recorte para.

    Returns:
        `(wavenumbers, absorption)` da janela recortada.

    Exemplo:
        >>> x, y = filterLine(gen.spectra, linecenter=2886.0, lineintensity=0.03,
        ...                   intensidade_relativa=0.01)
    """
    wavenumbers = np.asarray(spectra['wavenumbers'])
    absorption = np.asarray(spectra['absorption'])

    center_idx = int(np.argmin(np.abs(wavenumbers - linecenter)))
    icentro = lineintensity

    esquerda_x: List[float] = []
    esquerda_y: List[float] = []
    i = 1
    anterior = absorption[center_idx]
    while center_idx - i >= 0:
        atual = absorption[center_idx - i]
        if atual / icentro < intensidade_relativa or atual > anterior:
            break
        esquerda_x.insert(0, wavenumbers[center_idx - i])
        esquerda_y.insert(0, absorption[center_idx - i])
        anterior = atual
        i += 1

    direita_x: List[float] = []
    direita_y: List[float] = []
    i = 1
    anterior = absorption[center_idx]
    while center_idx + i < len(absorption):
        atual = absorption[center_idx + i]
        if atual / icentro < intensidade_relativa or atual > anterior:
            break
        direita_x.append(wavenumbers[center_idx + i])
        direita_y.append(absorption[center_idx + i])
        anterior = atual
        i += 1

    pontos_x = esquerda_x + [wavenumbers[center_idx]] + direita_x
    pontos_y = esquerda_y + [absorption[center_idx]] + direita_y

    return pontos_x, pontos_y


def resultFit(result: Optional[ModelResult], successful: bool) -> Dict[str, Optional[float]]:
    """Extrai os parâmetros do ajuste de Voigt.

    Args:
        result: resultado de `VoigtModel.fit`, ou `None` se o ajuste falhou.
        successful: se `False`, retorna todos os campos como NaN.

    Returns:
        dict com `evals`, `npoints`, `r2` e os campos em `_FIT_FIELDS`
        (`amplitude`, `center`, `sigma`, `gamma`, `fwhm`, `height` e seus `_stderr`).
    """
    if not successful:
        return {'evals': np.nan, 'npoints': np.nan, 'r2': np.nan,
                **{key: np.nan for key in _FIT_FIELDS}}

    return {
        'evals': result.nfev,
        'npoints': result.ndata,
        'r2': result.rsquared,
        'amplitude': result.params['amplitude'].value,
        'amplitude_stderr': result.params['amplitude'].stderr,
        'center': result.params['center'].value,
        'center_stderr': result.params['center'].stderr,
        'sigma': result.params['sigma'].value,
        'sigma_stderr': result.params['sigma'].stderr,
        'gamma': result.params['gamma'].value,
        'gamma_stderr': result.params['gamma'].stderr,
        'fwhm': result.params['fwhm'].value,
        'fwhm_stderr': result.params['fwhm'].stderr,
        'height': result.params['height'].value,
        'height_stderr': result.params['height'].stderr,
    }


def singleLineFit(data: Sequence[float], ex: Sequence[float], chute_centro: float, chute_sigma: float,
                  chute_gamma: float, vgamma: bool,
                  vsigma: bool) -> Tuple[np.ndarray, Optional[ModelResult], Dict[str, Optional[float]], bool]:
    """Ajusta um perfil de Voigt a uma única linha espectral.

    Args:
        data: valores de absorção (y) da janela recortada (ex.: `filterLine`).
        ex: números de onda (x) correspondentes.
        chute_centro: palpite inicial do centro da linha.
        chute_sigma: palpite inicial da largura Doppler (sigma).
        chute_gamma: palpite inicial da largura de pressão (gamma).
        vgamma: se `True`, `gamma` é livre no ajuste; se `False`, fica fixo em `chute_gamma`.
        vsigma: idem para `sigma`.

    Returns:
        `(final, result, params, successful)` — curva ajustada, `ModelResult`
        do lmfit (ou `None`), dict de parâmetros (ver `resultFit`) e flag de sucesso.

    Exemplo:
        >>> final, result, params, ok = singleLineFit(y, x, chute_centro=2886.0,
        ...     chute_sigma=8e-3, chute_gamma=8e-3, vgamma=True, vsigma=True)
    """
    data = np.array(data)
    ex = np.array(ex)

    successful = True
    result = None

    try:
        modelo = VoigtModel()
        pars = modelo.guess(data, x=ex)

        sigma_min, sigma_max = _SIGMA_GAMMA_BOUNDS
        pars['gamma'].set(value=chute_gamma, vary=vgamma, expr='', min=sigma_min, max=sigma_max)
        pars['sigma'].set(value=chute_sigma, vary=vsigma, expr='', min=sigma_min, max=sigma_max)

        pars['amplitude'].set(value=5e-3, vary=True, expr='', min=0, max=1)
        pars['center'].set(value=chute_centro, vary=True, expr='')

        result = modelo.fit(data, pars, x=ex, nan_policy='propagate')
        final = data + result.residual
    except Exception:
        successful = False
        final = np.array([])

    params = resultFit(result, successful)

    return final, result, params, successful


def _describeLine(line: pd.Series) -> str:
    """Descrição da linha para log, incluindo temperatura/pressão quando disponíveis."""
    description = f"linha em {line['wavenumber']:.4f} cm⁻¹"
    if 'temperature' in line and 'pressure' in line:
        description += f" ({line['temperature']}K, {line['pressure']}atm)"
    return description


def multipleLineFit(spectra: Dict[str, Optional[np.ndarray]], lines: pd.DataFrame,
                    linepercentage: float = 0.01) -> pd.DataFrame:
    """Ajusta um perfil de Voigt a cada linha, buscando via Optuna quando o R² fica baixo.

    Args:
        spectra: `{'wavenumbers': array, 'absorption': array}`.
        lines: colunas `wavenumber`/`absorption` (saída de `detectLines`);
            `temperature`/`pressure` são opcionais e só enriquecem o log.
        linepercentage: limiar relativo passado a `filterLine` (`intensidade_relativa`).
            Padrão `0.01`.

    Returns:
        DataFrame com uma linha por pico, colunas de `resultFit`.

    Exemplo:
        >>> fits = multipleLineFit(gen.spectra, lines)
    """
    rows: List[Dict[str, Optional[float]]] = []
    for _, line in lines.iterrows():
        center = line['wavenumber']
        absorption = line['absorption']
        description = _describeLine(line)

        xfilterline, yfilterline = filterLine(spectra, center, absorption, linepercentage)

        chute = 8e-3
        final, result, params, successful = singleLineFit(yfilterline, xfilterline, center, chute, chute, vgamma=True, vsigma=True)

        if successful and result.rsquared < 0.99:
            print(f"{description} -> {result.rsquared}-------> R² insuficiente, habilitando o optuna")
            best_params, _ = optimize(xfilterline, yfilterline, center)
            final, result, params, successful = singleLineFit(yfilterline, xfilterline, center,
                                                               best_params['chute_sigma'], best_params['chute_gamma'],
                                                               vgamma=True, vsigma=True)

            if successful:
                print(f"{description} -> {result.rsquared}-------> Melhor R² após o optuna")
            else:
                print(f"{description} -> ajuste falhou mesmo após o optuna")
        elif successful:
            print(f"{description} -> {result.rsquared}")
        else:
            print(f"{description} -> ajuste falhou")

        rows.append(params)

    return pd.DataFrame(rows)


def objective(trial: Trial, xfilterline: List[float], yfilterline: List[float], center: float) -> float:
    """Função objetivo do Optuna: minimiza 1 - R² do ajuste de Voigt.

    Args:
        trial: trial do Optuna (sugere `chute_sigma`/`chute_gamma`).
        xfilterline, yfilterline: janela recortada da linha (ver `filterLine`).
        center: palpite inicial do centro da linha.

    Returns:
        `1 - r2`, ou `1.0` se o ajuste falhar.
    """
    sigma_min, sigma_max = _SIGMA_GAMMA_BOUNDS
    chute_sigma = trial.suggest_float('chute_sigma', sigma_min, sigma_max)
    chute_gamma = trial.suggest_float('chute_gamma', sigma_min, sigma_max)

    _, result, _, successful = singleLineFit(yfilterline, xfilterline, center, chute_sigma, chute_gamma, vgamma=True, vsigma=True)

    return 1 - result.rsquared if successful else 1.0


def optimize(xfilterline: List[float], yfilterline: List[float], center: float) -> Tuple[Dict[str, float], float]:
    """Busca via Optuna o melhor palpite inicial de sigma/gamma para o ajuste de Voigt.

    Args:
        xfilterline, yfilterline: janela recortada da linha (ver `filterLine`).
        center: palpite inicial do centro da linha.

    Returns:
        `(best_params, best_value)` — `best_params` tem `chute_sigma`/`chute_gamma`.

    Exemplo:
        >>> best_params, best_value = optimize(x, y, center=2886.0)
    """
    objective_partial = partial(objective, xfilterline=xfilterline, yfilterline=yfilterline, center=center)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_partial, n_trials=50)

    return study.best_params, study.best_value
