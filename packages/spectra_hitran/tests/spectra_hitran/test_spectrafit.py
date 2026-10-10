import numpy as np
import pandas as pd
import pytest
from lmfit.models import VoigtModel

from SpectraHitran.SpectraFit import spectrafit


def _voigt_spectrum(center=50.0, sigma=0.3, gamma=0.3, amplitude=5e-3, npoints=2000):
    x = np.linspace(0, 100, npoints)
    y = VoigtModel().func(x, amplitude=amplitude, center=center, sigma=sigma, gamma=gamma)
    return {"wavenumbers": x, "absorption": y}


def test_filterLine_stops_at_relative_intensity_threshold():
    spectra = _voigt_spectrum()
    center = 50.0
    peak_idx = int(np.argmin(np.abs(spectra["wavenumbers"] - center)))
    peak_intensity = spectra["absorption"][peak_idx]
    threshold = 0.5

    xs, ys = spectrafit.filterLine(spectra, center, peak_intensity, intensidade_relativa=threshold)

    # todos os pontos retornados respeitam o limiar de intensidade relativa ao pico
    assert all(y / peak_intensity >= threshold for y in ys)
    # o pico central está incluído e é o de maior intensidade
    assert peak_intensity in ys
    assert max(ys) == pytest.approx(peak_intensity)
    # o primeiro ponto fora da janela (não incluído) já está abaixo do limiar
    first_excluded_idx = spectra["wavenumbers"].tolist().index(xs[0]) - 1
    if first_excluded_idx >= 0:
        assert spectra["absorption"][first_excluded_idx] / peak_intensity < threshold


def test_filterLine_does_not_raise_when_peak_is_near_the_edges():
    """Regressão: a implementação anterior podia estourar o índice ao caminhar para
    a esquerda/direita perto das bordas do array."""
    spectra = _voigt_spectrum(center=1.0, npoints=200)  # pico perto do início do array
    xs, ys = spectrafit.filterLine(spectra, 1.0, spectra["absorption"][0], intensidade_relativa=0.01)
    assert len(xs) == len(ys)

    spectra = _voigt_spectrum(center=99.0, npoints=200)  # pico perto do fim do array
    xs, ys = spectrafit.filterLine(spectra, 99.0, spectra["absorption"][-1], intensidade_relativa=0.01)
    assert len(xs) == len(ys)


def test_filterLine_stops_at_valley_between_weak_neighboring_lines():
    """Regressão: para linhas muito fracas (ex. N2 ~6000cm-1), o limiar absoluto
    (intensidade_relativa * lineintensity) fica tão baixo que a versão antiga
    caminhava até engolir a linha vizinha inteira. Agora o recorte também para
    no vale (mínimo local) entre as duas linhas, independente da intensidade."""
    x = np.linspace(0, 100, 4000)
    amplitude = 1e-6  # linha muito fraca
    y = (VoigtModel().func(x, amplitude=amplitude, center=50.0, sigma=0.3, gamma=0.3)
         + VoigtModel().func(x, amplitude=amplitude, center=52.0, sigma=0.3, gamma=0.3))
    spectra = {"wavenumbers": x, "absorption": y}

    peak_idx = int(np.argmin(np.abs(x - 50.0)))
    peak_intensity = y[peak_idx]

    # limiar relativo minúsculo: sem a checagem de vale, o recorte atravessaria
    # o vale entre as duas linhas e continuaria pela linha vizinha
    xs, ys = spectrafit.filterLine(spectra, 50.0, peak_intensity, intensidade_relativa=1e-6)

    assert max(xs) < 51.0  # não deve alcançar o pico vizinho em 52.0
    # à direita do centro, a absorção recortada é não-crescente até o vale
    # (nunca volta a subir rumo à linha vizinha)
    direita_y = [y for x_i, y in zip(xs, ys) if x_i >= 50.0]
    assert all(a >= b for a, b in zip(direita_y, direita_y[1:]))


def test_resultFit_failed_returns_nan_without_raising():
    """Regressão: o ramo de falha tentava ler result.nfev/np.null (result era o int 0) e quebrava."""
    params = spectrafit.resultFit(result=None, successful=False)

    expected_keys = {"evals", "npoints", "r2", "amplitude", "amplitude_stderr", "center", "center_stderr",
                     "sigma", "sigma_stderr", "gamma", "gamma_stderr", "fwhm", "fwhm_stderr",
                     "height", "height_stderr"}
    assert set(params) == expected_keys
    assert all(np.isnan(value) for value in params.values())


def test_resultFit_successful_extracts_expected_fields():
    spectra = _voigt_spectrum()
    result = VoigtModel().fit(spectra["absorption"], x=spectra["wavenumbers"],
                              amplitude=5e-3, center=50.0, sigma=0.3, gamma=0.3)

    params = spectrafit.resultFit(result, successful=True)

    assert params["center"] == pytest.approx(50.0, abs=0.5)
    assert params["r2"] == result.rsquared


def test_singleLineFit_bounds_match_optuna_search_range():
    """Regressão: os limites de sigma/gamma do ajuste (antes 0-5) e da busca do Optuna
    (antes 0-1) estavam desalinhados; agora compartilham `_SIGMA_GAMMA_BOUNDS`."""
    spectra = _voigt_spectrum()
    xs, ys = spectrafit.filterLine(spectra, 50.0, spectra["absorption"][1000], intensidade_relativa=0.01)

    _, result, _, successful = spectrafit.singleLineFit(ys, xs, 50.0, 0.3, 0.3, vgamma=True, vsigma=True)

    assert successful
    sigma_min, sigma_max = spectrafit._SIGMA_GAMMA_BOUNDS
    assert result.params["gamma"].min == sigma_min
    assert result.params["gamma"].max == sigma_max
    assert result.params["sigma"].min == sigma_min
    assert result.params["sigma"].max == sigma_max


def test_objective_suggests_within_singleLineFit_bounds():
    calls = {}

    class FakeTrial:
        def suggest_float(self, name, low, high):
            calls[name] = (low, high)
            return 0.3

    spectra = _voigt_spectrum()
    xs, ys = spectrafit.filterLine(spectra, 50.0, spectra["absorption"][1000], intensidade_relativa=0.01)

    spectrafit.objective(FakeTrial(), xs, ys, 50.0)

    assert calls["chute_sigma"] == spectrafit._SIGMA_GAMMA_BOUNDS
    assert calls["chute_gamma"] == spectrafit._SIGMA_GAMMA_BOUNDS


def test_multipleLineFit_returns_one_row_per_line_without_raising():
    spectra = _voigt_spectrum(center=50.0, sigma=0.5, gamma=0.5)
    peak_idx = int(np.argmin(np.abs(spectra["wavenumbers"] - 50.0)))

    lines = pd.DataFrame([{
        "wavenumber": 50.0,
        "absorption": spectra["absorption"][peak_idx],
        "temperature": 296.0,
        "pressure": 1.0,
    }])

    results = spectrafit.multipleLineFit(spectra, lines, linepercentage=0.01)

    assert len(results) == 1


def test_multipleLineFit_does_not_raise_when_optuna_fallback_fit_also_fails(monkeypatch):
    """Regressão: quando o primeiro ajuste tinha R² baixo e o reajuste guiado pelo
    Optuna também falhava (successful=False), o código acessava result.rsquared
    incondicionalmente e lançava AttributeError em vez de logar a falha."""
    spectra = _voigt_spectrum(center=50.0, sigma=0.5, gamma=0.5)
    peak_idx = int(np.argmin(np.abs(spectra["wavenumbers"] - 50.0)))
    lines = pd.DataFrame([{"wavenumber": 50.0, "absorption": spectra["absorption"][peak_idx]}])

    calls = {"n": 0}

    class FakeResult:
        rsquared = 0.5
        nfev = 1
        ndata = 1
        params = {key: type("P", (), {"value": 0.0, "stderr": 0.0})()
                 for key in ("amplitude", "center", "sigma", "gamma", "fwhm", "height")}

    def fake_singleLineFit(data, ex, chute_centro, chute_sigma, chute_gamma, vgamma, vsigma):
        calls["n"] += 1
        if calls["n"] == 1:
            # primeiro ajuste: sucesso, mas com R² baixo o suficiente pra acionar o optuna
            fake = FakeResult()
            return np.array([]), fake, spectrafit.resultFit(fake, True), True
        # reajuste guiado pelo optuna: falha
        return np.array([]), None, spectrafit.resultFit(None, False), False

    monkeypatch.setattr(spectrafit, "singleLineFit", fake_singleLineFit)
    monkeypatch.setattr(spectrafit, "optimize", lambda *a, **k: ({"chute_sigma": 0.3, "chute_gamma": 0.3}, 0.5))

    results = spectrafit.multipleLineFit(spectra, lines, linepercentage=0.01)

    assert len(results) == 1
    assert np.isnan(results.loc[0, "r2"])


def test_multipleLineFit_works_without_optional_metadata_columns():
    """`lines` só precisa de wavenumber/absorption; temperature/pressure são opcionais."""
    spectra = _voigt_spectrum(center=50.0, sigma=0.5, gamma=0.5)
    peak_idx = int(np.argmin(np.abs(spectra["wavenumbers"] - 50.0)))

    lines = pd.DataFrame([{"wavenumber": 50.0, "absorption": spectra["absorption"][peak_idx]}])

    results = spectrafit.multipleLineFit(spectra, lines, linepercentage=0.01)

    assert len(results) == 1
