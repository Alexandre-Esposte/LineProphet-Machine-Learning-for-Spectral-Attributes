from pathlib import Path

import numpy as np

from SpectraHitran.SpectraGenerator import spectragenerator
from SpectraHitran.SpectraGenerator.spectragenerator import SpectraGenerator


def test_init_always_calls_db_begin_even_if_folder_already_exists(monkeypatch, tmp_path):
    """Regressão: antes, db_begin() era pulado quando a pasta já existia, deixando o
    cache do HAPI apontando para o diretório corrente em vez da pasta configurada."""
    calls = []
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: calls.append(db))

    tmp_path.mkdir(exist_ok=True)  # a pasta já existe antes da instanciação
    SpectraGenerator(database_path=str(tmp_path))

    assert calls == [str(tmp_path)]


def test_default_database_path_is_absolute_and_project_anchored(monkeypatch):
    calls = []
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: calls.append(db))

    SpectraGenerator()

    assert len(calls) == 1
    resolved = Path(calls[0])
    assert resolved.is_absolute()
    assert resolved.name == "hitran_database"


def test_downloadMolecule_calls_fetch_by_ids_with_expected_args(monkeypatch):
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: None)
    calls = []
    monkeypatch.setattr(spectragenerator, "fetch_by_ids", lambda *a, **kw: calls.append((a, kw)))

    generator = SpectraGenerator()
    generator.downloadMolecule("hcl", (1, 2), (100, 200))

    (args, kwargs) = calls[0]
    assert args == ("hcl", (1, 2), 100, 200)
    assert kwargs["ParameterGroups"] == ["160-char", "voigt_air", "voigt_self"]


def test_downloadMolecule_fetches_custom_diluents(monkeypatch):
    """Deve ser possível baixar coeficientes de alargamento por outros gases (CO2, H2, He, H2O)."""
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: None)
    calls = []
    monkeypatch.setattr(spectragenerator, "fetch_by_ids", lambda *a, **kw: calls.append((a, kw)))

    generator = SpectraGenerator()
    generator.downloadMolecule("co2", (1,), (2200, 2400), diluents=("air", "co2", "h2"))

    (_, kwargs) = calls[0]
    assert kwargs["ParameterGroups"] == ["160-char", "voigt_air", "voigt_co2", "voigt_h2"]
    assert generator._fetched_diluents["co2"] == {"air", "co2", "h2"}


def test_downloadMolecule_rejects_unsupported_diluent(monkeypatch):
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: None)
    calls = []
    monkeypatch.setattr(spectragenerator, "fetch_by_ids", lambda *a, **kw: calls.append((a, kw)))

    generator = SpectraGenerator()
    generator.downloadMolecule("hcl", (1,), (100, 200), diluents=("air", "argonio"))

    assert calls == []  # não deve nem tentar baixar


def test_downloadMolecule_swallows_exceptions(monkeypatch):
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: None)

    def boom(*a, **kw):
        raise RuntimeError("falha de rede simulada")

    monkeypatch.setattr(spectragenerator, "fetch_by_ids", boom)

    generator = SpectraGenerator()
    generator.downloadMolecule("hcl", (1,), (100, 200))  # não deve levantar exceção


def _patch_voigt_and_convolve(monkeypatch, wavenumbers, coef):
    monkeypatch.setattr(spectragenerator, "absorptionCoefficient_Voigt", lambda **kw: (wavenumbers, coef))
    monkeypatch.setattr(spectragenerator, "convolveSpectrum",
                        lambda wn, absorption, **kw: (wn, absorption, None, None, None))


def test_simulateSpectra_computes_absorption_from_coefficient(monkeypatch):
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: None)

    wavenumbers = np.linspace(0, 10, 50)
    coef = np.full_like(wavenumbers, 0.5)
    _patch_voigt_and_convolve(monkeypatch, wavenumbers, coef)

    generator = SpectraGenerator()
    generator.simulateSpectra("hcl", {"air": 0, "self": 1}, {"l": 2, "p": 1, "T": 296})

    # Beer-Lambert com ln: profundidade ótica -ln(T) = coef * l
    expected_absorption = -np.log(np.exp(-coef * 2))
    np.testing.assert_allclose(generator.spectra["absorption"], expected_absorption)
    np.testing.assert_allclose(generator.spectra["wavenumbers"], wavenumbers)


def test_simulateSpectra_passes_wavenumber_range_to_hapi(monkeypatch):
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: None)
    calls = []

    def fake_voigt(**kw):
        calls.append(kw)
        return np.linspace(0, 10, 50), np.full(50, 0.5)

    monkeypatch.setattr(spectragenerator, "absorptionCoefficient_Voigt", fake_voigt)

    generator = SpectraGenerator()
    generator.simulateSpectra("hcl", {"self": 1}, {"l": 2, "p": 1, "T": 296}, convolve=False)
    generator.simulateSpectra("hcl", {"self": 1}, {"l": 2, "p": 1, "T": 296}, convolve=False,
                              wavenumber_range=(2600, 3100))

    assert calls[0]["WavenumberRange"] is None
    assert calls[1]["WavenumberRange"] == (2600, 3100)


def test_simulateSpectra_quantity_coefficient_returns_raw_array(monkeypatch):
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: None)

    wavenumbers = np.linspace(0, 10, 50)
    coef = np.full_like(wavenumbers, 0.5)
    _patch_voigt_and_convolve(monkeypatch, wavenumbers, coef)

    generator = SpectraGenerator()
    generator.simulateSpectra("hcl", {"air": 0, "self": 1}, {"l": 2, "p": 1, "T": 296}, quantity="coefficient")

    np.testing.assert_allclose(generator.spectra["absorption"], coef)


def test_simulateSpectra_quantity_transmittance_uses_hapi_function(monkeypatch):
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: None)

    wavenumbers = np.linspace(0, 10, 50)
    coef = np.full_like(wavenumbers, 0.5)
    _patch_voigt_and_convolve(monkeypatch, wavenumbers, coef)

    generator = SpectraGenerator()
    generator.simulateSpectra("hcl", {"air": 0, "self": 1}, {"l": 2, "p": 1, "T": 296}, quantity="transmittance")

    expected = np.exp(-coef * 2)
    np.testing.assert_allclose(generator.spectra["absorption"], expected)


def test_simulateSpectra_rejects_unknown_quantity_and_resets_spectra(monkeypatch):
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: None)

    generator = SpectraGenerator()
    generator.simulateSpectra("hcl", {"air": 0, "self": 1}, {"l": 2, "p": 1, "T": 296}, quantity="inexistente")

    assert generator.spectra["wavenumbers"] is None
    assert generator.spectra["absorption"] is None


def test_simulateSpectra_rejects_profile_not_in_literal(monkeypatch):
    """`profile` é tipado como Literal['voigt', 'lorentz', 'doppler', 'priority'],
    mas o Python não impõe isso em runtime — a checagem contra
    _PROFILE_FUNCTION_NAMES continua sendo a defesa real (ex.: perfis mais
    específicos do HITRAN como 'ht'/'sdvoigt'/'galatry', que só têm parâmetro
    real pra air/self em qualquer molécula, foram deliberadamente deixados de
    fora do conjunto suportado por simulateSpectra)."""
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: None)

    generator = SpectraGenerator()
    generator.simulateSpectra("hcl", {"air": 0, "self": 1}, {"l": 2, "p": 1, "T": 296}, profile="galatry")

    assert generator.spectra["wavenumbers"] is None
    assert generator.spectra["absorption"] is None


def test_simulateSpectra_rejects_diluent_fractions_not_summing_to_one(monkeypatch):
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: None)

    generator = SpectraGenerator()
    # soma 0.7 — sub-alarga todas as linhas silenciosamente se não for barrado
    generator.simulateSpectra("hcl", {"air": 0.5, "self": 0.2}, {"l": 2, "p": 1, "T": 296})

    assert generator.spectra["wavenumbers"] is None
    assert generator.spectra["absorption"] is None


def test_simulateSpectra_accepts_diluent_fractions_within_tolerance(monkeypatch):
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: None)

    wavenumbers = np.linspace(0, 10, 50)
    coef = np.full_like(wavenumbers, 0.5)
    _patch_voigt_and_convolve(monkeypatch, wavenumbers, coef)
    monkeypatch.setattr(spectragenerator, "LOCAL_TABLE_CACHE",
                        {"hcl": {"data": {"gamma_air": np.array([0.05]), "gamma_self": np.array([0.06])}}})

    generator = SpectraGenerator()
    # 0.3 + 0.7 pode não bater exatamente 1.0 em ponto flutuante — deve passar mesmo assim
    generator.simulateSpectra("hcl", {"air": 0.3, "self": 0.7}, {"l": 2, "p": 1, "T": 296})

    assert generator.spectra["wavenumbers"] is not None


def test_simulateSpectra_rejects_diluent_not_downloaded(monkeypatch):
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: None)
    monkeypatch.setattr(spectragenerator, "fetch_by_ids", lambda *a, **kw: None)

    generator = SpectraGenerator()
    generator.downloadMolecule("hcl", (1,), (100, 200), diluents=("air", "self"))
    generator.simulateSpectra("hcl", {"air": 0.5, "co2": 0.5}, {"l": 2, "p": 1, "T": 296})

    # CO2 não foi baixado -> valida e falha de forma controlada (reset, sem exceção propagada)
    assert generator.spectra["wavenumbers"] is None
    assert generator.spectra["absorption"] is None


def test_simulateSpectra_resets_spectra_on_failure(monkeypatch):
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: None)

    def boom(**kw):
        raise RuntimeError("falha simulada")

    monkeypatch.setattr(spectragenerator, "absorptionCoefficient_Voigt", boom)

    generator = SpectraGenerator()
    generator.simulateSpectra("hcl", {"air": 0, "self": 1}, {"l": 2, "p": 1, "T": 296})

    assert generator.spectra["wavenumbers"] is None
    assert generator.spectra["absorption"] is None


def test_simulateSpectra_no_warning_when_diluent_has_real_data(monkeypatch):
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: None)

    wavenumbers = np.linspace(0, 10, 50)
    coef = np.full_like(wavenumbers, 0.5)
    _patch_voigt_and_convolve(monkeypatch, wavenumbers, coef)
    monkeypatch.setattr(spectragenerator, "LOCAL_TABLE_CACHE",
                        {"hcl": {"data": {"gamma_self": np.array([0.05, 0.06, 0.055])}}})

    generator = SpectraGenerator()
    generator.simulateSpectra("hcl", {"self": 1}, {"l": 2, "p": 1, "T": 296})

    assert generator.diluent_warnings == []


def test_simulateSpectra_warns_when_diluent_has_no_real_data(monkeypatch):
    """Regressão: HITRAN pode não ter alargamento medido para um gás/molécula
    (coluna toda '#'/NaN) sem que isso gere nenhum erro — o HAPI silenciosamente
    trata a contribuição como zero. O usuário precisa ser avisado disso."""
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: None)

    wavenumbers = np.linspace(0, 10, 50)
    coef = np.full_like(wavenumbers, 0.5)
    _patch_voigt_and_convolve(monkeypatch, wavenumbers, coef)
    monkeypatch.setattr(spectragenerator, "LOCAL_TABLE_CACHE",
                        {"hcl": {"data": {"gamma_h2o": np.array([np.nan, np.nan, np.nan])}}})

    generator = SpectraGenerator()
    generator.simulateSpectra("hcl", {"h2o": 1}, {"l": 2, "p": 1, "T": 296})

    assert len(generator.diluent_warnings) == 1
    assert "h2o" in generator.diluent_warnings[0]
    # o aviso não impede a simulação de completar
    assert generator.spectra["wavenumbers"] is not None


def test_simulateSpectra_warns_when_absorbance_saturates_to_zero_transmittance(monkeypatch):
    """Regressão: coeficiente de absorção alto o bastante (ou 'l' longo o bastante)
    faz transmitance = exp(-coef*l) chegar a 0 e -ln(0) virar +inf silenciosamente
    (RuntimeWarning, não exceção) — o usuário precisa ser avisado disso."""
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: None)

    wavenumbers = np.linspace(0, 10, 50)
    coef = np.full_like(wavenumbers, 1000.0)  # suficientemente alto pra saturar exp(-coef*l)
    _patch_voigt_and_convolve(monkeypatch, wavenumbers, coef)
    monkeypatch.setattr(spectragenerator, "LOCAL_TABLE_CACHE",
                        {"hcl": {"data": {"gamma_air": np.array([0.05]), "gamma_self": np.array([0.06])}}})

    generator = SpectraGenerator()
    generator.simulateSpectra("hcl", {"air": 0, "self": 1}, {"l": 2, "p": 1, "T": 296})

    assert len(generator.diluent_warnings) == 1
    assert "saturada" in generator.diluent_warnings[0]
    # o espectro ainda é preenchido (com +inf nos pontos saturados: -ln(0) = +inf), não é resetado
    assert generator.spectra["wavenumbers"] is not None
    assert np.isposinf(generator.spectra["absorption"]).all()


def test_simulateSpectra_no_saturation_warning_for_normal_absorbance(monkeypatch):
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: None)

    wavenumbers = np.linspace(0, 10, 50)
    coef = np.full_like(wavenumbers, 0.5)
    _patch_voigt_and_convolve(monkeypatch, wavenumbers, coef)
    monkeypatch.setattr(spectragenerator, "LOCAL_TABLE_CACHE",
                        {"hcl": {"data": {"gamma_air": np.array([0.05]), "gamma_self": np.array([0.06])}}})

    generator = SpectraGenerator()
    generator.simulateSpectra("hcl", {"air": 0, "self": 1}, {"l": 2, "p": 1, "T": 296})

    assert generator.diluent_warnings == []


def test_plot_does_not_raise_with_valid_spectrum(monkeypatch):
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: None)

    generator = SpectraGenerator()
    generator.spectra["wavenumbers"] = np.linspace(0, 10, 20)
    generator.spectra["absorption"] = np.zeros(20)

    generator.plot()  # não deve levantar exceção


def test_simulateSpectra_convolves_transmittance_not_absorbance(monkeypatch):
    """A fenda deve receber a transmitância (valores em [0, 1]), não a absorbância."""
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: None)
    wavenumbers = np.linspace(0, 10, 101)
    coef = np.full_like(wavenumbers, 0.5)
    monkeypatch.setattr(spectragenerator, "absorptionCoefficient_Voigt", lambda **kw: (wavenumbers, coef))
    received = {}

    def fake_convolve(wn, values, **kw):
        received["values"] = values
        return wn, values, None, None, None

    monkeypatch.setattr(spectragenerator, "convolveSpectrum", fake_convolve)
    generator = SpectraGenerator()
    generator.simulateSpectra("hcl", {"self": 1}, {"l": 2, "p": 1, "T": 296}, convolve=True, resolution=0.1)

    assert np.all((received["values"] >= 0) & (received["values"] <= 1))
    np.testing.assert_allclose(generator.spectra["absorption"], coef * 2)


def test_simulateSpectra_convolution_keeps_original_grid(monkeypatch):
    """Mesmo que a convolução corte as bordas (como o HAPI faz), a saída fica na grade original."""
    monkeypatch.setattr(spectragenerator, "db_begin", lambda db=None: None)
    wavenumbers = np.linspace(0, 10, 101)
    coef = np.full_like(wavenumbers, 0.5)
    monkeypatch.setattr(spectragenerator, "absorptionCoefficient_Voigt", lambda **kw: (wavenumbers, coef))

    def trimming_convolve(wn, values, AF_wing=None, **kw):
        keep = (wn >= wn[0] + AF_wing) & (wn <= wn[-1] - AF_wing)
        return wn[keep], values[keep], None, None, None

    monkeypatch.setattr(spectragenerator, "convolveSpectrum", trimming_convolve)
    generator = SpectraGenerator()
    generator.simulateSpectra("hcl", {"self": 1}, {"l": 2, "p": 1, "T": 296}, convolve=True, resolution=0.5)

    np.testing.assert_allclose(generator.spectra["wavenumbers"], wavenumbers)
    assert len(generator.spectra["absorption"]) == len(wavenumbers)


def test_convolve_on_grid_with_real_hapi_preserves_grid_and_weak_line_area():
    """Com o convolveSpectrum real: grade preservada e área de uma linha fraca conservada."""
    wavenumbers = np.arange(0, 20, 0.01)
    absorbance = 1e-3 * 0.05 ** 2 / ((wavenumbers - 10) ** 2 + 0.05 ** 2)   # linha fraca (lorentziana)
    transmittance = np.exp(-absorbance)
    conv = SpectraGenerator._convolve_on_grid(wavenumbers, transmittance, 1.0, spectragenerator.SLIT_TRIANGULAR, 0.5, 5.0)

    assert conv.shape == wavenumbers.shape
    area_before = np.trapezoid(1 - transmittance, wavenumbers)
    area_after = np.trapezoid(1 - conv, wavenumbers)
    np.testing.assert_allclose(area_after, area_before, rtol=0.02)
    assert (1 - conv).max() < (1 - transmittance).max()   # a linha alarga e o pico cai
