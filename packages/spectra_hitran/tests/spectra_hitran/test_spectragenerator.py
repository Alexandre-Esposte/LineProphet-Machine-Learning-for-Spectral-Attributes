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

    # fórmula original (Beer-Lambert decádica), intocada
    expected_absorption = -np.log10(np.exp(-coef * 2))
    np.testing.assert_allclose(generator.spectra["absorption"], expected_absorption)
    np.testing.assert_allclose(generator.spectra["wavenumbers"], wavenumbers)


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
    faz transmitance = exp(-coef*l) chegar a 0 e -log10(0) virar +inf silenciosamente
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
    # o espectro ainda é preenchido (com +inf nos pontos saturados: -log10(0) = +inf), não é resetado
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
