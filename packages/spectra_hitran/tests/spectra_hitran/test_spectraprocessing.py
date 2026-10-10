import numpy as np

from SpectraHitran.SpectraProcessing import spectraprocessing


def test_detectLines_finds_peak_centers_and_heights():
    wavenumbers = np.linspace(0, 100, 1001)
    absorption = np.zeros_like(wavenumbers)

    peak_positions = [20.0, 40.0, 60.0, 80.0]
    for pos in peak_positions:
        idx = np.argmin(np.abs(wavenumbers - pos))
        absorption[idx] = 1.0

    spectra = {"wavenumbers": wavenumbers, "absorption": absorption}
    find_peaks_params = {"height": 0.5, "width": 0, "distance": 1}

    lines = spectraprocessing.detectLines(spectra, find_peaks_params)

    assert len(lines) == len(peak_positions)
    assert set(lines.columns) == {"wavenumber", "absorption"}
    np.testing.assert_allclose(sorted(lines["wavenumber"]), peak_positions, atol=0.2)


def test_detectLines_returns_empty_dataframe_when_no_peak_matches():
    wavenumbers = np.linspace(0, 100, 1001)
    absorption = np.zeros_like(wavenumbers)
    spectra = {"wavenumbers": wavenumbers, "absorption": absorption}

    lines = spectraprocessing.detectLines(spectra, {"height": 0.5, "width": 0, "distance": 1})

    assert len(lines) == 0
