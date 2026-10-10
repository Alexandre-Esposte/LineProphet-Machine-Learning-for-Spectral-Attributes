from typing import Dict, Optional
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Spectrum = Dict[str, Optional[np.ndarray]]


def detectLines(spectra: Spectrum, findPeaksParams: Dict[str, float]) -> pd.DataFrame:
    """Detecta picos de absorção no espectro.

    Args:
        spectra: `{'wavenumbers': array, 'absorption': array}`, ex. saída de
            `SpectraGenerator.simulateSpectra`.
        findPeaksParams: `{'height', 'width', 'distance'}` repassados a
            `scipy.signal.find_peaks`.

    Returns:
        DataFrame com colunas `wavenumber`/`absorption`, ordenado por `wavenumber`.

    Exemplo:
        >>> lines = detectLines(gen.spectra, {'height': 0.01, 'width': 1, 'distance': 5})
    """
    peaks, _ = find_peaks(spectra['absorption'],
                          height=findPeaksParams['height'],
                          width=findPeaksParams['width'],
                          distance=findPeaksParams['distance'])

    lines = pd.DataFrame()
    lines['wavenumber'] = spectra['wavenumbers'][peaks]
    lines['absorption'] = spectra['absorption'][peaks]

    return lines.sort_values(by='wavenumber').reset_index(drop=True)


def plotSpectrum(real: Spectrum, lines: pd.DataFrame) -> None:
    """Plota o espectro com os centros de linha detectados marcados.

    Args:
        real: `{'wavenumbers': array, 'absorption': array}`.
        lines: saída de `detectLines` (colunas `wavenumber`/`absorption`).

    Exemplo:
        >>> plotSpectrum(gen.spectra, lines)
    """
    plt.plot(real['wavenumbers'], real['absorption'])
    plt.scatter(lines['wavenumber'], lines['absorption'], color='tab:red', zorder=3)

    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Absorption (A.U)')
    plt.show()

