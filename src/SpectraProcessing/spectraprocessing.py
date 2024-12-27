from typing import Dict, Any
from scipy.signal import find_peaks
import pandas as pd
import numpy as np

def branchNumber(spectra: Dict[str,float] , thresoldBranch: float, findPeaksParams: Dict[str, float]) -> pd.DataFrame:

    lines = pd.DataFrame()

    peaks, _ = find_peaks(spectra['absorption'],width=10,height=0.02)

    lines['center'] = spectra['wavenumbers'][peaks]

    lines['intensity'] = spectra['absorption'][peaks]

    lines['branch'] = np.where(lines['center'] > thresoldBranch, 'R','P')

    return lines