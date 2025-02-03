from typing import Dict, Any, Union, List, Tuple
from scipy.signal import find_peaks, spectrogram
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def transitionClassification(lines: pd.DataFrame) -> pd.DataFrame:

    pbranch = lines.query('branch == "P"').sort_values(by='wavenumber', ascending=False)

    rbranch = lines.query('branch == "R"').sort_values(by='wavenumber',ascending= True)


    pbranch['j'] = [i+1 for i in range(pbranch.shape[0])]
    pbranch['m'] = -1 * pbranch['j']

    rbranch['j'] = [i for i in range(rbranch.shape[0])]
    rbranch['m'] = rbranch['j'] + 1

    lines = pd.concat([pbranch,rbranch])

    return lines.sort_values(by='wavenumber')


def branchClassification(spectra: Dict[str,float] , thresoldBranch: float, findPeaksParams: Dict[str, float]) -> pd.DataFrame:

    lines = pd.DataFrame()

    peaks, _ = find_peaks(spectra['absorption'],height=0.03,width = 10 )

    lines['wavenumber'] = spectra['wavenumbers'][peaks]

    lines['absorption'] = spectra['absorption'][peaks]

    lines['branch'] = np.where(lines['wavenumber'] > thresoldBranch, 'R','P')

    lines = transitionClassification(lines)

    return lines


def plotSpectrum(real: Dict[str,float], lines: pd.DataFrame) -> None:

    plt.plot(real['wavenumbers'],real['absorption'])
    sns.scatterplot(x=lines['wavenumber'],y=lines['absorption'],hue=lines['branch'])


    for _,line in lines.iterrows():
        plt.text(x= line['wavenumber'], y= line['absorption'] + 0.03, s= f"{line['branch']}({line['j']})", fontsize=8)

    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Absorption (A.U)' )
    plt.show()

def _interferogram(spectra: Dict[str, Union[List[float], np.array]]) -> np.array:

    interferogram = np.fft.irfft(spectra['absorption'])

    interferogram = np.fft.ifftshift(interferogram)

    return interferogram

def spectrogramFromSpectra(spectra:  Dict[str, Union[List[float], np.array]])-> Tuple[np.array, np.array , np.ndarray, np.array]:

    interferogram = _interferogram(spectra)

    f, t, sxx = spectrogram(x=interferogram , window= 'blackmanharris' ,mode='magnitude', scaling  = 'spectrum',nperseg = 3000, nfft= 4096, noverlap=500)

    return f, t, 10 * np.log10(sxx + 1e-11), interferogram