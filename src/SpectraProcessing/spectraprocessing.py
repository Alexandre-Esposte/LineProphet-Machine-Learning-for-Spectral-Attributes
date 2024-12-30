from typing import Dict, Any
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def transitionClassification(lines: pd.DataFrame) -> pd.DataFrame:

    pbranch = lines.query('branch == "P"').sort_values(by='center', ascending=False)

    rbranch = lines.query('branch == "R"').sort_values(by='center',ascending= True)


    pbranch['j'] = [i+1 for i in range(pbranch.shape[0])]
    pbranch['m'] = -1 * pbranch['j']

    rbranch['j'] = [i for i in range(rbranch.shape[0])]
    rbranch['m'] = rbranch['j'] + 1

    lines = pd.concat([pbranch,rbranch])

    return lines.sort_values(by='center')


def branchClassification(spectra: Dict[str,float] , thresoldBranch: float, findPeaksParams: Dict[str, float]) -> pd.DataFrame:

    lines = pd.DataFrame()

    peaks, _ = find_peaks(spectra['absorption'],height=0.01)

    lines['center'] = spectra['wavenumbers'][peaks]

    lines['intensity'] = spectra['absorption'][peaks]

    lines['branch'] = np.where(lines['center'] > thresoldBranch, 'R','P')

    lines = transitionClassification(lines)

    return lines


def plotSpectrum(real: Dict[str,float], lines: pd.DataFrame) -> None:

    plt.plot(real['wavenumbers'],real['absorption'])
    sns.scatterplot(x=lines['center'],y=lines['intensity'],hue=lines['branch'])


    for _,line in lines.iterrows():
        plt.text(x= line['center'] - 5, y= line['intensity']+0.02, s= f"{line['branch']}({line['j']})", fontsize=8)

    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Absorption (A.U)' )
    plt.show()