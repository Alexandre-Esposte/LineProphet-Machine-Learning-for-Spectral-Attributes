import numpy as np
from scipy.special import voigt_profile
import matplotlib.pyplot as plt


def statistics_calculus(wavenumbers, intensities):
    
    data = {}

    data['wv_mean'] = np.mean(wavenumbers)
    data['wv_std'] = np.std(wavenumbers)

    data['int_mean'] = np.mean(intensities)
    data['int_std'] = np.std(intensities)
    data['int_max'] = np.max(intensities)

    return data

def voigt_profile_generator():
    """Gera um perfil voigt através de parâmetros aleatórios"""

    sigma = np.random.uniform(0,10)
    gamma = np.random.uniform(0,10)

    fator = (sigma+gamma)/2
    #fator = (2*sigma*gamma)/(sigma+gamma)

    fl = 2 * gamma
    fg = 2 * sigma * np.sqrt(2 * np.log(2))
    fwhm = 0.5346*fl + np.sqrt(0.21666*fl**2+ fg**2)

    intensity_factor = np.random.uniform(0,10)
    center = np.random.uniform(1000,50000+1)

    n_points = 1000

    x = np.linspace(center-fator*10,center+fator*10, n_points)
    y = intensity_factor * voigt_profile(x - center, sigma,gamma)

    statistics = statistics_calculus(x,y)

    #print(f"\nsigma: {sigma}\ngamma: {gamma}\ncenter: {center}\nintensity: {intensity_factor}\nfwhm: {fwhm}\ndata: {data}")

    plt.plot(x,y)
    plt.show()

    return {'sigma':sigma,'gamma':gamma,'center':center,'intensity_factor':intensity_factor,'fwhm':fwhm,'statistics':statistics}


if __name__ == '__main__':
    print(voigt_profile_generator())