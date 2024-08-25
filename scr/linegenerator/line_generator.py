import numpy as np
from scipy.special import voigt_profile
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from . import linefit

def gaussian(x, sigma, A, center): 
  return (A/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x-center)**2  / (2*sigma**2)) )

def lorentzian(x, gamma, A, center):
   return (A/np.pi) * ( gamma/ ( (x - center)**2 + (gamma**2) ))

def derivar(x,y):
  derivada = []

  for indice , _ in enumerate(x):

      if indice == 0:
        # Derivada progressiva
        derivada.append(  (y[indice+1] - y[indice])/(x[indice+1] - x[indice]) )

      elif indice == len(x)-1:
        # Derivada regressiva
        derivada.append(  (y[indice] - y[indice-1])/(x[indice] - x[indice-1]) )

      else:
        # Derivada central

        derivada.append( (y[indice+1] - y[indice-1])/(x[indice+1] - x[indice-1])  )
  return(derivada)

def statistics_calculus(wavenumbers, intensities):
    
    data = {}

    data['wv_mean'] = np.mean(wavenumbers)
    data['wv_std'] = np.std(wavenumbers)
    data['wv_min'] = np.min(wavenumbers)
    data['wv_max'] = np.max(wavenumbers)
    data['wv_kurt'] = kurtosis(wavenumbers)
    data['wv_skew'] = skew(wavenumbers)

    data['int_mean'] = np.mean(intensities)
    data['int_std'] = np.std(intensities)
    data['int_max'] = np.max(intensities)
    data['int_min'] = np.min(intensities)
    data['int_kurt'] = kurtosis(intensities)
    data['int_skew'] = skew(intensities)


    return data

def voigt_profile_generator():
    """Gera um perfil voigt através de parâmetros aleatórios"""

    sigma = np.random.exponential(0.1)
    gamma = np.random.exponential(0.1)

    noise_mean = 0

    noise_std = np.abs( np.random.normal(0,1e-4))


    fator = (sigma+gamma)/2
    f = np.random.randint(5,10)
    #fator = (2*sigma*gamma)/(sigma+gamma)

    fl = 2 * gamma
    fg = 2 * sigma * np.sqrt(2 * np.log(2))
    fwhm = 0.5346*fl + np.sqrt(0.2166*fl**2 + fg**2)

    intensity_factor = np.random.uniform(0,1)
    center = np.random.uniform(1000,30000)

    n_points = 50

    x = np.linspace(center-fator*f,center+fator*f, n_points)
    y = intensity_factor * voigt_profile(x - center, sigma,gamma)
    y_with_noise = y + np.random.normal(noise_mean ,noise_std, n_points)

    ylinha = derivar(x,y_with_noise)
    ylinha2 = derivar(x,ylinha)
    
    statistics = statistics_calculus(x,y)


    #final, result, succesful = linefit.fit_raia(data = y_with_noise,ex = x,chute_centro = center,chute_sigma = 0.5, chute_gamma=0.5, model='Voigt')


    return {'x':list(x),
            'y':list(y_with_noise),
            'primeira_derivada': list(ylinha),
            'segunda_derivada': list(ylinha2),         
            'sigma':sigma,
            'gamma':gamma,
            'center':center,
            'intensity_factor':intensity_factor,
            'fwhm':fwhm,
            'statistics':statistics,
            'curve_type': 'voigt'}


def gaussian_profile_generator():

    sigma = np.random.exponential(0.1)

    noise_mean = 0

    noise_std = np.abs( np.random.normal(0,1e-4))
    
    fator = sigma / 2
    f = np.random.randint(5,10)
    
    intensity_factor = np.random.uniform(0,1)
    center = np.random.uniform(1000,30000)

    fwhm = 2.3548* sigma
    
    n_points = 50

    x = np.linspace(center-fator*f,center+fator*f, n_points)
    y = gaussian(x,sigma,intensity_factor, center)
    y_with_noise = y + np.random.normal(noise_mean,noise_std, n_points)

    ylinha = derivar(x,y_with_noise)
    ylinha2 = derivar(x,ylinha)
    
    statistics = statistics_calculus(x,y)
    #final, result, succesful = linefit.fit_raia(data = y_with_noise,ex = x,chute_centro = center,chute_sigma = 0.5, chute_gamma=0.5, model='Gaussian')


    
    return {'x':list(x),
            'y':list(y_with_noise),
            'primeira_derivada': list(ylinha),
            'segunda_derivada': list(ylinha2),         
            'sigma':sigma,
            'gamma':0,
            'center':center,
            'intensity_factor':intensity_factor,
            'fwhm':fwhm,
            'statistics':statistics,
            'curve_type': 'gaussian'}


def lorentzian_profile_generator():

    gamma = np.random.exponential(0.1)

    noise_mean = 0

    noise_std = np.abs( np.random.normal(0,1e-4))
    
    fator = gamma / 2
    f = np.random.randint(5,10)
    intensity_factor = np.random.uniform(0,1)
    center = np.random.uniform(1000,30000)

    fwhm = 2*gamma
    n_points = 50

    x = np.linspace(center-fator*f,center+fator*f, n_points)
    y = lorentzian(x,gamma,intensity_factor, center)
    y_with_noise = y + np.random.normal(noise_mean,noise_std, n_points)

    ylinha = derivar(x,y_with_noise)
    ylinha2 = derivar(x,ylinha)
    
    statistics = statistics_calculus(x,y)
    #final, result, succesful = linefit.fit_raia(data = y_with_noise,ex = x,chute_centro = center,chute_sigma = 0.5, chute_gamma=0.5, model='Lorentz')


    

    return {'x':list(x),
            'y':list(y_with_noise),
            'primeira_derivada': list(ylinha),
            'segunda_derivada': list(ylinha2),
            'sigma': 0,            
            'gamma':gamma,
            'center':center,
            'intensity_factor':intensity_factor,
            'fwhm':fwhm,
            'statistics':statistics,
            'curve_type': 'lorentz'}

if __name__ == '__main__':
    print(lorentzian_profile_generator())