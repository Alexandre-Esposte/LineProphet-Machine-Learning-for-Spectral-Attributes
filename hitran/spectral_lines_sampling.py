import numpy as np
import pandas as pd
from lmfit import minimize, Parameters , Minimizer, report_fit
from lmfit.models import *
from scipy import interpolate
from scipy.stats import kurtosis, skew


def interpolar(x, y, qte=50):
    f = interpolate.interp1d(x, y, kind='cubic')

    xnew = np.linspace(x[0], x[-1], qte)
    ynew = f(xnew)

    return xnew, ynew

def fit_raia(data, ex, chute_centro, chute_sigma= 0.01, chute_gamma = 0.01, vgamma = True, vsigma = True):

    data = np.array(data)
    ex = np.array(ex)

    successful = True
    
    #print(f"data: {type(data)}, ex: {type(ex)}, chute_centro: {type(chute_centro)}, chute_sigma: {type(chute_sigma)}, chute_gama: {type(chute_gamma)}")
   
    modelo = VoigtModel()
    pars = modelo.guess(data, x=ex)

    pars['gamma'].set(value=chute_gamma, vary=vgamma, expr='', min = 0)
    pars['sigma'].set(value=chute_sigma, vary=vsigma, expr='', min = 0)

    pars['amplitude'].set(value=1, vary=True, expr='', min = 0)
    pars['center'].set(value=chute_centro, vary=True, expr='')

    try:
        result = modelo.fit(data, pars, x=ex, nan_policy='propagate')

        final = data + result.residual

    except:
        successful = False
        final = result = 0

    print(f'Sucesso ajuste: {successful}')
    return final, result, successful


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

def separa_pontos_manual(intensidade_relativa = 0.05 ,centro = 0,linha = 0, spec_mensurado = None):


    x_espec = list(spec_mensurado['wavenumber'])
    y_espec = list(spec_mensurado['intensity'])
    

    pontosx = []
    pontosy = []

    # intensidade do centro e dos pontos adjacentes
    icentro = float(linha)
    iponto = float(linha)# só inicializando a variável, preciso disso para entrar no while (1 > intensidade_relativa)

    esquerda_x = []
    esquerda_y = []

    direita_x = []
    direita_y = []


    i = 1

    # para a esquerda
    while (iponto / icentro >= intensidade_relativa):


        # aqui é de fato o valor da intensidade dos pontos adjacentes
        iponto = y_espec[y_espec.index(icentro) - i]



        if iponto / icentro >= intensidade_relativa:
            esquerda_x.insert(-i, x_espec[x_espec.index(centro) - i])
            esquerda_y.insert(-i, y_espec[y_espec.index(icentro) - i])


        i = i + 1

    # Para direita

    icentro = float(linha)
    iponto  = float(linha)
    i = 1
    while (iponto / icentro >= intensidade_relativa):

        iponto = y_espec[y_espec.index(icentro) + i]


        if iponto / icentro >= intensidade_relativa:
            direita_x.insert(i, x_espec[x_espec.index(centro) + i])
            direita_y.insert(i, y_espec[y_espec.index(icentro) + i])


        i = i + 1

    pontos_x = esquerda_x + [centro] + direita_x

    pontos_y = esquerda_y + [icentro] + direita_y


   
    ylinha = derivar(pontos_x,pontos_y)
    ylinha2 = derivar(pontos_x,ylinha)
    

    return {'ex':pontos_x, 'ey':pontos_y} 



if __name__ == '__main__':
    print('Sucessful')