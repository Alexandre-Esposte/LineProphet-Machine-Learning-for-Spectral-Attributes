import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from lmfit import minimize, Parameters , Minimizer, report_fit
from lmfit.models import *
from lmfit.model import ModelResult
from functools import partial
import optuna
from optuna.trial import Trial


optuna.logging.set_verbosity(optuna.logging.WARNING)



def filterLine(spectra: Dict[str, float], linecenter: float, lineintensity: float, intensidade_relativa: float) -> Tuple[List[float], List[float]]:

    x_espec = list(spectra['wavenumbers'])
    y_espec = list(spectra['absorption'])

    intensidade = lineintensity

    centro = linecenter

    pontosx = []
    pontosy = []

    # intensidade do centro e dos pontos adjacentes
    icentro = intensidade
    iponto =  intensidade# só inicializando a variável, preciso disso para entrar no while (1 > intensidade_relativa)

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

    icentro = intensidade
    iponto  = intensidade
    i = 1
    while (iponto / icentro >= intensidade_relativa):

        
        iponto = y_espec[y_espec.index(icentro) + i]


        if iponto / icentro >= intensidade_relativa:
            direita_x.insert(i, x_espec[x_espec.index(centro) + i])
            direita_y.insert(i, y_espec[y_espec.index(icentro) + i])


        i = i + 1

    pontos_x = esquerda_x + [centro] + direita_x

    pontos_y = esquerda_y + [icentro] + direita_y

    #pontos_x, pontos_y = interpolar(pontos_x, pontos_y, qte=3000)

    return pontos_x, pontos_y



def resultFit(result: ModelResult, successful: bool) -> Dict[str, float]:

    params = dict()

    if successful:
        params['evals'] = result.nfev
        params['npoints'] = result.ndata
        params['r2'] = result.rsquared
        
        params['amplitude'] = result.params['amplitude'].value
        params['amplitude_stderr'] = result.params['amplitude'].stderr

        params['center'] = result.params['center'].value
        params['center_std'] = result.params['center'].stderr

        params['sigma'] = result.params['sigma'].value
        params['sigma_stderr'] = result.params['sigma'].stderr


        params['gamma'] = result.params['gamma'].value
        params['gamma_stderr'] = result.params['gamma'].stderr

        params['fwhm'] = result.params['fwhm'].value
        params['fwhm_stderr'] = result.params['fwhm'].stderr


        params['height'] = result.params['height'].value
        params['height_stderr'] = result.params['height'].stderr

        return params
    


    params['evals'] = result.nfev
    params['npoints'] = result.ndata
    params['r2'] = result.rsquared
        
    params['amplitude'] = np.null
    params['amplitude_stderr'] = np.null

    params['center'] = np.null
    params['center_std'] = np.null

    params['sigma'] = np.null
    params['sigma_stderr'] = np.null


    params['gamma'] = np.null
    params['gamma_stderr'] = np.null

    params['fwhm'] = np.null
    params['fwhm_stderr'] = np.null


    params['height'] = np.null
    params['height_stderr'] = np.null

    return params


def singleLineFit(data, ex, chute_centro: float, chute_sigma: float, chute_gamma: float, vgamma: bool, vsigma:bool) -> Tuple[np.array, ModelResult, Dict[str, float], bool]:

    data = np.array(data)
    ex = np.array(ex)

    successful = True
 
    modelo = VoigtModel()
    pars = modelo.guess(data, x=ex)
 
    pars['gamma'].set(value = chute_gamma, vary = vgamma, expr = '', min = 0,max = 5)
    pars['sigma'].set(value = chute_sigma, vary = vsigma, expr = '', min = 0,max = 5)

    pars['amplitude'].set(value = 5e-3, vary = True, expr = '', min = 0, max = 1)
    pars['center'].set(value = chute_centro, vary = True, expr = '')



    try:
        result = modelo.fit(data, pars, x=ex, nan_policy='propagate')

        final = data + result.residual

    except:
        successful = False
        final = result = 0

    params = resultFit(result, successful)
    

    return final, result, params, successful


def multipleLineFit(spectra: Dict[str, float], lines: pd.DataFrame, linepercentage: float = 0.01) -> pd.DataFrame:

    results = pd.DataFrame()
    for indice,line in lines.iterrows():
        
        center = line['wavenumber']
        absorption = line['absorption']

        xfilterline, yfilterline = filterLine(spectra, center, absorption, linepercentage)

        chute= 8e-3
        final, result, params, successful = singleLineFit(yfilterline, xfilterline, center, chute, chute, vgamma = True, vsigma = True)

        # se r² for menor que 0.99, será efetuado uma busca pelo chute inicial ótimo para melhorar o ajuste
        if result.rsquared < 0.99:
            print(f"{line['branch']}({line['j']}), {line['temperature']}K, {line['pressure']}atm -> {result.rsquared}-------> R² insuficiente, habilitando o optuna")
            best_params , best_value = optimize(xfilterline, yfilterline, center)
            chute_sigma = best_params['chute_sigma']
            chute_gamma = best_params['chute_gamma']
            final, result, params, successful = singleLineFit(yfilterline, xfilterline, center, chute_sigma, chute_gamma, vgamma = True, vsigma = True)

            print(f"{line['branch']}({line['j']}), {line['temperature']}K, {line['pressure']}atm -> {result.rsquared}-------> Melhor R² após o optuna")

        else:
            print(f"{line['branch']}({line['j']}), {line['temperature']}K, {line['pressure']}atm -> {result.rsquared}")

        
        results = pd.concat([results, pd.DataFrame({key: [value] for key, value in params.items()})], axis=0)

    return results.reset_index(drop=True)


def objective(trial: Trial, xfilterline: List[float], yfilterline: List[float], center: float, **kwargs: Dict[str, Any]) -> float:
    
    # Sugere um valor para o chute inicial
    chute_sigma = trial.suggest_float('chute_sigma', 0.000001, 1)
    chute_gamma = trial.suggest_float('chute_gamma', 0.000001, 1)

    # Executa o ajuste com os parâmetros fornecidos
    final, result, params, successful = singleLineFit(yfilterline, xfilterline, center, chute_sigma, chute_gamma, vgamma=True, vsigma=True)

    # Retorna o r² como métrica para o otimizador
    return 1 - result.rsquared  # Minimizar a diferença para o melhor ajuste (r² = 1)

# Exemplo de como usar
def optimize(xfilterline: List[float], yfilterline: List[float], center: float) -> Tuple[Dict[str, float], float]:
    
    # Define a função objetivo com parâmetros fixos
    objective_partial = partial(objective, xfilterline=xfilterline, yfilterline=yfilterline, center=center)

    # Configura o estudo do Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_partial, n_trials=50)

    # Resultado do melhor ajuste
    best_params = study.best_params
    best_value = study.best_value

    return best_params, best_value