from lmfit import minimize, Parameters , Minimizer, report_fit
from lmfit.models import *


def fit_raia(data, ex, chute_centro, chute_sigma, chute_gamma, model, vgamma=True, vsigma=True):

    data = np.array(data)
    ex = np.array(ex)

    successful = True

    #print(f"data: {type(data)}, ex: {type(ex)}, chute_centro: {type(chute_centro)}, chute_sigma: {type(chute_sigma)}, chute_gama: {type(chute_gamma)},model: {type(model)}")

    if model == 'Gaussian':
        #print("Ajustando com a Gaussiana")
        modelo = GaussianModel()
        pars = modelo.guess(data, x=ex)

        pars['sigma'].set(value=chute_sigma, vary=True, expr='')
        pars['amplitude'].set(value=1, vary=True, expr='')
        pars['center'].set(value=chute_centro, vary=True, expr='')

    elif model == 'Lorentz':
        #print("Ajustando com a Lorentz")
        modelo = LorentzianModel()
        pars = modelo.guess(data, x=ex)

        pars['sigma'].set(value=chute_sigma, vary=True, expr='')
        pars['amplitude'].set(value=1, vary=True, expr='')
        pars['center'].set(value=chute_centro, vary=True, expr='')

    elif model == 'Voigt':
        #print("Ajustando com a Voigt")
        modelo = VoigtModel()
        pars = modelo.guess(data, x=ex)

        pars['gamma'].set(value=chute_gamma, vary=vgamma, expr='')
        pars['sigma'].set(value=chute_sigma, vary=vsigma, expr='')

        pars['amplitude'].set(value=1, vary=True, expr='')
        pars['center'].set(value=chute_centro, vary=True, expr='')



    try:
        result = modelo.fit(data, pars, x=ex, nan_policy='propagate')

        final = data + result.residual

    except:
        successful = False
        final = result = 0

    return final, result, successful