import pandas as pd
import numpy as np
from typing import Dict, Tuple
from lmfit import minimize, Parameters , Minimizer, report_fit
from lmfit.models import *
from lmfit.model import ModelResult



def resultFit(result: ModelResult, successful: bool) -> Dict[str, float]:

    params = dict()

    if successful:
        params['evals'] = result.nfev
        params['npoints'] = result.ndata
        params['r2'] = result.rsquared
        
        params['slope'] = result.params['slope'].value
        params['slope_stderr'] = result.params['slope'].stderr

        params['intercept'] = result.params['intercept'].value
        params['intercept_std'] = result.params['intercept'].stderr

        return params
    

    params['evals'] = result.nfev
    params['npoints'] = result.ndata
    params['r2'] = result.rsquared
        
    params['slope'] = np.null
    params['slope_stderr'] = np.null

    params['intercept'] = np.null
    params['intercept_std'] = np.null

    return params

def linearFit(data, ex) -> Tuple[np.array, ModelResult, Dict[str, float], bool]:

    data = np.array(data)
    ex = np.array(ex)

    successful = True
 
    modelo = LinearModel()
    pars = modelo.guess(data, x=ex)

    pars['intercept'].set(value = 0, vary = False, expr = '')
 
    try:
        result = modelo.fit(data, pars, x=ex, nan_policy='propagate')

        final = data + result.residual

    except:
        successful = False
        final = result = 0

    params = resultFit(result, successful)
    

    return final, result, params, successful


def selfBroadening(lines: pd.DataFrame) -> pd.DataFrame:

    for m in lines['m']:
        lines_for_every_m = lines.query(f"m == {m}").reset_index(drop=True).sort_values(by = "pressure")