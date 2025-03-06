import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold
import os


def r2Filter(df,limiar):
    return df.query(f'r2 > {limiar}')



def bootstrap_normal_ci(predictions, n_bootstrap=10000, ci= 95):
    """
    Calcula um intervalo de confiança normal para a média das previsões usando bootstrap.

    Parameters:
        predictions (array-like): Lista com as estimativas do modelo para um espectro.
        n_bootstrap (int): Número de amostras bootstrap.
        ci (float): Nível de confiança (ex: 95 para um intervalo de 95%).

    Returns:
        tuple: (média estimada, limite inferior, limite superior)
    """
    medias_bootstrap = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(predictions, size=len(predictions), replace=True)
        medias_bootstrap.append(np.mean(sample))

    mean_pred = np.mean(medias_bootstrap)  # Média das médias bootstrap
    std_pred = np.std(medias_bootstrap, ddof=1)  # Desvio padrão das médias bootstrap
    se_pred = std_pred / np.sqrt(n_bootstrap)

    z_score = norm.ppf(1 - (1 - ci / 100) / 2)  # Quantil da normal (ex: 1.96 para 95%)

    lower = mean_pred - z_score * se_pred
    upper = mean_pred + z_score * se_pred

    return mean_pred, lower, upper



def crossVal(splits = 5, X = None, y = None, model = None):
 

    kf = KFold(n_splits = splits, random_state = 42, shuffle = True)

    folds = kf.split(X, y)

    mae = 0
    mse = 0
    mape = 0
    
    for i, (itrain, itest) in enumerate(folds):

        X_train = X.iloc[itrain, :]
        X_test  = X.iloc[itest, :]

        y_train = y.iloc[itrain]
        y_test  = y.iloc[itest]


        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)


        mse  += root_mean_squared_error(y_test, y_hat)
        mae  += mean_absolute_error(y_test, y_hat)
        mape += np.mean(np.abs((y_test - y_hat) / y_test)) * 100

    return mse/splits, mae/splits, mape/splits


def validacaoCruzada(X_train, y_train, model, preprocessor):

    results = {'model': [],
               'mae': [],
               'mse': [],
               'mape': []
              }
    

    for name, model in model.items():

            step = [('preprocessor', preprocessor),
                    ('Model', model)]
            
            model_pipe = Pipeline(steps = step)

            print(name)
            mse, mae, mape = crossVal(X = X_train, y = y_train, model = model_pipe)

            results['model'].append(name)
            results['mae'].append(mae)
            results['mse'].append(mse)
            results['mape'].append(mape)

    return pd.DataFrame(results)


def eng_features(df):
     
    # proporções e razoes
    df['gamma_sigma_ratio'] = df['gamma'] / df['sigma']
    df['amplitude_sigma_ratio'] = df['amplitude'] / df['sigma']
    df['amplitude_gamma_ratio'] = df['amplitude'] / df['gamma']
    df['fwhm_height_ratio'] = df['fwhm'] / df['height']
    df['sigma_plus_gamma'] = df['sigma'] + df['gamma']

    # features nao lineares
    df['sigma_gamma_interaction'] = df['sigma'] * df['gamma']
    df['amplitude_fwhm_interaction'] = df['amplitude'] * df['fwhm']
    df['sigma2'] = df['sigma']**2
    df['gamma2'] = df['gamma']**2

    df['ratio_lg'] = df['gamma']/ (df['sigma'] + df['gamma'])


    return df
    

def integrar_dados():
    arqs = os.listdir('../database/lines/train')

    data = pd.DataFrame()
    for arq in arqs:
        data = pd.concat([data,pd.read_csv("../database/lines/train/"+arq)])

    data = data.reset_index(drop=True)
    data = data.drop(columns=['Unnamed: 0'])
    print(data.shape)
    print('Dados integrados.')
    data.to_csv('../database/lines/lines_train.csv', index=None)