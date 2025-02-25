import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold



def r2Filter(df,limiar):
    return df.query(f'r2 > {limiar}')

def crossValMultioutput(splits = 5, X = None, y = None, model = None):
 

    kf = KFold(n_splits = splits, random_state = 42, shuffle = True)

    folds = kf.split(X, y)

    mse_pressure = 0
    mse_temperature = 0
    mae_pressure = 0
    mae_temperature = 0
    
    for i, (itrain, itest) in enumerate(folds):

        X_train = X.iloc[itrain, :]
        X_test  = X.iloc[itest, :]

        y_train = y.iloc[itrain, :]
        y_test  = y.iloc[itest , :]


        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)



        mse_pressure  += root_mean_squared_error(y_test['pressure'], y_hat[:, 1])
        mae_pressure  += mean_absolute_error(y_test['pressure'], y_hat[:, 1])

        mse_temperature += root_mean_squared_error(y_test['temperature'], y_hat[:, 0])
        mae_temperature += mean_absolute_error(y_test['temperature'], y_hat[:, 0])



    return mse_temperature/splits, mae_temperature/splits, mse_pressure/splits, mae_pressure/splits

def loss(y_real, y_pred):
    mse_pressure  = root_mean_squared_error(y_real['pressure'], y_pred[:, 1])
    mae_pressure  = mean_absolute_error(y_real['pressure'], y_pred[:, 1])

    mse_temperature = root_mean_squared_error(y_real['temperature'], y_pred[:, 0])
    mae_temperature = mean_absolute_error(y_real['temperature'], y_pred[:, 0])

    return mae_temperature, mse_temperature, mae_pressure, mse_pressure


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


def validacaoCruzadaMultioutput(X_train, y_train, model, preprocessor):

    results = {'model': [],
        'mae_temperature': [],
        'mse_temperature': [],
        'mae_pressure':[],
        'mse_pressure':[]}
    

    for name, model in model.items():

            step = [('preprocessor', preprocessor),
                    ('Model', model)]
            
            model_pipe = Pipeline(steps = step)

            print(name)
            mse_temp, mae_temp, mse_pres, mae_pres  = crossValMultioutput(X = X_train, y = y_train, model = model_pipe)

            results['model'].append(name)
            results['mae_temperature'].append(mae_temp)
            results['mse_temperature'].append(mse_temp)
            results['mae_pressure'].append(mae_pres)
            results['mse_pressure'].append(mse_pres)

    return pd.DataFrame(results)





def crossVal(splits = 5, X = None, y = None, model = None):
 

    kf = KFold(n_splits = splits, random_state = 42, shuffle = True)

    folds = kf.split(X, y)

    mae = 0
    mse = 0
    
    for i, (itrain, itest) in enumerate(folds):

        X_train = X.iloc[itrain, :]
        X_test  = X.iloc[itest, :]

        y_train = y.iloc[itrain]
        y_test  = y.iloc[itest]


        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)



        mse  += root_mean_squared_error(y_test, y_hat)
        mae  += mean_absolute_error(y_test, y_hat)


    return mse/splits, mae/splits


def validacaoCruzada(X_train, y_train, model, preprocessor):

    results = {'model': [],
               'mae': [],
               'mse': []
              }
    

    for name, model in model.items():

            step = [('preprocessor', preprocessor),
                    ('Model', model)]
            
            model_pipe = Pipeline(steps = step)

            print(name)
            mse, mae = crossVal(X = X_train, y = y_train, model = model_pipe)

            results['model'].append(name)
            results['mae'].append(mae)
            results['mse'].append(mse)

    return pd.DataFrame(results)