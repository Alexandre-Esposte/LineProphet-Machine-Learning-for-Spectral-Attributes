import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline


class ModelCrafter:

    def __init__(self) -> None:
        
        self.models = dict()

        self.results = dict()

        self.results_per_fold = dict()

        self.kf = KFold(n_splits=5) 

    def AddModel(self, modelos : list = []) -> None:
        """Método para adicionar modelos ao objeto. A estrutura é uma lista de tuplas onde a tupla segue o seguinte esquema: (nome do modelo, modelo instanciado)"""
        
        for modelo in modelos:
            self.models[modelo[0]] = modelo[1]

    def RemoveModel(self, nome: str = None, tipo: str = None) -> None:
        """Remove modelos do objeto"""
        
        if tipo == 'all':
            self.models=dict()
            return

        del self.models[nome]

          
    def Validacao(self, X_train: pd.DataFrame = None, X_test: pd.DataFrame = None , y_train: pd.Series = None, y_test: pd.Series = None, pipe: Pipeline = None):
       
        if len(self.models) == 0:
            return "Nenhum modelo adicionado na estrutura"

        resultados = {'modelo':[],'mae_treino':[], 'mae_teste':[], 'rms_treino':[], 'rms_teste':[],'mape_treino':[],'mape_teste':[]}

        for aux in self.models.items():
            nome_modelo = aux[0]
            modelo = aux[1]

            if len(pipe.steps) > 1:
                pipe.steps.pop()

            
            print(f"-----{nome_modelo}-----")
            pipe.steps.append(("Model",modelo))
            modelo = pipe

            modelo.fit(X_train,y_train)

            pred_train = modelo.predict(X_train)
            pred_test = modelo.predict(X_test)


            mae_train  =  mean_absolute_error(y_train,pred_train)
            rms_train =  mean_squared_error(y_train,pred_train, squared = False)
            mape_train = mean_absolute_percentage_error(y_train,pred_train)

            mae_test = mean_absolute_error(y_test,pred_test)
            rms_test = mean_squared_error(y_test, pred_test, squared= False)
            mape_test = mean_absolute_percentage_error(y_test, pred_test)

            
            

            resultados['modelo'].append(nome_modelo)
            resultados['mae_treino'].append(mae_train)
            resultados['mae_teste'].append(mae_test)
            resultados['rms_treino'].append(rms_train)
            resultados['rms_teste'].append(rms_test)
            resultados['mape_treino'].append(mape_train)
            resultados['mape_teste'].append(mape_test)

        return pd.DataFrame(resultados)
        

    def ValidacaoCruzada(self, X: np.ndarray, y: np.array, pipe: Pipeline = None) -> None:
        """Treina todos os modelos inseridos no objeto através de validação cruzada"""
		
        if len(self.models) == 0:
            return "Nenhum modelo adicionado na estrutura"
        
        for aux in self.models.items():
            nome_modelo = aux[0]
            modelo = aux[1]

            if len(pipe.steps) > 1:
                pipe.steps.pop()

            mae = 0
            rms = 0
            mape=0
            
            resultados_aux = []

            print(f"-----{nome_modelo}-----")
            pipe.steps.append(("Model",modelo))
            modelo = pipe
            for i, (train_index, test_index) in enumerate(self.kf.split(X)):
                #print(f"Fold {i}:")
                
                #print(f"  Train: index={train_index}")
                #print(f"  Test:  index={test_index}")

                X_train = X.loc[train_index,:]
                y_train = y.loc[train_index]

                
                X_test = X.loc[test_index,:]
                y_test = y.loc[test_index]

                
                modelo.fit(X_train,y_train)

                predito = modelo.predict(X_test)
  
                
                resultados_aux.append((y_test,predito)) 
                
                mae  += mean_absolute_error(y_test,predito)
                mape += mean_absolute_percentage_error(y_test,predito)
                rms += mean_squared_error(y_test,predito,squared = False)
               

                
            self.results[nome_modelo]=[mae/5, mape/5 ,rms/5]
            self.results_per_fold[nome_modelo] = resultados_aux

        return self._gerar_resultado()

    def _gerar_resultado(self) -> None:
        """Gera os resultados em uma estrutura DataFrame"""
        
        indices = ['mae','mape','rms']
        #display(pd.DataFrame(self.results,index=indices).T)
        return pd.DataFrame(self.results,index=indices).T