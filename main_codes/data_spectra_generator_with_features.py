from SpectraHitran.SpectraGenerator.spectragenerator import SpectraGenerator
from SpectraHitran.SpectraProcessing.spectraprocessing import branchClassification, plotSpectrum
from SpectraHitran.SpectraFit.spectrafit import multipleLineFit

import numpy as np

import pandas as pd
from multiprocessing import Pool
import os
from scipy.stats import qmc
import matplotlib.pyplot as plt


def process_temperature_pressure(args):
    """Função para simular espectro e ajustar linhas para uma combinação de temperatura e pressão."""
    temperature, pressure, optical_length = args
    s = SpectraGenerator()
    s.downloadMolecule('hcl',(52,53),(5200,5900))
    print(f'-----------Fitting for {temperature-273.15}°C and {pressure*1013} mbar, {optical_length} cm, process: {os.getpid()}-----------')
    
    # Simulando o espectro
    s.simulateSpectra('hcl', {'air': 0, 'self': 1}, {'l': optical_length, 'p': pressure, 'T': temperature})
    #s.spectra['absorption'] = s.spectra['absorption']/ np.max(s.spectra['absorption'])
    
    # Simulando ruidos aleatorios no espectro
    valor = np.random.uniform(0,1)
    ruido_std = valor * np.mean(s.spectra['absorption'])  
    s.spectra['absorption'] = np.random.normal(s.spectra['absorption'], ruido_std)

    if pressure*1013 < 80:
        height = 0.02

    elif pressure*1013 < 300:
        height = 0.08

    elif pressure*1013 <500:
        height = 0.1

    elif pressure*1013 < 600:
        height = 0.2
    
    else:
        height = 0.3
 
    print(f'HEIGHT---------------------------------------->{height}')
    # Organizando as linhas
    lines = branchClassification(s.spectra, findPeaksParams={'height': height, 'width': 5, 'distance': None},thresoldBranch=5665)
    lines['pressure'] = pressure
    lines['temperature'] = temperature


    # Ajustando múltiplas linhas
    fitted_params = multipleLineFit(s.spectra, lines, 0.01)
    lines = pd.concat([lines, fitted_params], axis=1)



    lines.to_csv(f'../database/lines/train/{temperature}_{pressure}.csv')

    return 1


def main():

    # Configurações
    processos = 8
    n_amostras = 24
    pressao_min, pressao_max = 20, 800  # mbar
    caminho_min, caminho_max = 8, 10  # cm

    temp_min, temp_max = 290, 318  


    temperatures = np.random.uniform(temp_min, temp_max, n_amostras)
    pressures = np.random.uniform(pressao_min,pressao_max,n_amostras)
    optical_lenght = np.random.uniform(caminho_min, caminho_max, n_amostras)

    # Conversão para atm (mantido)
    pressures = pressures / 1013

    # Ruído de temperatura (ajustado)
    ruido_temperatura = np.random.normal(0, 3.0, n_amostras)  # Reduzi o ruído
    temperatures += ruido_temperatura

    # Caminho óptico (mantido)
    ruido_caminho = np.random.normal(0, 0.15, n_amostras)
    optical_lenght = np.clip(optical_lenght + ruido_caminho, caminho_min, caminho_max)


    print(f"Temperatures (K): {temperatures}\nPressures (atm): {pressures}\nOptical lenght (cm): {optical_lenght}\n\n")

    

    temperatures = temperatures.tolist()
    pressures = pressures.tolist()
    optical_lenght = optical_lenght.tolist()

    # Criando combinações de temperatura e pressão
    tasks = tuple(zip(temperatures,pressures,optical_lenght))
    

    # Paralelizando as tarefas
    with Pool(processes= processos) as pool:
        results_list = pool.map(process_temperature_pressure, tasks)

    # Concatenando todos os resultados
    #results = pd.concat(results_list, axis=0, ignore_index=True)

    # Salvando os resultados
    #results.to_csv('../database/lines/lines_valid.csv', index=None)
    print("Processamento concluído!")


if __name__ == "__main__":
    main()