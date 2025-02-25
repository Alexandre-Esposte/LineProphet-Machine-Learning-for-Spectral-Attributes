from SpectraHitran.SpectraGenerator.spectragenerator import SpectraGenerator
from SpectraHitran.SpectraProcessing.spectraprocessing import branchClassification
from SpectraHitran.SpectraFit.spectrafit import multipleLineFit

import numpy as np

import pandas as pd
from multiprocessing import Pool
import os


def process_temperature_pressure(args):
    """Função para simular espectro e ajustar linhas para uma combinação de temperatura e pressão."""
    temperature, pressure, optical_length = args
    s = SpectraGenerator()
    s.downloadMolecule('hcl',(52,53),(5200,5900))
    print(f'-----------Fitting for {temperature}K and {pressure} atm, process: {os.getpid()}-----------')
    
    # Simulando o espectro
    s.simulateSpectra('hcl', {'air': 0, 'self': 1}, {'l': optical_length, 'p': pressure, 'T': temperature})
    s.spectra['absorption'] = s.spectra['absorption']/ np.max(s.spectra['absorption'])
    s.spectra['absorption'] = np.random.normal(s.spectra['absorption'],0.001)

    # Organizando as linhas
    lines = branchClassification(s.spectra, findPeaksParams={'height': 0.01, 'width': 1, 'distance': None},thresoldBranch=5665)
    lines['pressure'] = pressure
    lines['temperature'] = temperature

    # Ajustando múltiplas linhas
    fitted_params = multipleLineFit(s.spectra, lines, 0.01)
    lines = pd.concat([lines, fitted_params], axis=1)
    lines.to_csv(f'../database/lines/test/{temperature}_{pressure}.csv')

    return 1


def main():
    # Lista de temperaturas e pressões
    
    data = np.loadtxt('../database/envs_test.txt')
        
    temperatures = data[:,0].tolist() 

    pressures    = data[:,1].tolist()

    #temperatures = [293, 295, 300, 315, 350, 373]  # Kelvin
    #pressures    = [0.1, 0.2, 0.4, 0.6, 0.8, 1]  # atm

    optical_length = [2]  # cm

    optic = optical_length * len(pressures)

    print(f"Temperatures (K): {temperatures}\nPressures (atm): {pressures}\n\n")

    # Criando combinações de temperatura e pressão
    tasks = tuple(zip(temperatures,pressures,optic))
    #tasks = [(temperature, pressure, optical_length) for temperature in temperatures for pressure in pressures]

    # Paralelizando as tarefas
    with Pool(processes= 8) as pool:
        results_list = pool.map(process_temperature_pressure, tasks)

    # Concatenando todos os resultados
    #results = pd.concat(results_list, axis=0, ignore_index=True)

    # Salvando os resultados
    #results.to_csv('../database/lines/lines_valid.csv', index=None)
    print("Processamento concluído!")


if __name__ == "__main__":
    main()