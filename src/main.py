from SpectraGenerator.spectragenerator import SpectraGenerator
from SpectraProcessing.spectraprocessing import branchClassification
from SpectraFit.spectrafit import multipleLineFit

import pandas as pd
from multiprocessing import Pool
import os


def process_temperature_pressure(args):
    """Função para simular espectro e ajustar linhas para uma combinação de temperatura e pressão."""
    temperature, pressure, optical_length = args
    s = SpectraGenerator()
    print(f'-----------Fitting for {temperature}K and {pressure} atm, process: {os.getpid()}-----------')
    
    # Simulando o espectro
    s.simulateSpectra('H35Cl', {'air': 0, 'self': 1}, {'l': optical_length, 'p': pressure, 'T': temperature})

    # Organizando as linhas
    lines = branchClassification(s.spectra, findPeaksParams={'height': 0}, thresoldBranch=5665)
    lines['pressure'] = pressure
    lines['temperature'] = temperature

    # Ajustando múltiplas linhas
    fitted_params = multipleLineFit(s.spectra, lines, 0.01)
    lines = pd.concat([lines, fitted_params], axis=1)

    return lines


def main():
    # Lista de temperaturas e pressões
    
    temperatures = [295, 300, 315, 350, 373]  # Kelvin
    pressures = [0.1, 0.2, 0.4, 0.6, 0.8, 1]  # atm

    optical_length = 10  # cm

    print(f"Temperatures (K): {temperatures}\nPressures (atm): {pressures}\n\n")

    # Criando combinações de temperatura e pressão
    tasks = [(temperature, pressure, optical_length) for temperature in temperatures for pressure in pressures]

    # Paralelizando as tarefas
    with Pool(processes= 10) as pool:
        results_list = pool.map(process_temperature_pressure, tasks)

    # Concatenando todos os resultados
    results = pd.concat(results_list, axis=0, ignore_index=True)

    # Salvando os resultados
    results.to_csv('../database/lines.csv', index=None)
    print("Processamento concluído! Resultados salvos em '../database/lines.csv'.")


if __name__ == "__main__":
    main()