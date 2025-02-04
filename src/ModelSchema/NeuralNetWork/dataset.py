import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import Tuple, Dict, Union, List

class SpectraDataset(Dataset):

    def __init__(self, path: str):

        self.path = path + "/"

        self.data = os.listdir(self.path)

    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx):

        spectra_name = self.data[idx]

        spectra = torch.tensor(np.load(self.path + spectra_name, allow_pickle=True)['spectra'].item()['absorption'], dtype=torch.float)

        spectrogram, interferogram = self._stft(spectra = spectra)

        spectrogram = spectrogram.unsqueeze(0)

        temperature = float(spectra_name.split("_")[1])
        temperature = (temperature - 273.15) / (373.15 - 273.15)
        

        pressure = float(spectra_name.split("_")[2].split('.npz')[0])
        pressure = (pressure - 0.01) / (1 - 0.01)
        

        target = torch.tensor([temperature,pressure], dtype = torch.float)


        return spectrogram, target
    

    def _stft(self, spectra: torch.tensor ) -> torch.tensor:
        
        interferogram = self._interferogram(signal = spectra)

        result = torch.stft(interferogram, n_fft = 512, return_complex = True, normalized = True, window=torch.hann_window(512, device='cpu'))
        magnitude = torch.abs(result)
        
        # Verificar se existem valores negativos ou NaNs no emagnitudea original
        if torch.any(magnitude < 0):
            print("Aviso: Valores negativos detectados na magnitude!")
        if torch.any(torch.isnan(magnitude)):
            print("Aviso: Valores NaN detectados na magnitude!")

        # Evitar log(0) adicionando epsilon
        magnitude_clamped = torch.clamp(magnitude, min=1e-11)

        # Converter para decibéis
        magnitude_db = 20 * torch.log10(magnitude_clamped)

        # Verificar se o cálculo gerou NaN ou Inf
        if torch.any(torch.isnan(magnitude_db)):
            print("Aviso: Valores NaN detectados após conversão para dB!")
        if torch.any(torch.isinf(magnitude_db)):
            print("Aviso: Valores infinitos detectados após conversão para dB!")

        # Substituir NaNs e Infinitos por 0
        magnitude_db = torch.nan_to_num(magnitude_db, nan=0.0, posinf=0.0, neginf=0.0)

        return magnitude, interferogram

    def _interferogram(self, signal: np.array) -> np.array:

        interferogram = torch.fft.ifft(signal)

        interferogram = torch.fft.ifftshift(interferogram)


        return interferogram