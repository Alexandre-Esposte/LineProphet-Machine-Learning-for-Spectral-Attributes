import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import Tuple

class SpectraDataset(Dataset):

    def __init__(self, path: str):

        self.path = path + "/"

        self.data = os.listdir(self.path)

    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx):

        spectrogram_name = self.data[idx]

        spectrogram = torch.tensor(np.load(self.path + spectrogram_name)['a'], dtype=torch.float)

        spectrogram = spectrogram.unsqueeze(0)


        temperature = float(spectrogram_name.split("_")[1])
        temperature = (temperature - 250) / (400 - 250)
        

        pressure = float(spectrogram_name.split("_")[2].split('.npz')[0])
        pressure = (pressure - 0.01) / (0.8 - 0.01)
        

        target = torch.tensor([temperature,pressure], dtype = torch.float)


        return spectrogram, target
    