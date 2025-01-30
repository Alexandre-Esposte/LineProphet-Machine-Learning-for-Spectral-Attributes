import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Optimizer
from typing import Tuple


import numpy as np


def train(model: nn.Module, dataloader: DataLoader, optimizer: Optimizer, loss_fn: nn.Module) -> float:

    size = len(dataloader.dataset)

    perdas = []
    model.train()
    for batch, data in enumerate(dataloader):
        
        X, target = data[0], data[1]

        pred = model(X)
        
        temperatura_loss = loss_fn(pred[0][0], target[0])

        pressure_loss = loss_fn(pred[0][1], target[1])

        total_loss = temperatura_loss + pressure_loss
        
        perdas.append(total_loss.item())
       
        # Backpropagation
        total_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return np.mean(perdas)

def test(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module) -> float:


    model.eval()
    num_batches = len(dataloader)

    perdas = []
    perdas_temp = []
    perdas_pres = []
    with torch.no_grad():
        for data in dataloader:
            X, target = data[0], data[1]


            pred = model(X)


            temperatura_loss = loss_fn(pred[0][0], target[0])
            pressure_loss = loss_fn(pred[0][1], target[1])
            total_loss = temperatura_loss + pressure_loss

            perdas.append(total_loss.item())
            perdas_temp.append(temperatura_loss.item())
            perdas_pres.append(pressure_loss.item())

    
    return np.mean(perdas), np.mean(perdas_temp), np.mean(perdas_pres)



def saveModel(model: nn.Module, name: str):
    torch.save(model,'models/'+name)