import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Optimizer

import numpy as np

def train(model: nn.Module, dataloader: DataLoader, optimizer: Optimizer, loss_fn: nn.Module) -> float:

    size = len(dataloader.dataset)

    perdas = []
    model.train()
    for batch, data in enumerate(dataloader):
        
        X, target = data[0], data[1]

        pred = model(X)
        
        loss = loss_fn(pred, target)
        
        perdas.append(loss.item())
       
        # Backpropagation
        loss.backward()


        optimizer.step()
        optimizer.zero_grad()

    return np.mean(perdas)

def test(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module) -> float:


    model.eval()
    num_batches = len(dataloader)

    perdas = []
    with torch.no_grad():
        for data in dataloader:
            X, target = data[0], data[1]


            pred = model(X)
            perdas.append(loss_fn(pred, target).item())

    
    return np.mean(perdas)



def saveModel(model: nn.Module, name: str):
    torch.save(model,'models/'+name)