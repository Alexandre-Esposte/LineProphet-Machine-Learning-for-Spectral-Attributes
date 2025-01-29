import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Optimizer

import numpy as np

def f1loss(real,pred):
    
    temp_error = torch.mean(torch.abs(real[0]- pred[0]))

    pres_error = torch.mean(torch.abs(real[1]- pred[1]))

    return (2*temp_error*pres_error)/(temp_error + pres_error)

def train(model: nn.Module, dataloader: DataLoader, optimizer: Optimizer, loss_fn: nn.Module) -> float:

    size = len(dataloader.dataset)

    perdas = []
    model.train()
    for batch, data in enumerate(dataloader):
        
        X, target = data[0], data[1]

        pred = model(X)
        
        loss = f1loss(pred, target)
        
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
            perdas.append(f1loss(pred, target).item())

    
    return np.mean(perdas)



def saveModel(model: nn.Module, name: str):
    torch.save(model,'models/'+name)