import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Optimizer
from typing import Tuple

import numpy as np



def train(model: nn.Module, dataloader: DataLoader, optimizer: Optimizer, loss_fn: nn.Module) -> float:

    size = len(dataloader.dataset)
    total_loss = 0
    model.train()
    for batch, data in enumerate(dataloader):
        
        X, target, _ = data[0], data[1], data[2]

        pred = model(X)

        pred_temp = pred[:, 0]
        #pred_pres = pred[:, 1]

        target_temp = target[:, 0]
        #target_pres = target[:, 1]
        
        temperatura_loss = loss_fn(pred_temp, target_temp)

        #pressure_loss = loss_fn(pred_pres, target_pres)

        #total_loss_batch = temperatura_loss + pressure_loss
        
        #total_loss += total_loss_batch
        total_loss = temperatura_loss
       
        # Backpropagation
        #total_loss_batch.backward()
        temperatura_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(dataloader)

def test(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module) -> float:


    model.eval()
    num_batches = len(dataloader)

    total_loss = 0.0
    total_loss_temp = 0.0
    total_loss_press = 0.0
    with torch.no_grad():
        for data in dataloader:
            X, target, _ = data[0], data[1], data[2]


            pred = model(X)


            pred_temp = pred[:, 0]
            #pred_press = pred[:, 1]
            
            target_temp = target[:, 0]
            #target_press = target[:, 1]
            
            loss_temp = loss_fn(pred_temp, target_temp)
            #loss_press = loss_fn(pred_press, target_press)
            #total_loss_batch = loss_temp + loss_press
            
            #total_loss += total_loss_batch.item()
            total_loss_temp += loss_temp.item()
            #total_loss_press += loss_press.item()


    
    return total_loss_temp / num_batches#total_loss / num_batches, total_loss_temp / num_batches, total_loss_press / num_batches

    


def train_autoencoder(model: nn.Module, dataloader: DataLoader, optimizer: Optimizer, loss_fn: nn.Module) -> float:

    total_loss = 0
    model.train()
    for batch, data in enumerate(dataloader):
        
        X, _, X_noise = data[0], data[1], data[2]

        pred = model(X_noise)

        total_loss_batch = loss_fn(X, pred)

        total_loss += total_loss_batch.item()
       
        # Backpropagation
        total_loss_batch.backward()

        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(dataloader)

def test_autoencoder(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module) -> float:

    model.eval()
    num_batches = len(dataloader)

    total_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            X, _, X_noise = data[0], data[1], data[2]


            pred = model(X_noise)

            total_loss_batch = loss_fn(X, pred)
            
            total_loss += total_loss_batch.item()


    
    return total_loss / num_batches