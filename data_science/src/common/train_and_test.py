import copy
import torch
import logging

from typing import Callable, Optional
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train(dataloader, model, loss_fn, optimizer, device) -> float:
    """Roda uma época de treino e retorna a loss média da época.

    Assume `loss_fn` com reduction='mean' e `dataloader` com
    `drop_last=True` (todo batch com o mesmo tamanho)

    Args:
        dataloader: DataLoader que produz pares (X, y).
        model: modelo a treinar (recebe X, retorna pred).
        loss_fn: função de perda (reduction='mean').
        optimizer: otimizador do modelo.
        device: dispositivo (cpu/cuda) para mover os tensores.

    Returns:
        Loss média da época.
    """
    num_batches = len(dataloader)
    perdas = 0
    model.train()
    for X, y in dataloader:
        X = X.to(device, non_blocking = True)
        y = y.to(device, non_blocking = True)

        pred = model(X)
        loss = loss_fn(pred, y)
        perdas += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return perdas / num_batches


def test(dataloader, model, loss_fn, device) -> float:
    """Avalia o modelo (sem backprop) e retorna a loss média da época.

    Mesma suposição de `train`: batches de tamanho igual (`drop_last=True`)
    para que a média por batch corresponda à loss média real do dataset.
   

    Args:
        dataloader: DataLoader que produz pares (X, y).
        model: modelo já treinado/em treino, em modo avaliação.
        loss_fn: função de perda (reduction='mean').
        device: dispositivo (cpu/cuda) para mover os tensores.

    Returns:
        Loss média da época.
    """
    num_batches = len(dataloader)
    perdas = 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device, non_blocking = True)
            y = y.to(device, non_blocking = True)

            pred = model(X)
            perdas += loss_fn(pred, y).item()

    return perdas / num_batches


def get_device() -> torch.device:
    """cuda se disponível, senão mps, senão cpu."""
    return torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")


def make_dataloaders(train_dataset, valid_dataset, batch_size: int, num_workers: int, device: torch.device) -> tuple[DataLoader, DataLoader]:
    """DataLoaders de treino (embaralhado) e validação, ambos com `drop_last=True`
    (exigido pela média por batch de `train`/`test`)."""
    pin = device.type == "cuda"
    train_dataloader = DataLoader(dataset = train_dataset,
                                  batch_size = batch_size,
                                  num_workers = num_workers,
                                  shuffle = True,
                                  drop_last = True,
                                  pin_memory = pin)
    valid_dataloader = DataLoader(dataset = valid_dataset,
                                  batch_size = batch_size,
                                  num_workers = num_workers,
                                  shuffle = False,
                                  drop_last = True,
                                  pin_memory = pin)
    return train_dataloader, valid_dataloader


def fit(model, train_dataloader, valid_dataloader, loss_fn, optimizer, device, epochs: int,
        writer: Optional[SummaryWriter] = None, log_histograms: bool = False,
        epoch_callback: Optional[Callable[[int, float], None]] = None, logger: Optional[logging.Logger] = None,
        eval_loss_fn = None) -> float:
    """Loop de épocas (treino + validação ao fim de cada época) e retorna a
    menor loss de validação. Ao final, o `model` volta para os pesos da época
    com menor loss de validação. O conjunto de teste não passa por aqui.

    Args:
        epochs: número de épocas.
        writer: se passado, registra losses e LR no TensorBoard.
        log_histograms: também registra gradientes e pesos (pesado; útil no
            treino manual, desnecessário na busca).
        epoch_callback: chamado com (epoch, valid_loss) ao fim de cada época.
            Usado pela busca para reportar ao Optuna e interromper trials ruins.
        logger: se passado, loga as losses de cada época.
        eval_loss_fn: métrica da validação, se diferente da loss de treino. Mantém
            a validação comparável entre losses de treino diferentes (ex: na busca).
    """
    if eval_loss_fn is None:
        eval_loss_fn = loss_fn

    best_valid_loss = float('inf')
    best_state = None
    for epoch in range(epochs):
        train_loss = train(dataloader = train_dataloader, model = model, loss_fn = loss_fn,
                           optimizer = optimizer, device = device)
        valid_loss = test(dataloader = valid_dataloader, model = model, loss_fn = eval_loss_fn, device = device)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())

        if logger is not None:
            logger.info(f"[{epoch}] train - {train_loss} / valid - {valid_loss}")

        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/valid", valid_loss, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

            if log_histograms:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_scalar(f"GradMean/{name}", param.grad.abs().mean(), epoch)
                        writer.add_histogram(f"Gradhist/{name}", param.grad, epoch)
                        writer.add_histogram(f"Weights/{name}", param, epoch)

        if epoch_callback is not None:
            epoch_callback(epoch, valid_loss)

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_valid_loss
