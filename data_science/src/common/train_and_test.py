import torch


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
