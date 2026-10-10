import torch


def absorption_loss(pred: torch.Tensor, y: torch.Tensor, mean: float = 0.0, std: float = 1.0,
                    alpha: float = 0.0, delta: float = 0.0) -> torch.Tensor:
    """MSE no canal de absorção (canal 1), no espaço normalizado.

    Com `alpha > 0`, cada ponto pesa `1 + alpha * A / max(A)`, com A a absorbância
    física do alvo e o máximo por espectro: o pico pesa `1 + alpha` em qualquer
    espectro. Com `delta > 0`, soma `delta * média(relu(-Â)²)`, com Â a absorbância
    física prevista, pune reconstrução negativa; medida em unidades de `std` para ficar
    na escala do MSE. Com `alpha = delta = 0` é o MSE puro (métrica de validação e teste).

    Args:
        pred: saída do modelo, shape (batch, canais, N).
        y: alvo, mesmo shape.
        mean: média da absorção usada na normalização.
        std: desvio da absorção usado na normalização.
        alpha: quanto o pico pesa a mais que a linha de base.
        delta: peso da penalidade de absorbância prevista negativa.
    """
    p, t = pred[:, 1], y[:, 1]
    err = (p - t) ** 2
    if alpha:
        # clamp: ruído numérico não pode dar absorbância < 0
        absorbance = (t * std + mean).clamp(min=0)
        peak = absorbance.amax(dim=-1, keepdim=True).clamp(min=1e-6)
        err = (1 + alpha * absorbance / peak) * err
    loss = err.mean()
    if delta:
        # Â < 0 <=> p < -mean/std; a distância até esse limite já está em unidades de std
        loss = loss + delta * torch.relu(-(p + mean / std)).pow(2).mean()
    return loss
