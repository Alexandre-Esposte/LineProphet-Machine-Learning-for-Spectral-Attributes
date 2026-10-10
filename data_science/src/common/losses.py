import torch
from torch import nn


# Todas recebem (pred, y) com shape (batch, canais, N), no espaço normalizado,
# e olham só para o canal de absorção: o wavenumber é o grid, conhecido de antemão.
class _AbsorptionLoss(nn.Module):
    """Base: seleciona o canal de absorção e desfaz a normalização quando preciso."""

    def __init__(self, channel: int = 1, mean: torch.Tensor | None = None, std: torch.Tensor | None = None):
        """
        Args:
            channel: canal de absorção.
            mean: média por canal usada na normalização, shape (2, 1). Só a
                SpectralLoss, que precisa da absorbância física, exige.
            std: desvio por canal usado na normalização, shape (2, 1).
        """
        super().__init__()
        self.channel = channel
        self.mean = None if mean is None else float(mean[channel])
        self.std = None if std is None else float(std[channel])

    def _split(self, pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return pred[:, self.channel], y[:, self.channel]

    def _absorbance(self, target: torch.Tensor) -> torch.Tensor:
        """Absorbância física do alvo; clamp: ruído numérico não pode dar valor < 0."""
        return (target * self.std + self.mean).clamp(min=0)


class AbsorptionMSE(_AbsorptionLoss):
    """MSE só no canal de absorção. Métrica de teste, igual para qualquer config de treino."""

    def forward(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p, t = self._split(pred, y)
        return torch.mean((p - t) ** 2)


class SpectralLoss(_AbsorptionLoss):
    """Loss geral: MSE ponderado + `beta` x derivada ponderada + `gamma` x erro de área.

    - Peso `1 + alpha * A / max(A)`, com A a absorbância do alvo e o máximo
      tomado por espectro: o pico pesa `1 + alpha` em qualquer espectro,
      independente da escala (ex: do caminho ótico).
    - Derivada: pune o achatamento de linhas estreitas, com o mesmo peso.
    - Área: `((Σpred - Σalvo) / Σalvo)²` na absorbância física, por espectro;
      conserva a absorção integrada. É sempre relativa.

    Os termos da última chamada ficam em `components`, para log.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, alpha: float = 5.0, beta: float = 0.0,
                 gamma: float = 0.0, channel: int = 1, eps: float = 1e-6):
        """
        Args:
            mean: média por canal usada na normalização, shape (2, 1).
            std: desvio por canal usado na normalização, shape (2, 1).
            alpha: quanto o pico pesa a mais que a linha de base (0 = sem peso).
            beta: peso do termo de derivada (0 = desligado).
            gamma: peso do termo de área (0 = desligado).
            channel: canal de absorção.
            eps: evita divisão por zero em espectros sem absorção.
        """
        super().__init__(channel, mean, std)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.components: dict[str, float] = {}

    def forward(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p, t = self._split(pred, y)
        absorbance = self._absorbance(t)
        peak = absorbance.amax(dim=-1, keepdim=True).clamp(min=self.eps)
        weight = 1 + self.alpha * absorbance / peak
        error = p - t

        mse = torch.mean(weight * error ** 2)
        total = mse
        self.components = {'mse_w': mse.item()}

        if self.beta:
            # Peso no ponto médio de cada par vizinho, alinhado com a diff
            weight_mid = (weight[..., 1:] + weight[..., :-1]) / 2
            derivative = torch.mean(weight_mid * torch.diff(error, dim=-1) ** 2)
            total = total + self.beta * derivative
            self.components['derivative_w'] = derivative.item()

        if self.gamma:
            # Σ(pred - alvo) na absorbância física: a média da normalização se cancela
            area_diff = self.std * (p - t).sum(dim=-1)
            area_true = absorbance.sum(dim=-1).clamp(min=self.eps)
            area = torch.mean((area_diff / area_true) ** 2)
            total = total + self.gamma * area
            self.components['area'] = area.item()

        self.components['total'] = total.item()
        return total


def build_loss(config: dict, mean: torch.Tensor, std: torch.Tensor) -> SpectralLoss:
    """Monta a SpectralLoss a partir do yaml, ex: {alpha: 5, beta: 0, gamma: 0}."""
    return SpectralLoss(mean = mean, std = std, **config)
