import math
import torch


# Classes com __call__ (não lambdas): com num_workers > 0 no Windows os
# workers são criados via spawn e precisam serializar (pickle) os transforms.
class GaussianNoise:
    """Data augmentation: ruído gaussiano em um único canal do espectro.

    Aplicado na escala bruta (antes da normalização), então o desvio está em
    unidades físicas do canal (ex: absorbância). Com `std_max`, o desvio de
    cada espectro é sorteado log-uniforme em [std_min, std_max], para o modelo
    ver vários níveis de ruído; sem `std_max`, o desvio é fixo em `std_min`.
    """

    def __init__(self, std_min: float, std_max: float | None = None, channel: int = 1):
        """
        Args:
            std_min: desvio mínimo (ou fixo, sem `std_max`), em unidades do canal.
            std_max: desvio máximo. None mantém o desvio fixo.
            channel: canal que recebe o ruído. Padrão 1 (absorção); o
                wavenumber é o grid de medição, não um sinal medido.
        """
        if std_max is not None and not 0 < std_min <= std_max:
            raise ValueError("É preciso 0 < std_min <= std_max (sorteio log-uniforme).")
        self.std_min = std_min
        self.std_max = std_max
        self.channel = channel

    def __call__(self, spectrum: torch.Tensor) -> torch.Tensor:
        std = self.std_min
        if self.std_max is not None:
            # log-uniforme: cobre igualmente ruído fraco e forte
            log_min, log_max = math.log(self.std_min), math.log(self.std_max)
            std = math.exp(log_min + (log_max - log_min) * torch.rand(1).item())

        noisy = spectrum.clone()
        noisy[self.channel] += torch.randn_like(noisy[self.channel]) * std
        return noisy


class MaskWindows:
    """Data augmentation: zera janelas aleatórias de um canal e acrescenta um canal de máscara.

    Com probabilidade `p`, sorteia `n_windows` janelas (tamanho uniforme em
    [min_size, max_size], posição uniforme; podem se sobrepor) e zera `channel`
    nelas. Sempre devolve um canal extra no fim, com 1 onde foi zerado e 0 no
    resto, para o modelo saber o que precisa reconstruir. Sem máscara
    (probabilidade 1 - p), o canal extra é todo zero, como no teste/produção.
    """

    def __init__(self, min_size: int, max_size: int, n_windows: int = 1, p: float = 0.5, channel: int = 1):
        """
        Args:
            min_size: tamanho mínimo da janela, em pontos.
            max_size: tamanho máximo da janela, em pontos.
            n_windows: quantas janelas por espectro mascarado.
            p: probabilidade de mascarar o espectro.
            channel: canal zerado. Padrão 1 (absorção); o wavenumber é o grid
                e precisa ficar intacto para o modelo saber onde está a lacuna.
        """
        if not 1 <= min_size <= max_size:
            raise ValueError("É preciso 1 <= min_size <= max_size.")
        self.min_size = min_size
        self.max_size = max_size
        self.n_windows = n_windows
        self.p = p
        self.channel = channel

    def __call__(self, spectrum: torch.Tensor) -> torch.Tensor:
        length = spectrum.shape[-1]
        mask = torch.zeros(1, length, dtype=spectrum.dtype)

        if torch.rand(1).item() < self.p:
            for _ in range(self.n_windows):
                size = int(torch.randint(self.min_size, min(self.max_size, length) + 1, (1,)))
                start = int(torch.randint(0, length - size + 1, (1,)))
                mask[0, start:start + size] = 1

        masked = spectrum.clone()
        masked[self.channel][mask[0].bool()] = 0
        return torch.cat([masked, mask], dim=0)


class BaselineShift:
    """Data augmentation: desloca a linha de base de um canal com offset e inclinação aleatórios.

    Com probabilidade `p`, soma `offset + slope * x` ao canal, com `x` indo de
    -0.5 a 0.5 ao longo do espectro; offset e slope são sorteados uniformes em
    [-offset_max, offset_max] e [-slope_max, slope_max]. Não deforma as linhas.
    Aplicado na escala bruta, antes do ruído; y fica com a linha de base original.
    """

    def __init__(self, offset_max: float, slope_max: float = 0.0, p: float = 0.5, channel: int = 1):
        """
        Args:
            offset_max: maior offset, em unidades do canal (ex: absorbância).
            slope_max: maior variação da rampa de ponta a ponta do espectro (0 = sem inclinação).
            p: probabilidade de deslocar o espectro.
            channel: canal deslocado. Padrão 1 (absorção).
        """
        if offset_max < 0 or slope_max < 0:
            raise ValueError("offset_max e slope_max devem ser >= 0.")
        self.offset_max = offset_max
        self.slope_max = slope_max
        self.p = p
        self.channel = channel

    def __call__(self, spectrum: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return spectrum
        offset = self.offset_max * (2 * torch.rand(1).item() - 1)
        slope = self.slope_max * (2 * torch.rand(1).item() - 1)
        x = torch.linspace(-0.5, 0.5, spectrum.shape[-1], dtype=spectrum.dtype)
        shifted = spectrum.clone()
        shifted[self.channel] += offset + slope * x
        return shifted


class Compose:
    """Aplica uma sequência de augmentations, na ordem dada."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, spectrum: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            spectrum = transform(spectrum)
        return spectrum
