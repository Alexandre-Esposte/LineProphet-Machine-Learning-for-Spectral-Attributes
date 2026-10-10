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


class SyntheticContamination:
    """Data augmentation: soma linhas pseudo-Voigt aleatórias a um canal (contaminação sintética).

    Com probabilidade `p`, soma N linhas (N uniforme em [n_lines_min, n_lines_max]) ao canal de
    absorção, imitando linhas de outras moléculas. Com probabilidade `cluster_p` as linhas ficam
    agrupadas numa sub-região de largura log-uniforme em [cluster_width_min, cluster_width_max] cm⁻¹,
    como uma banda; senão, espalhadas pela janela. Larguras lorentziana e gaussiana (HWHM, cm⁻¹) e a
    altura relativa ao maior pico do espectro são log-uniformes. Aplicado só em X, na escala bruta:
    y fica sem as linhas, e o modelo aprende a ignorar o que não é da molécula. Absorbâncias se
    somam (Beer-Lambert), então a mistura é física. Supõe número de onda crescente no canal `wn_channel`.
    """

    def __init__(self, p: float = 0.5, n_lines_min: int = 1, n_lines_max: int = 150, cluster_p: float = 0.5,
                 cluster_width_min: float = 20.0, cluster_width_max: float = 200.0,
                 lorentz_hwhm_min: float = 0.005, lorentz_hwhm_max: float = 0.3,
                 gauss_hwhm_min: float = 0.003, gauss_hwhm_max: float = 0.02,
                 rel_height_min: float = 0.01, rel_height_max: float = 1.5,
                 wing: float = 50.0, channel: int = 1, wn_channel: int = 0):
        """
        Args:
            p: probabilidade de contaminar o espectro.
            n_lines_min, n_lines_max: faixa do número de linhas.
            cluster_p: probabilidade de agrupar as linhas numa sub-região (banda).
            cluster_width_min, cluster_width_max: largura da sub-região, em cm⁻¹.
            lorentz_hwhm_min, lorentz_hwhm_max: meia largura lorentziana (pressão), em cm⁻¹.
            gauss_hwhm_min, gauss_hwhm_max: meia largura gaussiana (Doppler), em cm⁻¹.
            rel_height_min, rel_height_max: altura do pico relativa ao maior pico do espectro.
            wing: cada linha só é calculada até `wing` meias larguras do centro (custo).
            channel: canal que recebe as linhas (absorção).
            wn_channel: canal de número de onda, usado para posicionar as linhas em cm⁻¹.
        """
        if not 1 <= n_lines_min <= n_lines_max:
            raise ValueError("É preciso 1 <= n_lines_min <= n_lines_max.")
        for lo, hi in [(cluster_width_min, cluster_width_max), (lorentz_hwhm_min, lorentz_hwhm_max),
                       (gauss_hwhm_min, gauss_hwhm_max), (rel_height_min, rel_height_max)]:
            if not 0 < lo <= hi:
                raise ValueError("Faixas log-uniformes precisam de 0 < mínimo <= máximo.")
        self.p = p
        self.n_lines_min = n_lines_min
        self.n_lines_max = n_lines_max
        self.cluster_p = cluster_p
        self.cluster_width = (cluster_width_min, cluster_width_max)
        self.lorentz_hwhm = (lorentz_hwhm_min, lorentz_hwhm_max)
        self.gauss_hwhm = (gauss_hwhm_min, gauss_hwhm_max)
        self.rel_height = (rel_height_min, rel_height_max)
        self.wing = wing
        self.channel = channel
        self.wn_channel = wn_channel

    @staticmethod
    def _log_uniform(bounds: tuple, n: int) -> torch.Tensor:
        lo, hi = math.log(bounds[0]), math.log(bounds[1])
        return torch.exp(lo + (hi - lo) * torch.rand(n))

    def __call__(self, spectrum: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return spectrum

        x = spectrum[self.wn_channel]
        y = spectrum[self.channel]
        x_min, x_max = x[0].item(), x[-1].item()
        # Altura de referência: maior pico acima da linha de base (mediana)
        ref = max((y.max() - y.median()).item(), 1e-3)

        n = int(torch.randint(self.n_lines_min, self.n_lines_max + 1, (1,)))
        if torch.rand(1).item() < self.cluster_p:
            width = self._log_uniform(self.cluster_width, 1).item()
            center = x_min + (x_max - x_min) * torch.rand(1).item()
            lo, hi = max(x_min, center - width / 2), min(x_max, center + width / 2)
        else:
            lo, hi = x_min, x_max
        centers = lo + (hi - lo) * torch.rand(n)
        gamma_l = self._log_uniform(self.lorentz_hwhm, n)
        gamma_g = self._log_uniform(self.gauss_hwhm, n)
        heights = ref * self._log_uniform(self.rel_height, n)

        # Pseudo-Voigt (Thompson et al., 1987): largura total e mistura η a partir das FWHM
        f_l, f_g = 2 * gamma_l, 2 * gamma_g
        f_v = (f_g ** 5 + 2.69269 * f_g ** 4 * f_l + 2.42843 * f_g ** 3 * f_l ** 2
               + 4.47163 * f_g ** 2 * f_l ** 3 + 0.07842 * f_g * f_l ** 4 + f_l ** 5) ** 0.2
        r = f_l / f_v
        eta = 1.36603 * r - 0.47719 * r ** 2 + 0.11116 * r ** 3
        gamma_v = f_v / 2

        contaminated = spectrum.clone()
        out = contaminated[self.channel]
        # Cada linha só é calculada numa janela local de ±wing meias larguras
        starts = torch.searchsorted(x, centers - self.wing * gamma_v)
        ends = torch.searchsorted(x, centers + self.wing * gamma_v)
        for k in range(n):
            s, e = int(starts[k]), int(ends[k])
            if e <= s:
                continue
            u = ((x[s:e] - centers[k]) / gamma_v[k]) ** 2
            profile = eta[k] / (1 + u) + (1 - eta[k]) * torch.exp(-math.log(2) * u)
            out[s:e] += (heights[k] * profile).to(out.dtype)
        return contaminated


class Compose:
    """Aplica uma sequência de augmentations, na ordem dada."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, spectrum: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            spectrum = transform(spectrum)
        return spectrum
