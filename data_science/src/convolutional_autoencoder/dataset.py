import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
from typing import Callable, Optional

class AutoencoderDataSet(Dataset):
    """Dataset de espectros para denoising autoencoder.

    Lê `wavenumbers`/`absorption` de cada .npz em `data_path` e empilha em
    2 canais. Normaliza X e y por canal com `mean`/`std` globais: se não
    forem passados, são calculados sobre os próprios dados (uso no treino);
    no teste, passe os do dataset de treino. `augmentation` é aplicado só
    em X, na escala bruta, antes da normalização; y fica limpo.

    Com `mask_channel=True`, X tem um 3º canal de máscara (1 = ponto apagado),
    que não é normalizado. Ele vem da augmentation (`MaskWindows`) ou, sem
    ela (teste/produção), é acrescentado todo zero. y continua com 2 canais.
    """

    def __init__(self, data_path: Path, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None, augmentation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None, mask_channel: bool = False):
        """
        Args:
            data_path: diretório raiz onde os .npz dos espectros estão.
            mean: média por canal, shape (2, 1). None calcula sobre `data_path`.
            std: desvio por canal, shape (2, 1). None calcula sobre `data_path`.
            augmentation: perturbação aplicada somente a X (ex: ruído).
            mask_channel: se True, X ganha o 3º canal de máscara.
        """
        if (mean is None) != (std is None):
            raise ValueError("mean e std devem ser passados juntos (ou nenhum dos dois).")

        self.data_path = data_path
        self.spectra_files = list(data_path.rglob('*.npz'))
        self.augmentation = augmentation
        self.mask_channel = mask_channel

        if mean is None:
            self.mean, self.std = self._compute_stats()
        else:
            self.mean = mean
            self.std = std

    def __len__(self) -> int:
        return len(self.spectra_files)

    def _load(self, file_path: Path) -> torch.Tensor:
        """Lê um .npz e retorna o espectro bruto com shape (2, N), N par."""
        # `with` garante que o handle do .npz seja sempre fechado
        with np.load(file_path) as data:
            # 2 canais: wavenumber e absorption -> shape (2, N)
            arr = np.stack([data['wavenumbers'], data['absorption']], axis=0)

        X = torch.from_numpy(arr.astype(np.float32))

        # A arquitetura exige comprimento par; descarta o último ponto se for ímpar
        if X.shape[-1] % 2 != 0:
            X = X[..., :-1]
        return X

    def _compute_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Média e desvio por canal sobre todos os pontos de todos os espectros.

        Acumula soma e soma dos quadrados arquivo a arquivo (float64), sem
        carregar o dataset inteiro na memória.
        """
        total = torch.zeros(2, dtype=torch.float64)
        total_sq = torch.zeros(2, dtype=torch.float64)
        count = 0
        for file_path in self.spectra_files:
            X = self._load(file_path).double()
            total += X.sum(dim=-1)
            total_sq += (X ** 2).sum(dim=-1)
            count += X.shape[-1]

        mean = total / count
        std = (total_sq / count - mean ** 2).sqrt()
        # shape (2, 1) para broadcast sobre o comprimento do espectro
        return mean.float().unsqueeze(-1), std.float().unsqueeze(-1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Carrega um espectro e monta o par (X perturbado, y limpo), normalizados.

        Args:
            idx: índice da amostra dentro de `self.spectra_files`.

        Returns:
            Tupla (X, y): X é a entrada do modelo (augmentation + normalização),
            y é o alvo de reconstrução (só normalização).
        """
        X = self._load(self.spectra_files[idx])
        y = X.clone()

        if self.augmentation is not None:
            X = self.augmentation(X)

        # Sem MaskWindows na augmentation, a máscara é toda zero (nada apagado)
        if self.mask_channel and X.shape[0] == 2:
            X = torch.cat([X, torch.zeros(1, X.shape[-1], dtype=X.dtype)], dim=0)

        # eps evita divisão por zero caso algum canal seja constante.
        # Só os 2 canais físicos são normalizados; a máscara fica em 0/1
        X = torch.cat([(X[:2] - self.mean) / (self.std + 1e-8), X[2:]], dim=0)
        y = (y - self.mean) / (self.std + 1e-8)

        return X, y
