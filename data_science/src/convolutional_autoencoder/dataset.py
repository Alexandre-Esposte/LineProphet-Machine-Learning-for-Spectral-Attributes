import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
from typing import Callable, Optional

class AutoencoderDataSet(Dataset):
    """Dataset de espectros para denoising autoencoder.

    Lê `wavenumbers`/`absorption` de cada .npz em `data_path` e empilha em
    2 canais. `transform` (ex: normalização) é aplicado em X e y igualmente,
    mantendo a mesma escala. `noisy_transform` é aplicado só em X, depois
    do `transform`, para gerar a entrada ruidosa a partir da qual o modelo
    deve reconstruir o alvo limpo (y).
    """

    def __init__(self, data_path: Path, transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None, noisy_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        """
        Args:
            data_path: diretório raiz onde os .npz dos espectros estão.
            transform: normalização (ou similar) aplicada igualmente a X e y.
            noisy_transform: ruído aplicado somente a X, após `transform`.
        """
        self.data_path = data_path
        self.spectra_files = list(data_path.rglob('*.npz'))
        self.transform = transform
        self.noisy_transform = noisy_transform

    def __len__(self) -> int:
        return len(self.spectra_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Carrega um espectro e monta o par (X ruidoso, y limpo).

        Args:
            idx: índice da amostra dentro de `self.spectra_files`.

        Returns:
            Tupla (X, y): X é a entrada do modelo (transform + noisy_transform,
            quando houver), y é o alvo de reconstrução (só transform, sem ruído).
        """
        file_path = self.spectra_files[idx]
        # `with` garante que o handle do .npz seja sempre fechado
        with np.load(file_path) as data:
            wavenumbers = data['wavenumbers']
            absorption = data['absorption']

            # 2 canais: wavenumber e absorption -> shape (2, N)
            arr = np.stack([wavenumbers, absorption], axis=0)
            arr = arr.astype(np.float32)

            X = torch.from_numpy(arr)

            # A arquitetura exigirá comprimento par; descarta o último ponto se for ímpar
            if X.shape[-1] % 2 != 0:
                X = X[..., :-1]


            if self.transform is not None:
                X = self.transform(X)
                
            y = X.clone()


            if self.noisy_transform is not None:
                X = self.noisy_transform(X)

        return X, y
