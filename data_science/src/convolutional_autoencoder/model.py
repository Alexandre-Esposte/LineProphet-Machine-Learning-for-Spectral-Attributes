import torch
from torch import nn
import torch.nn.functional as F

import logging

logging.basicConfig(level=logging.INFO)

class ConvolutionalBlock(nn.Module):
    """Bloco do encoder: `num_layers` x (Conv1d + GELU) e um MaxPool1d que divide o comprimento por 2.

    Args:
        in_channels: canais de entrada.
        out_channels: canais de saída.
        kernel_size_max_pooling: kernel do pooling (par).
        kernel_size_conv: kernel das convoluções (ímpar).
        num_layers: número de convoluções (>= 1).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size_max_pooling: int = 2, kernel_size_conv: int = 3, num_layers: int = 3) -> None:
        super(ConvolutionalBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_max_pooling = kernel_size_max_pooling
        self.kernel_size_conv = kernel_size_conv
        # Mesmo fator do UpSampleConvolutionalBlock, para o decoder desfazer a redução
        self.scale_factor = 2

        # Kernel ímpar permite padding simétrico, mantendo o comprimento
        if self.kernel_size_conv % 2 != 0:
            self.padding_conv = (self.kernel_size_conv - 1) // 2

        else:
            raise ValueError("kernel_size_conv deve ser ímpar para manter a dimensão espacial.")


        # Com kernel par, este padding dá L_out = L_in // 2
        if self.kernel_size_max_pooling % 2 == 0:
            self.padding_max_pooling = (self.kernel_size_max_pooling - 2) // 2
        else:
            raise ValueError("kernel_size_max_pooling deve ser par para reduzir a dimensão espacial pela metade.")


        if num_layers < 1:
            raise ValueError("num_layers deve ser >= 1.")

        layers = []
        for layer in range(num_layers):
            if layer == 0:
                layers.append(nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size_conv, stride=1, padding=self.padding_conv))
            else:
                layers.append(nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size_conv, stride=1, padding=self.padding_conv))
            layers.append(nn.GELU())

        layers.append(nn.MaxPool1d(kernel_size=self.kernel_size_max_pooling, stride=self.scale_factor, padding=self.padding_max_pooling))
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, in_channels, L) -> (batch, out_channels, L // 2)."""
        return self.conv_block(x)

class UpSampleConvolutionalBlock(nn.Module):
    """Bloco do decoder: Upsample x2 (linear), `num_layers - 1` x (Conv1d + GELU) e uma Conv1d 1x1 final sem ativação.

    Args:
        in_channels: canais de entrada.
        out_channels: canais de saída.
        kernel_size_conv: kernel das convoluções (ímpar).
        num_layers: total de convoluções, incluindo a 1x1 final (>= 1).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size_conv: int, num_layers: int = 2) -> None:
        super(UpSampleConvolutionalBlock,self).__init__()

        # Mesmo fator do ConvolutionalBlock
        self.scale_factor = 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_conv = kernel_size_conv

        # Kernel ímpar permite padding simétrico, mantendo o comprimento
        if self.kernel_size_conv % 2 != 0:
            self.padding_conv = (self.kernel_size_conv - 1) // 2
        else:
            raise ValueError("kernel_size_conv deve ser ímpar para manter a dimensão espacial.")


        if num_layers < 1:
            raise ValueError("num_layers deve ser >= 1.")

        layers = []

        layers.append(nn.Upsample(scale_factor = self.scale_factor, mode = 'linear', align_corners = False))

        for layer in range(num_layers-1):
            if layer == 0:
                layers.append(nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size_conv, stride=1, padding=self.padding_conv))
            else:
                layers.append(nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size_conv, stride=1, padding=self.padding_conv))
            layers.append(nn.GELU())

        # Com num_layers == 1 o loop não roda e o tensor ainda tem in_channels
        final_conv_in_channels = self.out_channels if num_layers > 1 else self.in_channels
        layers.append(nn.Conv1d(in_channels=final_conv_in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0))

        self.upconv_block = nn.Sequential(*layers)



    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """(batch, in_channels, L) -> (batch, out_channels, L * 2)."""
        return self.upconv_block(X)

class ConvolutionalAutoencoder(nn.Module):
    """Autoencoder convolucional 1D para denoising de espectros.

    Fluxo: encoder convolucional -> AdaptiveAvgPool1d + Linear (vetor latente)
    -> Linear -> decoder espelhado -> interpolação para o comprimento de entrada.

    Args:
        architecture_config: uma tupla por bloco do encoder, `(in_channels,
            out_channels, kernel_size_max_pooling, kernel_size_conv, num_layers)`.
            O decoder usa a mesma lista invertida, com os canais trocados.
        pooling_output_size: comprimento após o AdaptiveAvgPool1d.
        latent_space_size: tamanho do vetor latente.
        output_channels: canais da reconstrução. None usa os mesmos da
            entrada; difere quando a entrada tem canais extras (ex: máscara).
    """

    def __init__(
        self,
        architecture_config: list[tuple[int, int, int, int, int]],
        pooling_output_size: int,
        latent_space_size: int,
        output_channels: int | None = None,
    ) -> None:
        super().__init__()

        self.architecture_config = architecture_config

        self.encoder_layers_qte = len(self.architecture_config)
        self.decoder_layers_qte = len(self.architecture_config)
        self.total_layers_qte = self.encoder_layers_qte + self.decoder_layers_qte
        self.pooling_output_size = pooling_output_size
        self.latent_space_size = latent_space_size
        self.output_channels = output_channels if output_channels is not None else self.architecture_config[0][0]


        encoder_layers = []
        for config in self.architecture_config:
            in_channels, out_channels, kernel_size_max_pooling, kernel_size_conv, num_layers = config
            encoder_layers.append(ConvolutionalBlock(in_channels, out_channels, kernel_size_max_pooling, kernel_size_conv, num_layers))

        self.encoder_layer = nn.Sequential(*encoder_layers)

        # Canais trocados na desempacotagem: o decoder percorre o caminho do encoder ao contrário
        decoder_layers = []
        for i, config in enumerate(reversed(self.architecture_config)):
            out_channels, in_channels, kernel_size_max_pooling, kernel_size_conv, num_layers = config
            # O último bloco do decoder devolve output_channels, não o in_channels do encoder
            if i == len(self.architecture_config) - 1:
                out_channels = self.output_channels
            decoder_layers.append(UpSampleConvolutionalBlock(in_channels, out_channels, kernel_size_conv, num_layers))

        self.decoder_layer = nn.Sequential(*decoder_layers)


        self.latent_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size = self.pooling_output_size),
            nn.Flatten(),
            nn.Linear(self.pooling_output_size * self.architecture_config[-1][1], self.latent_space_size),
          
        )

        self.to_decoder_layer = nn.Sequential(
              nn.Linear(self.latent_space_size, self.pooling_output_size * self.architecture_config[-1][1]),
              nn.Unflatten(1, (self.architecture_config[-1][1], self.pooling_output_size))
        )

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Só os blocos convolucionais do encoder, sem o vetor latente."""
        return self.encoder_layer(x)

    def decoder(self, x: torch.Tensor) -> torch.Tensor:
        """Só os blocos de upsampling, sem `to_decoder` nem a interpolação final."""
        return self.decoder_layer(x)

    def latent_space(self, x: torch.Tensor) -> torch.Tensor:
        """Saída do encoder -> vetor latente (batch, latent_space_size)."""
        return self.latent_layer(x)

    def to_decoder(self, x: torch.Tensor) -> torch.Tensor:
        """Vetor latente -> (batch, canais, pooling_output_size), entrada do decoder."""
        return self.to_decoder_layer(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, in_channels, L) -> reconstrução com o mesmo shape."""

        input_batch, input_in_channels, input_length = x.shape

        logging.debug(f"Input shape: {x.shape}")       

        # encoder
        x = self.encoder_layer(x)

        #bottleneck
        x = self.latent_space(x)

        # decoder
        x = self.to_decoder(x)
        x = self.decoder_layer(x)

        final_batch, final_channels, final_length = x.shape


        # O pooling adaptativo não é divisão exata, então o decoder não volta ao comprimento original
        x = F.interpolate(x, size=input_length, mode='linear', align_corners=False)

        logging.debug(f"Output shape: {x.shape}")

        return x