import torch
from torch import nn
import torch.nn.functional as F

class ConvolutionalBlock(nn.Module):
    """Bloco convolucional 1D para o encoder.

    Composto por `num_layers` camadas `Conv1d` (kernel ímpar, stride 1,
    padding "same"), cada uma seguida de GELU, que preservam a dimensão
    espacial de entrada. Ao final, um `MaxPool1d` (kernel par, stride 2)
    reduz essa dimensão pela metade.

    Args:
        in_channels: número de canais de entrada.
        out_channels: número de canais de saída (usado também nas
            convoluções seguintes).
        kernel_size_max_pooling: tamanho do kernel do max pooling; deve
            ser par para garantir a redução exata pela metade.
        kernel_size_conv: tamanho do kernel das convoluções; deve ser
            ímpar para permitir padding simétrico "same".
        num_layers: quantidade de camadas `Conv1d` + GELU no bloco;
            deve ser >= 1.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size_max_pooling: int = 2, kernel_size_conv: int = 3, num_layers: int = 3):
        super(ConvolutionalBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_max_pooling = kernel_size_max_pooling
        self.kernel_size_conv = kernel_size_conv
        # Fator de redução fixo em 2 (não exposto como parâmetro), igual ao
        # scale_factor de aumento usado em UpSampleConvolutionalBlock.
        self.scale_factor = 2

        # Kernel ímpar garante padding simétrico igual em ambos os lados,
        # preservando exatamente o comprimento da sequência (stride=1).
        if self.kernel_size_conv % 2 != 0:
            self.padding_conv = (self.kernel_size_conv - 1) // 2

        else:
            raise ValueError("kernel_size_conv deve ser ímpar para manter a dimensão espacial.")


        # Kernel par + esse padding específico garante L_out = L_in // 2
        # (halving exato) para qualquer L_in par.
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

    def forward(self, x):
        """Aplica as camadas convolucionais (com GELU) seguidas do max pooling.

        Args:
            x: tensor de entrada com shape (batch, in_channels, length).

        Returns:
            Tensor com shape (batch, out_channels, length // 2).
        """
        return self.conv_block(x)

class UpSampleConvolutionalBlock(nn.Module):
    """Bloco de upsampling 1D para o decoder.

    Aumenta a dimensão espacial por `scale_factor` via `nn.Upsample`
    (interpolação linear) e em seguida refina a representação com
    `num_layers - 1` convoluções `Conv1d` (kernel ímpar, padding "same")
    seguidas de GELU, finalizando sempre com uma convolução 1x1 de
    projeção de canais (sem ativação) — totalizando `num_layers`
    convoluções no bloco.

    Args:
        in_channels: número de canais de entrada.
        out_channels: número de canais de saída (mantido pelas camadas
            seguintes e pela convolução 1x1 final).
        kernel_size_conv: tamanho do kernel das convoluções de refinamento;
            deve ser ímpar para permitir padding simétrico "same".
        num_layers: quantidade total de convoluções do bloco, incluindo
            a projeção 1x1 final; deve ser >= 1.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size_conv: int, num_layers: int = 2):
        super(UpSampleConvolutionalBlock,self).__init__()

        ## UpSample
        # Fator de aumento fixo em 2 (não exposto como parâmetro), igual ao
        # fator de redução usado em ConvolutionalBlock.
        self.scale_factor = 2

        ## Convolucional
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_conv = kernel_size_conv

        # Kernel ímpar garante padding simétrico igual em ambos os lados,
        # preservando exatamente o comprimento da sequência (stride=1).
        if self.kernel_size_conv % 2 != 0:
            self.padding_conv = (self.kernel_size_conv - 1) // 2
        else:
            raise ValueError("kernel_size_conv deve ser ímpar para manter a dimensão espacial.")


        if num_layers < 1:
            raise ValueError("num_layers deve ser >= 1.")

        layers = []

        # Aumenta a dimensão espacial por scale_factor. mode='linear' (não
        # 'bilinear') porque a entrada é 1D: (batch, channels, length).
        layers.append(nn.Upsample(scale_factor = self.scale_factor, mode = 'linear', align_corners = False))

        # Camadas de refinamento: a primeira projeta in_channels -> out_channels,
        # as demais mantêm out_channels, preservando a dimensão espacial.
        for layer in range(num_layers-1):
            if layer == 0:
                layers.append(nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size_conv, stride=1, padding=self.padding_conv))
            else:
                layers.append(nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size_conv, stride=1, padding=self.padding_conv))
            layers.append(nn.GELU())

        # Convolução 1x1 de projeção final (sem ativação). Se o loop acima não
        # rodou (num_layers == 1), o tensor ainda está com in_channels canais,
        # já que o Upsample não altera a quantidade de canais.
        final_conv_in_channels = self.out_channels if num_layers > 1 else self.in_channels
        layers.append(nn.Conv1d(in_channels=final_conv_in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0))

        self.upconv_block = nn.Sequential(*layers)



    def forward(self,X):
        """Aplica upsampling seguido das convoluções de refinamento.

        Args:
            X: tensor de entrada com shape (batch, in_channels, length).

        Returns:
            Tensor com shape (batch, out_channels, length * scale_factor).
        """
        return self.upconv_block(X)

class ConvolutionalAutoencoder(nn.Module):
    """Autoencoder convolucional 1D para denoising de espectros.

    Empilha `ConvolutionalBlock`s (encoder) a partir de `architecture_config`,
    na ordem em que a lista é dada, e espelha a mesma lista (ordem inversa,
    canais trocados) para montar os `UpSampleConvolutionalBlock`s (decoder).
    Isso garante o vínculo necessário para reconstrução: o decoder desfaz
    exatamente as reduções espaciais e projeções de canal feitas pelo
    encoder, bloco a bloco, na ordem inversa.

    O espaço latente (bottleneck) ainda não é uma camada própria — hoje é
    simplesmente a saída do último bloco do encoder, com shape
    `(batch, C, L)` onde `C` é o `out_channels` do último bloco e `L` é
    `length // (2 ** len(architecture_config))` (assumindo halving exato
    em todos os blocos).

    Args:
        architecture_config: lista de tuplas, uma por bloco do encoder, no
            formato `(in_channels, out_channels, kernel_size_max_pooling,
            kernel_size_conv, num_layers)`. A quantidade de blocos é dada
            pelo tamanho da lista. O decoder é construído percorrendo essa
            mesma lista em ordem inversa, com `in_channels`/`out_channels`
            trocados em cada tupla, para espelhar o caminho de canais do
            encoder (que vai de `in_channels` a `out_channels`) de volta
            (`out_channels` a `in_channels`).
    """

    def __init__(
        self,
        architecture_config: list[tuple[int, int, int, int, int]], # (in_channels, out_channels, kernel_size_max_pooling, kernel_size_conv, num_layers)
    ):
        super().__init__()

        self.architecture_config = architecture_config

        self.encoder_layers_qte = len(self.architecture_config)
        self.decoder_layers_qte = len(self.architecture_config)
        self.total_layers_qte = self.encoder_layers_qte + self.decoder_layers_qte


        # Encoder: um ConvolutionalBlock por tupla, na ordem dada (canais
        # crescentes), cada um reduzindo a dimensão espacial pela metade.
        encoder_layers = []
        for config in self.architecture_config:
            in_channels, out_channels, kernel_size_max_pooling, kernel_size_conv, num_layers = config
            encoder_layers.append(ConvolutionalBlock(in_channels, out_channels, kernel_size_max_pooling, kernel_size_conv, num_layers))

        self.encoder_layer = nn.Sequential(*encoder_layers)

        # Decoder: espelha a lista em ordem inversa. in_channels/out_channels
        # são trocados na desempacotagem porque o decoder percorre o caminho
        # de canais do encoder ao contrário (de out_channels de volta a
        # in_channels), enquanto kernel_size_conv e num_layers permanecem
        # os mesmos do bloco espelhado.
        decoder_layers = []
        for config in reversed(self.architecture_config):
            out_channels, in_channels, kernel_size_max_pooling, kernel_size_conv, num_layers = config
            decoder_layers.append(UpSampleConvolutionalBlock(in_channels, out_channels, kernel_size_conv, num_layers))

        self.decoder_layer = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        """Codifica x até o espaço latente (saída do último bloco do encoder).

        Útil para inspecionar/analisar o espaço latente isoladamente, sem
        rodar o decoder.

        Args:
            x: tensor de entrada com shape (batch, in_channels, length).

        Returns:
            Tensor latente com shape (batch, C, L) — ver docstring da classe.
        """
        return self.encoder_layer(x)

    def decoder(self, x):
        """Decodifica um tensor latente, reconstruindo o espectro original.

        Args:
            x: tensor latente, com o mesmo shape retornado por `encoder`.

        Returns:
            Tensor reconstruído com shape (batch, in_channels, length).
        """
        return self.decoder_layer(x)

    def forward(self, x):
        """Codifica e decodifica x, reconstruindo o espectro de entrada.

        Args:
            x: tensor de entrada com shape (batch, in_channels, length).

        Returns:
            Tensor reconstruído com shape (batch, in_channels, length).
        """
        # encoder
        x = self.encoder_layer(x)

        # decoder
        x = self.decoder_layer(x)

        return x