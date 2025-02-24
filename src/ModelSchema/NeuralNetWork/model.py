
from torch import nn
from torchvision import models
import torch


class ConvolutionalNet(nn.Module):

    def __init__(self, base_model):
        
        super().__init__()

        

        # Preciso criar uma adaptação para o input no mobilenet. Poderia mudar na própria rede, mas prefiro nao fazer
        self.adaptador = nn.Conv2d(1,3,2)


        self.base_model = base_model

        # Eu quero treinar somente as camadas fully connected, para isso eu preciso desabilitar o calculo de gradiente para as demais camadas
        # A camada fc é nomeada como classifier no mobilenet
        for name, param in self.base_model.named_parameters():
            
            if 'classifier' not in name:
                param.requires_grad = False

        self.top = nn.Sequential(
            nn.Linear(2,1)
        )


    def forward(self,X):
        
        entrada_adaptada = self.adaptador(X)

        y = self.base_model(entrada_adaptada)

        y = self.top(y)

        return y
    

class ConvolutionalAutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()


        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # Redução mais eficiente
            nn.ReLU(),
            nn.BatchNorm2d(8),
            
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),  # Mais canais para melhor representação
            nn.ReLU(),
            nn.BatchNorm2d(8),
        )
        
        # Decoder 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            
        )
        
    def forward(self, x):
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded
    
