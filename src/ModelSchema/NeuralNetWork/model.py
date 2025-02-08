
from torch import nn
from torchvision import models


class ConvolutionalNet(nn.Module):

    def __init__(self, base_model):
        
        super().__init__()

        

        # Preciso criar uma adaptação para o input no mobilenet. Poderia mudar na própria rede, mas prefiro nao fazer
        self.adaptador = nn.Conv2d(1,3,2)


        self.base_model = base_model

        # Eu quero treinar somente as camadas fully connected, para isso eu preciso desabilitar o calculo de gradiente para as demais camadas
        # A camada fc é nomeada como classifier no mobilenet
        for name, param in self.base_model.named_parameters():
            
            if 'fc' not in name:
                param.requires_grad = False



    def forward(self,X):
        
        entrada_adaptada = self.adaptador(X)

        y = self.base_model(entrada_adaptada)

        return y