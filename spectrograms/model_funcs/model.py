
from torch import nn
from torchvision import models


class ConvolutionalNet(nn.Module):

    def __init__(self):
        
        super().__init__()

        # Preciso criar uma adaptação para o input no mobilenet. Poderia mudar na própria rede, mas prefiro nao fazer
        self.adaptador = nn.Conv2d(1,3,4)


        # Carregando o mobilenet
        self.base_model = models.get_model('mobilenet_v2',weights = 'MobileNet_V2_Weights.IMAGENET1K_V2')


        # Eu quero treinar somente as camadas fully connected, para isso eu preciso desabilitar o calculo de gradiente para as demais camadas
        # A camada fc é nomeada como classifier no mobilenet
        for name, param in self.base_model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
            
        # Preciso que a saída seja apenas 2 neuronios
        in_features = self.base_model.classifier[-1].in_features
        self.base_model.classifier[-1] = nn.Linear(in_features, 2)



    def forward(self,X):
        
        entrada_adaptada = self.adaptador(X)

        y = self.base_model(entrada_adaptada)

        return y