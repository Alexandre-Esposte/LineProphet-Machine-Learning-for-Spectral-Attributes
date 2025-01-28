from model_funcs import dataset

from model_funcs import train_and_test

from model_funcs.model import ConvolutionalNet

from torch.utils.data import DataLoader

from torch import nn
from torch.optim import Adam
from torchsummary import summary
import logging
logging.basicConfig(filename='train.log',level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

logging.debug('\n----------------------Novo treinamento iniciado-------------------------------------------\n')


batch_size = 16
learning_rate = 1e-3
epochs = 10


train = dataset.SpectraDataset('train')
test = dataset.SpectraDataset('test')

loader_train = DataLoader(train, batch_size= batch_size, pin_memory= True, shuffle = True)
loader_test = DataLoader(test, batch_size= batch_size, pin_memory= True, shuffle = True)



model = ConvolutionalNet()
summary(model, (1,2049, 476))


loss_function = nn.L1Loss()
optimizer = Adam(params= model.parameters(),lr = learning_rate, weight_decay= 1e-4)


logging.info(f'Dados separados e preparados com sucesso, hiperparâmetros ajustados com sucesso. Serão {len(train)} para treino e {len(test)} para teste')
logging.info('Iniciando treinamento')

for epoch in range(epochs):
    perda_treino = train_and_test.train(model, loader_train, optimizer, loss_function)
    perda_teste  = train_and_test.test(model, loader_test, loss_function)
    logging.debug(f"------------------Epoch {epoch + 1}/{epochs} --> train loss: {perda_treino} / test loss: {perda_teste} --------------------------")


logging.info('Treinamento finalizado com sucesso, salvando arquivo e finalizando treinamento')
train_and_test.saveModel(model,f'convolutionalnet.pth')