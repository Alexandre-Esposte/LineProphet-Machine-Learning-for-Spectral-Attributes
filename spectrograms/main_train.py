from model_funcs import dataset

from model_funcs import train_and_test

from model_funcs.model import ConvolutionalNet

from torch.utils.data import DataLoader

from torch import nn
from torch.optim import Adam
from torchinfo import summary
import logging
logging.basicConfig(filename='train.log',level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

logging.debug('\n----------------------Novo treinamento iniciado-------------------------------------------\n')


batch_size = 8
learning_rate = 1e-4
epochs = 50


train = dataset.SpectraDataset('train')
test = dataset.SpectraDataset('test')

loader_train = DataLoader(train, batch_size= batch_size, pin_memory= True, shuffle = True)
loader_test = DataLoader(test, batch_size= batch_size, pin_memory= True, shuffle = True)



model = ConvolutionalNet()
sumario =  summary(model, input_size=(batch_size, 1, 2049, 476))


loss_function = nn.L1Loss()
optimizer = Adam(params= model.parameters(),lr = learning_rate, weight_decay= 1e-4)


logging.info(f'Dados separados e preparados com sucesso, hiperparâmetros ajustados com sucesso. Serão {len(train)} para treino e {len(test)} para teste')
logging.info(f'Sumario da arquitetura a ser utilizada:\n {sumario}')

logging.info(f'Configurações básicas: batch size: {batch_size} / learning rate: {learning_rate} / epocas: {epochs}')

logging.info('Iniciando treinamento')


menor_erro = {'epoch': 0, 'error':9999, 'pesos': None} 
for epoch in range(epochs):
    perda_treino = train_and_test.train(model, loader_train, optimizer, loss_function)
    perda_teste  = train_and_test.test(model, loader_test, loss_function)

    if perda_teste < menor_erro['error']:
        menor_erro['error'] = perda_teste
        menor_erro['epoch'] = epoch + 1
        menor_erro['pesos'] = model.state_dict()

    logging.debug(f"------------------Epoch {epoch + 1}/{epochs} --> train loss: {perda_treino} / test loss: {perda_teste} -------------- Menor erro: {menor_erro['error']} na epoca {menor_erro['epoch']}")

logging.info(f'O menor erro ocorreu na epoca {menor_erro["epoch"]} com o erro {menor_erro["error"]}. Transferindo pesos associados ao menor erro para a arquitetura atual')
model.load_state_dict(menor_erro['pesos'])

logging.info('Treinamento finalizado com sucesso, salvando arquivo e finalizando treinamento')
train_and_test.saveModel(model,f'convolutionalnet.pth')