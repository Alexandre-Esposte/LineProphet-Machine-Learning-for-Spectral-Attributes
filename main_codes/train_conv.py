from ModelSchema.NeuralNetWork.model import ConvolutionalNet
from ModelSchema.NeuralNetWork import dataset
from ModelSchema.NeuralNetWork import train_and_test
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.optim import Adam, RMSprop
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import logging
import yaml



with open("NNconfig.yaml", "r") as file:
    config = yaml.safe_load(file) 


logging.basicConfig(filename='train.log',level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

logging.debug('\n----------------------Novo treinamento iniciado-------------------------------------------\n')


batch_size = config['batch_size']
learning_rate = config['learning_rate']
epochs = config['epochs']
name_base_model = config['name_base_model']
name_to_save = config['name_to_save']



train = dataset.SpectraDataset('../database/real_spectras/train')
test = dataset.SpectraDataset( '../database/real_spectras/test' )
#valid = dataset.SpectraDataset('../database/spectras/valid')
valid=[]

loader_train = DataLoader(train, batch_size= batch_size, pin_memory= True, shuffle = True)
loader_test = DataLoader(test, batch_size= batch_size, pin_memory= True, shuffle = True)
#loader_valid = DataLoader(test, batch_size= batch_size, pin_memory= True, shuffle = True)


base_model = torch.load(f'../models/base_models/{name_base_model}.pth', weights_only=False)
model = ConvolutionalNet(base_model)
sumario =  summary(model, input_size=(batch_size, 1, 200, 200))


loss_function = nn.MSELoss()




logging.info(f'Dados separados e preparados com sucesso, hiperparâmetros ajustados com sucesso. Serão {len(train)} para treino, {len(valid)} para validacao e {len(test)} para teste')
logging.info(f'Sumario da arquitetura a ser utilizada:\n {sumario}')


logging.info(f'Configurações básicas --> batch size: {batch_size} / learning rate: {learning_rate} / epocas: {epochs}')


for name, param in model.named_parameters():

    if param.requires_grad:
        logging.info(f"Nome: {name} | Requer gradiente: {param.requires_grad}")

logging.info('Iniciando treinamento')




optimizer = Adam(params= model.parameters(),lr = learning_rate, weight_decay= 1e-4)
menor_erro = {'epoch': 0, 'error':9999, 'pesos': None} 
for epoch in range(epochs):
    perda_treino = train_and_test.train(model, loader_train, optimizer, loss_function)
    temperature_loss  = train_and_test.test(model, loader_test, loss_function)

    perda_valid = temperature_loss

    if perda_valid < menor_erro['error']:
        menor_erro['error'] = perda_valid
        menor_erro['epoch'] = epoch + 1
        menor_erro['pesos'] = model.state_dict()

    logging.debug(f"------------------Epoch {epoch + 1}/{epochs} --> train loss: {perda_treino} / test loss: {perda_valid} -------------- "
                  f"temperature loss: {temperature_loss} " 
                  f"\n-------Menor erro: {menor_erro['error']} na epoca {menor_erro['epoch']}")

logging.info(f'O menor erro ocorreu na epoca {menor_erro["epoch"]} com o erro {menor_erro["error"]}. Transferindo pesos associados ao menor erro para a arquitetura atual')
model.load_state_dict(menor_erro['pesos'])


logging.info('Treinamento finalizado com sucesso, salvando arquivo e finalizando treinamento')

torch.save(model,'../models/trained_models/'+ name_to_save +".pth")