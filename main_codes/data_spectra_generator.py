from SpectraHitran.SpectraGenerator.spectragenerator import SpectraGenerator
from SpectraHitran.SpectraProcessing.spectraprocessing import branchClassification, plotSpectrum, spectrogramFromSpectra
from SpectraHitran.SpectraFit.spectrafit import filterLine,singleLineFit, multipleLineFit

import numpy as np


train = np.loadtxt('../database/envs_train.txt')

test =  np.loadtxt('../database/envs_test.txt')

valid =  np.loadtxt('../database/envs_valid.txt')


if __name__ == "__main__":

    s = SpectraGenerator()
    s.downloadMolecule('hcl',(52,53),(5200,5900))

    optical_length = [2] #cm

    #train_size = 1000

    #test_size = 200

    for i, vars in enumerate(train):

        print(f'Train: {i+1}/{len(train)}')

        temperature = vars[0] 

        pressure    = vars[1]

        s.simulateSpectra('hcl',{'air':0, 'self':1}, {'l':optical_length,'p':pressure,'T':temperature})

        np.savez_compressed(f'../database/spectras/train/{i+1}_{temperature}_{pressure}', spectra = s.spectra)

    for i, vars in enumerate(test):

        
        print(f'Test: {i+1}/{len(test)}')
        
        temperature = vars[0] 
        
        pressure    = vars[1] 
        
        s.simulateSpectra('hcl',{'air':0, 'self':1}, {'l':optical_length,'p':pressure,'T':temperature})
          
        np.savez_compressed(f'../database/spectras/test/{i+1}_{temperature}_{pressure}', spectra = s.spectra)

        
    for i, vars in enumerate(valid):

        
        print(f'valid: {i+1}/{len(test)}')
        
        temperature = vars[0] 
        
        pressure    = vars[1] 
        
        s.simulateSpectra('hcl',{'air':0, 'self':1}, {'l':optical_length,'p':pressure,'T':temperature})
          
        np.savez_compressed(f'../database/spectras/valid/{i+1}_{temperature}_{pressure}', spectra = s.spectra)