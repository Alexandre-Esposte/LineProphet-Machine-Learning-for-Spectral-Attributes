from SpectraHitran.SpectraGenerator.spectragenerator import SpectraGenerator
from SpectraHitran.SpectraProcessing.spectraprocessing import branchClassification, plotSpectrum, spectrogramFromSpectra
from SpectraHitran.SpectraFit.spectrafit import filterLine,singleLineFit, multipleLineFit

import numpy as np



if __name__ == "__main__":

    s = SpectraGenerator()
    s.downloadMolecule('hcl',(52,53),(5200,5900))

    optical_length = 2 #cm

    train_size = 1000

    test_size = 200

    for i in range(train_size):

        print(f'Train: {i+1}/{train_size}')

        temperature = np.random.uniform(273.15, 373.15) 

        pressure    = np.random.uniform(0.01, 1) 

        s.simulateSpectra('hcl',{'air':0, 'self':1}, {'l':optical_length,'p':pressure,'T':temperature})

        np.savez_compressed(f'../database/spectras/train/{i+1}_{temperature}_{pressure}', spectra = s.spectra)

    for i in range(test_size):

        
        print(f'Test: {i+1}/{test_size}')
        
        temperature = np.random.uniform(273.15, 373.15) 
        
        pressure    = np.random.uniform(0.01, 1) 
        
        s.simulateSpectra('hcl',{'air':0, 'self':1}, {'l':optical_length,'p':pressure,'T':temperature})
          
        np.savez_compressed(f'../database/spectras/test/{i+1}_{temperature}_{pressure}', spectra = s.spectra)