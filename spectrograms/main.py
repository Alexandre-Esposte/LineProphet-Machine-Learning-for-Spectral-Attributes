from SpectraHitran.SpectraGenerator.spectragenerator import SpectraGenerator
from SpectraHitran.SpectraProcessing.spectraprocessing import branchClassification, plotSpectrum, spectrogramFromSpectra
from SpectraHitran.SpectraFit.spectrafit import filterLine,singleLineFit, multipleLineFit

import numpy as np



if __name__ == "__main__":

    s = SpectraGenerator()
    s.downloadMolecule('hitran_database/HCl', (52,53), (5300,5900))

    optical_length = 2 #cm

    train_size = 1000

    test_size = 0.2 * train_size

    for i in range(train_size):

        print(f'Train: {i+1}/{train_size}')

        temperature = np.random.uniform(250, 400) 

        pressure    = np.random.uniform(0.01, 0.8) 

        s.simulateSpectra('hitran_database/HCl',{'air':0, 'self':1}, {'l':optical_length,'p':pressure,'T':temperature})

        f, t, spec, interferogram = spectrogramFromSpectra(s.spectra)

        np.savez_compressed(f'train/{i+1}_{temperature}_{pressure}', a = spec)

    for i in range(test_size):

        
        print(f'Test: {i+1}/{test_size}')
        
        temperature = np.random.uniform(250, 400) 
        
        pressure    = np.random.uniform(0.01, 0.8) 
        
        s.simulateSpectra('HCl',{'air':0, 'self':1}, {'l':optical_length,'p':pressure,'T':temperature})
        
        f, t, spec, interferogram = spectrogramFromSpectra(s.spectra)
        
        np.savez_compressed(f'test/{i+1}_{temperature}_{pressure}',a = spec)