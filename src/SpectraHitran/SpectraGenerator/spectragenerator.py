from hapi import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import traceback
import os
from typing import Tuple, Dict



class SpectraGenerator():

    def __init__(self):

        
        print("Inicializando")

        print('Base de dados hitran já existe') if "hitran_database" in os.listdir() else db_begin('hitran_database')

        
        self.spectra = {'wavenumbers':0, 'absorption':0}


        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.spines.top'] = False

        mpl.rcParams['figure.figsize'] = (12,6)

       

    def downloadMolecule(self,moleculeName: str, isotopologues: Tuple[int], wavenumberRange: Tuple[int,int]) -> None:

        try:
            fetch_by_ids(moleculeName,isotopologues, wavenumberRange[0], wavenumberRange[1], ParameterGroups=['160-char','Voigt_Air','Voigt_Self'])

        except:
            traceback.print_exc()
            

    def simulateSpectra(self,moleculeName: str, diluent: Dict[str, float], enviroment: Dict[str, float], step: float = 0.01) -> None:


        try: 
            wavenumbers, absorptioncoef = absorptionCoefficient_Voigt(SourceTables = moleculeName,
                                                                      Diluent= diluent, 
                                                                      Environment= enviroment, 
                                                                      HITRAN_units = False,
                                                                      WavenumberStep = step,
                                                                      WavenumberWing = 50)
            
            self.spectra['wavenumbers'], self.spectra['absorption'] = absorptionSpectrum(wavenumbers,
                                                                                         absorptioncoef,
                                                                                         Environment = enviroment)
            
            #self.spectra['wavenumbers'], self.spectra['absorption'], _, _, _ = convolveSpectrum(self.spectra['wavenumbers'],
            #                                                                                    self.spectra['absorption'],
            #                                                                                    SlitFunction= SLIT_MICHELSON,
            #                                                                                    Resolution = 0.1,
            #                                                                                    AF_wing = 1 ) 
            
            
        except:
            self.wavenumbers = 0
            self.absorption = 0
            traceback.print_exc()


    def plot(self) -> None:

        try: 
            plt.plot(self.spectra['wavenumbers'],self.spectra['absorption'])
            plt.xlabel('Wavenumber (cm⁻¹)')
            plt.ylabel('Absorption (A.U)' )
            plt.show()
            

        except:
            traceback.print_exc()
