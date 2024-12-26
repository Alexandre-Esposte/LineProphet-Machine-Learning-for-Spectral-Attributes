from hapi import *
import matplotlib.pyplot as plt
import matplotlib as mpl

class SpectraGenerator():

    def __init__(self):

        db_begin('hitran_database')

        self.wavenumbers = 0
        self.absorption = 0

        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.spines.top'] = False

        mpl.rcParams['figure.figsize'] = (12,6)



    def downloadMolecule(self,moleculeName: str, isotopologues: tuple, wavenumberRange: tuple):

        try:
            fetch_by_ids(moleculeName,isotopologues, wavenumberRange[0], wavenumberRange[1], ParameterGroups=['160-char','Voigt_Air','Voigt_Self','All'])

        except Exception as err:
            print(err)

    def simulateSpectra(self,moleculeName: str, diluent: dict, enviroment: dict, step: float):


        try: 
            wavenumbers, absorptioncoef = absorptionCoefficient_Voigt(SourceTables = moleculeName,
                                                                      Diluent= diluent, 
                                                                      Environment= enviroment, 
                                                                      HITRAN_units = False,
                                                                      WavenumberStep = step,
                                                                      WavenumberWing = 50)
            
            self.wavenumbers , self.absorption = absorptionSpectrum(wavenumbers,
                                                          absorptioncoef,
                                                          Environment = enviroment)
            
            
        except Exception as err:
            self.wavenumbers = 0
            self.absorption = 0
            print(err)

    def convolveSpectrum(self, resolution: float, af_wing: float):

        try:
            self.wavenumbers, self.absorption, i1,i2,slit = convolveSpectrum(self.wavenumbers,
                                                                             self.absorption,
                                                                             SlitFunction = SLIT_RECTANGULAR,
                                                                             Resolution= resolution,
                                                                             AF_wing= af_wing)
        except Exception as err:
            print(err)


    def plot(self):

        try: 
            plt.plot(self.wavenumbers,self.absorption)
            plt.xlabel('Wavenumber (cm⁻¹)')
            plt.ylabel('Absorption (A.U)' )

        except Exception as err:
            print(err)
