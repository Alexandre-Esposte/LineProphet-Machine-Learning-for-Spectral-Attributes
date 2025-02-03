import argparse
from SpectraHitran.SpectraGenerator.spectragenerator import SpectraGenerator
parser = argparse.ArgumentParser(description='')

parser.add_argument('--name', type = str, required = True, help = 'Molecule name')

parser.add_argument('--isotopes', type = lambda x: tuple(map(int, x.split(','))), help = 'Isotopes')

parser.add_argument('--range', type = lambda x: tuple(map(int,x.split(','))), help = 'Wavenumber Range')

args = parser.parse_args()

params = vars(args)


print(f'Input parameters: ')

for key, value in params.items():
    print(key,': ',value)

print('\nDowload molecular data from Hitran database')

specra_obj = SpectraGenerator()

specra_obj.downloadMolecule(moleculeName = 'hitran_database/'+params['name'],isotopologues = params['isotopes'], wavenumberRange = params['range'])

print('Done')