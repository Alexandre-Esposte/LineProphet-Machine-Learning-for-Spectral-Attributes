from .line_generator import voigt_profile_generator, lorentzian_profile_generator, gaussian_profile_generator
import pandas as pd
import json


def save_json(file,path):
    with open(path, 'w') as f:
        json.dump(file, f)

def database_generator(samples: int, dir_name: str):

    db = {'sigma':[],'gamma':[],'fwhm':[],'wv_mean':[],'wv_std':[], 'int_mean':[],'int_std':[],'int_max':[], "curve_type": []}
    for n_sample in range(samples):

        profile = voigt_profile_generator()
        print(n_sample)
        #print('voigt')
        save_json(profile, f'{dir_name}/voigt_{n_sample+1}.json')
        #db['sigma'].append(profile['sigma'])
        #db['gamma'].append(profile['gamma'])
        #db['fwhm'].append(profile['fwhm'])
        #db['wv_mean'].append(profile['statistics']['wv_mean'])
        #db['wv_std'].append(profile['statistics']['wv_std'])
        #db['int_mean'].append(profile['statistics']['int_mean'])
        #db['int_std'].append(profile['statistics']['int_std'])
        #db['int_max'].append(profile['statistics']['int_max'])
        #db['curve_type'].append(0)
        
        profile = gaussian_profile_generator()
        #print('gauss')
        save_json(profile, f'{dir_name}/gauss_{n_sample}.json')
        #db['sigma'].append(profile['sigma'])
        #db['gamma'].append(None)
        #db['fwhm'].append(profile['fwhm'])
        #db['wv_mean'].append(profile['statistics']['wv_mean'])
        #db['wv_std'].append(profile['statistics']['wv_std'])
        #db['int_mean'].append(profile['statistics']['int_mean'])
        #db['int_std'].append(profile['statistics']['int_std'])
        #db['int_max'].append(profile['statistics']['int_max'])
        #db['curve_type'].append(1)

        profile = lorentzian_profile_generator()
        #print('lorentz')
        save_json(profile, f'{dir_name}/lorentz_{n_sample}.json')
        #db['sigma'].append(None)
        #db['gamma'].append(profile['gamma'])
        #db['fwhm'].append(profile['fwhm'])
        #db['wv_mean'].append(profile['statistics']['wv_mean'])
        #db['wv_std'].append(profile['statistics']['wv_std'])
        #db['int_mean'].append(profile['statistics']['int_mean'])
        #db['int_std'].append(profile['statistics']['int_std'])
        #db['int_max'].append(profile['statistics']['int_max'])
        #db['curve_type'].append(2)
    #db = pd.DataFrame(db)
    #db.to_csv(path,index=None)


if __name__ == "__main__":
    database_generator(10,'test')
    print('Done')