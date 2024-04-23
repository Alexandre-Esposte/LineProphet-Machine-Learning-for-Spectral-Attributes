from .line_generator import voigt_profile_generator
import pandas as pd


def database_generator(samples: int, path: str):

    db = {'sigma':[],'gamma':[],'fwhm':[],'wv_mean':[],'wv_std':[], 'int_mean':[],'int_std':[],'int_max':[]}
    for _ in range(samples):

        profile = voigt_profile_generator()
        print(_)
        db['sigma'].append(profile['sigma'])
        db['gamma'].append(profile['gamma'])
        db['fwhm'].append(profile['fwhm'])
        db['wv_mean'].append(profile['statistics']['wv_mean'])
        db['wv_std'].append(profile['statistics']['wv_std'])
        db['int_mean'].append(profile['statistics']['int_mean'])
        db['int_std'].append(profile['statistics']['int_std'])
        db['int_max'].append(profile['statistics']['int_max'])

    db = pd.DataFrame(db)
    db.to_csv(path,index=None)


if __name__ == "__main__":
    database_generator(10)
    print('Done')