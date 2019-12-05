import pandas as pd
import numpy as np
import sys
sys.path.append('../pp_1st')
import pp1st_pipeline
import pp1st_func
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


def replace(data):
#     data['lnd_ar'].replace(0, np.nan, inplace=True)
#     data['ttl_ar'].replace(0, np.nan, inplace=True)
#     data['bldng_ar'].replace(0, np.nan, inplace=True)
    data['wnd_drctn'].replace(0, np.nan, inplace=True)
    return data

def label_encoding(data):
    #fr_yn
    data.loc[data[data['fr_yn']=='Y'].index, 'fr_yn'] = 1
    data.loc[data[data['fr_yn']=='N'].index, 'fr_yn'] = 0
    
    #emd_nm
    data['emd_nm'] = data['emd_nm'].astype(str).apply(lambda x : x[4:] if x[:4]=='경상남도' else x )
    data['emd_nm_big'] = data['emd_nm'].apply(lambda x : x.split()[0] if x!='nan' else x )
    data['emd_nm_small'] = data['emd_nm'].apply(lambda x : x.split()[1] if x!='nan' else x)
    
    data['dt_of_fr_yr'] = data['dt_of_fr'].astype('datetime64').apply(lambda x : x.year)
    data['dt_of_fr_hr'] = data['dt_of_fr'].astype('datetime64').apply(lambda x : x.hour)
    data['dt_of_fr_mth'] = data['dt_of_fr'].astype('datetime64').apply(lambda x : x.month)
    data = pp1st_func.seasoning(data)
    data.loc[data[data['season']=='봄'].index, 'season'] = 0
    data.loc[data[data['season']=='여름'].index, 'season'] = 1
    data.loc[data[data['season']=='가을'].index, 'season'] = 2
    data.loc[data[data['season']=='겨울'].index, 'season'] = 3
    
    return data


def emd_nm_encoding(data):
    data['emd_nm'] = data['emd_nm'].astype(str).apply(lambda x : x[4:] if x[:4]=='경상남도' else x )
    data['emd_nm_big'] = data['emd_nm'].apply(lambda x : x.split()[0] if x.split()[0]!='창원시' else x.split()[0]+x.split()[1])
    data['emd_nm_small'] = data['emd_nm'].apply(lambda x : x.split()[1] if (x.split()[0]!='창원시') & (x!='nan') else x)
    data['emd_nm_small'] = data['emd_nm_small'].apply(lambda x : x.split()[2] if x.split()[0]=='창원시' else x )
    
    return data