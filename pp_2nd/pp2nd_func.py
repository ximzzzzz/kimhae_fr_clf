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
    data['lnd_ar'].replace(0, np.nan, inplace=True)
    data['ttl_ar'].replace(0, np.nan, inplace=True)
    data['bldng_ar'].replace(0, np.nan, inplace=True)
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
    
#     data['season_mean_differ'] = 0
    data.loc[data['season']==0, 'season_mean_differ'] = data[data['season']==0]['tmprtr'].apply(lambda x : x - 13.893263003271683).get_values()
    data.loc[data['season']==1, 'season_mean_differ'] = data[data['season']==1]['tmprtr'].apply(lambda x : x - 24.737439966590106).get_values()
    data.loc[data['season']==2, 'season_mean_differ'] = data[data['season']==2]['tmprtr'].apply(lambda x : x - 15.416313269493845).get_values()
    data.loc[data['season']==3, 'season_mean_differ'] = data[data['season']==3]['tmprtr'].apply(lambda x : x - 2.671166732361056).get_values()
    
    return data


def emd_nm_encoding(data):
    data['emd_nm'] = data['emd_nm'].astype(str).apply(lambda x : x[4:] if x[:4]=='경상남도' else x )
    data['emd_nm_big'] = data['emd_nm'].apply(lambda x : x.split()[0] if x.split()[0]!='창원시' else x.split()[0]+x.split()[1])
    data['emd_nm_small'] = data['emd_nm'].apply(lambda x : x.split()[1] if (x.split()[0]!='창원시') & (x!='nan') else x)
    data['emd_nm_small'] = data['emd_nm_small'].apply(lambda x : x.split()[2] if x.split()[0]=='창원시' else x )
    
    return data

def dt_of_athrztn(data, train):
    if train:
        data.loc[32635, 'dt_of_athrztn'] = 20020227 
    data['dt_of_athrztn'].fillna(0, inplace=True)
    data['dt_of_athrztn'] = data['dt_of_athrztn'].astype(float).astype(int)
    data['dt_of_athrztn'] = data['dt_of_athrztn'].astype(str).apply(lambda x : x[:4] if len(x)==8 else x )
    data['dt_of_athrztn'] = data['dt_of_athrztn'].astype(str).apply(lambda x : x[:-2] if len(x)==6 else x )
    data['dt_of_athrztn'] = data['dt_of_athrztn'].astype(str).apply(lambda x : x[:4] if len(x)==7 else x )
    data['dt_of_athrztn'] = data['dt_of_athrztn'].astype(str).apply(lambda x : '1'+ x[:3] if x[0]=='9' else x )
#     data['dt_of_athrztn'].replace('0', np.nan, inplace=True)
#     data['dt_of_athrztn'] = data['dt_of_athrztn'].astype(int)
    
    return data

def wnd_drctn_enc(data):
    data['wnd_drctn_enc'] = 0
    data.loc[(data['wnd_drctn'] >=0) & (data['wnd_drctn'] < 45), 'wnd_drctn_enc'] = 0
    data.loc[(data['wnd_drctn'] >=45) & (data['wnd_drctn'] < 90), 'wnd_drctn_enc'] = 1
    data.loc[(data['wnd_drctn'] >=90) & (data['wnd_drctn'] < 135), 'wnd_drctn_enc'] = 2
    data.loc[(data['wnd_drctn'] >=135) & (data['wnd_drctn'] < 180), 'wnd_drctn_enc'] = 3
    data.loc[(data['wnd_drctn'] >=180) & (data['wnd_drctn'] < 225), 'wnd_drctn_enc'] = 4
    data.loc[(data['wnd_drctn'] >=225) & (data['wnd_drctn'] < 270), 'wnd_drctn_enc'] = 5
    data.loc[(data['wnd_drctn'] >=270) & (data['wnd_drctn'] < 315), 'wnd_drctn_enc'] = 6
    data.loc[(data['wnd_drctn'] >=315) & (data['wnd_drctn'] < 360), 'wnd_drctn_enc'] = 7
    
    return data

def bldng_archtctr_enc(data):
    ##목구조 통일
    tree_idx = data[data['bldng_archtctr'].isin(['통나무구조','목구조','일반목구조'])].index
    data.loc[tree_idx, 'bldng_archtctr'] = '목구조'
    #철골구조 통일
    steel_idx = data[data['bldng_archtctr'].isin(['경량철골구조','일반철골구조','철골콘크리트구조','기타강구조','강파이프구조','철골철근콘크리트구조'])].index
    data.loc[steel_idx, 'bldng_archtctr'] = '철골구조'
    #콘크리트구조 통일
    concrete_idx = data[data['bldng_archtctr'].isin(['철근콘크리트구조','프리케스트콘크리트구조','기타콘크리트구조'])].index
    data.loc[concrete_idx, 'bldng_archtctr'] = '콘크리트구조'
    #기타조직구조
    etc_idx = data[data['bldng_archtctr'].isin(['기타구조','기타조적구조','조적구조'])].index
    data.loc[etc_idx, 'bldng_archtctr'] = '기타조적구조'
    
    return data