import pandas as pd
import numpy as np
import sys
sys.path.append('../pp_1st')
import pp1st_pipeline
import pp1st_func
import pp2nd_func
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def weather_mean(data):
    data.loc[data['tmprtr'].isnull(), 'tmprtr'] = data['tmprtr'].mean()
    data.loc[data['hmdt'].isnull(), 'hmdt'] = data['hmdt'].mean()
    return data


def energy_mean(data):
    ele_col = data.columns[data.columns.str.contains('ele')]
    gas_col = data.columns[data.columns.str.contains('gas')]
    
    data['gas_mean'] = np.nan
    for idx, row in data[(data[gas_col].mean(1)!=0) & (data[gas_col].mean(1).notna())].iterrows():
        use_period=0
        use_total=0
        for col in gas_col:
            if row[col]!=0:
                use_period+=1
                use_total+=row[col]
#                 print('idx : ',idx)
#                 print('use_period : ', use_period, 'columns : ', col)
#                 print('total : ',use_total)
        use_mean = use_total / use_period
        data.loc[idx,'gas_mean'] = use_mean
    data.loc[(data[gas_col].mean(1)==0), 'gas_mean'] = 0
    data.loc[(data[gas_col].mean(1)==0) & (data['us_yn']=='Y'), 'gas_mean'] = np.nan
#     data['gas_mean'].fillna(data['gas_mean'].mean())

    
    data['ele_mean'] = np.nan
    for idx, row in data[(data[ele_col].mean(1)!=0) & (data[ele_col].mean(1).notna())].iterrows():
        use_period=0
        use_total=0
        for col in ele_col:
            if row[col]!=0:
                use_period+=1
                use_total+=row[col]
        use_mean = use_total / use_period
        data.loc[idx,'ele_mean'] = use_mean
    data.loc[(data[ele_col].mean(1)==0), 'ele_mean'] = 0    
    data.loc[(data[ele_col].mean(1)==0) & (data['us_yn']=='Y'), 'ele_mean'] = np.nan
#     data['ele_mean'].fillna(data['ele_mean'].mean())


    data = data.join( pd.get_dummies(data['jmk'],prefix='jmk'))
    
    return data

def Hm_cnt_log_enc(data):
    data.loc[(data['hm_cnt_log'] < 8),'hm_cnt_log_enc'] = 0
    data.loc[(data['hm_cnt_log'] >= 8) & (data['hm_cnt_log'] <9) ,'hm_cnt_log_enc'] = 1
    data.loc[(data['hm_cnt_log'] >= 9) & (data['hm_cnt_log'] <10) ,'hm_cnt_log_enc'] = 2
    data.loc[(data['hm_cnt_log'] >= 10) ,'hm_cnt_log_enc'] = 3
    return data

def Hm_cnt(data):
    data_nn = data[data['hm_cnt'].notna()]
    for small in data[data['hm_cnt'].isnull()]['emd_nm_small'].unique():
        hm_cnt_mean = data_nn[data_nn['emd_nm_small']==small]['hm_cnt'].mean()
        data.loc[(data['hm_cnt'].isnull()) & (data['emd_nm_small']==small), 'hm_cnt'] = hm_cnt_mean
    data['hm_cnt_log'] = np.log(data['hm_cnt'])

#     data.loc[(data['hm_cnt_log'] < 7),'hm_cnt_log_enc'] = 0
#     data.loc[(data['hm_cnt_log'] >= 7) & (data['hm_cnt_log'] <8) ,'hm_cnt_log_enc'] = 1
#     data.loc[(data['hm_cnt_log'] >= 8) & (data['hm_cnt_log'] <9) ,'hm_cnt_log_enc'] = 2
#     data.loc[(data['hm_cnt_log'] >= 9) & (data['hm_cnt_log'] <10) ,'hm_cnt_log_enc'] = 3
#     data.loc[(data['hm_cnt_log'] >= 10) & (data['hm_cnt_log'] <11) ,'hm_cnt_log_enc'] = 4
#     data.loc[(data['hm_cnt_log'] >= 11),'hm_cnt_log_enc'] = 5

    data = Hm_cnt_log_enc(data)
    
    return data


def date_author_pre(data, train):
    data = pp2nd_func.dt_of_athrztn(data,train)
    data['dt_of_athrztn'] = data['dt_of_athrztn'].astype(int)
    data['dt_of_athrztn'].replace(0, np.nan , inplace=True)
    data['dt_of_athrztn_enc'] = np.nan
    
    data.loc[data['dt_of_athrztn'] < 1950, 'dt_of_athrztn_enc'] = 0
    data.loc[(data['dt_of_athrztn'] > 1950) & (data['dt_of_athrztn'] <= 1970), 'dt_of_athrztn_enc'] = 1
    data.loc[(data['dt_of_athrztn'] > 1970) & (data['dt_of_athrztn'] <= 1990), 'dt_of_athrztn_enc'] = 2
    data.loc[(data['dt_of_athrztn'] > 1990) & (data['dt_of_athrztn'] <= 2010), 'dt_of_athrztn_enc'] = 3
    data.loc[data['dt_of_athrztn'] > 2010, 'dt_of_athrztn_enc'] = 4
    return data


def Wnd_spd_log(data):
    data['wnd_spd_log'] = np.log(data['wnd_spd'] +1)
    return data


def Weather(data, weather_imputer):
    data.reset_index(drop=True, inplace=True)
#     data['fr_yn'] = data['fr_yn'].astype('category')
    data['season'] = data['season'].astype('category')
    weather_data = data[['season','season_mean_differ','hmdt','wnd_drctn_enc','wnd_spd','tmprtr']]
    weather_imp_res = weather_imputer.transform(weather_data)
    weather_imp_df = pd.DataFrame(weather_imp_res, columns=weather_data.columns)
    data = pd.concat([data.drop(columns=['season','season_mean_differ','hmdt','wnd_drctn_enc','wnd_spd','tmprtr']),
                      weather_imp_df[['season','season_mean_differ','hmdt','wnd_drctn_enc','wnd_spd','tmprtr']]], axis=1)
#     data = pp2nd_func.wnd_drctn_enc(data)
    return data


def Jmk_enc(data):
    data.loc[data['jmk'].isin(['과','목','전','답']), 'jmk_enc'] = 0 #농업용
    data.loc[data['jmk'].isin(['임','광']), 'jmk_enc'] = 1 #자연그자체
    data.loc[data['jmk'].isin(['공','체','원','학']), 'jmk_enc'] = 2 #공원
    data.loc[data['jmk'].isin(['대','차']), 'jmk_enc'] = 3 #상업
    data.loc[data['jmk'].isin(['구','천','유','양','염','수','제']), 'jmk_enc'] = 4 # 수자원시설
    data.loc[data['jmk'].isin(['종','사','묘','잡','도','철']), 'jmk_enc'] = 5 # 잡시설
    data.loc[data['jmk'].isin(['장','주','창','잡']), 'jmk_enc'] = 5 # 잡시설
    return data

def Fr_mn_cnt(data):
    for big in data[data['fr_mn_cnt'].isnull()]['emd_nm_big'].unique():
        fr_mean = data[data['emd_nm_big']==big]['fr_mn_cnt'].mean()
        data.loc[(data['fr_mn_cnt'].isnull()) & (data['emd_nm_big']==big), 'fr_mn_cnt'] = fr_mean
    return data





def regional_name_pre(data):
    data.loc[data['rgnl_ar_nm'].str.contains('주거지역',na=False), 'rgnl_ar_nm_enc'] = 0 #주거지역
    data.loc[data['rgnl_ar_nm'].str.contains('녹지지역',na=False), 'rgnl_ar_nm_enc'] = 1 #녹지지역
    data.loc[data['rgnl_ar_nm'].str.contains('개발제한',na=False), 'rgnl_ar_nm_enc'] = 1 #녹지지역
    data.loc[data['rgnl_ar_nm'].str.contains('보전',na=False),  'rgnl_ar_nm_enc'] =    1 #녹지지역
    data.loc[data['rgnl_ar_nm'].str.contains('상업지역',na=False), 'rgnl_ar_nm_enc'] = 2 #상업지역
    data.loc[data['rgnl_ar_nm'].str.contains('공업지역',na=False), 'rgnl_ar_nm_enc'] = 3 #공업지역
    data.loc[data['rgnl_ar_nm'].str.contains('관리지역',na=False), 'rgnl_ar_nm_enc'] = 4 #관리지역
    data.loc[data['rgnl_ar_nm'].str.contains('농림지역',na=False), 'rgnl_ar_nm_enc'] = 5 #녹지지역
    data.loc[(data['rgnl_ar_nm_enc'].isnull()) & (data['bldng_us_clssfctn']=='상업용'),'rgnl_ar_nm_enc']=2
    
    data['rgnl_ar_nm_enc'] = data['rgnl_ar_nm_enc'].astype('category')
    data['jmk_enc'] = data['jmk_enc'].astype('category')
    data['mlt_us_yn'] = data['mlt_us_yn'].astype('category')
    data['dt_of_athrztn_enc'] = data['dt_of_athrztn_enc'].astype('category')
    
    return data




def building_price(data, bldng_ar_prc_imputer):
    data.reset_index(drop=True, inplace=True)
    bldng_ar_prc_impres=bldng_ar_prc_imputer.transform(data[['bldng_ar_prc_log','dt_of_athrztn_enc','dt_of_athrztn_rflct_log','hm_cnt_log','bldng_ar','ttl_ar','ttl_grnd_flr','ttl_dwn_flr','fr_wthr_fclt_dstnc','tbc_rtl_str_dstnc','ele_mean','rgnl_ar_nm_enc']])
    
    bldng_ar_prc_df = pd.DataFrame(bldng_ar_prc_impres, columns = ['bldng_ar_prc_log','dt_of_athrztn_enc','dt_of_athrztn_rflct_log','hm_cnt_log','bldng_ar','ttl_ar','ttl_grnd_flr','ttl_dwn_flr','fr_wthr_fclt_dstnc','tbc_rtl_str_dstnc','ele_mean','rgnl_ar_nm_enc'])
    data = pd.concat([data.drop(columns = ['bldng_ar_prc_log','hm_cnt_log']), bldng_ar_prc_df[['bldng_ar_prc_log','hm_cnt_log']]], axis=1)
    
    return data


def regional_name_post(data, regional_imputer):
    data.reset_index(drop=True, inplace=True)
    rgnl_nm_impres = regional_imputer.transform(data[['rgnl_ar_nm_enc','bldng_ar_prc',
                                                     'bldng_ar_prc','jmk_enc','hm_cnt_log','hm_cnt','bldng_ar','ttl_ar','jmk_enc']])
    rgnl_nm_df = pd.DataFrame(rgnl_nm_impres, columns = ['rgnl_ar_nm_enc','bldng_ar_prc','bldng_ar_prc','jmk_enc','hm_cnt_log','hm_cnt','bldng_ar',
                                                         'ttl_ar','jmk_enc'])
    data = pd.concat([data.drop(columns = 'rgnl_ar_nm_enc'), rgnl_nm_df['rgnl_ar_nm_enc']], axis=1)
    return data
            
    
def date_author_post(data, date_author_imputer):
    data.reset_index(drop=True, inplace=True)
    date_impres = date_author_imputer.transform(data[['dt_of_athrztn_enc','dt_of_athrztn_rflct_log','bldng_cnt','bldng_ar','jmk_enc','gas_mean','ele_mean','hm_cnt',
                                                      'hm_cnt_log','tbc_rtl_str_dstnc','bldng_ar_prc_log','bldng_ar_prc','rgnl_ar_nm_enc']])
    
    date_df = pd.DataFrame(date_impres, columns = ['dt_of_athrztn_enc','dt_of_athrztn_rflct_log','bldng_cnt','bldng_ar',
                                                   'jmk_enc','gas_mean','ele_mean','hm_cnt','hm_cnt_log','tbc_rtl_str_dstnc',
                                                   'bldng_ar_prc_log','bldng_ar_prc', 'rgnl_ar_nm_enc'])
    data = pd.concat([data.drop(columns = ['dt_of_athrztn_enc','dt_of_athrztn_rflct_log']), 
                      date_df[['dt_of_athrztn_enc','dt_of_athrztn_rflct_log']]] ,axis=1)
    return data
    
    
def log_maker(data, var):
    data[var+'_log'] = np.log(data[var])
    return data

def area_predict(data, ar_imputer):
    data.reset_index(drop=True, inplace=True)
    ar_impres = ar_imputer.transform(data[['ttl_ar_log','bldng_ar_log','lnd_ar_log','hm_cnt_log','fr_mn_cnt','rgnl_ar_nm_enc','dt_of_athrztn_enc',
                               'jmk_enc','bldng_ar_prc_log','fr_wthr_fclt_dstnc','tbc_rtl_str_dstnc']])
    ar_df = pd.DataFrame(ar_impres, columns=['ttl_ar_log','bldng_ar_log','lnd_ar_log','hm_cnt_log','fr_mn_cnt','rgnl_ar_nm_enc',
                                            'dt_of_athrztn_enc','jmk_enc','bldng_ar_prc_log','fr_wthr_fclt_dstnc','tbc_rtl_str_dstnc'])
    data = pd.concat([data.drop(columns = ['ttl_ar_log','bldng_ar_log','lnd_ar_log']), ar_df[['ttl_ar_log','bldng_ar_log','lnd_ar_log']]], axis=1)
    return data


def from_category_to_int(dataset):
    dataset['dt_of_athrztn_enc']  = dataset['dt_of_athrztn_enc'].astype('int')
    dataset['rgnl_ar_nm_enc']  = dataset['rgnl_ar_nm_enc'].astype('int')
    dataset['wnd_drctn_enc']  = dataset['wnd_drctn_enc'].astype('int')
    dataset['jmk_enc']  = dataset['jmk_enc'].astype('int')
#     dataset['fr_yn']  = dataset['fr_yn'].astype('int')
    dataset['season']  = dataset['season'].astype('int')
    
    return dataset

def from_int_to_category(dataset):
    dataset['dt_of_athrztn_enc']  = dataset['dt_of_athrztn_enc'].astype('category')
    dataset['rgnl_ar_nm_enc']  = dataset['rgnl_ar_nm_enc'].astype('category')
    dataset['wnd_drctn_enc']  = dataset['wnd_drctn_enc'].astype('category')
    dataset['jmk_enc']  = dataset['jmk_enc'].astype('category')
#     dataset['fr_yn']  = dataset['fr_yn'].astype('int')
    dataset['season']  = dataset['season'].astype('category')
    dataset['dt_of_fr_hr_enc']  = dataset['dt_of_fr_hr_enc'].astype('category')
    dataset['dt_of_fr_hr']  = dataset['dt_of_fr_hr'].astype('category')
    dataset['dt_of_fr_mth']  = dataset['dt_of_fr_mth'].astype('category')
    dataset['hm_cnt_log_enc']  = dataset['hm_cnt_log_enc'].astype('category')
    dataset['dt_of_fr_mth']  = dataset['dt_of_fr_mth'].astype('category')
    return dataset