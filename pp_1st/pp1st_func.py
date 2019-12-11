import numpy as np
import pandas as pd


def Dt_of_fr(data):
    data['dt_of_fr_hr'] = data['dt_of_fr'].astype('datetime64').apply(lambda x : x.hour)
    data['dt_of_fr_mth'] = data['dt_of_fr'].astype('datetime64').apply(lambda x : x.month)
    return data

def Fr_age(data):
    data['dt_of_athrztn'].fillna(0, inplace=True)
    data['dt_of_athrztn'] = data['dt_of_athrztn'].astype(float).astype(int)
    data['dt_of_athrztn'] = data['dt_of_athrztn'].astype(str).apply(lambda x : x[:4] if len(x)==8 else x )
    data['dt_of_fr_yr'] = data['dt_of_fr'].astype('datetime64').apply(lambda x : x.year)
    data['화재시점나이'] = data['dt_of_fr_yr'] - data['dt_of_athrztn'].astype(int)
    data['화재시점나이'] = data['화재시점나이'].apply(lambda x : np.nan if len(str(x))==4 else x) # nan값은 그대로 nan값으로
    return data

def available_ar(data):
    data['lnd_ar'].replace(0.00, 0.01, inplace=True) #임시로 0.01로 변경
    data['한채당가용면적'] = data['lnd_ar']/ data['bldng_cnt']
    data.loc[data[data['lnd_ar']==0.01].index, '한채당가용면적']=np.nan
    return data
    
def expected_cnt(data):
    data['bldng_ar'].replace(0, 0.02, inplace=True)
    data['면적대비예상채수'] = data['lnd_ar'] / data['bldng_ar']
    data.loc[data[data['lnd_ar']==0.01].index, '면적대비예상채수']=np.nan
    data['예상채수초과']=0
    data.loc[data[data['bldng_cnt'] > data['면적대비예상채수']].index, '예상채수초과']=1
    return data

def floor_avg(data):
    data['한채당평균지상층수'] = data['ttl_grnd_flr'] / data['bldng_cnt']
    data['한채당평균지하층수'] = data['ttl_dwn_flr'] / data['bldng_cnt']
    data['지상지하층합'] = data['ttl_grnd_flr'] + data['ttl_dwn_flr']
    data['예상평균층당면적'] = data['ttl_ar'] / data['지상지하층합']
    return data

def seasoning(data):
    data['season'] = data['dt_of_fr_mth'].apply(lambda x : '겨울'  if (x ==12) | (x==1) | (x==2)  else x)
    data['season'] = data['season'].apply(lambda x : '봄'  if (x ==3) | (x==4) | (x==5)  else x)
    data['season'] = data['season'].apply(lambda x : '여름'  if (x ==6) | (x==7) | (x==8)  else x)
    data['season'] = data['season'].apply(lambda x : '가을'  if (x ==9) | (x==10) | (x==11)  else x)
    
#     data['계절평균온도차'] = 0
#     data.loc[data[data['season']=='봄']['tmprtr'].keys(), '계절평균온도차'] = data[data['season']=='봄']['tmprtr'].apply(lambda x : x - 13.893263003271683).get_values()
#     data.loc[data[data['season']=='여름']['tmprtr'].keys(), '계절평균온도차'] = data[data['season']=='여름']['tmprtr'].apply(lambda x : x - 24.737439966590106).get_values()
#     data.loc[data[data['season']=='가을']['tmprtr'].keys(), '계절평균온도차'] = data[data['season']=='가을']['tmprtr'].apply(lambda x : x - 15.416313269493845).get_values()
#     data.loc[data[data['season']=='겨울']['tmprtr'].keys(), '계절평균온도차'] = data[data['season']=='겨울']['tmprtr'].apply(lambda x : x - 2.671166732361056).get_values()
    
    return data


def against_wind_direction(data):
    data['반대풍향여부']=1
    inv_idx = data[(data['wnd_drctn'] < 250 ) & (data['wnd_drctn']>70)].index
    data.loc[inv_idx, '반대풍향여부']=0
    return data


def miss_bd_cls(data):
    ## 주거용
    apt_idx = data[(data['lnd_us_sttn_nm']=='아파트') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('주거지역')) | (data['bldng_us'].str.contains('주택')))].index
    data.loc[apt_idx, 'bldng_us_clssfctn']='주거용'
    apt_idx = data[(data['lnd_us_sttn_nm']=='주거기타') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('주거지역')) | (data['bldng_us'].str.contains('주택')))].index
    data.loc[apt_idx, 'bldng_us_clssfctn']='주거용'
    apt_idx = data[(data['lnd_us_sttn_nm']=='다세대') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('주거지역')) | (data['bldng_us'].str.contains('주택')))].index
    data.loc[apt_idx, 'bldng_us_clssfctn']='주거용'
    apt_idx = data[(data['lnd_us_sttn_nm']=='주상용') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('주거지역')) | (data['bldng_us'].str.contains('주택')))].index
    data.loc[apt_idx, 'bldng_us_clssfctn']='주거용'
    apt_idx = data[(data['lnd_us_sttn_nm']=='주거나지') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('주거지역')) | (data['bldng_us'].str.contains('주택')))].index
    data.loc[apt_idx, 'bldng_us_clssfctn']='주거용'
    apt_idx = data[(data['lnd_us_sttn_nm']=='주상기타') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('주거지역')) | (data['bldng_us'].str.contains('주택')))].index
    data.loc[apt_idx, 'bldng_us_clssfctn']='주거용'
    apt_idx = data[(data['lnd_us_sttn_nm']=='연립') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('주거지역')) | (data['bldng_us'].str.contains('주택')))].index
    data.loc[apt_idx, 'bldng_us_clssfctn']='주거용'
    apt_idx = data[(data['lnd_us_sttn_nm']=='단독') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('주거지역')) | (data['bldng_us'].str.contains('주택')))].index
    data.loc[apt_idx, 'bldng_us_clssfctn']='주거용'
    apt_idx = data[(data['lnd_us_sttn_nm']=='주상나지') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('주거지역')) | (data['bldng_us'].str.contains('주택')))].index
    data.loc[apt_idx, 'bldng_us_clssfctn']='주거용'
    
    ##공업용
    factory_idx = data[(data['lnd_us_sttn_nm']=='공업용') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('공업지역')) | (data['bldng_us'].str.contains('공장')))].index
    data.loc[factory_idx, 'bldng_us_clssfctn']='공업용'
    factory_idx = data[(data['lnd_us_sttn_nm']=='공업기타') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('공업지역')) | (data['bldng_us'].str.contains('공장')))].index
    data.loc[factory_idx, 'bldng_us_clssfctn']='공업용'
    factory_idx = data[(data['lnd_us_sttn_nm']=='공업나지') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('공업지역')) | (data['bldng_us'].str.contains('공장')))].index
    data.loc[factory_idx, 'bldng_us_clssfctn']='공업용'
    
    
    ##상업용
    com_idx = data[(data['lnd_us_sttn_nm']=='상업용') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('상업지역')) | (data['bldng_us'].str.contains('근린생활시설')))].index
    data.loc[com_idx, 'bldng_us_clssfctn']='상업용'
    com_idx = data[(data['lnd_us_sttn_nm']=='주상용') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('상업지역')) | (data['bldng_us'].str.contains('근린생활시설')))].index
    data.loc[com_idx, 'bldng_us_clssfctn']='상업용'
    com_idx = data[(data['lnd_us_sttn_nm']=='주상기타') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('상업지역')) | (data['bldng_us'].str.contains('근린생활시설')))].index
    data.loc[com_idx, 'bldng_us_clssfctn']='상업용'
    com_idx = data[(data['lnd_us_sttn_nm']=='상업기타') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('상업지역')) | (data['bldng_us'].str.contains('근린생활시설')))].index
    data.loc[com_idx, 'bldng_us_clssfctn']='상업용'
    com_idx = data[(data['lnd_us_sttn_nm']=='주상나지') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('상업지역')) | (data['bldng_us'].str.contains('근린생활시설')))].index
    data.loc[com_idx, 'bldng_us_clssfctn']='상업용'
    com_idx = data[(data['lnd_us_sttn_nm']=='상업나지') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('상업지역')) | (data['bldng_us'].str.contains('근린생활시설')))].index
    data.loc[com_idx, 'bldng_us_clssfctn']='상업용'
    com_idx = data[(data['lnd_us_sttn_nm']=='여객자동차터미널') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('상업지역')) | (data['bldng_us'].str.contains('근린생활시설')))].index
    data.loc[com_idx, 'bldng_us_clssfctn']='상업용'
    
    ##농업용
    farm_idx = data[(data['lnd_us_sttn_nm']=='답') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('농림지역')))].index
    data.loc[farm_idx, 'bldng_us_clssfctn']='농수산용'
    farm_idx = data[(data['lnd_us_sttn_nm']=='과수원') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('농림지역')))].index
    data.loc[farm_idx, 'bldng_us_clssfctn']='농수산용'
    
    ##문교사회용
    cul_idx = data[(data['lnd_us_sttn_nm']=='공원등') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('녹지지역')))].index
    data.loc[cul_idx, 'bldng_us_clssfctn']='문교사회용'
    cul_idx = data[(data['lnd_us_sttn_nm']=='운동장등') & (data['bldng_us_clssfctn'].isnull()) & ((data['rgnl_ar_nm'].str.contains('녹지지역')))].index
    data.loc[cul_idx, 'bldng_us_clssfctn']='문교사회용'
    
    return data


def bd_arch_group(data_):
    data_.reset_index(drop=True, inplace=True)
    ##목구조 통일
    tree_idx = data_[data_['bldng_archtctr'].isin(['통나무구조','목구조','일반목구조'])].index
    data_.loc[tree_idx, 'bldng_archtctr'] = '목구조'
    #철골구조 통일
    steel_idx = data_[data_['bldng_archtctr'].isin(['경량철골구조','일반철골구조','철골콘크리트구조','기타강구조','강파이프구조'])].index
    data_.loc[steel_idx, 'bldng_archtctr'] = '철골구조'
    #콘크리트구조 통일
    concrete_idx = data_[data_['bldng_archtctr'].isin(['철근콘크리트구조','프리케스트콘크리트구조',])].index
    data_.loc[concrete_idx, 'bldng_archtctr'] = '콘크리트구조'
    #기타조직구조
    etc_idx = data_[data_['bldng_archtctr'].isin(['기타구조','기타조적구조','조적구조'])].index
    data_.loc[etc_idx, 'bldng_archtctr'] = '기타조적구조'
    return data_

def bd_use_group(data_):
    data_.loc[data_[data_['bldng_us']=='판매시설'].index, 'bldng_us'] = '제2종근린생활시설'
    data_.loc[data_[data_['bldng_us']=='근린생활시설'].index, 'bldng_us'] = '제1종근린생활시설'
    data_.loc[data_[data_['bldng_us']=='방송통신시설'].index, 'bldng_us'] = '제1종근린생활시설'
    data_.loc[data_[data_['bldng_us']=='자동차관련시설'].index, 'bldng_us'] = '업무시설'
    data_.loc[data_[data_['bldng_us'].isin(['위험물저장및처리시설','분뇨.쓰레기처리시설'])].index, 'bldng_us'] = '기피시설'
    data_.loc[data_[data_['bldng_us'].isin(['문화및집회시설','종교시설','교육연구시설','노유자시설'])].index, 'bldng_us'] = '문화종교교육시설'
    data_.loc[data_[data_['bldng_us'].isin(['공장','창고시설'])].index, 'bldng_us'] = '공장창고시설'
    return data_

def side_road(data_):
    data_.reset_index(drop=True, inplace=True)
    data_['rd_sd_nm'].fillna('맹지', inplace=True)
    data_['측면도로크기'] = data_['rd_sd_nm'].apply(lambda x : x[:2])
    data_['차량통행가능여부']=1
    car_idx = data_[data_['rd_sd_nm'].isin(['세로한면(불)','맹지','세로각지(불)'])].index
    data_.loc[car_idx, '차량통행가능여부'] = 0
    return data_