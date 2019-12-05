from pp1st_func import *
from sklearn.preprocessing import LabelEncoder as lbe
import numpy as np
import pandas as pd


def Pipeline_var(data):
    data = Dt_of_fr(data)
    data = Fr_age(data)
    data = available_ar(data)
    data = expected_cnt(data)
    data = floor_avg(data)
    data = seasoning(data)
    data = against_wind_direction(data)
    data = miss_bd_cls(data)
    data = bd_arch_group(data)
    data = bd_use_group(data)
    data = side_road(data)
    
    dataset = data[['fr_yn','bldng_us','bldng_archtctr', 'bldng_cnt', 'bldng_ar','ttl_ar','lnd_ar','ttl_grnd_flr','ttl_dwn_flr','bldng_us_clssfctn',
                'tmprtr','wnd_spd','wnd_drctn','hmdt','jmk','rgnl_ar_nm','lnd_us_sttn_nm','fr_sttn_dstnc','bldng_ar_prc','fr_wthr_fclt_dstnc',
                'fr_mn_cnt', 'mlt_us_yn','cctv_dstnc', 'fr_wthr_fclt_in_100m','cctv_in_100m','tbc_rtl_str_dstnc','sft_emrgnc_bll_dstnc','ahsm_dstnc',
                'no_tbc_zn_dstnc','bldng_cnt_in_50m','dt_of_fr_hr','dt_of_fr_mth','화재시점나이','한채당가용면적','면적대비예상채수','예상채수초과',
                '한채당평균지상층수','한채당평균지하층수','예상평균층당면적','season','계절평균온도차','반대풍향여부','측면도로크기','차량통행가능여부']]
    
    return dataset

def Pipeline_le(dataset):
    fr_yn_lb = lbe()
    dataset['fr_yn'] = fr_yn_lb.fit_transform(dataset['fr_yn'])
    
    
    bldng_us_lb = lbe()
    dataset['bldng_us'] = bldng_us_lb.fit_transform(dataset['bldng_us'])
    bldng_archtctr_lb = lbe()
    dataset['bldng_archtctr'] = bldng_archtctr_lb.fit_transform(dataset['bldng_archtctr'])
    bldng_us_clssfctn_lb = lbe()
    dataset['bldng_us_clssfctn'] = bldng_us_clssfctn_lb.fit_transform(dataset['bldng_us_clssfctn'])
    jmk_lb = lbe()
    dataset['jmk'] = jmk_lb.fit_transform(dataset['jmk'])
    rgnl_ar_nm_lb = lbe()
    dataset['rgnl_ar_nm'] = rgnl_ar_nm_lb.fit_transform(dataset['rgnl_ar_nm'])
    lnd_us_sttn_nm_lb = lbe()
    dataset['lnd_us_sttn_nm'] = lnd_us_sttn_nm_lb.fit_transform(dataset['lnd_us_sttn_nm'])
    mlt_us_yn_lb = lbe()
    dataset['mlt_us_yn'] = mlt_us_yn_lb.fit_transform(dataset['mlt_us_yn'])
    season_lb = lbe()
    dataset['season'] = season_lb.fit_transform(dataset['season'])
    측면도로크기_lb = lbe()
    dataset['측면도로크기'] = 측면도로크기_lb.fit_transform(dataset['측면도로크기'])
    
    return dataset
    