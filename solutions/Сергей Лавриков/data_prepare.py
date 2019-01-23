import pandas as pd
import numpy as np
from datetime import datetime
import os
import gc

from data_loading import CACHE_DIR, load_data_session, load_csi_train, load_avg_kpi, load_chnn_kpi, load_csi_test, \
    load_voice_session, load_consumption

pd.set_option('display.expand_frame_repr', False)


def add_features(df, features_df):
    features_2m_df = features_mean(features_df, 2)
    features_3m_df = features_mean(features_df, 3)
    features_6m_df = features_mean(features_df, 6)

    features_diff_1m_df = features_df[['SK_ID',
                                       'REVENUE', 'ITC', 'VAS', 'RENT_CHANNEL', 'ROAM', 'COST',
                                       'COM_CAT#17', 'COM_CAT#18', 'COM_CAT#19', 'COM_CAT#20', 'COM_CAT#21',
                                       'COM_CAT#22', 'COM_CAT#23', 'COM_CAT#27', 'COM_CAT#28', 'COM_CAT#29',
                                       'COM_CAT#30', 'COM_CAT#31', 'COM_CAT#32', 'COM_CAT#33']] \
        .set_index('SK_ID') \
        .diff(periods=-1) \
        .reset_index(drop=False)
    features_diff_2m_df = features_diff_1m_df \
        .shift(-1) \
        .drop_duplicates(['SK_ID'])
    features_diff_3m_df = features_diff_1m_df \
        .shift(-2) \
        .drop_duplicates(['SK_ID'])
    features_diff_1m_df = features_diff_1m_df \
        .drop_duplicates(['SK_ID'])

    features_df = features_df \
        .drop_duplicates(['SK_ID'])

    features_div_1m_2m_df = features_divide(features_diff_1m_df, features_2m_df)
    features_div_1m_3m_df = features_divide(features_diff_1m_df, features_3m_df)
    features_div_1m_6m_df = features_divide(features_diff_1m_df, features_6m_df)

    res_df = pd.merge(df, features_df, on='SK_ID', how='left')

    res_df = pd.merge(res_df, features_2m_df, on='SK_ID', how='left', suffixes=('', '_2m'))
    res_df = pd.merge(res_df, features_3m_df, on='SK_ID', how='left', suffixes=('', '_3m'))
    res_df = pd.merge(res_df, features_6m_df, on='SK_ID', how='left', suffixes=('', '_6m'))

    res_df = pd.merge(res_df, features_diff_1m_df, on='SK_ID', how='left', suffixes=('', '_diff_1m'))
    res_df = pd.merge(res_df, features_diff_2m_df, on='SK_ID', how='left', suffixes=('', '_diff_2m'))
    res_df = pd.merge(res_df, features_diff_3m_df, on='SK_ID', how='left', suffixes=('', '_diff_3m'))

    res_df = pd.merge(res_df, features_div_1m_2m_df, on='SK_ID', how='left', suffixes=('', '_div_1m_2m'))
    res_df = pd.merge(res_df, features_div_1m_3m_df, on='SK_ID', how='left', suffixes=('', '_div_1m_3m'))
    res_df = pd.merge(res_df, features_div_1m_6m_df, on='SK_ID', how='left', suffixes=('', '_div_1m_6m'))

    return res_df.fillna(0)


def features_divide(features_diff_1m_df, features_2m_df):
    features_div_1m_2m_df = features_diff_1m_df \
        .set_index('SK_ID') \
        .divide(features_2m_df.set_index('SK_ID'), fill_value=1.0) \
        .reset_index(drop=False)
    return features_div_1m_2m_df


def features_mean(features_df, window_size):
    features_mean_df = features_df[['SK_ID',
                                    'REVENUE', 'ITC', 'VAS', 'RENT_CHANNEL', 'ROAM', 'COST',
                                    'COM_CAT#17', 'COM_CAT#18', 'COM_CAT#19', 'COM_CAT#20', 'COM_CAT#21',
                                    'COM_CAT#22', 'COM_CAT#23', 'COM_CAT#27', 'COM_CAT#28', 'COM_CAT#29',
                                    'COM_CAT#30', 'COM_CAT#31', 'COM_CAT#32', 'COM_CAT#33']] \
        .rolling(window_size) \
        .mean() \
        .drop_duplicates(['SK_ID'])
    features_mean_df = features_mean_df[features_mean_df['SK_ID'] == features_mean_df['SK_ID'] // 1]
    features_mean_df['SK_ID'] = features_mean_df['SK_ID'].astype(np.uint32)
    return features_mean_df


def session_kpi(csi_df, session_df, kpi_df, name: str):
    if os.path.isfile(os.path.join(CACHE_DIR, f'{name}.feather')):
        session_df = pd.read_feather(os.path.join(CACHE_DIR, f'{name}.feather'))

        session_df[session_df.select_dtypes(include=[np.float32]).columns] = \
            session_df[session_df.select_dtypes(include=[np.float32]).columns].astype(np.float16)
    else:
        session_df = session_df[session_df['START_TIME'] >= '2018-03-01']
        session_df.loc[:, 'START_TIME'] = session_df.loc[:, 'START_TIME'].dt.date
        session_df.loc[:, 'COUNT'] = 1.0
        session_df = session_df.groupby(by=['SK_ID', 'CELL_LAC_ID', 'START_TIME'], as_index=False).sum()
        session_df = pd.merge(session_df, csi_df[['SK_ID', 'CONTACT_DATE']], on='SK_ID', how='left')
        session_df = session_df[session_df['CONTACT_DATE'].dt.date - session_df['START_TIME'] <= pd.Timedelta(days=30)]
        session_df.loc[:, 'T_DATE'] = pd.to_datetime(session_df['START_TIME'])
        session_df.drop('START_TIME', inplace=True, axis=1)
        session_df.loc[:, 'CELL_LAC_ID'] = session_df.loc['CELL_LAC_ID'].astype(np.uint32)

        kpi_df['T_DATE'] = pd.to_datetime(kpi_df['T_DATE'])

        session_df = pd.merge(session_df, kpi_df, on=['CELL_LAC_ID', 'T_DATE'], how='left')

        session_df[session_df.select_dtypes(include=[np.float16]).columns] = \
            session_df[session_df.select_dtypes(include=[np.float16]).columns].astype(np.float32)
        session_df[session_df.select_dtypes(include=['datetime64']).columns] = \
            session_df[session_df.select_dtypes(include=['datetime64']).columns].astype(str)

        session_df.to_feather(os.path.join(CACHE_DIR, f'{name}.feather'))

    return session_df


def main_cell_kpi(csi_df, cons_df, kpi_df, name: str):
    if os.path.isfile(os.path.join(CACHE_DIR, f'{name}.feather')):
        cons_df = pd.read_feather(os.path.join(CACHE_DIR, f'{name}.feather'))

        cons_df[cons_df.select_dtypes(include=[np.float32]).columns] = \
            cons_df[cons_df.select_dtypes(include=[np.float32]).columns].astype(np.float16)
    else:
        cons_df = cons_df[cons_df['MON'] >= '2018-03-01']
        cons_df = pd.merge(cons_df, csi_df[['SK_ID', 'CONTACT_DATE']], on='SK_ID', how='left')
        cons_df = cons_df[cons_df['CONTACT_DATE'].dt.to_period('M') == cons_df['MON'].dt.to_period('M')].drop(
            ['CONTACT_DATE'], axis=1)
        cons_df['MON'] = (cons_df['MON'] - pd.offsets.DateOffset(months=1)).dt.to_period('M')
        cons_df['CELL_LAC_ID'] = cons_df['CELL_LAC_ID'].astype(np.uint32)
        cons_df = cons_df\
            .groupby(by=['SK_ID', 'CELL_LAC_ID'], as_index=False)\
            .sum()
        voice_cons_df = cons_df\
            .sort_values(by=['SK_ID', 'SUM_MINUTES'], ascending=False)\
            .drop_duplicates('SK_ID')[['SK_ID', 'CELL_LAC_ID']]
        data_cons_df = cons_df\
            .sort_values(by=['SK_ID', 'SUM_DATA_MB'], ascending=False)\
            .drop_duplicates('SK_ID')[['SK_ID', 'CELL_LAC_ID']]
        data_min_cons_df = cons_df\
            .sort_values(by=['SK_ID', 'SUM_DATA_MIN'], ascending=False)\
            .drop_duplicates('SK_ID')[['SK_ID', 'CELL_LAC_ID']]

        cons_df = pd.merge(voice_cons_df, data_cons_df, on='SK_ID', how='outer', suffixes=('_voice', '_data'))
        cons_df = pd.merge(cons_df, data_min_cons_df, on='SK_ID', how='outer')\
            .rename(index=str, columns={'CELL_LAC_ID': 'CELL_LAC_ID_data_min'})
        print(cons_df.head())

        mean_kpi_df = kpi_df.groupby(by=['CELL_LAC_ID']).mean()

        cons_df = pd.merge(cons_df, mean_kpi_df,
                           left_on='CELL_LAC_ID_voice',
                           right_on='CELL_LAC_ID',
                           how='left',
                           suffixes=('', f'_voice_main'))

        cons_df = pd.merge(cons_df, mean_kpi_df,
                           left_on='CELL_LAC_ID_data',
                           right_on='CELL_LAC_ID',
                           how='left',
                           suffixes=('', f'_data_main'))

        cons_df = pd.merge(cons_df, mean_kpi_df,
                           left_on='CELL_LAC_ID_data_min',
                           right_on='CELL_LAC_ID',
                           how='left',
                           suffixes=('', f'_data_min_main'))

        cons_df[cons_df.select_dtypes(include=[np.float16]).columns] = \
            cons_df[cons_df.select_dtypes(include=[np.float16]).columns].astype(np.float32)

        cons_df.to_feather(os.path.join(CACHE_DIR, f'{name}.feather'))

    return cons_df


def add_consumption(df, cons_df):
    last_cons_df = cons_df\
        .groupby(by=['SK_ID', 'MON'], as_index=False)\
        .sum()\
        .sort_values(by=['MON'], ascending=False)\
        .drop_duplicates(['SK_ID'])[['SK_ID', 'SUM_MINUTES', 'SUM_DATA_MB', 'SUM_DATA_MIN']]
    mean_cons_df = cons_df\
        .groupby(by=['SK_ID', 'MON'], as_index=False)\
        .sum()\
        .groupby(by=['SK_ID'], as_index=False)\
        .mean()[['SK_ID', 'SUM_MINUTES', 'SUM_DATA_MB', 'SUM_DATA_MIN']]
    mean_cons_df = pd.merge(mean_cons_df, last_cons_df, on='SK_ID', how='left', suffixes=('', '_last'))
    mean_cons_df['SUM_MINUTES_rate'] = mean_cons_df['SUM_MINUTES_last'] / mean_cons_df['SUM_MINUTES']
    mean_cons_df['SUM_DATA_MB_rate'] = mean_cons_df['SUM_DATA_MB_last'] / mean_cons_df['SUM_DATA_MB']
    mean_cons_df['SUM_DATA_MIN_rate'] = mean_cons_df['SUM_DATA_MIN_last'] / mean_cons_df['SUM_DATA_MIN']

    cons_df = cons_df[cons_df['MON'] >= '2018-03-01']
    cons_df = pd.merge(cons_df, df[['SK_ID', 'CONTACT_DATE']], on='SK_ID', how='left')
    cons_df = cons_df[cons_df['CONTACT_DATE'].dt.to_period('M') == cons_df['MON'].dt.to_period('M')].drop(
        ['CONTACT_DATE'], axis=1)
    cons_df['CELL_LAC_ID'] = cons_df['CELL_LAC_ID'].astype(np.uint32)
    cons_df['DATA_SPEED'] = cons_df['SUM_DATA_MB'] / cons_df['SUM_DATA_MIN']

    gr_cons_df = cons_df.groupby(by=['SK_ID'])
    cons_df = gr_cons_df.mean()
    cons_df['DATA_SPEED_weighted_data'] = gr_cons_df.apply(weighted_mean, 'DATA_SPEED', 'SUM_DATA_MB')
    cons_df['DATA_SPEED_weighted_data_min'] = gr_cons_df.apply(weighted_mean, 'DATA_SPEED', 'SUM_DATA_MIN')
    cons_df['DATA_SPEED_weighted_voice'] = gr_cons_df.apply(weighted_mean, 'DATA_SPEED', 'SUM_MINUTES')
    cons_df['DATA_SPEED_weighted_data_rel'] = cons_df['DATA_SPEED_weighted_data'] / cons_df['DATA_SPEED']
    cons_df['DATA_SPEED_weighted_data_min_rel'] = cons_df['DATA_SPEED_weighted_data_min'] / cons_df['DATA_SPEED']
    cons_df['DATA_SPEED_weighted_voice_rel'] = cons_df['DATA_SPEED_weighted_voice'] / cons_df['DATA_SPEED']

    cons_df = pd.merge(cons_df, mean_cons_df, on='SK_ID', how='left', suffixes=('', '_mean'))

    return pd.merge(df, cons_df, on='SK_ID', how='left')


def kpi_groupby(cons_kpi_df, last_days):
    return cons_kpi_df[
        pd.to_datetime(cons_kpi_df['CONTACT_DATE']) - pd.to_datetime(cons_kpi_df['T_DATE']) <= pd.Timedelta(
            days=last_days)] \
        .groupby(by=['SK_ID'], as_index=False)


def add_kpi(df, avg_kpi_df, chnn_kpi_df, type: str):
    res_df = pd.merge(df, kpi_groupby(avg_kpi_df, 3).min(), on='SK_ID', how='left',
                      suffixes=('', f'_{type}_avg_min_3d'))
    res_df = pd.merge(res_df, kpi_groupby(avg_kpi_df, 3).max(), on='SK_ID', how='left',
                      suffixes=('', f'_{type}_avg_max_3d'))
    res_df = pd.merge(res_df, kpi_groupby(avg_kpi_df, 3).mean(), on='SK_ID', how='left',
                      suffixes=('', f'_{type}_avg_mean_3d'))
    res_df = pd.merge(res_df, kpi_groupby(avg_kpi_df, 7).min(), on='SK_ID', how='left',
                      suffixes=('', f'_{type}_avg_min_7d'))
    res_df = pd.merge(res_df, kpi_groupby(avg_kpi_df, 7).max(), on='SK_ID', how='left',
                      suffixes=('', f'_{type}_avg_max_7d'))
    res_df = pd.merge(res_df, kpi_groupby(avg_kpi_df, 7).mean(), on='SK_ID', how='left',
                      suffixes=('', f'_{type}_avg_mean_7d'))
    res_df = pd.merge(res_df, kpi_groupby(avg_kpi_df, 14).min(), on='SK_ID', how='left',
                      suffixes=('', f'_{type}_avg_min_14d'))
    res_df = pd.merge(res_df, kpi_groupby(avg_kpi_df, 14).max(), on='SK_ID', how='left',
                      suffixes=('', f'_{type}_avg_max_14d'))
    res_df = pd.merge(res_df, kpi_groupby(avg_kpi_df, 14).mean(), on='SK_ID', how='left',
                      suffixes=('', f'_{type}_avg_mean_14d'))

    res_df = pd.merge(res_df, kpi_groupby(chnn_kpi_df, 3).min(), on='SK_ID', how='left',
                      suffixes=('', f'_{type}_chnn_min_3d'))
    res_df = pd.merge(res_df, kpi_groupby(chnn_kpi_df, 3).max(), on='SK_ID', how='left',
                      suffixes=('', f'_{type}_chnn_max_3d'))
    res_df = pd.merge(res_df, kpi_groupby(chnn_kpi_df, 3).mean(), on='SK_ID', how='left',
                      suffixes=('', f'_{type}_chnn_mean_3d'))
    res_df = pd.merge(res_df, kpi_groupby(chnn_kpi_df, 7).min(), on='SK_ID', how='left',
                      suffixes=('', f'_{type}_chnn_min_7d'))
    res_df = pd.merge(res_df, kpi_groupby(chnn_kpi_df, 7).max(), on='SK_ID', how='left',
                      suffixes=('', f'_{type}_chnn_max_7d'))
    res_df = pd.merge(res_df, kpi_groupby(chnn_kpi_df, 7).mean(), on='SK_ID', how='left',
                      suffixes=('', f'_{type}_chnn_mean_7d'))
    res_df = pd.merge(res_df, kpi_groupby(chnn_kpi_df, 14).min(), on='SK_ID', how='left',
                      suffixes=('', f'_{type}_chnn_min_14d'))
    res_df = pd.merge(res_df, kpi_groupby(chnn_kpi_df, 14).max(), on='SK_ID', how='left',
                      suffixes=('', f'_{type}_chnn_max_14d'))
    res_df = pd.merge(res_df, kpi_groupby(chnn_kpi_df, 14).mean(), on='SK_ID', how='left',
                      suffixes=('', f'_{type}_chnn_mean_14d'))

    return res_df


def as_category(df):
    for c in categorical:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype('category')
    return df


def add_weekday(df, data_field: str):
    df[data_field + '_WEEKDAY'] = df[data_field].dt.weekday
    return df


HOLIDAYS = [
    datetime(2018, 4, 30),
    datetime(2018, 5, 1),
    datetime(2018, 5, 2),
    datetime(2018, 5, 9),
    datetime(2018, 5, 15),
]


def add_holidays(df, data_field: str):
    for i, date in enumerate(HOLIDAYS):
        df[data_field + f'_{i}_HOLIDAYS'] = ((df[data_field] - HOLIDAYS[i]).dt.days).clip(-1, 3)
    return df


def merge_all(df, feat_df, cons_df,
              data_avg_df, data_chnn_df, voice_avg_df, voice_chnn_df,
              main_cell_avg_df, main_cell_chnn_df):
    data_rel_avg_df = mult_by_column(data_avg_df, 'DATA_VOL_MB')
    data_rel_chnn_df = mult_by_column(data_chnn_df, 'DATA_VOL_MB')
    voice_rel_avg_df = mult_by_column(voice_avg_df, 'VOICE_DUR_MIN')
    voice_rel_chnn_df = mult_by_column(voice_chnn_df, 'VOICE_DUR_MIN')
    gc.collect()

    data_count_avg_df = mult_by_column(data_avg_df, 'COUNT')
    data_count_chnn_df = mult_by_column(data_chnn_df, 'COUNT')
    voice_count_avg_df = mult_by_column(voice_avg_df, 'COUNT')
    voice_count_chnn_df = mult_by_column(voice_chnn_df, 'COUNT')
    gc.collect()

    df = add_features(df, feat_df)
    df = add_consumption(df, cons_df)
    df = add_kpi(df, data_avg_df, data_chnn_df, 'data')
    df = add_kpi(df, voice_avg_df, voice_chnn_df, 'voice')
    df = add_kpi(df, data_rel_avg_df, data_rel_chnn_df, 'data_rel')
    df = add_kpi(df, voice_rel_avg_df, voice_rel_chnn_df, 'voice_rel')
    df = add_kpi(df, data_count_avg_df, data_count_chnn_df, 'data_count')
    df = add_kpi(df, voice_count_avg_df, voice_count_chnn_df, 'voice_count')

    df = pd.merge(df, main_cell_avg_df, on='SK_ID', how='left', suffixes=('', f'_avg'))
    df = pd.merge(df, main_cell_chnn_df, on='SK_ID', how='left', suffixes=('', f'_chnn'))

    return df


def mult_by_column(df, col_name: str):
    res_df = df[df.select_dtypes(include=[np.float16]).columns] \
        .multiply(df[col_name], axis="index")
    res_df['SK_ID'] = df['SK_ID']
    res_df['CONTACT_DATE'] = df['CONTACT_DATE']
    res_df['T_DATE'] = df['T_DATE']
    return res_df


def weighted_mean(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()


categorical = [
    'COM_CAT#1',
    'COM_CAT#2',
    'COM_CAT#3',
    'BASE_TYPE',
    'ACT',
    'ARPU_GROUP',
    'COM_CAT#7',
    'COM_CAT#8',
    'DEVICE_TYPE_ID',
    'INTERNET_TYPE_ID',
    'COM_CAT#25',
    'COM_CAT#26',
    'COM_CAT#34',
]

features = [
    #float
    'SUM_MINUTES',
    'SUM_DATA_MB',
    'SUM_DATA_MIN',
    'DATA_SPEED',
    'DATA_SPEED_weighted_data',
    'DATA_SPEED_weighted_data_min',
    'DATA_SPEED_weighted_voice',
    'DATA_SPEED_weighted_data_rel',
    'DATA_SPEED_weighted_data_min_rel',
    'DATA_SPEED_weighted_voice_rel',

    'SUM_MINUTES_mean',
    'SUM_DATA_MB_mean',
    'SUM_DATA_MIN_mean',
    'SUM_MINUTES_last',
    'SUM_DATA_MB_last',
    'SUM_DATA_MIN_last',
    'SUM_MINUTES_rate',
    'SUM_DATA_MB_rate',
    'SUM_DATA_MIN_rate',

    'VAS',
    'COM_CAT#29',
    'COM_CAT#30',
    'RENT_CHANNEL_2m',
    'COM_CAT#21_2m',
    # 'COM_CAT#23_2m',
    'COM_CAT#31_2m',
    # 'COM_CAT#23_3m',
    # 'COM_CAT#27_3m',
    'COM_CAT#33_6m',
    'VAS_diff_1m',
    'RENT_CHANNEL_diff_1m',
    # 'COM_CAT#18_diff_1m',
    'COM_CAT#19_diff_1m',
    # 'COM_CAT#23_diff_1m',
    # 'COM_CAT#27_diff_1m',
    'COST_diff_2m',
    'COM_CAT#32_diff_2m',
    'COM_CAT#33_diff_2m',
    'COM_CAT#27_diff_3m',
    'RENT_CHANNEL_div_1m_2m',
    'REVENUE_div_1m_3m',
    'VAS_div_1m_3m',
    'COM_CAT#31_div_1m_3m',
    # 'COM_CAT#31_div_1m_6m',
    'COM_CAT#32_div_1m_6m',
    'ERAB_PS_BLOCKING_RATE_LTE',
    'PSSR_2G',
    'RBU_AVAIL_DL_LTE',
    'RBU_OTHER_UL',
    'RRC_BLOCKING_RATE_LTE',
    'UTIL_SUBUNITS_3G',
    'PSSR_2G_data_avg_max_3d',
    'RAB_CS_BLOCKING_RATE_3G_data_avg_max_3d',
    'RBU_OWN_UL_data_avg_max_3d',
    'RTWP_3G_data_avg_max_3d',
    'UTIL_SUBUNITS_3G_data_avg_max_3d',
    'TOTAL_DL_VOLUME_3G_data_avg_max_3d',
    'PART_MCS_QPSK_LTE_data_avg_mean_3d',
    'PSSR_2G_data_avg_mean_3d',
    'RBU_AVAIL_UL_data_avg_mean_3d',
    # 'RBU_OTHER_DL_data_avg_mean_3d',
    'RBU_OTHER_UL_data_avg_mean_3d',
    'UTIL_CE_DL_3G_data_avg_mean_3d',
    'UTIL_CE_UL_3G_data_avg_mean_3d',
    'DATA_VOL_MB_data_avg_min_7d',
    'CSSR_2G_data_avg_min_7d',
    'ERAB_PS_BLOCKING_RATE_PLMN_LTE_data_avg_min_7d',
    'HSPDSCH_CODE_UTIL_3G_data_avg_min_7d',
    'RBU_AVAIL_UL_data_avg_min_7d',
    'RTWP_3G_data_avg_min_7d',
    'UTIL_BRD_CPU_3G_data_avg_min_7d',
    'UTIL_CE_DL_3G_data_avg_min_7d',
    'TOTAL_DL_VOLUME_3G_data_avg_min_7d',
    'DATA_VOL_MB_data_avg_max_7d',
    'CSSR_2G_data_avg_max_7d',
    'HSPDSCH_CODE_UTIL_3G_data_avg_max_7d',
    'NODEB_CNBAP_LOAD_HARDWARE_data_avg_max_7d',
    'PSSR_2G_data_avg_max_7d',
    # 'RBU_AVAIL_DL_LTE_data_avg_max_7d',
    'RBU_OTHER_DL_data_avg_max_7d',
    'RBU_OTHER_UL_data_avg_max_7d',
    # 'RBU_OWN_UL_data_avg_max_7d',
    'UTIL_CE_HW_DL_3G_data_avg_max_7d',
    'UL_VOLUME_LTE_data_avg_max_7d',
    'DATA_VOL_MB_data_avg_mean_7d',
    'ERAB_PS_BLOCKING_RATE_PLMN_LTE_data_avg_mean_7d',
    'RAB_CS_BLOCKING_RATE_3G_data_avg_mean_7d',
    # 'RAB_PS_DROP_RATE_3G_data_avg_mean_7d',
    'RBU_AVAIL_DL_data_avg_mean_7d',
    'RBU_AVAIL_UL_data_avg_mean_7d',
    'RBU_OTHER_UL_data_avg_mean_7d',
    'RBU_OWN_DL_data_avg_mean_7d',
    'UTIL_CE_DL_3G_data_avg_mean_7d',
    'DL_VOLUME_LTE_data_avg_mean_7d',
    'COUNT_data_avg_min_14d',
    'ERAB_PS_BLOCKING_RATE_LTE_data_avg_min_14d',
    'RAB_CS_DROP_RATE_3G_data_avg_min_14d',
    'DL_VOLUME_LTE_data_avg_min_14d',
    'PART_CQI_QPSK_LTE_data_avg_max_14d',
    'PROC_LOAD_3G_data_avg_max_14d',
    # 'RAB_CS_DROP_RATE_3G_data_avg_max_14d',
    'RAB_PS_DROP_RATE_3G_data_avg_max_14d',
    'UTIL_CE_UL_3G_data_avg_max_14d',
    'DL_VOLUME_LTE_data_avg_max_14d',
    'PSSR_LTE_data_avg_mean_14d',
    # 'RAB_PS_BLOCKING_RATE_3G_data_avg_mean_14d',
    'RAB_PS_DROP_RATE_3G_data_avg_mean_14d',
    'RBU_AVAIL_UL_data_avg_mean_14d',
    'RRC_BLOCKING_RATE_3G_data_avg_mean_14d',
    'TCH_DROP_RATE_2G_data_avg_mean_14d',
    # 'AVR_TX_POWER_3G',
    'RBU_USED_DL',
    'UL_MEAN_USER_THROUGHPUT_LTE',
    'AVEUSERNUMBER_data_chnn_max_3d',
    'AVR_DL_USER_LTE_data_chnn_max_3d',
    'HSDPA_USERS_3G_data_chnn_max_3d',
    'UL_AVR_THROUGHPUT_R99_data_chnn_max_3d',
    'AVR_UL_HSPA_USER_data_chnn_mean_3d',
    'AVR_UL_R99_USER_data_chnn_mean_3d',
    'DL_AVR_THROUGHPUT_R99_data_chnn_mean_3d',
    'HSDPA_USERS_3G_data_chnn_mean_3d',
    'RBU_USED_UL_data_chnn_mean_3d',
    'UL_AVR_THROUGHPUT_3G_data_chnn_mean_3d',
    'UL_AVR_THROUGHPUT_LTE_data_chnn_mean_3d',
    'UL_MEAN_USER_THROUGHPUT_LTE_data_chnn_mean_3d',
    'AVEUSERNUMBER_PLMN_data_chnn_min_7d',
    'AVR_DL_USER_3G_data_chnn_min_7d',
    'AVR_DL_USER_LTE_data_chnn_min_7d',
    'DL_AVR_THROUGHPUT_R99_data_chnn_min_7d',
    'DL_MEAN_USER_THROUGHPUT_LTE_data_chnn_min_7d',
    'UL_MEAN_USER_THROUGHPUT_LTE_data_chnn_min_7d',
    'COUNT_data_chnn_max_7d',
    # 'AVR_UL_USER_3G_data_chnn_max_7d',
    'DL_MEAN_USER_THROUGHPUT_DL_2G_data_chnn_max_7d',
    'RELATIVE_TX_POWER_3G_data_chnn_max_7d',
    'UL_AVR_THROUGHPUT_LTE_data_chnn_max_7d',
    'UL_MEAN_USER_THROUGHPUT_HS3G_data_chnn_max_7d',
    # 'AVR_DL_R99_USER_3G_data_chnn_mean_7d',
    'AVR_DL_USER_3G_data_chnn_mean_7d',
    # 'DL_AVR_THROUGHPUT_3G_data_chnn_mean_7d',
    'RBU_USED_UL_data_chnn_mean_7d',
    'RELATIVE_RBU_USED_DL_data_chnn_mean_7d',
    'RELATIVE_TX_POWER_3G_data_chnn_mean_7d',
    'UL_MEAN_USER_THROUGHPUT_HS3G_data_chnn_mean_7d',
    'AVR_DL_USER_3G_data_chnn_min_14d',
    'RBU_USED_DL_data_chnn_min_14d',
    'UL_AVR_THROUGHPUT_LTE_data_chnn_min_14d',
    'AVR_DL_HSPA_USER_3G_data_chnn_max_14d',
    'AVR_DL_USER_LTE_data_chnn_max_14d',
    # 'DL_MEAN_USER_THROUGHPUT_HSPA3G_data_chnn_max_14d',
    'DL_MEAN_USER_THROUGHPUT_PLTE_data_chnn_max_14d',
    'HSDPA_USERS_3G_data_chnn_max_14d',
    'RBU_USED_DL_data_chnn_max_14d',
    'UL_AVR_THROUGHPUT_LTE_data_chnn_max_14d',
    'COUNT_data_chnn_mean_14d',
    'AVR_UL_HSPA_USER_data_chnn_mean_14d',
    'DL_AVR_THROUGHPUT_LTE_data_chnn_mean_14d',
    'HSUPA_USERS_3G_data_chnn_mean_14d',
    'RBU_USED_UL_data_chnn_mean_14d',
    'RELATIVE_TX_POWER_3G_data_chnn_mean_14d',
    'UL_MEAN_USER_THROUGHPUT_REL93G_data_chnn_mean_14d',
    'VOICE_DUR_MIN',
    'CELL_AVAILABILITY_4G_voice_avg_min_3d',
    'CSSR_2G_voice_avg_min_3d',
    'RRC_BLOCKING_RATE_3G_voice_avg_min_3d',
    'UL_VOLUME_LTE_voice_avg_min_3d',
    'VOICE_DUR_MIN_voice_avg_max_3d',
    'CELL_AVAILABILITY_2G_voice_avg_max_3d',
    'PSSR_2G_voice_avg_max_3d',
    'RBU_AVAIL_DL_voice_avg_max_3d',
    'RBU_OWN_UL_voice_avg_max_3d',
    # 'UTIL_CE_UL_3G_voice_avg_max_3d',
    'UTIL_SUBUNITS_3G_voice_avg_max_3d',
    'VOICE_DUR_MIN_voice_avg_mean_3d',
    'CSSR_2G_voice_avg_mean_3d',
    'CSSR_3G_voice_avg_mean_3d',
    'PART_CQI_QPSK_LTE_voice_avg_mean_3d',
    'PSSR_2G_voice_avg_mean_3d',
    'RAB_PS_DROP_RATE_3G_voice_avg_mean_3d',
    'UTIL_CE_DL_3G_voice_avg_mean_3d',
    'UTIL_CE_HW_DL_3G_voice_avg_mean_3d',
    'UTIL_CE_UL_3G_voice_avg_mean_3d',
    'UL_VOLUME_LTE_voice_avg_mean_3d',
    'ERAB_PS_BLOCKING_RATE_LTE_voice_avg_min_7d',
    'HSPDSCH_CODE_UTIL_3G_voice_avg_min_7d',
    'SHO_FACTOR_voice_avg_min_7d',
    'UTIL_CE_DL_3G_voice_avg_min_7d',
    'UTIL_CE_UL_3G_voice_avg_min_7d',
    'UL_VOLUME_LTE_voice_avg_min_7d',
    'DL_VOLUME_LTE_voice_avg_min_7d',
    'CELL_AVAILABILITY_3G_voice_avg_max_7d',
    'ERAB_PS_BLOCKING_RATE_LTE_voice_avg_max_7d',
    'HSPDSCH_CODE_UTIL_3G_voice_avg_max_7d',
    # 'RAB_CS_DROP_RATE_3G_voice_avg_max_7d',
    'RBU_AVAIL_UL_voice_avg_max_7d',
    'RBU_OTHER_DL_voice_avg_max_7d',
    'RRC_BLOCKING_RATE_LTE_voice_avg_max_7d',
    'UTIL_SUBUNITS_3G_voice_avg_max_7d',
    'DL_VOLUME_LTE_voice_avg_max_7d',
    # 'RAB_PS_DROP_RATE_3G_voice_avg_mean_7d',
    # 'RBU_OWN_DL_voice_avg_mean_7d',
    'RBU_OWN_UL_voice_avg_mean_7d',
    'RRC_BLOCKING_RATE_LTE_voice_avg_mean_7d',
    'RTWP_3G_voice_avg_mean_7d',
    'TBF_DROP_RATE_2G_voice_avg_mean_7d',
    'UTIL_SUBUNITS_3G_voice_avg_mean_7d',
    'UL_VOLUME_LTE_voice_avg_mean_7d',
    'CSSR_2G_voice_avg_min_14d',
    'ERAB_PS_BLOCKING_RATE_PLMN_LTE_voice_avg_min_14d',
    'HSPDSCH_CODE_UTIL_3G_voice_avg_min_14d',
    'PART_MCS_QPSK_LTE_voice_avg_min_14d',
    'SHO_FACTOR_voice_avg_min_14d',
    'UTIL_CE_DL_3G_voice_avg_min_14d',
    'UTIL_SUBUNITS_3G_voice_avg_min_14d',
    'CELL_AVAILABILITY_2G_voice_avg_max_14d',
    'ERAB_PS_DROP_RATE_LTE_voice_avg_max_14d',
    'PSSR_3G_voice_avg_max_14d',
    'PSSR_LTE_voice_avg_max_14d',
    'RBU_AVAIL_DL_voice_avg_max_14d',
    'RBU_OWN_DL_voice_avg_max_14d',
    # 'RRC_BLOCKING_RATE_LTE_voice_avg_max_14d',
    'RTWP_3G_voice_avg_max_14d',
    'UTIL_CE_UL_3G_voice_avg_max_14d',
    'UTIL_SUBUNITS_3G_voice_avg_max_14d',
    'UL_VOLUME_LTE_voice_avg_max_14d',
    'ERAB_PS_BLOCKING_RATE_LTE_voice_avg_mean_14d',
    'PART_MCS_QPSK_LTE_voice_avg_mean_14d',
    'PROC_LOAD_3G_voice_avg_mean_14d',
    'RAB_CS_BLOCKING_RATE_3G_voice_avg_mean_14d',
    'RAB_PS_BLOCKING_RATE_3G_voice_avg_mean_14d',
    'RBU_OWN_DL_voice_avg_mean_14d',
    'AVR_UL_USER_LTE_voice_chnn_min_3d',
    'DL_MEAN_USER_THROUGHPUT_LTE_voice_chnn_min_3d',
    'RELATIVE_TX_POWER_3G_voice_chnn_min_3d',
    'UL_MEAN_USER_THROUGHPUT_PLTE_voice_chnn_min_3d',
    'COUNT_voice_chnn_max_3d',
    'AVEUSERNUMBER_PLMN_voice_chnn_max_3d',
    'AVR_DL_USER_LTE_voice_chnn_max_3d',
    'AVR_TX_POWER_3G_voice_chnn_max_3d',
    'DL_AVR_THROUGHPUT_LTE_voice_chnn_max_3d',
    'DL_MEAN_USER_THROUGHPUT_LTE_voice_chnn_max_3d',
    'DL_MEAN_USER_THROUGHPUT_DL_2G_voice_chnn_max_3d',
    'RBU_USED_UL_voice_chnn_max_3d',
    'RELATIVE_RBU_USED_DL_voice_chnn_max_3d',
    'AVEUSERNUMBER_PLMN_voice_chnn_mean_3d',
    'DL_AVR_THROUGHPUT_R99_voice_chnn_mean_3d',
    'UL_AVR_THROUGHPUT_3G_voice_chnn_mean_3d',
    'AVR_TX_POWER_3G_voice_chnn_min_7d',
    'DL_AVR_THROUGHPUT_LTE_voice_chnn_min_7d',
    'DL_MEAN_USER_THROUGHPUT_LTE_voice_chnn_min_7d',
    'DL_AVR_THROUGHPUT_LTE_voice_chnn_max_7d',
    'DL_MEAN_USER_THROUGHPUT_PLTE_voice_chnn_max_7d',
    'HSDPA_USERS_3G_voice_chnn_max_7d',
    'RBU_USED_DL_voice_chnn_max_7d',
    'RELATIVE_TX_POWER_3G_voice_chnn_max_7d',
    'AVR_DL_HSPA_USER_3G_voice_chnn_mean_7d',
    'AVR_DL_R99_USER_3G_voice_chnn_mean_7d',
    'HSUPA_USERS_3G_voice_chnn_mean_7d',
    'UL_AVR_THROUGHPUT_3G_voice_chnn_mean_7d',
    'UL_AVR_THROUGHPUT_R99_voice_chnn_mean_7d',
    'AVR_TX_POWER_3G_voice_chnn_min_14d',
    'RELATIVE_RBU_USED_DL_voice_chnn_min_14d',
    'UL_MEAN_USER_THROUGHPUT_PLTE_voice_chnn_min_14d',
    'AVR_DL_R99_USER_3G_voice_chnn_max_14d',
    'AVR_UL_R99_USER_voice_chnn_max_14d',
    'DL_MEAN_USER_THROUGHPUT_REL93G_voice_chnn_max_14d',
    'RBU_USED_UL_voice_chnn_max_14d',
    'AVR_DL_USER_LTE_voice_chnn_mean_14d',
    'AVR_UL_USER_LTE_voice_chnn_mean_14d',
    'DL_AVR_THROUGHPUT_LTE_voice_chnn_mean_14d',
    'DL_MEAN_USER_THROUGHPUT_LTE_voice_chnn_mean_14d',
    # 'HSDPA_USERS_3G_voice_chnn_mean_14d',
    # 'HSUPA_USERS_3G_voice_chnn_mean_14d',
    'UL_AVR_THROUGHPUT_LTE_voice_chnn_mean_14d',
    'UL_MEAN_USER_THROUGHPUT_PLTE_voice_chnn_mean_14d',
    'HSPDSCH_CODE_UTIL_3G_data_rel_avg_min_3d',
    'PART_CQI_QPSK_LTE_data_rel_avg_min_3d',
    'PSSR_2G_data_rel_avg_min_3d',
    'RAB_CS_DROP_RATE_3G_data_rel_avg_min_3d',
    'RBU_OTHER_DL_data_rel_avg_min_3d',
    'RBU_OWN_DL_data_rel_avg_min_3d',
    'RRC_BLOCKING_RATE_3G_data_rel_avg_min_3d',
    'TBF_DROP_RATE_2G_data_rel_avg_min_3d',
    'UTIL_CE_UL_3G_data_rel_avg_min_3d',
    'UL_VOLUME_LTE_data_rel_avg_min_3d',
    'ERAB_PS_BLOCKING_RATE_PLMN_LTE_data_rel_avg_max_3d',
    'RBU_AVAIL_UL_data_rel_avg_max_3d',
    'RRC_BLOCKING_RATE_3G_data_rel_avg_max_3d',
    'TOTAL_UL_VOLUME_3G_data_rel_avg_max_3d',
    'PART_CQI_QPSK_LTE_data_rel_avg_mean_3d',
    'PROC_LOAD_3G_data_rel_avg_mean_3d',
    'RAB_PS_DROP_RATE_3G_data_rel_avg_mean_3d',
    'RBU_OTHER_UL_data_rel_avg_mean_3d',
    'UTIL_CE_HW_DL_3G_data_rel_avg_mean_3d',
    'UL_VOLUME_LTE_data_rel_avg_mean_3d',
    'TOTAL_UL_VOLUME_3G_data_rel_avg_mean_3d',
    'CSSR_2G_data_rel_avg_min_7d',
    # 'RAB_CS_DROP_RATE_3G_data_rel_avg_min_7d',
    'RAB_PS_DROP_RATE_3G_data_rel_avg_min_7d',
    'UL_VOLUME_LTE_data_rel_avg_min_7d',
    'DL_VOLUME_LTE_data_rel_avg_min_7d',
    'TOTAL_UL_VOLUME_3G_data_rel_avg_min_7d',
    'DATA_VOL_MB_data_rel_avg_max_7d',
    'CSSR_2G_data_rel_avg_max_7d',
    'ERAB_PS_BLOCKING_RATE_PLMN_LTE_data_rel_avg_max_7d',
    'PSSR_2G_data_rel_avg_max_7d',
    'RAB_CS_DROP_RATE_3G_data_rel_avg_max_7d',
    'RBU_OWN_UL_data_rel_avg_max_7d',
    # 'RRC_BLOCKING_RATE_LTE_data_rel_avg_max_7d',
    'SHO_FACTOR_data_rel_avg_max_7d',
    'UTIL_CE_HW_DL_3G_data_rel_avg_max_7d',
    'CELL_AVAILABILITY_3G_data_rel_avg_mean_7d',
    'PSSR_3G_data_rel_avg_mean_7d',
    'CSSR_3G_data_rel_avg_min_14d',
    'PSSR_LTE_data_rel_avg_min_14d',
    'RBU_OTHER_DL_data_rel_avg_min_14d',
    'UTIL_BRD_CPU_3G_data_rel_avg_min_14d',
    'CSSR_2G_data_rel_avg_max_14d',
    'NODEB_CNBAP_LOAD_HARDWARE_data_rel_avg_max_14d',
    'PSSR_3G_data_rel_avg_max_14d',
    'RBU_OTHER_DL_data_rel_avg_max_14d',
    'SHO_FACTOR_data_rel_avg_max_14d',
    'UTIL_BRD_CPU_3G_data_rel_avg_max_14d',
    'UTIL_CE_DL_3G_data_rel_avg_max_14d',
    'CSSR_2G_data_rel_avg_mean_14d',
    'NODEB_CNBAP_LOAD_HARDWARE_data_rel_avg_mean_14d',
    'PART_MCS_QPSK_LTE_data_rel_avg_mean_14d',
    'PSSR_2G_data_rel_avg_mean_14d',
    'RBU_OWN_UL_data_rel_avg_mean_14d',
    'RRC_BLOCKING_RATE_3G_data_rel_avg_mean_14d',
    'TOTAL_UL_VOLUME_3G_data_rel_avg_mean_14d',
    'AVEUSERNUMBER_data_rel_chnn_min_3d',
    'AVR_UL_HSPA_USER_data_rel_chnn_min_3d',
    'DL_AVR_THROUGHPUT_3G_data_rel_chnn_min_3d',
    'DL_AVR_THROUGHPUT_R99_data_rel_chnn_min_3d',
    'HSUPA_USERS_3G_data_rel_chnn_min_3d',
    'AVEUSERNUMBER_PLMN_data_rel_chnn_max_3d',
    'AVR_UL_HSPA_USER_data_rel_chnn_max_3d',
    'UL_AVR_THROUGHPUT_3G_data_rel_chnn_max_3d',
    'AVR_DL_R99_USER_3G_data_rel_chnn_mean_3d',
    'AVR_UL_USER_LTE_data_rel_chnn_mean_3d',
    'UL_AVR_THROUGHPUT_3G_data_rel_chnn_mean_3d',
    'UL_AVR_THROUGHPUT_R99_data_rel_chnn_mean_3d',
    'UL_MEAN_USER_THROUGHPUT_HS3G_data_rel_chnn_mean_3d',
    'UL_MEAN_USER_THROUGHPUT_REL93G_data_rel_chnn_mean_3d',
    'AVR_UL_R99_USER_data_rel_chnn_min_7d',
    'DL_MEAN_USER_THROUGHPUT_REL93G_data_rel_chnn_min_7d',
    'RELATIVE_RBU_USED_UL_data_rel_chnn_min_7d',
    'UL_MEAN_USER_THROUGHPUT_LTE_data_rel_chnn_min_7d',
    # 'DATA_VOL_MB_data_rel_chnn_max_7d',
    'DL_AVR_THROUGHPUT_R99_data_rel_chnn_max_7d',
    'DL_MEAN_USER_THROUGHPUT_REL93G_data_rel_chnn_max_7d',
    'UL_AVR_THROUGHPUT_3G_data_rel_chnn_max_7d',
    'UL_MEAN_USER_THROUGHPUT_LTE_data_rel_chnn_max_7d',
    'AVR_UL_USER_3G_data_rel_chnn_mean_7d',
    'HSDPA_USERS_3G_data_rel_chnn_mean_7d',
    'UL_AVR_THROUGHPUT_LTE_data_rel_chnn_mean_7d',
    'UL_MEAN_USER_THROUGHPUT_HS3G_data_rel_chnn_mean_7d',
    'AVR_TX_POWER_3G_data_rel_chnn_min_14d',
    'DL_AVR_THROUGHPUT_R99_data_rel_chnn_min_14d',
    'DL_MEAN_USER_THROUGHPUT_LTE_data_rel_chnn_min_14d',
    'RBU_USED_DL_data_rel_chnn_min_14d',
    # 'RBU_USED_UL_data_rel_chnn_min_14d',
    'AVR_DL_USER_LTE_data_rel_chnn_max_14d',
    # 'AVR_UL_USER_LTE_data_rel_chnn_max_14d',
    'DL_MEAN_USER_THROUGHPUT_LTE_data_rel_chnn_max_14d',
    'RELATIVE_RBU_USED_DL_data_rel_chnn_max_14d',
    'AVEUSERNUMBER_data_rel_chnn_mean_14d',
    'UL_MEAN_USER_THROUGHPUT_PLTE_data_rel_chnn_mean_14d',
    'UL_MEAN_USER_THROUGHPUT_REL93G_data_rel_chnn_mean_14d',
    'VOICE_DUR_MIN_voice_rel_avg_min_3d',
    'ERAB_PS_BLOCKING_RATE_PLMN_LTE_voice_rel_avg_min_3d',
    'PSSR_2G_voice_rel_avg_min_3d',
    'RBU_OWN_UL_voice_rel_avg_min_3d',
    'UTIL_SUBUNITS_3G_voice_rel_avg_min_3d',
    'TOTAL_UL_VOLUME_3G_voice_rel_avg_min_3d',
    'ERAB_PS_DROP_RATE_LTE_voice_rel_avg_max_3d',

    'CELL_AVAILABILITY_2G_avg',
    'CELL_AVAILABILITY_3G_avg',
    'CELL_AVAILABILITY_4G_avg',
    'CSSR_2G_avg',
    'CSSR_3G_avg',
    'ERAB_PS_BLOCKING_RATE_LTE_avg',
    'ERAB_PS_BLOCKING_RATE_PLMN_LTE_avg',
    'ERAB_PS_DROP_RATE_LTE_avg',
    'HSPDSCH_CODE_UTIL_3G_avg',
    'NODEB_CNBAP_LOAD_HARDWARE_avg',
    'PART_CQI_QPSK_LTE_avg',
    'PART_MCS_QPSK_LTE_avg',
    'PROC_LOAD_3G_avg',
    'PSSR_2G_avg',
    'PSSR_3G_avg',
    'PSSR_LTE_avg',
    'RAB_CS_BLOCKING_RATE_3G_avg',
    'RAB_CS_DROP_RATE_3G_avg',
    'RAB_PS_BLOCKING_RATE_3G_avg',
    'RAB_PS_DROP_RATE_3G_avg',
    'RBU_AVAIL_DL_avg',
    'RBU_AVAIL_DL_LTE_avg',
    'RBU_AVAIL_UL_avg',
    'RBU_OTHER_DL_avg',
    'RBU_OTHER_UL_avg',
    'RBU_OWN_DL_avg',
    'RBU_OWN_UL_avg',
    'RRC_BLOCKING_RATE_3G_avg',
    'RRC_BLOCKING_RATE_LTE_avg',
    'RTWP_3G_avg',
    'SHO_FACTOR_avg',
    'TBF_DROP_RATE_2G_avg',
    'TCH_DROP_RATE_2G_avg',
    'UTIL_BRD_CPU_3G_avg',
    'UTIL_CE_DL_3G_avg',
    'UTIL_CE_HW_DL_3G_avg',
    'UTIL_CE_UL_3G_avg',
    'UTIL_SUBUNITS_3G_avg',
    'UL_VOLUME_LTE_avg',
    'DL_VOLUME_LTE_avg',
    'TOTAL_DL_VOLUME_3G_avg',
    'TOTAL_UL_VOLUME_3G_avg',
    'CELL_AVAILABILITY_2G_data_main',
    'CELL_AVAILABILITY_3G_data_main',
    'CELL_AVAILABILITY_4G_data_main',
    'CSSR_2G_data_main',
    'CSSR_3G_data_main',
    'ERAB_PS_BLOCKING_RATE_LTE_data_main',
    'ERAB_PS_BLOCKING_RATE_PLMN_LTE_data_main',
    'ERAB_PS_DROP_RATE_LTE_data_main',
    'HSPDSCH_CODE_UTIL_3G_data_main',
    'NODEB_CNBAP_LOAD_HARDWARE_data_main',
    'PART_CQI_QPSK_LTE_data_main',
    'PART_MCS_QPSK_LTE_data_main',
    'PROC_LOAD_3G_data_main',
    'PSSR_2G_data_main',
    'PSSR_3G_data_main',
    'PSSR_LTE_data_main',
    'RAB_CS_BLOCKING_RATE_3G_data_main',
    'RAB_CS_DROP_RATE_3G_data_main',
    'RAB_PS_BLOCKING_RATE_3G_data_main',
    'RAB_PS_DROP_RATE_3G_data_main',
    'RBU_AVAIL_DL_data_main',
    'RBU_AVAIL_DL_LTE_data_main',
    'RBU_AVAIL_UL_data_main',
    'RBU_OTHER_DL_data_main',
    'RBU_OTHER_UL_data_main',
    'RBU_OWN_DL_data_main',
    'RBU_OWN_UL_data_main',
    'RRC_BLOCKING_RATE_3G_data_main',
    'RRC_BLOCKING_RATE_LTE_data_main',
    'RTWP_3G_data_main',
    'SHO_FACTOR_data_main',
    'TBF_DROP_RATE_2G_data_main',
    'TCH_DROP_RATE_2G_data_main',
    'UTIL_BRD_CPU_3G_data_main',
    'UTIL_CE_DL_3G_data_main',
    'UTIL_CE_HW_DL_3G_data_main',
    'UTIL_CE_UL_3G_data_main',
    'UTIL_SUBUNITS_3G_data_main',
    'UL_VOLUME_LTE_data_main',
    'DL_VOLUME_LTE_data_main',
    'TOTAL_DL_VOLUME_3G_data_main',
    'TOTAL_UL_VOLUME_3G_data_main',
    'CELL_AVAILABILITY_2G_data_min_main',
    'CELL_AVAILABILITY_3G_data_min_main',
    'CELL_AVAILABILITY_4G_data_min_main',
    'CSSR_2G_data_min_main',
    'CSSR_3G_data_min_main',
    'ERAB_PS_BLOCKING_RATE_LTE_data_min_main',
    'ERAB_PS_BLOCKING_RATE_PLMN_LTE_data_min_main',
    'ERAB_PS_DROP_RATE_LTE_data_min_main',
    'HSPDSCH_CODE_UTIL_3G_data_min_main',
    'NODEB_CNBAP_LOAD_HARDWARE_data_min_main',
    'PART_CQI_QPSK_LTE_data_min_main',
    'PART_MCS_QPSK_LTE_data_min_main',
    'PROC_LOAD_3G_data_min_main',
    'PSSR_2G_data_min_main',
    'PSSR_3G_data_min_main',
    'PSSR_LTE_data_min_main',
    'RAB_CS_BLOCKING_RATE_3G_data_min_main',
    'RAB_CS_DROP_RATE_3G_data_min_main',
    'RAB_PS_BLOCKING_RATE_3G_data_min_main',
    'RAB_PS_DROP_RATE_3G_data_min_main',
    'RBU_AVAIL_DL_data_min_main',
    'RBU_AVAIL_DL_LTE_data_min_main',
    'RBU_AVAIL_UL_data_min_main',
    'RBU_OTHER_DL_data_min_main',
    'RBU_OTHER_UL_data_min_main',
    'RBU_OWN_DL_data_min_main',
    'RBU_OWN_UL_data_min_main',
    'RRC_BLOCKING_RATE_3G_data_min_main',
    'RRC_BLOCKING_RATE_LTE_data_min_main',
    'RTWP_3G_data_min_main',
    'SHO_FACTOR_data_min_main',
    'TBF_DROP_RATE_2G_data_min_main',
    'TCH_DROP_RATE_2G_data_min_main',
    'UTIL_BRD_CPU_3G_data_min_main',
    'UTIL_CE_DL_3G_data_min_main',
    'UTIL_CE_HW_DL_3G_data_min_main',
    'UTIL_CE_UL_3G_data_min_main',
    'UTIL_SUBUNITS_3G_data_min_main',
    'UL_VOLUME_LTE_data_min_main',
    'DL_VOLUME_LTE_data_min_main',
    'TOTAL_DL_VOLUME_3G_data_min_main',
    'TOTAL_UL_VOLUME_3G_data_min_main',
    'AVEUSERNUMBER_chnn',
    'AVEUSERNUMBER_PLMN_chnn',
    'AVR_DL_HSPA_USER_3G_chnn',
    'AVR_DL_R99_USER_3G_chnn',
    'AVR_DL_USER_3G_chnn',
    'AVR_DL_USER_LTE_chnn',
    'AVR_TX_POWER_3G_chnn',
    'AVR_UL_HSPA_USER_chnn',
    'AVR_UL_R99_USER_chnn',
    'AVR_UL_USER_3G_chnn',
    'AVR_UL_USER_LTE_chnn',
    'DL_AVR_THROUGHPUT_3G_chnn',
    'DL_AVR_THROUGHPUT_LTE_chnn',
    'DL_AVR_THROUGHPUT_R99_chnn',
    'DL_MEAN_USER_THROUGHPUT_LTE_chnn',
    'DL_MEAN_USER_THROUGHPUT_DL_2G_chnn',
    'DL_MEAN_USER_THROUGHPUT_HSPA3G_chnn',
    'DL_MEAN_USER_THROUGHPUT_PLTE_chnn',
    'DL_MEAN_USER_THROUGHPUT_REL93G_chnn',
    'HSDPA_USERS_3G_chnn',
    'HSUPA_USERS_3G_chnn',
    'RBU_USED_DL_chnn',
    'RBU_USED_UL_chnn',
    'RELATIVE_RBU_USED_DL_chnn',
    'RELATIVE_RBU_USED_UL_chnn',
    'RELATIVE_TX_POWER_3G_chnn',
    'UL_AVR_THROUGHPUT_3G_chnn',
    'UL_AVR_THROUGHPUT_LTE_chnn',
    'UL_AVR_THROUGHPUT_R99_chnn',
    'UL_MEAN_USER_THROUGHPUT_LTE_chnn',
    'UL_MEAN_USER_THROUGHPUT_HS3G_chnn',
    'UL_MEAN_USER_THROUGHPUT_PLTE_chnn',
    'UL_MEAN_USER_THROUGHPUT_REL93G_chnn',
    'AVEUSERNUMBER_data_main',
    'AVEUSERNUMBER_PLMN_data_main',
    'AVR_DL_HSPA_USER_3G_data_main',
    'AVR_DL_R99_USER_3G_data_main',
    'AVR_DL_USER_3G_data_main',
    'AVR_DL_USER_LTE_data_main',
    'AVR_TX_POWER_3G_data_main',
    'AVR_UL_HSPA_USER_data_main',
    'AVR_UL_R99_USER_data_main',
    'AVR_UL_USER_3G_data_main',
    'AVR_UL_USER_LTE_data_main',
    'DL_AVR_THROUGHPUT_3G_data_main',
    'DL_AVR_THROUGHPUT_LTE_data_main',
    'DL_AVR_THROUGHPUT_R99_data_main',
    'DL_MEAN_USER_THROUGHPUT_LTE_data_main',
    'DL_MEAN_USER_THROUGHPUT_DL_2G_data_main',
    'DL_MEAN_USER_THROUGHPUT_HSPA3G_data_main',
    'DL_MEAN_USER_THROUGHPUT_PLTE_data_main',
    'DL_MEAN_USER_THROUGHPUT_REL93G_data_main',
    'HSDPA_USERS_3G_data_main',
    'HSUPA_USERS_3G_data_main',
    'RBU_USED_DL_data_main',
    'RBU_USED_UL_data_main',
    'RELATIVE_RBU_USED_DL_data_main',
    'RELATIVE_RBU_USED_UL_data_main',
    'RELATIVE_TX_POWER_3G_data_main',
    'UL_AVR_THROUGHPUT_3G_data_main',
    'UL_AVR_THROUGHPUT_LTE_data_main',
    'UL_AVR_THROUGHPUT_R99_data_main',
    'UL_MEAN_USER_THROUGHPUT_LTE_data_main',
    'UL_MEAN_USER_THROUGHPUT_HS3G_data_main',
    'UL_MEAN_USER_THROUGHPUT_PLTE_data_main',
    'UL_MEAN_USER_THROUGHPUT_REL93G_data_main',
    'AVEUSERNUMBER_data_min_main',
    'AVEUSERNUMBER_PLMN_data_min_main',
    'AVR_DL_HSPA_USER_3G_data_min_main',
    'AVR_DL_R99_USER_3G_data_min_main',
    'AVR_DL_USER_3G_data_min_main',
    'AVR_DL_USER_LTE_data_min_main',
    'AVR_TX_POWER_3G_data_min_main',
    'AVR_UL_HSPA_USER_data_min_main',
    'AVR_UL_R99_USER_data_min_main',
    'AVR_UL_USER_3G_data_min_main',
    'AVR_UL_USER_LTE_data_min_main',
    'DL_AVR_THROUGHPUT_3G_data_min_main',
    'DL_AVR_THROUGHPUT_LTE_data_min_main',
    'DL_AVR_THROUGHPUT_R99_data_min_main',
    'DL_MEAN_USER_THROUGHPUT_LTE_data_min_main',
    'DL_MEAN_USER_THROUGHPUT_DL_2G_data_min_main',
    'DL_MEAN_USER_THROUGHPUT_HSPA3G_data_min_main',
    'DL_MEAN_USER_THROUGHPUT_PLTE_data_min_main',
    'DL_MEAN_USER_THROUGHPUT_REL93G_data_min_main',
    'HSDPA_USERS_3G_data_min_main',
    'HSUPA_USERS_3G_data_min_main',
    'RBU_USED_DL_data_min_main',
    'RBU_USED_UL_data_min_main',
    'RELATIVE_RBU_USED_DL_data_min_main',
    'RELATIVE_RBU_USED_UL_data_min_main',
    'RELATIVE_TX_POWER_3G_data_min_main',
    'UL_AVR_THROUGHPUT_3G_data_min_main',
    'UL_AVR_THROUGHPUT_LTE_data_min_main',
    'UL_AVR_THROUGHPUT_R99_data_min_main',
    'UL_MEAN_USER_THROUGHPUT_LTE_data_min_main',
    'UL_MEAN_USER_THROUGHPUT_HS3G_data_min_main',
    'UL_MEAN_USER_THROUGHPUT_PLTE_data_min_main',
    'UL_MEAN_USER_THROUGHPUT_REL93G_data_min_main',


    # category
    'COM_CAT#1',
    # 'COM_CAT#2',
    # 'COM_CAT#3',
    'BASE_TYPE',
    'ACT',
    # 'ARPU_GROUP',
    'COM_CAT#7',
    # 'COM_CAT#8',
    'DEVICE_TYPE_ID',
    'INTERNET_TYPE_ID',
    'COM_CAT#25',
    'COM_CAT#26',
    # 'COM_CAT#34',
]

if __name__ == '__main__':
    if not os.path.isfile(os.path.join(CACHE_DIR, 'train_data_avg.feather')):
        cons_df = load_data_session('train')
        train_df = load_csi_train()
        avg_df = load_avg_kpi()
        session_avg_df = session_kpi(train_df, cons_df, avg_df, 'train_data_avg')
        print(session_avg_df.info())
    gc.collect()

    if not os.path.isfile(os.path.join(CACHE_DIR, 'test_data_avg.feather')):
        cons_df = load_data_session('test')
        train_df = load_csi_test()
        avg_df = load_avg_kpi()
        session_avg_df = session_kpi(train_df, cons_df, avg_df, 'test_data_avg')
        print(session_avg_df.info())
    gc.collect()

    if not os.path.isfile(os.path.join(CACHE_DIR, 'train_data_chnn.feather')):
        cons_df = load_data_session('train')
        train_df = load_csi_train()
        chnn_df = load_chnn_kpi()
        session_chnn_df = session_kpi(train_df, cons_df, chnn_df, 'train_data_chnn')
        print(session_chnn_df.info())
    gc.collect()

    if not os.path.isfile(os.path.join(CACHE_DIR, 'test_data_chnn.feather')):
        cons_df = load_data_session('test')
        train_df = load_csi_test()
        chnn_df = load_chnn_kpi()
        session_chnn_df = session_kpi(train_df, cons_df, chnn_df, 'test_data_chnn')
        print(session_chnn_df.info())
    gc.collect()

    if not os.path.isfile(os.path.join(CACHE_DIR, 'train_voice_avg.feather')):
        cons_df = load_voice_session('train')
        train_df = load_csi_train()
        avg_df = load_avg_kpi()
        session_avg_df = session_kpi(train_df, cons_df, avg_df, 'train_voice_avg')
        print(session_avg_df.info())
    gc.collect()

    if not os.path.isfile(os.path.join(CACHE_DIR, 'test_voice_avg.feather')):
        cons_df = load_voice_session('test')
        train_df = load_csi_test()
        avg_df = load_avg_kpi()
        session_avg_df = session_kpi(train_df, cons_df, avg_df, 'test_voice_avg')
        print(session_avg_df.info())
    gc.collect()

    if not os.path.isfile(os.path.join(CACHE_DIR, 'train_voice_chnn.feather')):
        cons_df = load_voice_session('train')
        train_df = load_csi_train()
        chnn_df = load_chnn_kpi()
        session_chnn_df = session_kpi(train_df, cons_df, chnn_df, 'train_voice_chnn')
        print(session_chnn_df.info())
    gc.collect()

    if not os.path.isfile(os.path.join(CACHE_DIR, 'test_voice_chnn.feather')):
        cons_df = load_voice_session('test')
        train_df = load_csi_test()
        chnn_df = load_chnn_kpi()
        session_chnn_df = session_kpi(train_df, cons_df, chnn_df, 'test_voice_chnn')
        print(session_chnn_df.info())
    gc.collect()


    if not os.path.isfile(os.path.join(CACHE_DIR, 'train_main_avg_kpi.feather')):
        avg_df = load_avg_kpi()

        train_cons_df = load_consumption('train')
        train_df = load_csi_train()
        res_df = main_cell_kpi(train_df, train_cons_df, avg_df, 'train_main_avg_kpi')
        print(res_df.info())
        gc.collect()

        test_cons_df = load_consumption('test')
        test_df = load_csi_test()
        res_df = main_cell_kpi(test_df, test_cons_df, avg_df, 'test_main_avg_kpi')
        print(res_df.info())
        gc.collect()

    if not os.path.isfile(os.path.join(CACHE_DIR, 'train_main_chnn_kpi.feather')):
        chnn_df = load_chnn_kpi()

        train_cons_df = load_consumption('train')
        train_df = load_csi_train()
        res_df = main_cell_kpi(train_df, train_cons_df, chnn_df, 'train_main_chnn_kpi')
        print(res_df.info())
        gc.collect()

        test_cons_df = load_consumption('test')
        test_df = load_csi_test()
        res_df = main_cell_kpi(test_df, test_cons_df, chnn_df, 'test_main_chnn_kpi')
        print(res_df.info())
        gc.collect()

    # if not os.path.isfile(os.path.join(CACHE_DIR, 'train_cons_avg.feather')):
    #     cons_df = load_consumption('train')
    #     train_df = load_csi_train()
    #     avg_df = load_avg_kpi()
    #     session_avg_df = cons_kpi(train_df, cons_df, avg_df, 'train_cons_avg')
    #     print(session_avg_df.info())
    #
    #     cons_df = load_consumption('test')
    #     test_df = load_csi_test()
    #     session_avg_df = cons_kpi(test_df, cons_df, avg_df, 'test_cons_avg')
    #     print(session_avg_df.info())
    # gc.collect()
    #
    # if not os.path.isfile(os.path.join(CACHE_DIR, 'train_cons_chnn.feather')):
    #     cons_df = load_consumption('train')
    #     train_df = load_csi_train()
    #     avg_df = load_chnn_kpi()
    #     session_avg_df = cons_kpi(train_df, cons_df, avg_df, 'train_cons_chnn')
    #     print(session_avg_df.info())
    #
    #     cons_df = load_consumption('test')
    #     test_df = load_csi_test()
    #     session_avg_df = cons_kpi(test_df, cons_df, avg_df, 'test_cons_chnn')
    #     print(session_avg_df.info())
    # gc.collect()
