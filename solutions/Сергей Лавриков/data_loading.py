import pandas as pd
import numpy as np
import os


TRAIN_DIR = './dataset/train'
TEST_DIR = './dataset/test'
BS_DIR = './dataset/'
CACHE_DIR = './cache/'

YEAR = '2018'
PREV_YEAR = '2017'


def load_consumption(tp: str):
    if tp == 'test':
        filepath = os.path.join(TEST_DIR, 'subs_bs_consumption_{}.csv'.format(tp))
    else:
        filepath = os.path.join(TRAIN_DIR, 'subs_bs_consumption_{}.csv'.format(tp))
    df = pd.read_csv(filepath,
                     delimiter=';',
                     decimal=',',
                     dtype={
                         'SK_ID': np.uint16,
                         'CELL_LAC_ID': np.uint32,
                         'MON': str,
                         'SUM_MINUTES': np.float16,
                         'SUM_DATA_MB': np.float16,
                         'SUM_DATA_MIN': np.float16,
                     })
    df['MON'] = pd.to_datetime(df['MON'] + '.' + YEAR,
                               dayfirst=True,
                               format='%d.%m.%Y',
                               infer_datetime_format=True,
                               cache=True)
    return df


def load_data_session(tp: str):
    if tp == 'test':
        filepath = os.path.join(TEST_DIR, 'subs_bs_data_session_{}.csv'.format(tp))
    else:
        filepath = os.path.join(TRAIN_DIR, 'subs_bs_data_session_{}.csv'.format(tp))
    df = pd.read_csv(filepath,
                     delimiter=';',
                     decimal=',',
                     dtype={
                         'SK_ID': np.uint16,
                         'CELL_LAC_ID': np.uint32,
                         'DATA_VOL_MB': np.float16,
                         'START_TIME': str,
                     })
    df['START_TIME'] = pd.to_datetime(df['START_TIME'] + ' ' + YEAR,
                                      dayfirst=True,
                                      format='%d.%m %H:%M:%S %Y',
                                      infer_datetime_format=True,
                                      cache=True)
    return df


def load_voice_session(tp: str):
    if tp == 'test':
        filepath = os.path.join(TEST_DIR, 'subs_bs_voice_session_{}.csv'.format(tp))
    else:
        filepath = os.path.join(TRAIN_DIR, 'subs_bs_voice_session_{}.csv'.format(tp))
    df = pd.read_csv(filepath,
                     delimiter=';',
                     decimal=',',
                     dtype={
                         'SK_ID': np.uint16,
                         'CELL_LAC_ID': np.uint32,
                         'VOICE_DUR_MIN': np.float16,
                         'START_TIME': str,
                     })
    df['START_TIME'] = pd.to_datetime(df['START_TIME'] + ' ' + YEAR,
                                      dayfirst=True,
                                      format='%d.%m %H:%M:%S %Y',
                                      infer_datetime_format=True,
                                      cache=True)
    return df


def load_csi_train():
    df = pd.read_csv(os.path.join(TRAIN_DIR, 'subs_csi_train.csv'),
                     delimiter=';',
                     dtype={
                         'SK_ID': np.uint16,
                         'CSI': np.uint8,
                         'CONTACT_DATE': str
                     })
    df['CONTACT_DATE'] = pd.to_datetime(df['CONTACT_DATE'] + '.' + YEAR,
                                        dayfirst=True,
                                        format='%d.%m',
                                        infer_datetime_format=True,
                                        cache=True)

    return df


def load_csi_test():
    df = pd.read_csv(os.path.join(TEST_DIR, 'subs_csi_test.csv'),
                     delimiter=';',
                     dtype={
                         'SK_ID': np.uint16,
                         'CONTACT_DATE': str
                     })
    df['CONTACT_DATE'] = pd.to_datetime(df['CONTACT_DATE'] + '.' + YEAR,
                                        dayfirst=True,
                                        format='%d.%m.%Y',
                                        infer_datetime_format=True,
                                        cache=True)

    return df


def load_features(tp: str):
    if tp == 'test':
        filepath = os.path.join(TEST_DIR, 'subs_features_{}.csv'.format(tp))
    else:
        filepath = os.path.join(TRAIN_DIR, 'subs_features_{}.csv'.format(tp))
    df = pd.read_csv(filepath,
                     delimiter=';',
                     decimal=',',
                     dtype={
                         'SNAP_DATE': str,
                         'COM_CAT#1': np.uint8,
                         'SK_ID': np.uint16,
                         'COM_CAT#2': np.uint8,
                         'COM_CAT#3': np.uint8,
                         'BASE_TYPE': np.uint8,
                         'ACT': np.uint8,
                         'ARPU_GROUP': np.float16,
                         'COM_CAT#7': np.uint8,
                         'COM_CAT#8': np.float32,
                         'DEVICE_TYPE_ID': np.float16,
                         'INTERNET_TYPE_ID': np.float16,
                         'REVENUE': np.float16,
                         'ITC': np.float16,
                         'VAS': np.float16,
                         'RENT_CHANNEL': np.float16,
                         'ROAM': np.float16,
                         'COST': np.float16,
                         'COM_CAT#17': np.float16,
                         'COM_CAT#18': np.float16,
                         'COM_CAT#19': np.float16,
                         'COM_CAT#20': np.float16,
                         'COM_CAT#21': np.float16,
                         'COM_CAT#22': np.float16,
                         'COM_CAT#23': np.float16,
                         'COM_CAT#24': str,
                         'COM_CAT#25': np.uint8,
                         'COM_CAT#26': np.uint8,
                         'COM_CAT#27': np.float16,
                         'COM_CAT#28': np.float16,
                         'COM_CAT#29': np.float16,
                         'COM_CAT#30': np.float16,
                         'COM_CAT#31': np.float16,
                         'COM_CAT#32': np.float16,
                         'COM_CAT#33': np.float16,
                         'COM_CAT#34': np.float16,
                     })
    df['SNAP_DATE'] = pd.to_datetime(df['SNAP_DATE'],
                                     dayfirst=True,
                                     format='%d.%m.%y',
                                     infer_datetime_format=True,
                                     cache=True)
    df['SNAP_DATE'] = df['SNAP_DATE']\
        .apply(lambda dt: dt.replace(year=2018) if dt.year == 2002 else dt.replace(year=2017))

    df['COM_CAT#24'] = pd.to_datetime(df['COM_CAT#24'] + '.' + PREV_YEAR,
                                      dayfirst=True,
                                      format='%d.%m.%y',
                                      infer_datetime_format=True,
                                      cache=True)

    df['ARPU_GROUP'] = df['ARPU_GROUP'].fillna(0).astype(np.uint8)
    df['COM_CAT#8'] = df['COM_CAT#8'].fillna(0).astype(np.uint16)
    df['DEVICE_TYPE_ID'] = df['DEVICE_TYPE_ID'].fillna(0).astype(np.uint8)
    df['INTERNET_TYPE_ID'] = df['INTERNET_TYPE_ID'].fillna(0).astype(np.uint8)
    df['COM_CAT#34'] = df['COM_CAT#34'].fillna(0).astype(np.uint8)

    df = df.sort_values(['SK_ID', 'SNAP_DATE'], ascending=False)
    return df


def load_avg_kpi():
    if os.path.isfile(os.path.join(CACHE_DIR, 'bs_avg_kpi.feather')):
        df = pd.read_feather(os.path.join(CACHE_DIR, 'bs_avg_kpi.feather'))

        df[df.select_dtypes(include=[np.float32]).columns] = \
            df[df.select_dtypes(include=[np.float32]).columns].astype(np.float16)
    else:
        df = pd.read_csv(os.path.join(BS_DIR, 'bs_avg_kpi.csv'),
                         delimiter=';',
                         decimal=',',
                         dtype={
                             'T_DATE': str,
                             'CELL_LAC_ID': np.uint32,
                             'CELL_AVAILABILITY_2G': np.float16,
                             'CELL_AVAILABILITY_3G': np.float16,
                             'CELL_AVAILABILITY_4G': np.float16,
                             'CSSR_2G': np.float16,
                             'CSSR_3G': np.float16,
                             'ERAB_PS_BLOCKING_RATE_LTE': np.float16,
                             'ERAB_PS_BLOCKING_RATE_PLMN_LTE': np.float16,
                             'ERAB_PS_DROP_RATE_LTE': np.float16,
                             'HSPDSCH_CODE_UTIL_3G': np.float16,
                             'NODEB_CNBAP_LOAD_HARDWARE': np.float16,
                             'PART_CQI_QPSK_LTE': np.float16,
                             'PART_MCS_QPSK_LTE': np.float16,
                             'PROC_LOAD_3G': np.float16,
                             'PSSR_2G': np.float16,
                             'PSSR_3G': np.float16,
                             'PSSR_LTE': np.float16,
                             'RAB_CS_BLOCKING_RATE_3G': np.float16,
                             'RAB_CS_DROP_RATE_3G': np.float16,
                             'RAB_PS_BLOCKING_RATE_3G': np.float16,
                             'RAB_PS_DROP_RATE_3G': np.float16,
                             'RBU_AVAIL_DL': np.float16,
                             'RBU_AVAIL_DL_LTE': np.float16,
                             'RBU_AVAIL_UL': np.float16,
                             'RBU_OTHER_DL': np.float16,
                             'RBU_OTHER_UL': np.float16,
                             'RBU_OWN_DL': np.float16,
                             'RBU_OWN_UL': np.float16,
                             'RRC_BLOCKING_RATE_3G': np.float16,
                             'RRC_BLOCKING_RATE_LTE': np.float16,
                             'RTWP_3G': np.float16,
                             'SHO_FACTOR': np.float16,
                             'TBF_DROP_RATE_2G': np.float16,
                             'TCH_DROP_RATE_2G': np.float16,
                             'UTIL_BRD_CPU_3G': np.float16,
                             'UTIL_CE_DL_3G': np.float16,
                             'UTIL_CE_HW_DL_3G': np.float16,
                             'UTIL_CE_UL_3G': np.float16,
                             'UTIL_SUBUNITS_3G': np.float16,
                             'UL_VOLUME_LTE': np.float16,
                             'DL_VOLUME_LTE': np.float16,
                             'TOTAL_DL_VOLUME_3G': np.float16,
                             'TOTAL_UL_VOLUME_3G': np.float16
                         })
        df['T_DATE'] = pd.to_datetime(df['T_DATE'] + '.' + YEAR,
                                      dayfirst=True,
                                      format='%d.%m.%y',
                                      infer_datetime_format=True,
                                      cache=True)
        df = df[df['T_DATE'] >= '2002-03-01'] \
            .reset_index(drop=True) \
            .fillna(0.0)

        df[df.select_dtypes(include=[np.float16]).columns] = \
            df[df.select_dtypes(include=[np.float16]).columns].astype(np.float32)
        df['T_DATE'] = df['T_DATE'].astype(str)

        df.to_feather(os.path.join(CACHE_DIR, 'bs_avg_kpi.feather'))

    return df


def load_chnn_kpi():
    if os.path.isfile(os.path.join(CACHE_DIR, 'bs_chnn_kpi.feather')):
        df = pd.read_feather(os.path.join(CACHE_DIR, 'bs_chnn_kpi.feather'))

        df[df.select_dtypes(include=[np.float32]).columns] = \
            df[df.select_dtypes(include=[np.float32]).columns].astype(np.float16)
    else:
        df = pd.read_csv(os.path.join(BS_DIR, 'bs_chnn_kpi.csv'),
                         delimiter=';',
                         decimal=',',
                         dtype={'T_DATE': str,
                                'CELL_LAC_ID': np.uint32,
                                'AVEUSERNUMBER': np.float16,
                                'AVEUSERNUMBER_PLMN': np.float16,
                                'AVR_DL_HSPA_USER_3G': np.float16,
                                'AVR_DL_R99_USER_3G': np.float16,
                                'AVR_DL_USER_3G': np.float16,
                                'AVR_DL_USER_LTE': np.float16,
                                'AVR_TX_POWER_3G': np.float16,
                                'AVR_UL_HSPA_USER': np.float16,
                                'AVR_UL_R99_USER': np.float16,
                                'AVR_UL_USER_3G': np.float16,
                                'AVR_UL_USER_LTE': np.float16,
                                'DL_AVR_THROUGHPUT_3G': np.float16,
                                'DL_AVR_THROUGHPUT_LTE': np.float16,
                                'DL_AVR_THROUGHPUT_R99': np.float16,
                                'DL_MEAN_USER_THROUGHPUT_LTE': np.float16,
                                'DL_MEAN_USER_THROUGHPUT_DL_2G': np.float16,
                                'DL_MEAN_USER_THROUGHPUT_HSPA3G': np.float16,
                                'DL_MEAN_USER_THROUGHPUT_PLTE': np.float16,
                                'DL_MEAN_USER_THROUGHPUT_REL93G': np.float16,
                                'HSDPA_USERS_3G': np.float16,
                                'HSUPA_USERS_3G': np.float16,
                                'RBU_USED_DL': np.float16,
                                'RBU_USED_UL': np.float16,
                                'RELATIVE_RBU_USED_DL': np.float16,
                                'RELATIVE_RBU_USED_UL': np.float16,
                                'RELATIVE_TX_POWER_3G': np.float16,
                                'UL_AVR_THROUGHPUT_3G': np.float16,
                                'UL_AVR_THROUGHPUT_LTE': np.float16,
                                'UL_AVR_THROUGHPUT_R99': np.float16,
                                'UL_MEAN_USER_THROUGHPUT_LTE': np.float16,
                                'UL_MEAN_USER_THROUGHPUT_HS3G': np.float16,
                                'UL_MEAN_USER_THROUGHPUT_PLTE': np.float16,
                                'UL_MEAN_USER_THROUGHPUT_REL93G': np.float16,
                                })
        df['T_DATE'] = pd.to_datetime(df['T_DATE'] + '.' + YEAR,
                                      dayfirst=True,
                                      format='%d.%m.%y',
                                      infer_datetime_format=True,
                                      cache=True)
        df = df[df['T_DATE'] >= '2002-03-01'] \
            .reset_index(drop=True) \
            .fillna(0.0)

        df[df.select_dtypes(include=[np.float16]).columns] = \
            df[df.select_dtypes(include=[np.float16]).columns].astype(np.float32)
        df['T_DATE'] = df['T_DATE'].astype(str)

        df.to_feather(os.path.join(CACHE_DIR, 'bs_chnn_kpi.feather'))

    return df


if __name__ == '__main__':
    pd.set_option('display.expand_frame_repr', False)

    train_df = load_csi_train()[['SK_ID', 'CSI']]
    train_cons_df = load_consumption('train')
    print(train_cons_df['CELL_LAC_ID'].value_counts())
    test_cons_df = load_consumption('test')
    print(test_cons_df['CELL_LAC_ID'].value_counts())
    print(len(set(train_cons_df['CELL_LAC_ID'].unique()).intersection(test_cons_df['CELL_LAC_ID'].unique())))

    cell_df = load_consumption('train')[['CELL_LAC_ID', 'SK_ID']]
    gr_cell_df = pd.merge(cell_df, train_df, on='SK_ID', how='left')[['CELL_LAC_ID', 'CSI']].groupby(by='CELL_LAC_ID')
    mean_cell_df = gr_cell_df.mean()
    mean_cell_df['count'] = gr_cell_df.count()
    mean_cell_df = mean_cell_df[mean_cell_df['count'] > 10][['CSI']].rename(index=str, columns={'CSI': 'fail_rate'})
    print(mean_cell_df.head())

    # chnn_df = load_chnn_kpi()
    # print(chnn_df['CELL_LAC_ID'].value_counts())
    # avg_df = load_avg_kpi()
    # print(avg_df['CELL_LAC_ID'].value_counts())

    # cons_df = cons_df[cons_df['MON'] >= '2018-03-01']
    # train_df = load_csi_test()
    # cons_df = pd.merge(cons_df, train_df[['SK_ID', 'CONTACT_DATE']], on='SK_ID', how='left')
    # cons_df = cons_df[cons_df['CONTACT_DATE'].dt.to_period('M') == cons_df['MON'].dt.to_period('M')].drop(['CONTACT_DATE'], axis=1)
    # cons_df['MON'] = (cons_df['MON'] - pd.offsets.DateOffset(months=1)).dt.to_period('M')
    # cons_df['CELL_LAC_ID'] = cons_df['CELL_LAC_ID'].astype(np.uint32)
    # cons_df['DATA_SPEED'] = cons_df['SUM_DATA_MB'] / cons_df['SUM_DATA_MIN']
    # print(cons_df.head())
    #
    # chnn_df = load_chnn_kpi()
    # chnn_df['T_DATE'] = pd.to_datetime(chnn_df['T_DATE'])
    # chnn_df['MON'] = chnn_df['T_DATE'].dt.to_period('M')
    # print(chnn_df.head())
    #
    # gr_chnn_df = chnn_df.groupby(by=['CELL_LAC_ID', 'MON'])


    # chnn_df = chnn_df.groupby(by=[''])
    # cons_df = pd.merge(cons_df, chnn_df, on=['CELL_LAC_ID', 'T_DATE'], how='left')
    # print(cons_df.info())

    # train_df = load_csi_train()
    # test_df = load_csi_test()
    # feat_df = load_features('test')
    # print(test_df[test_df['SK_ID'] == 6184].head(10))
    # print(feat_df[feat_df['SK_ID'] == 6184].head(10))
    #
    # print(test_df[test_df['SK_ID'] == 1927].head(10))
    # print(feat_df[feat_df['SK_ID'] == 1927].head(10))

    # train_df = load_csi_train()
    # train_feat_df = load_features('train')
    #
    # train_df = merge_features(train_df, train_feat_df)
    #
    # print(train_df.head(100))
    #
    # print(train_df.groupby(['CONTACT_DATE']).mean()['CSI'] * train_df.groupby(['CONTACT_DATE']).count()['CSI'])
    # print(train_df.groupby(['CONTACT_DATE']).mean()['CSI'])
    # print(train_df.groupby(['CONTACT_DATE']).count()['CSI'])
    #
    # feat_df = load_features('test')

    # print(train_df['CONTACT_DATE'].value_counts())
    # print(test_df['CONTACT_DATE'].value_counts())
    # print(test_df['CONTACT_DATE'].value_counts())
    # print(test_df.info())
    #
    # field = 'CONTACT_DATE'
    # train_vc = train_df[field].value_counts()
    # test_vc = test_df[field].value_counts()
    # print(train_vc)
    # print(test_vc)
    # print('Not in test', set(train_vc.index) - set(test_vc.index))
    # print('Not in train', set(test_vc.index) - set(train_vc.index))

    # cons_df = load_data_session('train')
    # print(cons_df['START_TIME'].describe())

