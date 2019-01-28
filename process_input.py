import math
import pandas as pd
import numpy as np
from pandas._libs.index import datetime


def normalizeData(data):
    data['date_account_created'] = normalizeDateColumn(data['date_account_created'], '%Y-%m-%d')

    first_active_min = min(data['timestamp_first_active'])
    data['timestamp_first_active'] = data['timestamp_first_active'].map(lambda x: x - first_active_min)

    genderMapping = {
        'MALE': 1,
        'FEMALE': 2,
        'OTHER': 3,
        '-unknown-': np.nan
    }
    data['gender'] = data['gender'].map(genderMapping)

    signupMapping = {
        'basic': 1,
        'facebook': 2,
        'google': 3
    }
    data['signup_method'] = data['signup_method'].map(signupMapping)

    data['language'] = mapStrToOrdinals(data['language'])

    data['language'] = mapStrToOrdinals(data['language'])
    data['affiliate_channel'] = mapStrToOrdinals(data['affiliate_channel'])
    data['affiliate_provider'] = mapStrToOrdinals(data['affiliate_provider'])
    data['first_affiliate_tracked'] = mapStrToOrdinals(data['first_affiliate_tracked'])
    data['signup_app'] = mapStrToOrdinals(data['signup_app'])
    data['first_device_type'] = mapStrToOrdinals(data['first_device_type'])
    data['first_browser'] = mapStrToOrdinals(data['first_browser'])

    return data

def mapStrToOrdinals(column):
    uniques = np.unique(column.fillna('abracadabra')).tolist()
    uniquesWithIndices = {k: v + 1 for v, k in enumerate(uniques)}  # TODO: sort by frequency?
    return column.map(uniquesWithIndices)

def normalizeDateColumn(column, dateFormat):
    column = pd.to_datetime(column, format=dateFormat)
    column = column.map(datetime.toordinal)
    # pandas maps nan to 1
    column.replace(1, math.nan, inplace=True)
    min_value = np.nanmin(column)
    column = column.map(lambda x: x - min_value)
    return column

data = pd.read_csv("data/train_users_2.csv").drop('date_first_booking', axis=1)
normalizeData(data).to_csv("data/train_users_2_norm.csv", index=False)
test = pd.read_csv("data/test_users.csv").drop('date_first_booking', axis=1)
normalizeData(test).to_csv("data/test_users_norm.csv", index=False)