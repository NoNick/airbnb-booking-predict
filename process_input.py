import math
import pandas as pd
import numpy as np
from pandas._libs.index import datetime

# if an action was used less than this number across all sessions, drop it from features
ACTION_COUNT_DROP_THRESHOLD = 300

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


def dropLowCountColumns(table, minCount):
    columnsToDrop = []
    for column in table:
        if np.issubdtype(table[column].dtype, np.number):
            if table[column].sum() < minCount:
                columnsToDrop.append(column)
    return table.drop(columnsToDrop, axis=1)

start = datetime.now()

train = pd.read_csv("data/train_users_2.csv").drop('date_first_booking', axis=1)
train = normalizeData(train)
test = pd.read_csv("data/test_users.csv").drop('date_first_booking', axis=1)
test = normalizeData(test)

actions = pd.read_csv("data/user_actions_count.csv")
tmp = len(actions.columns)
actions = dropLowCountColumns(actions, ACTION_COUNT_DROP_THRESHOLD)
actions.set_index('user_id', inplace=True)
print("Dropped %d actions due to rare use" % (tmp - len(actions.columns)))

# train = train.set_index('user_id').join(actions, how='left')
destinations = train.pop('country_destination')
train['country_destination'] = destinations  # move column to the end
train.to_csv("data/train_users_2_norm.csv", index=False)
# test = test.set_index('id').join(actions, how='left')
test.to_csv("data/test_users_norm.csv", index=False)
print("%d features in total" % (len(train.columns) - 1))

print("Completed in " + str(datetime.now() - start))