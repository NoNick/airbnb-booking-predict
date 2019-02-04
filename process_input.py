import math
import re

import pandas as pd
import numpy as np
from pandas._libs.index import datetime

# if an action was used less than this number across all sessions, drop it from features
from sklearn.preprocessing import LabelEncoder

ACTION_COUNT_DROP_THRESHOLD = 50

def normalizeData(data):
    DAC = pd.to_datetime(data.pop('date_account_created'), format='%Y-%m-%d')
    data['DAC_year'] = DAC.dt.year
    data['DAC_month'] = DAC.dt.month
    data['DAC_day_of_month'] = DAC.dt.day
    data['DAC_yearmonthday'] = DAC.dt.strftime('%Y%m%d')
    data['DAC_yearweek'] = DAC.dt.strftime('%Y%U')
    data['DAC_weekday'] = DAC.dt.weekday
    data['DAC_season'] = DAC.dt.month.apply(lambda x: (x % 12 + 3) // 3)

    TFA = pd.to_datetime(data.pop('timestamp_first_active'), format='%Y%m%d%H%M%S')
    data['TFA_year'] = TFA.dt.year
    data['TFA_month'] = TFA.dt.month
    data['TFA_day_of_month'] = TFA.dt.day
    data['TFA_yearmonthday'] = TFA.dt.strftime('%Y%m%d')
    data['TFA_yearweek'] = TFA.dt.strftime('%Y%U')
    data['TFA_weekday'] = TFA.dt.weekday
    data['TFA_season'] = TFA.dt.month.apply(lambda x: (x % 12 + 3) // 3)
    data['TFA_hour_in_day'] = TFA.dt.hour

    genderMapping = {
        'MALE': 1,
        'FEMALE': 2,
        'OTHER': 3,
        '-unknown-': np.nan
    }
    data['gender'] = data['gender'].map(genderMapping)
    data['gender_copy'] = data['gender_copy'].map(genderMapping)

    langLE = LabelEncoder()
    data['language'] = langLE.fit_transform(data['language'])
    data['language_copy'] = langLE.fit_transform(data['language_copy'])

    data.loc[data['first_browser'] == '-unknown-', 'first_browser'] = np.nan
    data.loc[data['first_affiliate_tracked'] == 'untracked', 'first_affiliate_tracked'] = np.nan
    data.loc[data['first_device_type'] == 'Other/Unknown', 'first_device_type'] = np.nan
    for columnName in ['signup_method', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
                       'signup_app', 'first_browser', 'first_device_type']:
        mapStrToOrdinals(data, columnName)

    data.loc[(14 > data.age) & (data.age > 100), 'age'] = np.nan

    return data


def mapStrToOrdinals(data, columnName):
    nanIndices = data[columnName].isnull()
    data[columnName].fillna('NA', inplace=True)
    data[columnName] = LabelEncoder().fit(data[columnName]).transform(data[columnName])
    data.loc[nanIndices, columnName] = np.nan


def dropLowCountColumns(table, minCount):
    columnsToDrop = []
    for column in table:
        if np.issubdtype(table[column].dtype, np.number):
            if table[column].astype(bool).sum(axis=0) < minCount:
                columnsToDrop.append(column)
    return table.drop(columnsToDrop, axis=1)


def getMostUsedDeviceByUserId():
    sessions = pd.read_csv("data/sessions.csv")\
        .drop(['action', 'action_type', 'action_detail', 'secs_elapsed'], axis=1)
    result = sessions.groupby('user_id').agg(lambda x: x['device_type'].value_counts().index[0])
    le = LabelEncoder()
    result['device_type'] = le.fit_transform(result['device_type'])
    return result


def getSecsElapsedByUserId():
    sessions = pd.read_csv("data/sessions.csv")\
        .drop(['action', 'action_type', 'action_detail', 'device_type'], axis=1).fillna(0)
    return sessions.groupby('user_id').agg(lambda x: x['secs_elapsed'].sum())


def getAgeGenderBuckets():
    buckets = pd.read_csv("data/age_gender_bkts.csv").drop(['year'], axis=1)  # all data is from 2015
    getBucketStartInt = lambda x: int(re.search(r'\d+', x['age_bucket']).group())
    buckets['age_start'] = buckets.apply(getBucketStartInt, axis=1)
    return buckets


def joinAgeGenderBucketFeatues(buckets, users):
    # move relevant features alongside new features
    ages = users['age']
    genders = users['gender']
    users['age_copy'] = ages
    users['gender_copy'] = genders
    users['language_copy'] = users['language']

    BUCKET_SIZE = 5  # years
    SUFFIX_1 = '_sameGender_population'
    SUFFIX_2 = '_oppositeGender_population'

    countries = sorted(buckets['country_destination'].unique())
    for country in countries:
        users[country + SUFFIX_1] = np.nan
        users[country + SUFFIX_2] = np.nan

    for _, bucket in buckets.iterrows():
        ageStart = bucket['age_start']
        ageEnd = ageStart + BUCKET_SIZE
        population = bucket['population_in_thousands']
        if bucket['gender'] == 'male':
            users.loc[(ageStart <= users['age']) & (users['age'] <= ageEnd) & (users['gender'] == 'MALE'),
                      bucket['country_destination'] + SUFFIX_1] = population
            users.loc[(ageStart <= users['age']) & (users['age'] <= ageEnd) & (users['gender'] == 'FEMALE'),
                      bucket['country_destination'] + SUFFIX_2] = population
        elif bucket['gender'] == 'female':
            users.loc[(ageStart <= users['age']) & (users['age'] <= ageEnd) & (users['gender'] == 'MALE'),
                      bucket['country_destination'] + SUFFIX_2] = population
            users.loc[(ageStart <= users['age']) & (users['age'] <= ageEnd) & (users['gender'] == 'FEMALE'),
                      bucket['country_destination'] + SUFFIX_1] = population

    return users


start = datetime.now()

train = pd.read_csv("data/train_users_2.csv").drop('date_first_booking', axis=1).set_index('user_id')
test = pd.read_csv("data/test_users.csv").drop('date_first_booking', axis=1).set_index('id')
print("Loaded train and test sets")

usedDevice = getMostUsedDeviceByUserId()
elapsed = getSecsElapsedByUserId()
train = train.join(usedDevice, how='left')#.join(elapsed, how='left')
test = test.join(usedDevice, how='left')#.join(elapsed, how='left')
print("Processed used devices")

ageGenderBuckets = getAgeGenderBuckets()
train = joinAgeGenderBucketFeatues(ageGenderBuckets, train)
test = joinAgeGenderBucketFeatues(ageGenderBuckets, test)
print("Joined with age and gender features per country")

train = normalizeData(train)
test = normalizeData(test)
print("Normalized numbers and labels")

actions = pd.read_csv("data/user_actions_count.csv")
tmp = len(actions.columns)
actions = dropLowCountColumns(actions, ACTION_COUNT_DROP_THRESHOLD)
actions.replace(0, np.nan, inplace=True)
actions.set_index('user_id', inplace=True)
print("Dropped %d actions due to rare use" % (tmp - len(actions.columns)))

train = train.join(actions, how='left')
destinations = train.pop('country_destination')
train['country_destination'] = destinations  # move column to the end
train.to_csv("data/train_users_2_norm.csv")
test = test.join(actions, how='left')
test.to_csv("data/test_users_norm.csv")
print("%d features in total" % (len(train.columns) - 2))

print("Completed in " + str(datetime.now() - start))
