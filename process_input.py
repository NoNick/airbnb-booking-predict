import math
import re

import pandas as pd
import numpy as np
from pandas._libs.index import datetime

# if an action was used less than this number across all sessions, drop it from features
from sklearn.preprocessing import LabelEncoder

ACTION_COUNT_DROP_THRESHOLD = 100

def normalizeData(data):
    DAC = pd.to_datetime(data.pop('date_account_created'), format='%Y-%m-%d')
    data['DAC_year'] = DAC.dt.year
    data['DAC_month'] = DAC.dt.month
    data['DAC_day_of_month'] = DAC.dt.day
    data['DAC_weekday'] = DAC.dt.weekday
    data['DAC_season'] = DAC.dt.month.apply(lambda x: (x % 12 + 3) // 3)

    TFA = pd.to_datetime(data.pop('timestamp_first_active'), format='%Y%m%d%H%M%S')
    data['TFA_year'] = TFA.dt.year
    data['TFA_month'] = TFA.dt.month
    data['TFA_day_of_month'] = TFA.dt.day
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

    signupMapping = {
        'basic': 1,
        'facebook': 2,
        'google': 3
    }
    data['signup_method'] = data['signup_method'].map(signupMapping)

    langLE = LabelEncoder()
    data['language'] = langLE.fit_transform(data['language'])
    data['language_copy'] = langLE.fit_transform(data['language_copy'])

    data['language'] = mapStrToOrdinals(data['language'])
    data['affiliate_channel'] = mapStrToOrdinals(data['affiliate_channel'])
    data['affiliate_provider'] = mapStrToOrdinals(data['affiliate_provider'])
    data['first_affiliate_tracked'] = mapStrToOrdinals(data['first_affiliate_tracked'])
    data['signup_app'] = mapStrToOrdinals(data['signup_app'])
    data['first_device_type'] = mapStrToOrdinals(data['first_device_type'])
    # data.drop('first_device_type', axis=1, inplace=True)
    data['first_browser'] = mapStrToOrdinals(data['first_browser'])

    data.loc[(14 > data.age) & (data.age > 100), 'age'] = np.nan

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
    ages = users.pop('age')
    genders = users.pop('gender')
    users['age'] = ages
    users['gender'] = genders
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

train = pd.read_csv("data/train_users_2.csv").drop('date_first_booking', axis=1)
test = pd.read_csv("data/test_users.csv").drop('date_first_booking', axis=1)
print("Loaded train and test sets")

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
actions.set_index('user_id', inplace=True)
print("Dropped %d actions due to rare use" % (tmp - len(actions.columns)))

usedDevice = getMostUsedDeviceByUserId()
elapsed = getSecsElapsedByUserId()
print("Processed session data")


def joinTables(src):
    return src.join(actions, how='left').join(usedDevice, how='left').join(elapsed, how='left')


train = joinTables(train.set_index('user_id'))
destinations = train.pop('country_destination')
train['country_destination'] = destinations  # move column to the end
train.to_csv("data/train_users_2_norm.csv")
test = joinTables(test.set_index('id'))
test.to_csv("data/test_users_norm.csv")
print("%d features in total" % (len(train.columns) - 2))

print("Completed in " + str(datetime.now() - start))
