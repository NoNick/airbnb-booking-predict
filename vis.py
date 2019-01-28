import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time


def cntByUser(users, actions):
    join = actions.set_index('user_id').join(users.set_index('user_id'), on='user_id', how='inner')
    join['cnt'] = join.groupby('user_id').size()
    return join.drop_duplicates()


def avgByCountry(cnt):
    return cnt.groupby('country_destination')['cnt'].mean()


train = pd.read_csv("data/train_users_2.csv")\
    .drop(['date_account_created', 'timestamp_first_active', 'date_first_booking', 'gender', 'age', 'signup_method',
           'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
           'signup_app', 'first_device_type', 'first_browser'], axis=1)
sessions = pd.read_csv("data/sessions.csv")\
    .drop(['action_type', 'action_detail', 'device_type', 'secs_elapsed'], axis=1)
# allActions = avgByCountry(cntByUser(train, sessions))
# searches = avgByCountry(cntByUser(train, sessions[sessions.action == 'search_results']))
# lookups = avgByCountry(cntByUser(train, sessions[sessions.action == 'lookup']))
# print(allActions.to_frame().
#       join(searches.to_frame(), on='country_destination', lsuffix='all', rsuffix='searches').
#       join(lookups.to_frame(), on='country_destination', lsuffix='', rsuffix='lookups'))

cnt = cntByUser(train, sessions[sessions.action == 'search_results'])
cntNDF = cnt[cnt.country_destination == 'NL']
print(cntNDF)
sns.set_style("white", {'ytick.major.size': 10.0})
sns.set_context("poster", font_scale=1.1)
sns.distplot(cnt[cnt.country_destination == 'NL'].cnt, kde=False, rug=True)
sns.distplot(cnt[cnt.country_destination == 'US'].cnt, kde=False, rug=True)
x = 1