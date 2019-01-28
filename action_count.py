import pandas as pd
import numpy as np
from datetime import datetime

sessions = pd.read_csv("data/sessions.csv").drop(['device_type', 'secs_elapsed'], axis=1)
sessions.dropna(subset=['user_id'], inplace=True)
sessions.sort_values('user_id', inplace=True)
sessions['action'].fillna('', inplace=True)
sessions['action_detail'].fillna('', inplace=True)
sessions['action_full'] = sessions['action'] + '$' + sessions['action_detail']
# make sure there are no more than one unique action_detail per action_full
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#      print(sessions.groupby('action_full').action_type.nunique())
sessions.drop(['action', 'action_type', 'action_detail'], axis=1, inplace=True)
actions = list(sessions['action_full'].unique())
actionsTotal = len(sessions['action_full'])
actN = len(actions)
usersN = sessions['user_id'].nunique()
result = pd.DataFrame(0, index=np.arange(usersN), columns=['user_id']+actions)
result.user_id = result.user_id.astype(object)

start = datetime.now()
user_id = ''
cntVector = []
pos = 0
i = 0
for _, session in sessions.iterrows():
    if user_id != session['user_id']:
        if user_id != '':
            pos = pos + 1
        user_id = session['user_id']
        result.iloc[pos, 0] = user_id
    actionNumber = actions.index(session['action_full']) + 1
    result.iloc[pos, actionNumber] = result.iloc[pos, actionNumber] + 1
    i = i + 1
    if i % 10000 == 0:
        print(("Processed %8d / %d (%.2f%%) actions, " + str(datetime.now() - start) + " per 10k")
              % (i, actionsTotal, i * 100 / actionsTotal))
        start = datetime.now()

result.to_csv("data/user_actions_count.csv", index=False)