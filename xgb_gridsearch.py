import numpy as np
import pandas as pd
import datetime
import xgboost
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

from classifiers import clfPos

NA_CONST = -1
THREADS = 4
FOLDS = 3

n_classes = 12  # Same number of classes as in Airbnb competition.
data = pd.read_csv('data/train_users_2_norm.csv').drop('user_id', axis=1).fillna(NA_CONST)
labels = data.pop('country_destination')

start = datetime.datetime.now()

le = LabelEncoder()
y = le.fit_transform(labels)

xgb_param = {
    'BaseFeatures': {
        'max_depth': [6],
        'learning_rate': [0.02],
        'silent': [True],
        'n_jobs': [1],
        'n_thread': [1],
        'subsample': [0.5],
        'colsample_bytree': [0.15],
        'objective': ['multi:softprob'],
        'num_class': [n_classes],
        'n_estimators': [225, 300],
        'missing': [NA_CONST]
    },
    'AgeGender': {
        'max_depth': [7],  # went up
        'learning_rate': [0.02],
        'silent': [True],
        'n_jobs': [1],
        'n_thread': [1],
        'subsample': [0.5],
        'colsample_bytree': [0.4],  # went up
        'objective': ['multi:softprob'],
        'num_class': [n_classes],
        'n_estimators': [250, 350],  # went up
        'missing': [NA_CONST]
    },
    # 'DAC': {
    #     'max_depth': [1],
    #     'learning_rate': [0.03],
    #     'silent': [True],
    #     'n_jobs': [1],
    #     'n_thread': [1],
    #     'subsample': [0.5],
    #     'colsample_bytree': [0.1],  # went down
    #     'objective': ['multi:softprob'],
    #     'num_class': [n_classes],
    #     'n_estimators': [100, 150, 200],  # went up
    #     'missing': [NA_CONST]
    # },
    # 'TFA': {
    #     'max_depth': [1],
    #     'learning_rate': [0.03],
    #     'silent': [True],
    #     'n_jobs': [1],
    #     'n_thread': [1],
    #     'subsample': [0.5],
    #     'colsample_bytree': [0.1],  # went down
    #     'objective': ['multi:softprob'],
    #     'num_class': [n_classes],
    #     'n_estimators': [100, 150, 200],  # went up
    #     'missing': [NA_CONST]
    # },
    'Actions': {
        'max_depth': [5, 6],
        'learning_rate': [0.02],
        'silent': [True],
        'n_jobs': [1],
        'n_thread': [1],
        'subsample': [0.5],
        'colsample_bytree': [0.15, 0.3],
        'objective': ['multi:softprob'],
        'num_class': [n_classes],
        'n_estimators': [150, 200, 250],
        'missing': [0]
    },
}

for name, params in xgb_param.items():
    clf = GridSearchCV(xgboost.XGBClassifier(), params, n_jobs=THREADS, cv=FOLDS, iid=False,
                   scoring='neg_log_loss',
                   verbose=2, refit=True, error_score=np.nan)
    beginColumn = data.columns.get_loc(clfPos[name][0])
    endColumn = data.columns.get_loc(clfPos[name][1])
    clf.fit(data.iloc[:, beginColumn:endColumn], y)

    print('=' * len(str(clf.best_params_)))
    print('\033[39mBest params for %s:' % name)
    print(clf.best_params_)
    print('\033[0m')
    print('=' * len(str(clf.best_params_)))
