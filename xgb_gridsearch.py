import numpy as np
import pandas as pd
import datetime
import xgboost
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate

from classifiers import clfPos, nDCG5_score

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
        'max_depth': [2, 3, 4],
        'learning_rate': [0.05],
        'silent': [True],
        'n_jobs': [THREADS],
        'n_thread': [THREADS],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5],
        'objective': ['multi:softprob'],
        'num_class': [n_classes],
        'n_estimators': [50, 200],
        'missing': [NA_CONST]
    },
    'AgeGender': {
        'max_depth': [2, 3, 4],
        'learning_rate': [0.05],
        'silent': [True],
        'n_jobs': [THREADS],
        'n_thread': [THREADS],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5],
        'objective': ['multi:softprob'],
        'num_class': [n_classes],
        'n_estimators': [50, 200],
        'missing': [NA_CONST]
    },
    'DAC': {
        'max_depth': [1, 2],
        'learning_rate': [0.05],
        'silent': [True],
        'n_jobs': [THREADS],
        'n_thread': [THREADS],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5],
        'objective': ['multi:softprob'],
        'num_class': [n_classes],
        'n_estimators': [50, 150],
        'missing': [NA_CONST]
    },
    'TFA': {
        'max_depth': [1, 2],
        'learning_rate': [0.05],
        'silent': [True],
        'n_jobs': [THREADS],
        'n_thread': [THREADS],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5],
        'objective': ['multi:softprob'],
        'num_class': [n_classes],
        'n_estimators': [50, 150],
        'missing': [NA_CONST]
    },
    'Actions': {
        'max_depth': [4, 5],
        'learning_rate': [0.05],
        'silent': [True],
        'n_jobs': [THREADS],
        'n_thread': [THREADS],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5],
        'objective': ['multi:softprob'],
        'num_class': [n_classes],
        'n_estimators': [50, 200],
        'missing': [NA_CONST]
    },
    'AllFeatures': {
        'max_depth': [6, 7],
        'learning_rate': [0.05],
        'silent': [True],
        'n_jobs': [THREADS],
        'n_thread': [THREADS],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5],
        'objective': ['multi:softprob'],
        'num_class': [n_classes],
        'n_estimators': [100, 300],
        'missing': [NA_CONST]
    }
}

for name, params in xgb_param.items():
    clf = GridSearchCV(xgboost.XGBClassifier(), params, n_jobs=1, cv=FOLDS, iid=False,
                       # scoring='neg_log_loss',
                       scoring=nDCG5_score,
                       verbose=2, refit=True, error_score=np.nan)
    beginColumn = data.columns.get_loc(clfPos[name][0])
    endColumn = data.columns.get_loc(clfPos[name][1])
    clf.fit(data.iloc[:, beginColumn:endColumn], y)

    print('=' * len(str(clf.best_params_)))
    print('\033[39m %s' % name)
    result = pd.DataFrame(clf.cv_results_).drop(['split0_train_score', 'split1_train_score', 'split2_train_score',
                                                'mean_train_score', 'std_train_score', 'mean_score_time',
                                                 'std_score_time', 'param_n_jobs', 'param_n_thread',
                                                 'param_num_class', 'param_objective', 'param_silent',
                                                 'param_missing', 'params', 'std_fit_time', 'std_test_score',
                                                 'split0_test_score', 'split1_test_score', 'split2_test_score'], axis=1)
    print(tabulate(result, headers='keys', tablefmt='psql'))
    print()
    print('Best params:')
    print(str(clf.best_params_))
    print('\033[0m')
    print('=' * len(str(clf.best_params_)))
