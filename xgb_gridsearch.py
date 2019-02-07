import numpy as np
import pandas as pd
import datetime
import xgboost
import warnings
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate

from classifiers import clfPos, nDCG5_score, accuracy, underweightClasses, getWeightsLowerClasses, \
    nDCG5_NoNDF_NoUS_score

NA_CONST = -1
THREADS = 4
FOLDS = 3

warnings.simplefilter(action='ignore', category=FutureWarning)

n_classes = 12  # Same number of classes as in Airbnb competition.
data = pd.read_csv('data/train_users_2_norm.csv').drop('user_id', axis=1).fillna(NA_CONST)
labels = data.pop('country_destination')

start = datetime.datetime.now()

le = LabelEncoder()
y = le.fit_transform(labels)

# ['AU' 'CA' 'DE' 'ES' 'FR' 'GB' 'IT' 'NDF' 'NL' 'PT' 'US' 'other']
scorers = {'nDCG5': nDCG5_score,
           'nDCG5_NoNDF_NoUS': nDCG5_NoNDF_NoUS_score
           # 'AU_acc': make_scorer(lambda y, y_pred: accuracy(0, y, y_pred), needs_proba=False),
           # 'CA_acc': make_scorer(lambda y, y_pred: accuracy(1, y, y_pred), needs_proba=False),
           # 'DE_acc': make_scorer(lambda y, y_pred: accuracy(2, y, y_pred), needs_proba=False),
           # 'ES_acc': make_scorer(lambda y, y_pred: accuracy(3, y, y_pred), needs_proba=False),
           # 'FR_acc': make_scorer(lambda y, y_pred: accuracy(4, y, y_pred), needs_proba=False),
           # 'GB_acc': make_scorer(lambda y, y_pred: accuracy(5, y, y_pred), needs_proba=False),
           # 'IT_acc': make_scorer(lambda y, y_pred: accuracy(6, y, y_pred), needs_proba=False),
           # 'NDF_acc': make_scorer(lambda y, y_pred: accuracy(7, y, y_pred), needs_proba=False),
           # 'NL_acc': make_scorer(lambda y, y_pred: accuracy(8, y, y_pred), needs_proba=False),
           # 'PT_acc': make_scorer(lambda y, y_pred: accuracy(9, y, y_pred), needs_proba=False),
           # 'US_acc': make_scorer(lambda y, y_pred: accuracy(10, y, y_pred), needs_proba=False),
           # 'other_acc': make_scorer(lambda y, y_pred: accuracy(11, y, y_pred), needs_proba=False)
           }

xgb_param = {
    'BaseFeatures': {
        'max_depth': [3, 4],
        'learning_rate': [0.05],
        'silent': [True],
        'n_jobs': [THREADS],
        'n_thread': [THREADS],
        'subsample': [0.5],
        'colsample_bytree': [0.5],
        'objective': ['rank:ndcg'],
        'eval_metric': ['ndcg@5'],
        'num_class': [n_classes],
        'n_estimators': [200],
        'missing': [NA_CONST]
    },
    'AgeGender': {
        'max_depth': [1],
        'learning_rate': [0.05],
        'silent': [True],
        'n_jobs': [THREADS],
        'n_thread': [THREADS],
        'subsample': [0.5],
        'colsample_bytree': [0.5],
        'objective': ['rank:ndcg'],
        'eval_metric': ['ndcg@5'],
        'num_class': [n_classes],
        'n_estimators': [50],
        'missing': [NA_CONST]
    },
    'DAC_TFA': {
        'max_depth': [1],
        'learning_rate': [0.05],
        'silent': [True],
        'n_jobs': [THREADS],
        'n_thread': [THREADS],
        'subsample': [0.5],
        'colsample_bytree': [0.5],
        'objective': ['rank:ndcg'],
        'eval_metric': ['ndcg@5'],
        'num_class': [n_classes],
        'n_estimators': [35, 50, 75],
        'missing': [NA_CONST]
    },
    'Actions': {
        'max_depth': [2],
        'learning_rate': [0.05],
        'silent': [True],
        'n_jobs': [THREADS],
        'n_thread': [THREADS],
        'subsample': [0.3, 0.5],
        'colsample_bytree': [0.5],
        'objective': ['rank:ndcg'],
        'eval_metric': ['ndcg@5'],
        'num_class': [n_classes],
        'n_estimators': [35, 50],
        'missing': [NA_CONST]
    }
}

for name, params in xgb_param.items():
    clf = GridSearchCV(xgboost.XGBClassifier(), params, n_jobs=1, cv=FOLDS, iid=False,
                       scoring=scorers,
                       verbose=2, refit=False, error_score=np.nan)
    beginColumn = data.columns.get_loc(clfPos[name][0])
    endColumn = data.columns.get_loc(clfPos[name][1])

    if name in underweightClasses:
        clf.fit(data.iloc[:, beginColumn:endColumn], y,
                sample_weight=getWeightsLowerClasses(underweightClasses.get(name), y))
    else:
        clf.fit(data.iloc[:, beginColumn:endColumn], y)

    print('\033[39m %s' % name)
    result = pd.DataFrame(clf.cv_results_)[['mean_fit_time', 'param_max_depth', 'param_n_estimators',
                                            'param_subsample', 'param_colsample_bytree',
                                            'mean_train_nDCG5', 'mean_test_nDCG5',
                                            'mean_train_nDCG5_NoNDF_NoUS', 'mean_test_nDCG5_NoNDF_NoUS']]
                                            # 'mean_test_AU_acc', 'mean_test_CA_acc', 'mean_test_DE_acc',
                                            # 'mean_test_ES_acc', 'mean_test_FR_acc', 'mean_test_GB_acc',
                                            # 'mean_test_IT_acc', 'mean_test_NDF_acc', 'mean_test_NL_acc',
                                            # 'mean_test_PT_acc', 'mean_test_US_acc', 'mean_test_other_acc']]
    print(tabulate(result, headers='keys', tablefmt='psql'))
    print('\033[0m')
