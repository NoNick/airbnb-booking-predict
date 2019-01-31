import numpy as np
import pandas as pd
import datetime
import xgboost
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

NA_CONST = -1

n_classes = 12  # Same number of classes as in Airbnb competition.
data = pd.read_csv('data/train_users_2_norm.csv').drop('user_id', axis=1).fillna(NA_CONST)
labels = data.pop('country_destination')

start = datetime.datetime.now()

le = LabelEncoder()
y = le.fit_transform(labels)

xgb_model = xgboost.XGBClassifier()
xgb_param = {
    'max_depth': [5, 6, 7],
    'silent': [True],
    'n_jobs': [1],
    'n_thread': [1],
    'subsample': [0.5],
    'colsample_bytree': [0.3],
    'objective': ['multi:softprob'],
    'num_class': [n_classes],
    'n_estimators': [35, 50, 75, 100],
    'missing': [-1]
}

clf = GridSearchCV(xgb_model, xgb_param, n_jobs=4, cv=4, iid=False,
                   scoring='neg_log_loss', # TODO: try merror
                   verbose=2, refit=True, error_score=np.nan)

clf.fit(data, y)

#trust your CV!
print('\n Best estimator:')
print(clf.best_estimator_)
print('\n Best hyperparameters:')
print(clf.best_params_)
