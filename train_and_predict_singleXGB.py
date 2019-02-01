import pandas as pd
import datetime
import xgboost
from sklearn.preprocessing import LabelEncoder

from submission_utils import saveResult

NA_CONST = -1
THREADS = 4

n_classes = 12  # Same number of classes as in Airbnb competition.
data = pd.read_csv('data/train_users_2_norm.csv').drop('user_id', axis=1).fillna(NA_CONST)
labels = data.pop('country_destination')

start = datetime.datetime.now()

le = LabelEncoder()
y = le.fit_transform(labels)
xgtrain = xgboost.DMatrix(data, label=y)

clf = xgboost.XGBClassifier(max_depth=5,
                            missing=NA_CONST,
                            n_estimators=35,
                            subsample=0.5,
                            colsample_bytree=0.3,
                            objective='multi:softprob',
                            num_class=n_classes,
                            silent=False,
                            n_jobs=THREADS,
                            nthread=THREADS)
clf.fit(data, y)

X_test = pd.read_csv('data/test_users_norm.csv').fillna(NA_CONST)
Xid = X_test.pop('id')

y_pred = clf.predict_proba(X_test)

saveResult(Xid, y_pred, le, 'predict/xgb.csv')

print(str(datetime.datetime.now() - start))
