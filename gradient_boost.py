import numpy as np
import pandas as pd
import datetime
import xgboost
from sklearn.preprocessing import LabelEncoder

NA_CONST = -1

n_classes = 12  # Same number of classes as in Airbnb competition.
data = pd.read_csv('data/train_users_2_norm.csv').drop('user_id', axis=1).fillna(NA_CONST)
labels = data.pop('country_destination')

start = datetime.datetime.now()

le = LabelEncoder()
y = le.fit_transform(labels)
xgtrain = xgboost.DMatrix(data, label=y)

clf = xgboost.XGBClassifier(max_depth=6,
                            silent=False,
                            n_jobs=4,
                            nthread=4,
                            subsample=0.5,
                            colsample_bytree=0.3,
                            objective='multi:softprob',
                            num_class=n_classes)
clf.fit(data, y)

X_test = pd.read_csv('data/test_users_norm.csv').fillna(NA_CONST)
Xid = X_test.pop('id')

y_pred = clf.predict_proba(X_test)

#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(Xid)):
    idx = Xid[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('predict/xgb.csv', index=False)

print(str(datetime.datetime.now() - start))
