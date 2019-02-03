import numpy as np
import pandas as pd
import datetime

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ensemble import EN_optB
from classifiers import getClassifiersList, getTrainTestValidPredictions, learn, predict, NA_CONST
from submission_utils import saveResult

FOLDS = 5

n_classes = 12  # Same number of classes as in Airbnb competition.
data = pd.read_csv('data/train_users_2_norm.csv').drop('user_id', axis=1).fillna(NA_CONST)
labels = data.pop('country_destination')
le = LabelEncoder()
labelsEncoded = le.fit_transform(labels)

# Splitting train data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(data, labelsEncoded, test_size=0.3, shuffle=True)

print('Data shape:')
print('X_train: %s, X_valid: %s \n' % (X_train.shape, X_valid.shape))

# Defining the classifiers
clfs = getClassifiersList(X_train)

print('Learning individual classifiers on X_train, predicting X_valid')
print('--------------------------------------------------------------')
p_valid = getTrainTestValidPredictions(clfs, X_train, y_train, X_valid)
print('')

print('Optimizing weights of ensemble')
print('--------------------------------------------------------------')

# Creating the data for the 2nd layer.
XV = np.hstack(p_valid)

start = datetime.datetime.now()
enB = EN_optB(n_classes)
enB.fit(XV, y_valid)
print("Fitted ensemble in " + str(datetime.datetime.now() - start))
print('')


#Calibrated version of EN_optB
print("Calibrating ensemble with %d folds" % FOLDS)
start = datetime.datetime.now()
cc_optB = CalibratedClassifierCV(enB, method='isotonic', cv=FOLDS)
cc_optB.fit(XV, y_valid)
print("Ensemble is calibrated in " + str(datetime.datetime.now() - start))
print('')

print('Predicting test set')
print('--------------------------------------------------------------')
# print("Learn classifiers on whole train set")
# learn(clfs, data, labels)

start = datetime.datetime.now()
test = pd.read_csv('data/test_users_norm.csv').fillna(NA_CONST)
Xid = test.pop('id')
XT = np.hstack(predict(clfs, test))
saveResult(Xid, cc_optB.predict_proba(XT), le, 'predict/ensemble.csv')
# saveResult(Xid, enB.predict_proba(XT), le, 'predict/ensemble.csv')
print("Ensemble submission is predicted and weighted in " + str(datetime.datetime.now() - start))
