import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import log_loss

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ensemble import EN_optA, EN_optB
from classifiers import genClassifiersList, learnEachTwoOnSingleFeature, predictEachTwoOnSingleFeature, random_state
from submission_utils import saveResult

NA_CONST = -1

n_classes = 12  # Same number of classes as in Airbnb competition.
data = pd.read_csv('data/train_users_2_norm.csv').drop('user_id', axis=1).fillna(NA_CONST)
labels = data.pop('country_destination')
le = LabelEncoder()
labelsEncoded = le.fit_transform(labels)

# Splitting data into train and test sets.
X, X_test, y, y_test = train_test_split(data, labelsEncoded, test_size=0.2, random_state=random_state)

# Splitting train data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=random_state)

print('Data shape:')
print('X_train: %s, X_valid: %s, X_test: %s \n' % (X_train.shape, X_valid.shape, X_test.shape))

# Defining the classifiers
clfs = genClassifiersList(4)

print('Performance of individual classifiers (1st layer) on X_test')
print('------------------------------------------------------------')
p_valid, p_test = learnEachTwoOnSingleFeature(clfs, X, y, X_train, y_train, X_valid, X_test, y_test)
print('')

print('Performance of optimization based ensemblers (2nd layer) on X_test')
print('------------------------------------------------------------')

# Creating the data for the 2nd layer.
XV = np.hstack(p_valid)
XT = np.hstack(p_test)

start = datetime.datetime.now()
# EN_optA
enA = EN_optA(n_classes)
enA.fit(XV, y_valid)
w_enA = enA.w
y_enA = enA.predict_proba(XT)
print('{:20s} {:2s} {:1.7f}'.format('EN_optA:', 'logloss  =>', log_loss(y_test, y_enA)))
print(datetime.datetime.now() - start)
start = datetime.datetime.now()

# Calibrated version of EN_optA
cc_optA = CalibratedClassifierCV(enA, method='isotonic', cv=5)
cc_optA.fit(XV, y_valid)
y_ccA = cc_optA.predict_proba(XT)
print('{:20s} {:2s} {:1.7f}'.format('Calibrated_EN_optA:', 'logloss  =>', log_loss(y_test, y_ccA)))
print(datetime.datetime.now() - start)
start = datetime.datetime.now()

#EN_optB
enB = EN_optB(n_classes)
enB.fit(XV, y_valid)
w_enB = enB.w
y_enB = enB.predict_proba(XT)
print('{:20s} {:2s} {:1.7f}'.format('EN_optB:', 'logloss  =>', log_loss(y_test, y_enB)))
print(datetime.datetime.now() - start)
start = datetime.datetime.now()

#Calibrated version of EN_optB
cc_optB = CalibratedClassifierCV(enB, method='isotonic', cv=5)
cc_optB.fit(XV, y_valid)
y_ccB = cc_optB.predict_proba(XT)
print('{:20s} {:2s} {:1.7f}'.format('Calibrated_EN_optB:', 'logloss  =>', log_loss(y_test, y_ccB)))
print(datetime.datetime.now() - start)
start = datetime.datetime.now()
print('')

y_3l = (y_enA * 4./9.) + (y_ccA * 2./9.) + (y_enB * 2./9.) + (y_ccB * 1./9.)
print('{:20s} {:2s} {:1.7f}'.format('3rd_layer:', 'logloss  =>', log_loss(y_test, y_3l)))
print(datetime.datetime.now() - start)

from tabulate import tabulate
print('               Weights of EN_optA:')
print('|---------------------------------------------|')
wA = np.round(w_enA, decimals=2).reshape(1, -1)
np.savetxt("predict/weightsA.csv", wA, delimiter=',')
print(tabulate(wA, headers=range(0, len(clfs)), tablefmt="orgtbl"))
print('')
print('                                    Weights of EN_optB:')
print('|-------------------------------------------------------------------------------------------|')
wB = np.round(w_enB.reshape((-1,n_classes)), decimals=2)
np.savetxt("predict/weightsB.csv", wB, delimiter=',')
wB = np.hstack((np.array(range(0, len(clfs)), dtype=str).reshape(-1, 1), wB))
print(tabulate(wB, headers=['y%s' % i for i in range(n_classes)], tablefmt="orgtbl"))

start = datetime.datetime.now()
X_final = pd.read_csv('data/test_users_norm.csv').fillna(NA_CONST)
Xid = X_final.pop('id')
X_predicted = predictEachTwoOnSingleFeature(clfs, X_final)
print("Test data classified in " + str(datetime.datetime.now() - start) + ", weighting and formatting now")

start = datetime.datetime.now()
saveResult(Xid, cc_optA.predict_proba(X_predicted), le, 'predict/enA.csv')
print("Ensemble A submission is ready in " + str(datetime.datetime.now() - start))
start = datetime.datetime.now()
saveResult(Xid, cc_optB.predict_proba(X_predicted), le, 'predict/enB.csv')
print("Ensemble B submission is ready in " + str(datetime.datetime.now() - start))
