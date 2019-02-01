import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import log_loss

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ensemble import EN_optB
from classifiers import getClassifiersList, getTrainTestValidPredictions, random_state

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
clfs = getClassifiersList(X)

print('Performance of individual classifiers (1st layer) on X_test')
print('------------------------------------------------------------')
p_valid, p_test = getTrainTestValidPredictions(clfs, X, y, X_train, y_train, X_valid, X_test, y_test)
print('')

print('Performance of optimization based ensemblers (2nd layer) on X_test')
print('------------------------------------------------------------')

# Creating the data for the 2nd layer.
XV = np.hstack(p_valid)
XT = np.hstack(p_test)

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


from tabulate import tabulate
print('                                    Weights of EN_optB:')
print('|-------------------------------------------------------------------------------------------|')
wB = np.round(w_enB.reshape((-1,n_classes)), decimals=2)
np.savetxt("predict/weightsB.csv", wB, delimiter=',')
wB = np.hstack((np.array(list(clfs.keys()), dtype=str).reshape(-1, 1), wB))
print(tabulate(wB, headers=['y%s' % i for i in range(n_classes)], tablefmt="orgtbl"))

np.save('predict/ensemble_weights', enB.w)

