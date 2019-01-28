import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import log_loss

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier

from ensemble import objf_ens_optA, EN_optA, objf_ens_optB, EN_optB


random_state = 1
n_classes = 12  # Same number of classes as in Airbnb competition.
data = pd.read_csv('data/train_users_2_norm.csv').drop('user_id', axis=1).fillna(0)
labels = data.pop('country_destination')

# Spliting data into train and test sets.
X, X_test, y, y_test = train_test_split(data, labels, test_size=0.2,
                                        random_state=random_state)

# Spliting train data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25,
                                                      random_state=random_state)

print('Data shape:')
print('X_train: %s, X_valid: %s, X_test: %s \n' % (X_train.shape, X_valid.shape,
                                                   X_test.shape))

# Defining the classifiers
clfs = {'LR': LogisticRegression(random_state=random_state),
        # 'SVM': SVC(probability=True, random_state=random_state),
#        'RF': RandomForestClassifier(n_estimators=100, n_jobs=-1,
#                                     random_state=random_state),
        'GBM': GradientBoostingClassifier(n_estimators=50,
                                          random_state=random_state)}
#        'ETC': ExtraTreesClassifier(n_estimators=100, n_jobs=-1,
#                                    random_state=random_state),
#        'KNN': KNeighborsClassifier(n_neighbors=30)}

# predictions on the validation and test sets
p_valid = []
p_test = []

print('Performance of individual classifiers (1st layer) on X_test')
print('------------------------------------------------------------')
start = datetime.datetime.now()
for nm, clf in clfs.items():
#     First run. Training on (X_train, y_train) and predicting on X_valid.
    clf.fit(X_train, y_train)
    yv = clf.predict_proba(X_valid)
    p_valid.append(yv)

    # Second run. Training on (X, y) and predicting on X_test.
    clf.fit(X, y)
    yt = clf.predict_proba(X_test)
    p_test.append(yt)

    # Printing out the performance of the classifier
    print('{:10s} {:2s} {:1.7f}'.format('%s: ' % (nm), 'logloss  =>', log_loss(y_test, yt)))
    print(datetime.datetime.now() - start)
    start = datetime.datetime.now()
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
cc_optA = CalibratedClassifierCV(enA, method='isotonic')
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
cc_optB = CalibratedClassifierCV(enB, method='isotonic')
cc_optB.fit(XV, y_valid)
y_ccB = cc_optB.predict_proba(XT)
print('{:20s} {:2s} {:1.7f}'.format('Calibrated_EN_optB:', 'logloss  =>', log_loss(y_test, y_ccB)))
print(datetime.datetime.now() - start)
start = datetime.datetime.now()
print('')

y_3l = (y_enA * 4./9.) + (y_ccA * 2./9.) + (y_enB * 2./9.) + (y_ccB * 1./9.)
print('{:20s} {:2s} {:1.7f}'.format('3rd_layer:', 'logloss  =>', log_loss(y_test, y_3l)))
print(datetime.datetime.now() - start)
start = datetime.datetime.now()

from tabulate import tabulate
print('               Weights of EN_optA:')
print('|---------------------------------------------|')
wA = np.round(w_enA, decimals=2).reshape(1,-1)
print(tabulate(wA, headers=clfs.keys(), tablefmt="orgtbl"))
print('')
print('                                    Weights of EN_optB:')
print('|-------------------------------------------------------------------------------------------|')
wB = np.round(w_enB.reshape((-1,n_classes)), decimals=2)
wB = np.hstack((np.array(list(clfs.keys()), dtype=str).reshape(-1,1), wB))
print(tabulate(wB, headers=['y%s'%(i) for i in range(n_classes)], tablefmt="orgtbl"))

# Xfinal = pd.read_csv('data/test_users_norm.csv')[0:10]
# Xid = Xfinal['id']
# Xfinal.drop('id', axis=1, inplace=True)
# p_final = []
# for nm, clf in clfs.items():
#     yv = clf.predict_proba(Xfinal)
#     p_final.append(yv)
# Xpredicted = np.hstack(p_final)
# (pd.DataFrame(cc_optA.predict_proba(Xpredicted))).to_csv('predict/enA_LR_GBM.csv', index=False)
# (pd.DataFrame(cc_optB.predict_proba(Xpredicted))).to_csv('predict/enB_LR_GBM.csv', index=False)
# (pd.DataFrame(lr.predict_proba(Xtest))).to_csv('predict/LR.csv', index=False)
