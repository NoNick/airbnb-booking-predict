import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from xgboost.sklearn import XGBClassifier
import xgboost as xgb

random_state = 1


def getPos(i):
    if i < 3:
        return i, i + 1
    return 4, 10


def genClassifiersList(N):
    result = []
    for _ in range(0, N):
        # result.append(LogisticRegression(
        #     random_state=random_state,
        #     solver='lbfgs',
        #     max_iter=1000,
        #     n_jobs=4,
        #     multi_class='multinomial'))
        result.append(XGBClassifier(
            max_depth=6,
            silent=True,
            n_jobs=4,
            nthread=4,
            subsample=.5,
            colsample_bytree=.3))
    return result


def learnEachTwoOnSingleFeature(clfs, X, y, X_train, y_train, X_valid, X_test, y_test):
    # predictions on the validation and test sets
    p_valid = []
    p_test = []

    clfN = len(clfs)
    i = 0
    start = datetime.now()
    timeDiffs = (start - start)
    for clf in clfs:
        # pos = int(i / 2)  # feature position
        startP, end = getPos(i)
        #     First run. Training on (X_train, y_train) and predicting on X_valid.
        clf.fit(X_train.iloc[:, startP:end], y_train)
        yv = clf.predict_proba(X_valid.iloc[:, startP:end])
        if i == 13:
            x = 1
        p_valid.append(yv)

        # Second run. Training on (X, y) and predicting on X_test.
        clf.fit(X.iloc[:, startP:end], y)
        yt = clf.predict_proba(X_test.iloc[:, startP:end])
        p_test.append(yt)

        print('{:10s} {:2s} {:1.7f}'.format('%dth: ' % i, 'logloss  =>', log_loss(y_test, yt)))
        i += 1
        timeDiffs += datetime.now() - start
        print(("Trained %3d/%d classifiers, " + str(datetime.now() - start) +
               " last one, " + str((clfN - i) * timeDiffs / i) + " ETA") % (i, clfN))

    return p_valid, p_test

def predictEachTwoOnSingleFeature(clfs, X):
    clfN = len(clfs)
    p = []
    i = 0
    for clf in clfs:
        # pos = int(i / 2)  # feature position
        start, end = getPos(i)
        yf = clf.predict_proba(X.iloc[:, start:end])
        p.append(yf)
        i += 1
        print("Got predictions from %3d/%d classifier" % (i, clfN))
    return np.hstack(p)
