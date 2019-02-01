from datetime import datetime
from xgboost.sklearn import XGBClassifier

random_state = 1

THREADS = 4

# 'clfName': classifierObj
_clfs = {
    'BaseFeatures': XGBClassifier(
        max_depth=5,
        learning_rate=0.01,
        n_estimators=50,
        subsample=0.5,
        colsample_bytree=0.3,
        missing=-1,
        objective='multi:softprob',
        num_class=12,
        n_jobs=THREADS,
        n_threads=THREADS),
    'AgeGender': XGBClassifier(
        max_depth=4,
        learning_rate=0.01,
        n_estimators=35,
        subsample=0.5,
        colsample_bytree=0.3,
        missing=-1,
        objective='multi:softprob',
        num_class=12,
        n_jobs=THREADS,
        n_threads=THREADS),
    'DAC': XGBClassifier(
        max_depth=2,
        learning_rate=0.01,
        n_estimators=25,
        subsample=0.5,
        colsample_bytree=0.3,
        missing=-1,
        objective='multi:softprob',
        num_class=12,
        n_jobs=THREADS,
        n_threads=THREADS),
    'TFA': XGBClassifier(
        max_depth=2,
        learning_rate=0.01,
        n_estimators=25,
        subsample=0.5,
        colsample_bytree=0.3,
        missing=-1,
        objective='multi:softprob',
        num_class=12,
        n_jobs=THREADS,
        n_threads=THREADS),
    'Actions': XGBClassifier(
        max_depth=6,
        learning_rate=0.01,
        n_estimators=35,
        subsample=0.5,
        colsample_bytree=0.3,
        missing=0,
        objective='multi:softprob',
        num_class=12,
        n_jobs=THREADS,
        n_threads=THREADS)
}
# 'clfName': (startColumnName, endColumnName
clfPos = {'BaseFeatures': ('signup_method', 'secs_elapsed'),
          'AgeGender': ('age', 'US_oppositeGender_population'),
          'DAC': ('DAC_year', 'DAC_season'),
          'TFA': ('TFA_year', 'TFA_hour_in_day'),
          'Actions': ('personalize$wishlist_content_update', 'phone_verification_error$-unknown-')}


def getClassifiersList(X):
    for name, clf in _clfs.items():
        clf.beginColumn = X.columns.get_loc(clfPos[name][0])
        clf.endColumn = X.columns.get_loc(clfPos[name][1])
    return _clfs


def getTrainTestValidPredictions(clfs, X_train, y_train, X_valid):
    # predictions on the validation and test sets
    p_valid = []

    clfN = len(clfs)
    i = 0
    for name, clf in clfs.items():
        start = datetime.now()
        startPos = clf.beginColumn
        endPos = clf.endColumn
        #     First run. Training on (X_train, y_train) and predicting on X_valid.
        clf.fit(X_train.iloc[:, startPos:endPos], y_train)
        yv = clf.predict_proba(X_valid.iloc[:, startPos:endPos])
        p_valid.append(yv)
        i += 1
        print(("Trained %3d/%d classifiers, " + str(datetime.now() - start) + " last one") % (i, clfN))

    return p_valid


def learn(clfs, X, y):
    clfN = len(clfs)
    i = 0
    for name, clf in clfs.items():
        start = datetime.now()
        startPos = clf.beginColumn
        endPos = clf.endColumn
        #     First run. Training on (X_train, y_train) and predicting on X_valid.
        clf.fit(X.iloc[:, startPos:endPos], y)
        i += 1
        print(("Trained %3d/%d classifiers, " + str(datetime.now() - start) + " last one") % (i, clfN))
    return clfs


def predict(clfs, X):
    clfN = len(clfs)
    p = []
    i = 0
    for name, clf in clfs.items():
        startPos = clf.beginColumn
        endPos = clf.endColumn
        yf = clf.predict_proba(X.iloc[:, startPos:endPos])
        p.append(yf)
        i += 1
        print("Got predictions from %3d/%d classifier" % (i, clfN))
    return p
