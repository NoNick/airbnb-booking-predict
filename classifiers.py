import numpy as np

from datetime import datetime
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import make_scorer

random_state = 1

NA_CONST = -1
THREADS = 4

# 'clfName': classifierObj
_clfs = {
    'BaseFeatures': XGBClassifier(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.3,
        colsample_bytree=0.15,
        missing=NA_CONST,
        objective='multi:softprob',
        num_class=12,
        n_jobs=THREADS,
        n_threads=THREADS),
    'AgeGender': XGBClassifier(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=75,
        # learning_rate=0.1,
        # n_estimators=70,
        subsample=0.3,
        colsample_bytree=0.3,
        missing=NA_CONST,
        objective='multi:softprob',
        num_class=12,
        n_jobs=THREADS,
        n_threads=THREADS),
    'DAC': XGBClassifier(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=50,
        # learning_rate=0.1,
        # n_estimators=45,
        subsample=0.5,
        colsample_bytree=0.3,
        missing=NA_CONST,
        objective='multi:softprob',
        num_class=12,
        n_jobs=THREADS,
        n_threads=THREADS),
    'TFA': XGBClassifier(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=50,
        # learning_rate=0.1,
        # n_estimators=45,
        subsample=0.5,
        colsample_bytree=0.3,
        missing=NA_CONST,
        objective='multi:softprob',
        num_class=12,
        n_jobs=THREADS,
        n_threads=THREADS),
    'Actions': XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=50,
        # max_depth=4,
        # learning_rate=0.1,
        # n_estimators=50,
        subsample=0.5,
        colsample_bytree=0.2,
        missing=NA_CONST,
        objective='multi:softprob',
        num_class=12,
        n_jobs=THREADS,
        n_threads=THREADS),
    'AllFeatures': XGBClassifier(
        max_depth=6,
        learning_rate=0.01,
        n_estimators=50,
        subsample=0.7,
        colsample_bytree=0.3,
        missing=NA_CONST,
        objective='multi:softprob',
        num_class=12,
        n_jobs=THREADS,
        n_threads=THREADS
    )
}
# 'clfName': (startColumnName, endColumnName
clfPos = {'BaseFeatures': ('gender', 'device_type'),
          'AgeGender': ('age_copy', 'US_oppositeGender_population'),
          'DAC': ('DAC_year', 'DAC_season'),
          'TFA': ('TFA_year', 'TFA_hour_in_day'),
          'Actions': ('personalize$wishlist_content_update', 'print_confirmation$-unknown-'),
          'AllFeatures': ('gender', 'print_confirmation$-unknown-')}


def getClassifiersList(X):
    for name, clf in _clfs.items():
        clf.beginColumn = X.columns.get_loc(clfPos[name][0])
        clf.endColumn = X.columns.get_loc(clfPos[name][1])
    return _clfs


def getTrainTestValidPredictions(clfs, X_train, y_train, X_valid):
    p_valid = []

    clfN = len(clfs)
    i = 0
    for name, clf in clfs.items():
        start = datetime.now()
        startPos = clf.beginColumn
        endPos = clf.endColumn
        clf.fit(X_train.iloc[:, startPos:endPos], y_train)

        X_valid_featured = X_valid.iloc[:, startPos:endPos]
        yv = clf.predict_proba(X_valid_featured)
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


dcg5_at_k = [1.0, 0.63092975, 0.5, 0.43067656, 0.38685281]


def nDCG5(y, y_pred):
    top5 = np.argsort(-y_pred, axis=1)[:, 0:5]
    correctDestination = np.equal(top5, np.tile(y, (5, 1)).transpose())
    return np.apply_along_axis(lambda x: (dcg5_at_k * x).sum(), 1, correctDestination).mean()


nDCG5_score = make_scorer(nDCG5, greater_is_better=True, needs_proba=True)


def accuracy(classNumber, y, y_pred):
    classIndices = (y == classNumber)
    return (y_pred[classIndices] == classNumber).mean()


