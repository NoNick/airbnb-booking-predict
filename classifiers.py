import numpy as np

from datetime import datetime
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import make_scorer

random_state = 1

NA_CONST = -1
THREADS = 4

# 'clfName': classifierObj
_clfs = {
    # 'BaseFeatures': XGBClassifier(
    #     max_depth=4,
    #     learning_rate=0.05,
    #     n_estimators=250,
    #     subsample=0.5,
    #     colsample_bytree=0.5,
    #     missing=NA_CONST,
    #     objective='rank:ndcg',
    #     eval_metric='ndcg@5',
    #     num_class=12,
    #     n_jobs=THREADS,
    #     n_threads=THREADS),
    # 'AgeGender': XGBClassifier(
    #     max_depth=1,
    #     learning_rate=0.05,
    #     n_estimators=50,
    #     subsample=0.5,
    #     colsample_bytree=0.5,
    #     missing=NA_CONST,
    #     objective='rank:ndcg',
    #     eval_metric='ndcg@5',
    #     num_class=12,
    #     n_jobs=THREADS,
    #     n_threads=THREADS),
    # 'DAC_TFA': XGBClassifier(
    #     max_depth=1,
    #     learning_rate=0.05,
    #     n_estimators=50,
    #     subsample=0.3,
    #     colsample_bytree=0.5,
    #     missing=NA_CONST,
    #     objective='rank:ndcg',
    #     eval_metric='ndcg@5',
    #     num_class=12,
    #     n_jobs=THREADS,
    #     n_threads=THREADS),
    # 'Actions': XGBClassifier(
    #     max_depth=1,
    #     learning_rate=0.025,
    #     n_estimators=75,
    #     subsample=0.3,
    #     colsample_bytree=0.5,
    #     missing=NA_CONST,
    #     objective='rank:ndcg',
    #     eval_metric='ndcg@5',
    #     num_class=12,
    #     n_jobs=THREADS,
    #     n_threads=THREADS),
    'AllFeatures': XGBClassifier(
        max_depth=8,
        missing=NA_CONST,
        n_estimators=250,
        subsample=0.5,
        colsample_bytree=0.5,
        objective='rank:ndcg',
        eval_metric='ndcg@5',
        num_class=12,
        silent=True,
        n_jobs=THREADS,
        nthread=THREADS
    ),
    'AllLowerNDFLowerUS': XGBClassifier(
        max_depth=8,
        missing=NA_CONST,
        n_estimators=250,
        subsample=0.5,
        colsample_bytree=0.5,
        objective='rank:ndcg',
        eval_metric='ndcg@5',
        num_class=12,
        silent=True,
        n_jobs=THREADS,
        nthread=THREADS
    )
}
# 'clfName': (startColumnName, endColumnName
clfPos = {'BaseFeatures': ('gender', 'device_type'),
          'AgeGender': ('age_copy', 'US_oppositeGender_population'),
          'DAC_TFA': ('DAC_year', 'TFA_hour_in_day'),
          'DAC': ('DAC_year', 'DAC_season'),
          'TFA': ('TFA_year', 'TFA_hour_in_day'),
          'Actions': ('personalize$wishlist_content_update', 'print_confirmation$-unknown-'),
          'AllFeatures': ('gender', 'print_confirmation$-unknown-'),
          'AllLowerNDFLowerUS': ('gender', 'print_confirmation$-unknown-')}
# ['AU' 'CA' 'DE' 'ES' 'FR' 'GB' 'IT' 'NDF' 'NL' 'PT' 'US' 'other']
underweightClasses = {'BaseFeatures': [7, 10],
                      'AgeGender': [7, 10],
                      'DAC_TFA': [7, 10],
                      'Actions': [7, 10],
                      'AllLowerNDFLowerUS': [7, 10]}

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


dcg5_at_k = np.array([1.0, 0.63092975, 0.5, 0.43067656, 0.38685281])


def nDCG5(y, y_pred):
    top5 = np.argsort(-y_pred, axis=1)[:, 0:5]
    sum = 0
    for i in range(len(top5)):
        sum += dcg5_at_k[top5[i] == y[i]].sum()
    return sum / len(y)


def nDCG5_NoNDF_NoUS(y, y_pred):
    indices = np.isin(y, [7, 10], invert=True)
    return nDCG5(y[indices], y_pred[indices])


def DCG5_sum(y, y_pred):
    top5 = np.argsort(-y_pred, axis=1)[:, 0:5]
    sum = 0
    for i in range(len(top5)):
        sum += dcg5_at_k[top5[i] == y[i]].sum()
    return sum


nDCG5_score = make_scorer(nDCG5, greater_is_better=True, needs_proba=True)
nDCG5_NoNDF_NoUS_score = make_scorer(nDCG5_NoNDF_NoUS, greater_is_better=True, needs_proba=True)


def accuracy(classNumber, y, y_pred):
    classIndices = (y == classNumber)
    return (y_pred[classIndices] == classNumber).mean()


# makes prediction of classesToExclude as valuable as prediction of least numbered class in y
def getWeightsLowerClasses(classesToUnderweight, y):
    unique, counts = np.unique(y, return_counts=True)
    countByUnique = dict(zip(unique, counts))

    result = np.ones(len(y))
    for classToUnderWeight in classesToUnderweight:
        newWeight = min(counts) / countByUnique[classToUnderWeight]
        result[y == classToUnderWeight] = newWeight
    return result


# formatted for StackingCVClassifier::fit
def getDictionaryWithWeights(y):
    result = {}
    for name, classes in underweightClasses.items():
        result[name + '__sample_weight'] = getWeightsLowerClasses(classes, y)
    return result
