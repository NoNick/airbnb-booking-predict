import numpy as np
import pandas as pd
from datetime import datetime
from mlxtend.classifier import StackingCVClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from ensemble_regression import EnsembleRegression
from classifiers import getClassifiersList, NA_CONST, nDCG5, getDictionaryWithWeights, DCG5_sum
from submission_utils import saveResult

FOLDS = 3
META_FOLDS = 3
THREADS = 4

totalStart = datetime.now()

data = pd.read_csv('data/train_users_2_norm.csv').drop('user_id', axis=1).fillna(NA_CONST)
labels = data.pop('country_destination')
le = LabelEncoder()
labelsEncoded = le.fit_transform(labels)
# xgboost on all features predicts NDF and US just fine
# make other classifier's predictions of NDF and US less valuable (sample_weight)
weightsPerClassifier = getDictionaryWithWeights(labelsEncoded)

clfs = getClassifiersList(data)
pipes = []
for name, clf in clfs.items():
    pipes.append(Pipeline([('clm_selector', ColumnSelector(cols=range(clf.beginColumn, clf.endColumn))), (name, clf)]))

start = datetime.now()
x0 = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1,
               #1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1,
               1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1,
               1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1,
               0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
metaClassifier = CalibratedClassifierCV(
    EnsembleRegression(x0, list(clfs.keys()), le.classes_),
    method='isotonic',
    cv=META_FOLDS)
sclf = StackingCVClassifier(classifiers=pipes, meta_classifier=metaClassifier, use_clones=False,
                            use_probas=True, cv=FOLDS, verbose=1)
sclf.fit(data.values, labelsEncoded, groups=None, **weightsPerClassifier)
print('StackingCV classifier is fitted in ' + str(datetime.now() - start))

start = datetime.now()
test = pd.read_csv('data/test_users_norm.csv').fillna(NA_CONST)
Xid = test.pop('id')
saveResult(Xid, sclf.predict_proba(test.values), le, 'predict/stacking.csv')
print('Submission predict/stacking.csv is predicted in ' + str(datetime.now() - start))

trainPredicted = sclf.predict_proba(data)
print('Test set nDCG5 score: ' + str(nDCG5(labelsEncoded, trainPredicted)))
print('Total time: ' + str(datetime.now() - totalStart))
