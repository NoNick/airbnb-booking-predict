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

FOLDS = 5
META_FOLDS = 3
THREADS = 4

totalStart = datetime.now()

data = pd.read_csv('data/train_users_2_norm.csv').drop('user_id', axis=1).fillna(NA_CONST)
labels = data.pop('country_destination')
le = LabelEncoder()
labelsEncoded = le.fit_transform(labels)
# weightsPerClassifier = getDictionaryWithWeights(labelsEncoded)

clfs = getClassifiersList(data)
pipes = []
for name, clf in clfs.items():
    pipes.append(Pipeline([('clm_selector', ColumnSelector(cols=range(clf.beginColumn, clf.endColumn))), (name, clf)]))

start = datetime.now()
x0 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
      0, 0, 1, 0, 0, 0, 0, 0.5, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
metaClassifier = CalibratedClassifierCV(
    EnsembleRegression(x0, DCG5_sum, list(clfs.keys()), le.classes_),
    method='isotonic',
    cv=META_FOLDS)
# metaClassifier = EnsembleRegression(x0, nDCG5, list(clfs.keys()))
sclf = StackingCVClassifier(classifiers=pipes, meta_classifier=metaClassifier, use_clones=False,
                            use_probas=True, cv=FOLDS, verbose=1)
sclf.fit(data.values, labelsEncoded)#, fit_params=weightsPerClassifier)
print('StackingCV classifier is fitted in ' + str(datetime.now() - start))

start = datetime.now()
test = pd.read_csv('data/test_users_norm.csv').fillna(NA_CONST)
Xid = test.pop('id')
saveResult(Xid, sclf.predict_proba(test.values), le, 'predict/stacking.csv')
print('Submission predict/stacking.csv is predicted in ' + str(datetime.now() - start))
print('Total time: ' + str(datetime.now() - totalStart))
