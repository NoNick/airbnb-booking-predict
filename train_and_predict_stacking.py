import pandas as pd
from datetime import datetime
from mlxtend.classifier import StackingCVClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder

from classifiers import getClassifiersList, NA_CONST
from submission_utils import saveResult

FOLDS = 5
META_FOLDS = 4
THREADS = 4

data = pd.read_csv('data/train_users_2_norm.csv').drop('user_id', axis=1).fillna(NA_CONST)
labels = data.pop('country_destination')
le = LabelEncoder()
labelsEncoded = le.fit_transform(labels)

clfs = getClassifiersList(data)
pipes = []
for _, clf in clfs.items():
    pipes.append(make_pipeline(ColumnSelector(cols=range(clf.beginColumn, clf.endColumn)), clf))

start = datetime.now()
metaClassifier = LogisticRegressionCV(cv=META_FOLDS, multi_class='multinomial', max_iter=1000,
                                      n_jobs=THREADS, solver='lbfgs', verbose=1)
sclf = StackingCVClassifier(classifiers=pipes, meta_classifier=metaClassifier,
                            use_probas=True, cv=FOLDS, verbose=1)
sclf.fit(data.values, labelsEncoded)
print('StackingCV classifier is fitted in ' + str(datetime.now() - start))

start = datetime.now()
test = pd.read_csv('data/test_users_norm.csv').fillna(NA_CONST)
Xid = test.pop('id')
saveResult(Xid, sclf.predict_proba(test.values), le, 'predict/stacking.csv')
print('Submission predict/stacking.csv is predicted in ' + str(datetime.now() - start))
