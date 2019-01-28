import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.base import BaseEstimator
from scipy.optimize import minimize

def objf_ens_optA(w, Xs, y, n_class=12):
    """
    Function to be minimized in the EN_optA ensembler.

    Parameters:
    ----------
    w: array-like, shape=(n_preds)
       Candidate solution to the optimization problem (vector of weights).
    Xs: list of predictions to combine
       Each prediction is the solution of an individual classifier and has a
       shape=(n_samples, n_classes).
    y: array-like sahpe=(n_samples,)
       Class labels
    n_class: int
       Number of classes in the problem (12 in Airbnb competition)

    Return:
    ------
    score: Score of the candidate solution.
    """
    w = np.abs(w)
    sol = np.zeros(Xs[0].shape)
    for i in range(len(w)):
        sol += Xs[i] * w[i]
    # Using log-loss as objective function (different objective functions can be used here).
    score = log_loss(y, sol)
    return score


class EN_optA(BaseEstimator):
    """
    Given a set of predictions $X_1, X_2, ..., X_n$,  it computes the optimal set of weights
    $w_1, w_2, ..., w_n$; such that minimizes $log\_loss(y_T, y_E)$,
    where $y_E = X_1*w_1 + X_2*w_2 +...+ X_n*w_n$ and $y_T$ is the true solution.
    """
    classes_ = ['US', 'FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL', 'DE', 'AU', 'NDF', 'other']

    def __init__(self, n_class=12):
        super(EN_optA, self).__init__()
        self.n_class = n_class

    def fit(self, X, y):
        """
        Learn the optimal weights by solving an optimization problem.

        Parameters:
        ----------
        Xs: list of predictions to be ensembled
           Each prediction is the solution of an individual classifier and has
           shape=(n_samples, n_classes).
        y: array-like
           Class labels
        """
        Xs = np.hsplit(X, X.shape[1] / self.n_class)
        # Initial solution has equal weight for all individual predictions.
        x0 = np.ones(len(Xs)) / float(len(Xs))
        # Weights must be bounded in [0, 1]
        bounds = [(0,1)]*len(x0)
        #All weights must sum to 1
        cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
        #Calling the solver
        res = minimize(objf_ens_optA, x0, args=(Xs, y, self.n_class),
                       method='SLSQP',
                       bounds=bounds,
                       constraints=cons
                       )
        self.w = res.x
        return self

    def predict_proba(self, X):
        """
        Use the weights learned in training to predict class probabilities.

        Parameters:
        ----------
        Xs: list of predictions to be blended.
            Each prediction is the solution of an individual classifier and has
            shape=(n_samples, n_classes).

        Return:
        ------
        y_pred: array_like, shape=(n_samples, n_class)
                The blended prediction.
        """
        Xs = np.hsplit(X, X.shape[1] / self.n_class)
        y_pred = np.zeros(Xs[0].shape)
        for i in range(len(self.w)):
            y_pred += Xs[i] * self.w[i]
        return y_pred


def objf_ens_optB(w, Xs, y, n_class=12):
    """
    Function to be minimized in the EN_optB ensembler.

    Parameters:
    ----------
    w: array-like, shape=(n_preds)
       Candidate solution to the optimization problem (vector of weights).
    Xs: list of predictions to combine
       Each prediction is the solution of an individual classifier and has a
       shape=(n_samples, n_classes).
    y: array-like sahpe=(n_samples,)
       Class labels
    n_class: int
       Number of classes in the problem, i.e. = 12

    Return:
    ------
    score: Score of the candidate solution.
    """
    # Constraining the weights for each class to sum up to 1.
    # This constraint can be defined in the scipy.minimize function, but doing
    # it here gives more flexibility to the scipy.minimize function
    # (e.g. more solvers are allowed).
    w_range = np.arange(len(w)) % n_class
    for i in range(n_class):
        w[w_range == i] = w[w_range == i] / np.sum(w[w_range == i])

    sol = np.zeros(Xs[0].shape)
    for i in range(len(w)):
        sol[:, i % n_class] += Xs[int(i / n_class)][:, i % n_class] * w[i]

        # Using log-loss as objective function (different objective functions can be used here).
    score = log_loss(y, sol)
    return score


class EN_optB(BaseEstimator):
    """
    Given a set of predictions $X_1, X_2, ..., X_n$, where each $X_i$ has
    $m=12$ clases, i.e. $X_i = X_{i1}, X_{i2},...,X_{im}$. The algorithm finds the optimal
    set of weights $w_{11}, w_{12}, ..., w_{nm}$; such that minimizes
    $log\_loss(y_T, y_E)$, where $y_E = X_{11}*w_{11} +... + X_{21}*w_{21} + ...
    + X_{nm}*w_{nm}$ and and $y_T$ is the true solution.
    """
    classes_ = ['US', 'FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL', 'DE', 'AU', 'NDF', 'other']

    def __init__(self, n_class=12):
        super(EN_optB, self).__init__()
        self.n_class = n_class

    def fit(self, X, y):
        """
        Learn the optimal weights by solving an optimization problem.

        Parameters:
        ----------
        Xs: list of predictions to be ensembled
           Each prediction is the solution of an individual classifier and has
           shape=(n_samples, n_classes).
        y: array-like
           Class labels
        """
        Xs = np.hsplit(X, X.shape[1]/self.n_class)
        #Initial solution has equal weight for all individual predictions.
        x0 = np.ones(self.n_class * len(Xs)) / float(len(Xs))
        #Weights must be bounded in [0, 1]
        bounds = [(0,1)]*len(x0)
        #Calling the solver (constraints are directly defined in the objective
        #function)
        res = minimize(objf_ens_optB, x0, args=(Xs, y, self.n_class),
                       method='L-BFGS-B',
                       bounds=bounds,
                       )
        self.w = res.x
        return self

    def predict_proba(self, X):
        """
        Use the weights learned in training to predict class probabilities.

        Parameters:
        ----------
        Xs: list of predictions to be ensembled
            Each prediction is the solution of an individual classifier and has
            shape=(n_samples, n_classes).

        Return:
        ------
        y_pred: array_like, shape=(n_samples, n_class)
                The ensembled prediction.
        """
        Xs = np.hsplit(X, X.shape[1] / self.n_class)
        y_pred = np.zeros(Xs[0].shape)
        for i in range(len(self.w)):
            y_pred[:, i % self.n_class] += \
                Xs[int(i / self.n_class)][:, i % self.n_class] * self.w[i]
        return y_pred


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
import datetime

random_state=1
n_classes = 12  # Same number of classes as in Airbnb competition.
data = pd.read_csv('data/train_users_2_norm.csv').drop('user_id', axis=1)[0:100000].fillna(0)
labels = data['country_destination']
data.drop('country_destination', axis=1, inplace=True)

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
