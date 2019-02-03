# Copy-paste from https://www.kaggle.com/svpons/three-level-classification-architecture
import numpy as np
from sklearn.base import BaseEstimator
from scipy.optimize import minimize, Bounds
from sklearn.preprocessing import LabelBinarizer
from tabulate import tabulate

from classifiers import _clfs

eps = 1e-15


def print_weights(w, n_classes=12, clfs_keys=_clfs.keys()):
    print('                                    Weights of ensemble:')
    print('|-------------------------------------------------------------------------------------------------|')
    wB = np.round(w.reshape((-1, n_classes)), decimals=2)
    wB = np.hstack((np.array(list(clfs_keys), dtype=str).reshape(-1, 1), wB))
    print(tabulate(wB, headers=['y%s' % i for i in range(n_classes)], tablefmt="orgtbl"))
    print()


def objf_ens_optB_with_Gradient(w, Xs, y, n_class=12):
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
    wLen = len(w)

    # Constraint of class weight sum
    w[w < 0] = 0
    w_range = np.arange(len(w)) % n_class
    for i in range(n_class):
        wi_sum = np.sum(w[w_range == i])
        if wi_sum != 0:
            w[w_range == i] = w[w_range == i] / wi_sum

    sol = np.zeros(Xs[0].shape)
    for i in range(wLen):
        sol[:, i % n_class] += Xs[int(i / n_class)][:, i % n_class] * w[i]

    lb = LabelBinarizer().fit(y)
    y_t = lb.transform(y)
    sol = np.clip(sol, eps, 1 - eps)
    loss = np.average(-(y_t * np.log(sol)).sum(axis=1))

    # grad = np.zeros(wLen)
    # for i in range(wLen):
    #     grad[i] = -np.average((y_t[:, i % n_class] * Xs[int(i / n_class)][:, i % n_class]) / sol[:, i % n_class])

    return loss#, grad


def sumOfClassWeights(w, c, n_class=12):
    w_range = np.arange(len(w)) % n_class
    return np.sum(w[w_range == c])


constraints = [
    {'type': 'eq', 'fun': lambda x: 1 - sumOfClassWeights(x, 0)},
    {'type': 'eq', 'fun': lambda x: 1 - sumOfClassWeights(x, 1)},
    {'type': 'eq', 'fun': lambda x: 1 - sumOfClassWeights(x, 2)},
    {'type': 'eq', 'fun': lambda x: 1 - sumOfClassWeights(x, 3)},
    {'type': 'eq', 'fun': lambda x: 1 - sumOfClassWeights(x, 4)},
    {'type': 'eq', 'fun': lambda x: 1 - sumOfClassWeights(x, 5)},
    {'type': 'eq', 'fun': lambda x: 1 - sumOfClassWeights(x, 6)},
    {'type': 'eq', 'fun': lambda x: 1 - sumOfClassWeights(x, 7)},
    {'type': 'eq', 'fun': lambda x: 1 - sumOfClassWeights(x, 8)},
    {'type': 'eq', 'fun': lambda x: 1 - sumOfClassWeights(x, 9)},
    {'type': 'eq', 'fun': lambda x: 1 - sumOfClassWeights(x, 10)},
    {'type': 'eq', 'fun': lambda x: 1 - sumOfClassWeights(x, 11)},
]


class EN_optB(BaseEstimator):
    """
    Given a set of predictions $X_1, X_2, ..., X_n$, where each $X_i$ has
    $m=12$ clases, i.e. $X_i = X_{i1}, X_{i2},...,X_{im}$. The algorithm finds the optimal
    set of weights $w_{11}, w_{12}, ..., w_{nm}$; such that minimizes
    $log\_loss(y_T, y_E)$, where $y_E = X_{11}*w_{11} +... + X_{21}*w_{21} + ...
    + X_{nm}*w_{nm}$ and and $y_T$ is the true solution.
    """
    classes_ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

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
        x0 = np.ones(self.n_class * len(Xs)) / float(len(Xs))
        # x0 = normalizeWeights(np.random.rand(self.n_class * len(Xs)))
        res = minimize(objf_ens_optB_with_Gradient, x0, args=(Xs, y, self.n_class),
                       # jac=True,
                       method='Nelder-Mead',
                       # method='L-BFGS-B',
                       # method='SLSQP',
                       # constraints=constraints,
                       # bounds=Bounds(0, 1),
                       options={
                           'disp': True,
                           'adaptive': True,
                           # 'eps': eps,
                           # 'ftol': 1e-8,
                           'maxiter': 60 * 400
                       })
        self.w = res.x
        print_weights(self.w)
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

