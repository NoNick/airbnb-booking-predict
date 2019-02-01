# Copy-paste from https://www.kaggle.com/svpons/three-level-classification-architecture
import numpy as np
from sklearn.metrics import log_loss
from sklearn.base import BaseEstimator
from scipy.optimize import minimize


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
        wi_sum = np.sum(w[w_range == i])
        if wi_sum != 0:
            w[w_range == i] = w[w_range == i] / wi_sum

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
        #Initial solution has equal weight for all individual predictions.
        x0 = np.ones(self.n_class * len(Xs)) / float(len(Xs))
        #Weights must be bounded in [0, 1]
        bounds = [(0, 1)] * len(x0)
        #Calling the solver (constraints are directly defined in the objective
        #function)
        res = minimize(objf_ens_optB, x0, args=(Xs, y, self.n_class),
                       # method='L-BFGS-B',
                       method='SLSQP',
                       bounds=bounds)
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
