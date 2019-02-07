import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from tabulate import tabulate


class EnsembleRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, x0, scoreFun, clfNames, yNames, n_classes=12):
        """
        :param x0: vector of  (n_classes * n_classifiers) elements with initial weights
        :param scoreFun: takes (y, y_pred) where y is sample labels vector and y_pred is samples * n_classes matrix
        :param clfNames: names of classifier for weights print
        :param clfNames: names of labels (y) for weights print
        :param n_classes:
        """
        self.scoreFun = scoreFun
        self.classesN = n_classes
        self.clfNames = clfNames
        self.yNames = yNames
        self.w = x0
        self.classes_ = range(len(yNames))

    def get_params(self, deep=True):
        return {
            'x0': self.w,
            'scoreFun': self.scoreFun,
            'n_classes': self.classesN,
            'clfNames': self.clfNames,
            'yNames': self.yNames
        }

    def _loss(self, w, X, y):
        # normalization
        # w[w < 0] = 0
        w_range = np.arange(len(w)) % self.classesN
        for i in range(self.classesN):
            wi_sum = np.sum(w[w_range == i])
            if wi_sum != 0:
                w[w_range == i] = w[w_range == i] / wi_sum

        probas = np.clip(self.predict_proba(X, w), 1e-8, 1 - 1e-8)
        y2D = LabelBinarizer().fit(y).transform(y)
        return np.average(-(y2D * np.log(probas)).sum(axis=1))
        #probas = self.predict_proba(X, w)
        #return X.shape[0] - self.scoreFun(y, probas)

    def fit(self, X, y):
        """
        :param X: samples x (n_classes * n_classifiers)
        :param y: sample labels (ints)
        :return:
        """
        result = minimize(self._loss, self.w, args=(X, y),
                          method='L-BFGS-B',
                          bounds=([(0, 1)] * len(self.w)),
                          options={
                              'disp': 10,
                              'eps': 0.02,
                              'maxiter': 500,
                              'ftol': 1e-9,
                              'gtol': 1e-9,
                              'maxls': 20
                          })
        self.w = result.x
        self.printWeights(self.w)

    def predict_proba(self, X, w=None):
        if w is None:
            w = self.w

        probas = np.zeros((X.shape[0], self.classesN))
        for i in range(len(w)):
            probas[:, i % self.classesN] += w[i] * X[:, i]
        return probas

    def printWeights(self, w):
        print('                                    Weights:')
        print('|-------------------------------------------------------------------------------------------------|')
        wB = np.round(w.reshape((-1, self.classesN)), decimals=2)
        wB = np.hstack((np.array(self.clfNames, dtype=str).reshape(-1, 1), wB))
        print(tabulate(wB, headers=self.yNames, tablefmt="orgtbl"))
        print()
