import numpy as np
from scipy.optimize import check_grad
from sklearn.preprocessing import LabelBinarizer

Xs = np.load('Xs.npy')
y = np.load('y.npy')
lb = LabelBinarizer().fit(y)
y_t = lb.transform(y)
n_class = 12
n_classifiers = 5
# x0 = np.ones(n_class * len(Xs)) / float(len(Xs))
x0 = np.random.rand(n_class * len(Xs))
w_range = np.arange(len(x0)) % n_class
for i in range(n_class):
    wi_sum = np.sum(x0[w_range == i])
    if wi_sum != 0:
        x0[w_range == i] = x0[w_range == i] / wi_sum
wLen = len(x0)

def fun(w):
    sol = np.zeros(Xs[0].shape)
    for i in range(wLen):
        sol[:, i % n_class] += Xs[int(i / n_class)][:, i % n_class] * w[i]

    loss_matrix = -(y_t * np.log(sol))

    return np.sum(loss_matrix.sum(axis=1))


def fun2(w):
    loss = 0.0
    for k in range(len(y)):
        for l in range(n_class):
            sum1 = 0
            for j in range(n_classifiers):
                sum1 += Xs[j][k][l] * w[j * n_class + l]
            loss += -y_t[k][l] * np.log(sum1)
    return loss #/ len(y)


def grad(w):
    sol = np.zeros(Xs[0].shape)
    for i in range(wLen):
        sol[:, i % n_class] += Xs[int(i / n_class)][:, i % n_class] * w[i]

    grad = np.zeros(wLen)
    for i in range(wLen):
        for k in range(len(y)):
            sum1 = 0.0
            for j in range(n_classifiers):
                sum1 += np.sum(Xs[j][k, i % n_class] * w[(j * n_class) + (i % n_class)])
        grad[i] += -np.sum((y_t[:, i % n_class] * Xs[int(i / n_class)][:, i % n_class]) /
                            Xs[int(i / n_class)][:, i % n_class] * w[i])

    return grad


print(check_grad(fun, grad, x0))
