import numpy as np
import pandas as pd
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso, Ridge

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import \
    Axes3D  # <--- This is important for 3d plotting
import pandas as pd
from pandas import DataFrame
# from plotnine import *

import matplotlib as mpl
import matplotlib.pyplot as plt

from random import random

from ex4_tools import DecisionStump


class abcModel:

    def __init__(self):
        self.mod = None

    def fit(self, X, y):
        self.mod.fit(self.bias(X), y.ravel())

    def predict(self, X):
        return self.mod.predict(self.bias(X))

    def get_hyperplan(self):
        pass

    def bias(self, X):
        return np.insert(X, 0, 1, 1)

    def draw(self):
        pass

    def score(self, X, y):
        return {
            "num_samples": 0,
            "error" : 0,
            "accuracy" : 0,
            "FPR": 0,
            "TPR": 0,
            "precision": 0,
            "recall": 0
        }


# [v]
class Perceptron(abcModel):
    def __init__(self):
        self.W = np.array([])
        super().__init__()

    def fit(self, _X, y):
        X = self.bias(_X)

        self.W = np.zeros(shape=X.shape[1])

        def not_classifiy():
            return [(x, y) for x in X if self.predict(x) != y]

        _updated = True
        while _updated:
            _updated = False
            for i in range(len(y)):
                if np.dot(self.W, X[i]) * y[i] <= 0:
                    self.W = self.W + (X[i] * y[i])
                    _updated = True
        return self.W

    def predict(self, X):
        return np.sign(self.W @ self.bias(X).transpose())


class LDA(abcModel):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        X = self.bias(X)

        def gen_delta_y(X, y, y_val):
            _X = X[y == y_val]
            lnP = np.log(len(y == y_val) / len(y))

            aritmetic_mean = np.array([np.mean(u) for u in _X.transpose()])
            _cov = np.cov(X.transpose())
            _inv_cov = np.linalg.pinv(_cov)

            def delta(x):
                return x.transpose() @ _inv_cov @ aritmetic_mean - \
                       0.5 * aritmetic_mean.transpose() @ _inv_cov @ aritmetic_mean + \
                       lnP

            return delta

        self.deltas = [gen_delta_y(X, y, y_val) for y_val in [-1, 1]]

    def predict(self, U):
        return np.array([{0: -1.0, 1: 1.0}[
                             np.argmax([_delta(u) for _delta in self.deltas])]
                         for u in self.bias(U)])


class SVM(abcModel):
    def __init__(self):
        super().__init__()
        self.mod = SVC(C=1e10, kernel='linear')

    def coef_(self):
        return self.mod.coef_


class Logistic(abcModel):
    def __init__(self):
        super().__init__()
        self.mod = LogisticRegression(solver='liblinear')
    
    def fit(self, X, y):
        super().fit(X, y.flatten())

    # def predict(self, X):
    #     return self.mod.predict(self.bias(X))

class DecisionTree(abcModel):
    def __init__(self, max_depth=2):
        super().__init__()
        self.mod = DecisionTreeClassifier(max_depth=2)


class KNearestNeighbor(abcModel):

    def __init__(self):
        super().__init__()
        self.mod = KNeighborsClassifier(n_neighbors=40)


class RidgeClassifier(abcModel):

    def __init__(self):
        super().__init__()
        self.mod = Ridge(alpha=0, normalize=True)


class LassoClassifier(abcModel):

    def __init__(self):
        super().__init__()
        self.mod = Lasso(alpha=0, normalize=True)

class DecisionStumpWarper(abcModel):
    def __init__(self):
        super().__init__()
        self.mod = DecisionStump( )

    def fit(self, X, y, D=None):
        self.mod.fit(D, X, y.flatten())
    
    def predict(self, X):
        return self.mod.predict(X)
    

def label_f(weight, bias):
    def _label_f(x):
        return np.sign(np.dot(weight, x) + bias)

    return _label_f


def draw_points(m, label_function=label_f(np.array([0.3, -0.5]), 0.1)):
    def _draw_points_n(m, n):
        return np.random.multivariate_normal(
            np.zeros(n), np.identity(n), m)

    X = _draw_points_n(m, 2)
    y = np.array([label_function(vec) for vec in X])
    return X, y


def analyze_clssifiers():
    # def generate_line_prec(prec):

    def plot(prec, _svm, X, y):
        plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1])
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1])

        min_x, max_x = min(X[:, 0]), max(X[:, 0])
        min_y, max_y = min(X[:, 1]), max(X[:, 1])

        _min_range, _max_range = min(min_x, min_y), max(max_x, max_y)
        xx = [_min_range, _max_range]

        def get_y(W, _x):
            return -(W[0] + _x * W[1]) / W[2] if W[2] != 0 else -W[0]

        def get_y_prep(_x):
            return get_y(prec.W, _x)

        def get_y_svm(_x):
            print(_svm.coef_()[0])
            return get_y(_svm.coef_()[0], _x)

        def get_true_y(_x):
            return 0.1 / 0.5 + 0.3 / 0.5 * _x

        plt.xlim([_min_range, _max_range])
        plt.ylim([_min_range, _max_range])

        middle = (_min_range + _max_range) / 2

        def print_line(msg, _f, _color):
            xx = [_min_range, _max_range]
            yy = [_f(_x) for _x in xx]
            _x = middle + 2 * (0.5 - random())
            _y = _f(_x)
            plt.plot(xx, yy, color=_color)
            plt.annotate(msg, color=_color,
                         xy=(_x, _y), xycoords='data',
                         xytext=(_x + 0.3, _y), textcoords='data',
                         arrowprops=dict(arrowstyle="->"))

        print_line("prep", get_y_prep, "C5")
        print_line("svm", get_y_svm, "C4")
        print_line("true plane", get_true_y, "C2")
        plt.title("svm vs prep")
        plt.xlabel("x)")
        plt.ylabel("y")
        plt.show()

    for m in [5, 10, 15, 25, 70]:
        X, y = draw_points(m)
        blues, reds = X[y == 1], X[y == -1]
        _modes = [Perceptron(), SVM()]
        for _model in _modes:
            _model.fit(deepcopy(X), y)

        plot(_modes[0], _modes[1], X, y)


def expanded_analyze_clssifiers():
    times, k = 7, 1000
    modles = []
    models_num = 3

    def genrate_real_plane(m):
        _f = label_f(np.array([random(), random()]), random())
        X, y = draw_points(m, label_function=_f)
        while (-1 not in y) or (1 not in y):
            X, y = draw_points(m, label_function=_f)
        return X, y, _f

    def accur(mod, _f, Z):
        _prob = 0
        for x, y in zip(map(_f, Z), mod.predict(Z)):
            if x == y:
                _prob += 1
        return _prob / len(Z)

    def one_iteraion(m):
        _modes = [Perceptron(), SVM(), LDA()]
        X, y, _f = genrate_real_plane(m)
        ret = []
        for _model in _modes:
            print(type(_model))
            _model.fit(deepcopy(X), y)
            Z, _ = draw_points(k)
            ret.append(accur(_model, _f, Z))
        return np.array(ret)

    def calc_mean_performance(M=[5, 10, 15, 25, 70]):
        ret = []
        for m in M:
            _mean = np.zeros(models_num)
            for _ in range(times):
                _mean += one_iteraion(m)
            ret.append(_mean / times)
        return M, np.array(ret)

    m, mean_performance = calc_mean_performance()
    for _model_num, _name in enumerate(["perc", "svm", "lda"]):
        print(_name)
        print(mean_performance)
        plt.plot(m, mean_performance[:, _model_num])
    plt.legend(["perc", "svm", "lda"])
    plt.title("calc_mean_performance")
    plt.xlabel("m (size of the given training data)")
    plt.ylabel("propability of successes")
    plt.show()


if __name__ == "__main__":
    X, y = draw_points(10)
    # p = Perceptron()

    from copy import deepcopy

    models_class = [Perceptron, SVM, Logistic, DecisionTree, LDA]
    models = []
    for mod in models_class:
        models.append(mod())
        print("{} init ".format(type(mod)))

    for mod in models:
        mod.fit(deepcopy(X), y)
        print("{} fit ".format(type(mod)))

    for mod in models:
        print(mod.predict(deepcopy(X)))
        print("{} predict ".format(type(mod)))

    # analyze_clssifiers()
    expanded_analyze_clssifiers()
