import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from models import Logistic, DecisionTree, KNearestNeighbor, SVM
from copy import deepcopy

import time

def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    train_images = np.logical_or((y_train == 0), (y_train == 1))
    test_images = np.logical_or((y_test == 0), (y_test == 1))
    x_train, y_train = x_train[train_images], y_train[train_images]
    x_test, y_test = x_test[test_images], y_test[test_images]

    return x_train, y_train, x_test, y_test


def rearrange_data(X):
    return np.array( [ x.flatten() for x in X ])


def expanded_analyze_clssifiers(x_train, y_train, x_test, y_test):
    times, k = 50, 100
    modles = []
    models_num = 4

    def generate( m, X, y ):
        indexs = [False] * len(X)
        _indexs = np.random.choice( list(range(len(X)) ), size=m)
        for _index in _indexs:
            indexs[_index] = True

        return X[indexs], y[indexs], indexs

    def accur(mod, indexs , Z):
        _prob = 0
        for x,y in zip(y_test[indexs], mod.predict(Z)):
            if x == y:
                _prob +=1
        return _prob/len(Z)

    def one_iteraion(m):
        _modes = [Logistic(), DecisionTree(), KNearestNeighbor(), SVM()]
        X, y, indexs = generate(m, x_train, y_train)

        # your code


        while (0 not in y) or (1 not in y) :
             X, y, indexs = generate(m, x_train, y_train)
        ret = []
        for _model in _modes:
            start_time = time.time()
            _model.fit(deepcopy(X),y)
            elapsed_time = time.time() - start_time
            print("train : {} takes {}".format(_model, elapsed_time))
            Z, _, indexs = generate(k, x_test, y_test)
            ret.append( accur(_model, indexs, Z) )
        return np.array(ret)

    def calc_mean_performance(M = [50, 100, 300, 500] ):
        ret = []
        for m in M:
            _mean = np.zeros(models_num)
            for _ in range(times):

                _mean += one_iteraion(m)
            ret.append( _mean/ times )
        return M, np.array( ret )

    m, mean_performance = calc_mean_performance()
    for _model_num, _name  in enumerate(["Logistic", "DecisionTree", "KNearestNeighbor", "soft-SVM"]):
        plt.plot( m , mean_performance[:,_model_num] )
    plt.legend( ["Logistic", "DecisionTree", "KNearestNeighbor", "soft-SVM"] )
    plt.title("calc_mean_performance")
    plt.xlabel("m (size of the given training data)")
    plt.ylabel("propability of successes")
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    expanded_analyze_clssifiers(rearrange_data(x_train),
     rearrange_data(y_train), rearrange_data(x_test), rearrange_data(y_test))
