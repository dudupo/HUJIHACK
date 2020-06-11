"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np
from ex4_tools import *
from matplotlib import pyplot as plt

class AdaBoost(object):

    def __init__(self, WL, T , support_wights = False):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = [ WL for _ in range(T)]
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights
        self.support_wights = support_wights

        # self._predict = np.vactorize(WL.prdeict)

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        y = y.flatten()
        D = np.ones( len( y ) ) * 1/len( y )

        def find_loss_D(h, D, X, y):
            _out = h.predict(X).flatten()
            return np.sum( D[_out != y.transpose()] ), _out

        def normalize(vec):
            r = np.sum(vec )
            return vec / r if r != 0 else 0

        self.w = []

        for i in range(self.T):
            self.h[i] = self.WL[i]()
            if self.support_wights:
                self.h[i].train(X, y, D)
            else:
                self.h[i].train(X, y)
                
            e_t, _out = find_loss_D(self.h[i], D, X, y)
            print("[%]et:")
            print(e_t)
            w = 0.5 * np.log( 1/e_t - 1 )
            D = normalize(D * np.e **( -w * y * _out ) )
            self.w.append(w)

        self.w = np.array(self.w)
        return D

    def predict(self, X, max_t=None):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        if max_t is None: 
            max_t = self.T
        return np.sign(
                 sum(
                  self.w[i] * self.h[i].predict(X) for i in range(max_t)) )

    def error(self, X, y, max_t=None):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the correct predictions when predict only with max_t weak learners (float)
        """
        if max_t is None: 
            max_t = self.T
        errors = sum(np.ones( len(y) )[ self.predict(X, max_t) != y ])
        return errors / len(y)


class AdaBoostList(AdaBoost):
    def __init__(self, WLs):
        self.WL = WLs
        self.T = len(WLs)
        self.h = [None]*self.T   # lisself. of base learners
        self.w = np.zeros(self.T)  # weights

    def prdict(self, X):
        return super().prdict(X, self.T)


def test():
    def plot_as_function_of_agents(A, X, y, max_t):
        yplot = [A.error(X, y, t) for t in range(max_t)]
        plt.plot([t for t in range(max_t)], yplot)

    def plot_points(A, X, y,
     classfiers=[ 5, 10, 50, 100, 200, 500 ], D_T=None, noise=0):

        for i ,T in enumerate(classfiers):
            plt.subplot( 320 + i + 1)
            decision_boundaries(A, X, y, num_classifiers=T, weights=D_T)

        size_str = True if D_T is not None else False
        plt.savefig(f"plot_points_{size_str}_{noise}.png")

    for noise in [0, 0.01, 0.4]:
        A = AdaBoost( DecisionStump, 500 )
        X,y = generate_data(5000, noise)
        A.train(X,y)


        plot_as_function_of_agents(A, X, y, 500)
        X,y = generate_data(200, noise)
        plot_as_function_of_agents(A, X, y, 500)
        plt.legend(["training error rate","testing error rate"])
        plt.xlabel("agents")
        plt.ylabel("error rate")
        plt.ylim([0,1])
        plt.title(f"error rate above distribution with nosie - {noise}")
        plt.savefig(f"plot_error_{noise}.png")


        D_T = A.w.transpose()
        D_T = 30* D_T/max(D_T)
        plt.figure()

        for d_T in [None, D_T ]:
            plot_points(A, X, y, D_T=d_T, noise=noise)
            plt.figure()



if __name__ == "__main__":
    test()
