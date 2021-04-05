"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None] * T  # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        m = X.shape[0]
        D = np.full((m,), 1.0 / m)
        for t in np.arange(0, self.T):
            self.h[t] = self.WL(D, X, y)
            y_hat = self.h[t].predict(X)
            error_t = np.sum(D[y_hat != y])
            self.w[t] = 0.5 * np.log((1.0 / error_t) - 1)
            D = D * np.exp(-self.w[t] * y_hat * y)
            D = np.divide(D, np.sum(D))
        return D

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        votes = np.zeros((X.shape[0]))
        for i in np.arange(0, max_t):
            votes += self.h[i].predict(X) * self.w[i]
        return np.sign(votes)

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the wrong predictions when predict only with max_t
        weak learners (float)
        """
        y_hat = self.predict(X, max_t)
        return np.mean(y_hat != y)
