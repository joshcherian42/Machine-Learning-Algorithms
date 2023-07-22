import numpy as np
from Regression.linear_regression import LinearRegression


class LassoRegression(LinearRegression):
    def __init__(self, regularization_coef=1.0, epochs=50):
        self.regularization_coef = regularization_coef
        self.epochs = epochs

    def _soft_thresholding(self, rho_j, z_j):
        '''
        Implements soft thresholding operator for coordinate descent

        :param rho_j
        :param z_j
        :return: coefficient
        '''
        if rho_j < -self.regularization_coef:
            return (rho_j + self.regularization_coef) / z_j
        elif rho_j > self.regularization_coef:
            return (rho_j - self.regularization_coef) / z_j
        else:
            return 0

    def _coordinate_descent(self, X, y, j):
        '''
        Coordinate descent update for jth feature

        :param X: training features
        :param y: training labels
        :param j: feature index
        :return: predicted value
        '''

        k = [i for i in range(self.cols) if i != j]  # columns except the jth column
        X_k = X[:, k]

        coefs_k = self.coefficients[k]
        rho_j = np.sum(np.multiply(X[:, j], y - np.sum(np.multiply(coefs_k, X_k), axis=1)), axis=0)
        z_j = np.sum(X[:, j] ** 2, axis=0)

        coef_j = self._soft_thresholding(rho_j, z_j)
        self.coefficients[j] = coef_j

    def _estimate_coefficients(self, X, y):
        '''
        Implements coordinate descent to find weights

        :param X: training features
        :param y: training labels
        :return: predicted value
        '''
        self.coefficients = np.zeros(self.cols)

        for e in range(self.epochs):
            for j in range(self.cols):
                self._coordinate_descent(X, y, j)

        return self.coefficients
