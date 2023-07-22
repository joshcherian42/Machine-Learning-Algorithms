import numpy as np
from Regression.linear_regression import LinearRegression


class RidgeRegression(LinearRegression):
    def __init__(self, regularization_coef=1.0):
        self.regularization_coef = regularization_coef

    def _estimate_coefficients(self, X, y):
        '''
        Helper function, implements Ordinary Least Squares with l2 regularization

        :param X: training features
        :param y: training labels
        :return: predicted value
        '''
        xT = X.transpose()
        coefficients = np.linalg.inv(xT.dot(X) + self.regularization_coef * np.eye(self.cols)).dot(xT).dot(y)

        return coefficients
