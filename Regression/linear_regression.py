import numpy as np


class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.cols = None

    def fit(self, X, y):
        '''
        Trains Linear Regression Model

        :param x: single observation
        :return: None
        '''
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)  # add in intercept
        self.cols = X.shape[1]
        coefficients = self._estimate_coefficients(X, y)
        self.intercept = coefficients[0]
        self.coefficients = coefficients[1:]

    def _estimate_coefficients(self, X, y):
        '''
        Helper function, implements Ordinary Least Squares

        :param X: training features
        :param y: training labels
        :return: predicted value
        '''
        xT = X.transpose()
        coefficients = np.linalg.inv(xT.dot(X)).dot(xT).dot(y)

        return coefficients

    def _predict(self, x):
        '''
        Helper function, predicts value for single row of features

        :param x: single observation
        :return: predicted value
        '''
        pred = sum(np.multiply(x, self.coefficients)) + self.intercept
        return pred

    def predict(self, X):
        '''
        Predicts values for given feature values

        :param X: feature values
        :return: predicted value
        '''
        return [self._predict(x) for x in X]
