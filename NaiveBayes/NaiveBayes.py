import numpy as np


class NaiveBayes:

    def fit(self, X, y):
        '''
        Trains Naive Bayes Model

        :param x: single observation
        :return: None
        '''

        self.classes = np.unique(y)
        self._calc_mean(X, y)
        self._calc_var(X, y)
        self._calc_priors_wrapper(X, y)

    def _calc_mean(self, X, y):
        '''
        Calculates variance

        :param X: training features
        :param y: training labels
        :return: None
        '''
        self.mean = np.array([X[y == i].mean(axis=0) for i in np.unique(y)])

    def _calc_var(self, X, y):
        '''
        Calculates variance

        :param X: training features
        :param y: training labels
        :return: None
        '''
        self.var = np.array([X[y == i].var(axis=0) for i in np.unique(y)])

    def _pdf(self, X, class_label):

        mean = self.mean[np.where(self.classes == class_label)]
        var = self.var[np.where(self.classes == class_label)]
        prob = np.exp((-1 / 2) * ((X - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)

        return prob

    def _calc_priors(self, X, y, class_label):
        return y[y == class_label].shape[0] / X.shape[0]

    def _calc_priors_wrapper(self, X, y):

        self.priors = np.array([self._calc_priors(X, y, i) for i in np.unique(y)])

    def _calc_prob(self, X, class_label):

        prior = self.priors[np.where(self.classes == class_label)]
        conditional = np.sum(self._pdf(X, class_label))

        return prior + conditional

    def _calc_probs_wrapper(self, X):

        class_probs = np.array([self._calc_prob(X, i) for i in self.classes])
        return self.classes[np.argmax(class_probs)]

    def predict(self, X):
        '''
        Predicts values for given feature values

        :param X: feature values
        :return: predicted value
        '''
        return [self._calc_probs_wrapper(x) for x in X]
