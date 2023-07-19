import numpy as np
from DecisionTree.decisiontree import DecisionTree
from collections import Counter
from multiprocessing import Pool


class RandomForest:
    '''
    A class that implements Random Forest algorithm from scratch.
    '''

    def __init__(self,
                 n_estimators=100, min_samples_split=2,
                 max_depth=5, criterion='gini',
                 random_state=None, min_impurity_decrease=0):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.decision_trees = []
        self.criterion = criterion
        self.random_state = np.random.RandomState(seed=random_state)
        self.min_impurity_decrease = min_impurity_decrease

    def _sample(self, X, y):
        '''
        Boostrap sampling

        :param X: np.array, features
        :param y: np.array, labels
        :return: tuple (sample of features, sample of labels)
        '''
        n_rows, _ = X.shape
        samples = self.random_state.choice(a=n_rows, size=n_rows, replace=True)

        return X[samples], y[samples]

    def _fit_tree(self, X, y):
        '''
        Creates and fits a decision tree on a sample of the data

        :param X: np.array, training features
        :param y: np.array, training labels
        :return: Decision Tree
        '''
        clf = DecisionTree(
            min_samples_split=self.min_samples_split,
            max_depth=self.max_depth,
            criterion=self.criterion,
            min_impurity_decrease=self.min_impurity_decrease
        )

        _X, _y = self._sample(X, y)
        clf.fit(_X, _y)

        return clf

    def fit(self, X, y):
        '''
        Trains a Random Forest classifier

        :param X: np.array, training features
        :param y: np.array, training labels
        :return: None
        '''
        if len(self.decision_trees) > 0:
            self.decision_trees = []

        print('Building Trees')

        pool = Pool()
        self.decision_trees = pool.starmap(self._fit_tree, [(X, y)])

    def _predict_helper(self, row):
        '''
        Wrapper function to get prediction for each tree in forest

        :param row: np.array, features used to generate prediction
        :return: list of predictions
        '''
        preds = []
        for tree in self.decision_trees:
            preds.append(tree.predict(row.reshape(1, -1))[0])
        return preds

    def predict(self, X):
        '''
        Predicts class labels for new data instances.
        :param X: np.array, new instances to predict
        :return: np.array of predictions
        '''
        y = [self._predict_helper(row) for row in X]

        predictions = []
        for preds in y:
            c = Counter(preds)
            predictions.append(c.most_common(1)[0][0])

        return np.array(predictions)
