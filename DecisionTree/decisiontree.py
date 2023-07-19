from DecisionTree.node import Node, Leaf
from DecisionTree.splitting_criterion import Splitter
from collections import Counter


class DecisionTree:

    def __init__(self, min_samples_split=2, max_depth=5, criterion='gini', min_impurity_decrease=0):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self.criterion = Splitter(criterion=criterion)
        self.min_impurity_decrease = min_impurity_decrease

    def _build(self, X, y, depth=0):
        '''
        Helper recursive function, used to build a decision tree from the input data.

        :param X: np.array, features
        :param y: np.array or list, target
        :param depth: current depth of a tree, used as a stopping criteria
        :return: Node
        '''

        n_rows, n_cols = X.shape

        if n_rows >= self.min_samples_split and (self.max_depth is None or depth < self.max_depth):

            best = self.criterion.best_split_wrapper(X, y, None)
            if best['gini'] > 0:
                n = Node()
                n.split_variable = best['feature_index']
                n.threshold = best['threshold']
                n.num_samples = n_rows
                n.gini = self.criterion.gini_entropy(y)
                left = self._build(
                    X=best['x_left'],
                    y=best['labels_left'],
                    depth=depth + 1,
                )
                right = self._build(
                    X=best['x_right'],
                    y=best['labels_right'],
                    depth=depth + 1,
                )

                n.left_child = left
                n.right_child = right

                return n

        return Leaf(
            value=Counter(y).most_common(1)[0][0],
            num_samples=n_rows
        )

    def fit(self, X, y):
        '''
        Function used to train a decision tree classifier model.

        :param X: np.array, features
        :param y: np.array or list, target
        :return: None
        '''
        self.criterion.fit(list(set(y)))
        self.root = self._build(X, y)

    def _predict(self, x, tree):
        '''
        Helper recursive function, used to predict a single instance (tree traversal).

        :param x: single observation
        :param tree: built tree
        :return: predicted class
        '''
        # Leaf node
        if isinstance(tree, Leaf):
            return tree.value
        else:
            feature_value = x[tree.split_variable]
            if feature_value < tree.threshold:
                return self._predict(x, tree.left_child)
            else:
                return self._predict(x, tree.right_child)

    def predict(self, X):
        '''
        Function used to classify new instances.

        :param X: np.array, features
        :return: list, predicted classes
        '''
        return [self._predict(x, self.root) for x in X]
