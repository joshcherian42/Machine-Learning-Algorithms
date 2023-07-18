import math
import numpy as np
import itertools
from numba import jit, prange


@jit(nopython=True, nogil=True, cache=True)
def left_mask(x, threshold):
    return x <= threshold


@jit(nopython=True, nogil=True, cache=True)
def right_mask(x, threshold):
    return x > threshold


@jit(nopython=True, cache=True, nogil=True)
def numba_bool(x, bool_mask):
    return x[bool_mask]


@jit(nopython=True, cache=True, nogil=True)
def mask_equals(x, k):
    return x == k


@jit(nopython=True, cache=True, nogil=True)
def sqrt_product(a, b):
    return math.sqrt(a * b)


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def sum_reduce(x):
    s = 0
    for i in prange(x.shape[0]):
        s += x[i]
    return s


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def prange_count(x):
    s = 0
    for i in prange(x.shape[0]):
        s += 1
    return s


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def numpy_prod(p_tj, p_j):
    return np.multiply(p_tj, p_j)


class Splitter:
    def __init__(self, criterion='gini'):
        self.criterion = criterion
        self.classes = []

    def fit(self, classes):
        '''
        Sets up Splitter

        :param classes: list, class values
        :return: None
        '''
        self.classes = classes
        self.n_classes = np.unique(classes).size
        self.class_map = {act: i for i, act in enumerate(self.classes)}
        self.inverse_map = {i: act for i, act in enumerate(self.classes)}

    def gini_entropy(self, y):
        '''
        Calculates gini entropy

        :param y: list, labels
        :return: float, calculated gini entropy
        '''
        _classes = np.unique(y)
        p = 0.0
        for c in _classes:
            P_i = np.count_nonzero(y == c) / len(y)
            p += P_i * (1 - P_i)
        return p

    def _calculate_criterion(self, labels, y_left, y_right):
        '''
        Calculates gini entropy

        :param y: list, labels
        :return: float, calculated gini entropy
        '''

        num_left = y_left.shape[0]
        num_right = y_right.shape[0]

        if self.criterion == 'gini':
            left_gini = self.gini_entropy(y_left)
            right_gini = self.gini_entropy(y_right)
            return (num_left * left_gini + num_right * right_gini) / labels.shape[0]
        # TODO: information gain

    def _process_candidate(self, features, labels, threshold):
        '''
        Wrapper to calculate potential split

        :param features: np.array, sorted feature values
        :param labels: np.array, sorted labels
        :param threshold: np.float, candidate threshold
        :return: float, calculated gini entropy
        '''
        y_left = numba_bool(labels, left_mask(features, threshold))
        y_right = numba_bool(labels, right_mask(features, threshold))

        if len(y_left) > 0 and len(y_right) > 0:
            return self._calculate_criterion(labels, y_left, y_right)

        return 1

    def _best_split(self, X, y, n_idx, parent_gini):
        '''
        Helper function to find the best split

        :param X: np.array, feature values
        :param y: np.array, labels
        :param n_idx: int, feature column
        :param parent_gini: np.float, gini of parent node
        :return: best split
        '''
        _X = X[:, n_idx]
        features, labels = zip(*sorted(zip(_X, y)))

        features = np.asarray(features, dtype=np.float64)
        if isinstance(labels[0], str):
            labels = np.asarray(labels, dtype=str)
        else:
            labels = np.asarray(labels, dtype=np.int64)

        shifted_values = np.roll(features, -1)
        shifted_labels = np.roll(labels, -1)

        candidates = list(itertools.compress([np.average([val, shifted_values[i]]) for i, val in enumerate(features)],
                                             [lbl != shifted_labels[i] for i, lbl in enumerate(labels)]))

        if (len(candidates) == 0):
            # Pure node
            best_split = {
                'threshold': None,
                'x_left': [],
                'labels_left': [],
                'x_right': [],
                'labels_right': [],
                'gini': parent_gini
            }

        else:
            ginis = []
            unique_candidates = np.unique(candidates)
            for threshold in unique_candidates:
                gini = self._process_candidate(features, labels, threshold)
                ginis.append(gini)

            gini = np.min(ginis)
            inv_ginis = ginis[::-1]
            threshold = unique_candidates[len(inv_ginis) - np.argmin(inv_ginis) - 1]

            x_left = numba_bool(X, left_mask(_X, threshold))
            labels_left = numba_bool(y, left_mask(_X, threshold))
            x_right = numba_bool(X, right_mask(_X, threshold))
            labels_right = numba_bool(y, right_mask(_X, threshold))

            best_split = {
                'threshold': threshold,
                'x_left': x_left,
                'labels_left': labels_left,
                'x_right': x_right,
                'labels_right': labels_right,
                'gini': gini
            }
        return best_split

    def best_split_wrapper(self, X, y, node=None):
        '''
        Wrapper function to iterate through features to find best split

        :param X: np.array, feature values
        :param y: np.array, labels
        :param node: Node, current node
        :return: best split
        '''
        if len(y) == 0:
            return None

        if node is None:
            num_parent = [np.sum(y == c) for c in range(self.n_classes)]

            best_gini = 1.0 - sum((n / len(y)) ** 2 for n in num_parent)

            for split_variable in range(X.shape[1]):
                best_split = self._best_split(X, y, split_variable, best_gini)
                if best_split['gini'] <= best_gini:
                    max_best_split = {
                        'feature_index': split_variable,
                        'threshold': best_split['threshold'],
                        'x_left': best_split['x_left'],
                        'labels_left': best_split['labels_left'],
                        'x_right': best_split['x_right'],
                        'labels_right': best_split['labels_right'],
                        'gini': best_gini
                    }

                    best_gini = best_split['gini']

        else:
            max_best_split = self._best_split(
                X, y, node.split_variable)

        return max_best_split
