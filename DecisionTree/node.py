class Node:
    '''
    Helper class which implements a single tree node.
    '''

    def __init__(self):
        self.split_variable = None
        self.left_child = None
        self.right_child = None
        self.threshold = None
        self.num_samples = None
        self.gini = None


class Leaf():
    '''
    Helper class which implements a single leaf node.
    '''

    def __init__(self, value, num_samples):
        self.value = value
        self.num_samples = num_samples
