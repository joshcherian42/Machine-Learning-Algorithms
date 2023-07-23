from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split

from DecisionTree.decisiontree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from RandomForest.random_forest import RandomForest
from sklearn.ensemble import RandomForestClassifier
from Regression.linear_regression import LinearRegression
from sklearn.linear_model import LinearRegression as lr
from NaiveBayes.NaiveBayes import NaiveBayes as NB
from sklearn.naive_bayes import GaussianNB
from Regression.ridge_regression import RidgeRegression
from sklearn.linear_model import Ridge
from Regression.lasso_regression import LassoRegression
from sklearn.linear_model import Lasso

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score


def load_dataset(dataset_type):
    if dataset_type == 'classification':
        return load_iris(return_X_y=True)
    elif dataset_type == 'regression':
        return load_diabetes(return_X_y=True)
    else:
        return None


def get_model(clf_type):

    if clf_type == 'Decision Tree':
        return DecisionTreeClassifier(random_state=0, min_samples_split=2, max_depth=5), DecisionTree()
    elif clf_type == 'Random Forest':
        return RandomForestClassifier(n_estimators=100, min_samples_split=2, max_depth=5, criterion='gini', random_state=0, min_impurity_decrease=0), RandomForest(random_state=0)
    elif clf_type == 'Linear Regression':
        return lr(), LinearRegression()
    elif clf_type == 'Gaussian':
        return GaussianNB(), NB()
    elif clf_type == 'Ridge':
        return Ridge(), RidgeRegression()
    elif clf_type == 'Lasso':
        return Lasso(), LassoRegression()
    else:
        print('Algorithm not found')
        exit()
        return None


def cmp(clf_type, dataset_type, sklearn_perf, custom_perf):
    metric = 'Accuracy' if dataset_type == 'classification' else 'R2 Score'
    diff = sklearn_perf - custom_perf

    print('')
    print(f'{clf_type}({metric}): Sklearn: {sklearn_perf} | Custom: {custom_perf} | Diff: {diff}')
    print('')


def eval_performance(y_pred, y_test, dataset_type):

    if dataset_type == 'classification':
        return accuracy_score(y_test, y_pred)
    elif dataset_type == 'regression':
        return r2_score(y_test, y_pred)


def run_clf(clf, X_train, X_test, y_train, y_test, dataset_type):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return eval_performance(y_pred, y_test, dataset_type)


if __name__ == '__main__':

    clf_type = 'Random Forest'
    dataset_type = 'classification'

    X, y = load_dataset(dataset_type)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    sklearn_clf, custom_clf = get_model(clf_type)

    sklearn_perf = run_clf(sklearn_clf, X_train, X_test, y_train, y_test, dataset_type)
    custom_perf = run_clf(custom_clf, X_train, X_test, y_train, y_test, dataset_type)

    cmp(clf_type, dataset_type, sklearn_perf, custom_perf)
