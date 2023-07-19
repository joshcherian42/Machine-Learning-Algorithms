from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from DecisionTree.decisiontree import DecisionTree
from RandomForest.random_forest import RandomForest

if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # clf = DecisionTreeClassifier(random_state=0, min_samples_split=2, max_depth=5)
    # clf = DecisionTree()
    # clf = RandomForestClassifier(n_estimators=100, min_samples_split=2, max_depth=5, criterion='gini', random_state=0, min_impurity_decrease=0)
    # clf = RandomForest(random_state=0)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(accuracy_score(y_test, y_pred))
