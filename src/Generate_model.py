from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from classifier_hyperparameter import all_classifier
import random

def generate_random_molde():
    all_clf = all_classifier()
    name, clf = random.choice(list(all_clf.items()))
    return name, clf
name, clf =generate_random_molde()
print(name, clf)

def train(x_train, y_train):
    clf.fit(x_train, y_train)


def fit(x_test, y_test):
        return clf.fit(x_test, y_test)


def feature_importances(x_test, y_test):
    return clf.fit(x_test, y_test).feature_importances_


def predict(x_test):
    return clf.predict(x_test)


def trainModel(model, x_train, y_train, x_test, n_folds, seed):
    cv = KFold(n_splits=n_folds, random_state=seed)
    scores = cross_val_score(model.clf, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores



"""class ClassifierModel(object):
    def __init__(self, clf, params=None):
        self.clf = clf(**params)


def train(self, x_train, y_train):
    self.clf.fit(x_train, y_train)


    def fit(self, x, y):
        return self.clf.fit(x, y)


    def feature_importances(self, x, y):
        return self.clf.fit(x, y).feature_importances_


    def predict(self, x):
        return self.clf.predict(x)


def trainModel(model, x_train, y_train, x_test, n_folds, seed):
    cv = KFold(n_splits=n_folds, random_state=seed)
    scores = cross_val_score(model.clf, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores"""