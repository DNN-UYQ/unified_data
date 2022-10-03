from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from classifier_hyperparameter import all_classifier
import random



# Model Class to be used for different ML algorithms
class ClassifierModel(object):
    """def __init__(self, clf, params=None):
        self.clf = clf(**params)"""
    def __init__(self, clf):
        self.clf = clf

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        return self.clf.fit(x, y).feature_importances_

    def predict(self, x):
        return self.clf.predict(x)
