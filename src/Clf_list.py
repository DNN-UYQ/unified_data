from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def classfier():
    clf1 = LogisticRegression()
    clf2 = KNeighborsClassifier()
    clf3 = svm.SVC()
    return [clf1, clf2, clf3]


def params():
    param1 = {"C":np.logspace(-3, 3, 7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
    param2 = {
        "n_neighbors": [i for i in range(1, 10)],
        "weights": ["uniform", "distance"]
    }
    param3 = {
    "kernel": ["linear", "sigmoid", "rbf", "poly"]
    }
    return [param1, param2, param3]