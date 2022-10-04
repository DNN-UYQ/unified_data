import sklearn.datasets

from sklearn.model_selection import RepeatedStratifiedKFold


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

# Here our classifiers list.
import random
def all_classifier():
    all_clf = {}
    cl1 = AdaBoostClassifier()
    cl2 = BernoulliNB()
    cl3 = DecisionTreeClassifier()
    cl4 = ExtraTreesClassifier()
    cl5 = GaussianNB()
    cl6 = HistGradientBoostingClassifier()
    cl7 = KNeighborsClassifier()
    cl8 = LinearDiscriminantAnalysis()
    cl9 = LinearSVC()
    cl10 = MLPClassifier()
    cl11 = MultinomialNB()
    cl12 = PassiveAggressiveClassifier()
    cl13 = QuadraticDiscriminantAnalysis()
    cl14 = RandomForestClassifier()
    cl15 = SGDClassifier()
    cl16 = SVC()
    all_clf = {
        "AdaBoostClassifier": cl1,
        "BernoulliNB": cl2,
        "DecisionTreeClassifier": cl3,
        "ExtraTreesClassifier": cl4,
        "GaussianNB": cl5,
        "HistGradientBoostingClassifier": cl6,
        "KNeighborsClassifier": cl7,
        "LinearDiscriminantAnalysis": cl8,
        "LinearSVC": cl9,
        "MLPClassifier": cl10,
        "MultinomialNB": cl11,
        "PassiveAggressiveClassifier": cl12,
        "QuadraticDiscriminantAnalysis": cl13,
        "RandomForestClassifier": cl14,
        "SGDClassifier": cl15,
        "SVC": cl16
    }
    return all_clf
"""#cl_list ==[cl1, cl2, cl3, cl4, cl5, cl6, cl7, cl8, cl9, cl10, cl11, cl12, cl13, cl14, cl15, cl16]
    #cl_list ==[cl1, cl2, cl3, cl4, cl5, cl6, cl7, cl8, cl9, cl10, cl11, cl12, cl13, cl14, cl15, cl16]
iris = sklearn.datasets.load_iris()
x, y = iris.data, iris.target

models_scores_results, models_names = list(), list()
print(">>>> Training started <<<<")
clf_list = all_classifier()
name, clf = random.choice(list(clf_list.items()))
print("name",name)
print("clf",clf)
print(type(clf))
scores = model_selection.cross_val_score(clf, x, y, cv=FOLDS, scoring='accuracy')
models_scores_results.append(scores)
models_names.append(name)
print("[%s] - accuracy: %0.5f " % (name, scores.mean()))
clf.fit(x, y)

# Save classifier for prediction
clf_list[name] =name"""



