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
# Here our classifiers list.
def all_classifier():
    cl1 = AdaBoostClassifier()
    cl2 = BernoulliNB()
    cl3 = DecisionTreeClassifier()
    cl4 = ExtraTreesClassifier()
    cl5 = GaussianNB()
    cl6 = HistGradientBoostingClassifier()
    cl7 = KNeighborsClassifier()
    cl8 = LinearDiscriminantAnalysis()
    cl9 = LinearSVC(kernel='linear', probability=True)
    cl10 = MLPClassifier()
    cl11 = PassiveAggressiveClassifier()
    cl12 = QuadraticDiscriminantAnalysis()
    cl13 = RandomForestClassifier()
    cl14 = SGDClassifier(loss='hinge')
    cl15 = SVC()
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
        "PassiveAggressiveClassifier": cl11,
        "QuadraticDiscriminantAnalysis": cl12,
        "RandomForestClassifier": cl13,
        "SGDClassifier": cl14,
        "SVC": cl15
    }
    return all_clf



