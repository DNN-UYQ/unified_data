from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
def best_models(X, y):
    # generate dataset
    X, y = make_classification(n_samples=100, n_features=50, n_informative=2)
    # define feature selection
    fs = SelectKBest(score_func=f_classif, k=5)
    # apply feature selection
    X_selected = fs.fit_transform(X, y)
    return X_selected