import random

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import optuna
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm
from optuna import trial
from functools import partial
import sklearn.datasets
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB




import numpy as np
b = 5
clf_list = ["SVC", "RandomForest", "BernoulliNB", "DecisionTreeClassifier", "GaussianNB"]
a = random.randint(0, 3)
print("hier ist a", a)


def objective(trial, X, y, a):
    print("alda5el",a)


    classifier_name = clf_list[a]

    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
    elif classifier_name == "RandomForest":
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        rf_criterion = trial.suggest_categorical("rf_criterion", ["gini", "entropy"])
        #rf_n_estimators= trial.suggest_int("n_estimators", 10, 100)
        rf_max_features= trial.suggest_uniform("max_features", 0.01, 1.0)

        classifier_obj = sklearn.ensemble.RandomForestClassifier(
            n_estimators=10,
            max_depth=rf_max_depth,
            max_features=rf_max_features,
            criterion=rf_criterion
        )

    elif classifier_name == "BernoulliNB":
        alpha = trial.suggest_loguniform("alpha", 1e-2, 100)
        fit_prior = trial.suggest_categorical("fit_prior", [True, False])

        classifier_obj = BernoulliNB(
            alpha=alpha,
            fit_prior=fit_prior

        )

    elif classifier_name == "DecisionTreeClassifier":
        dt_criterion = trial.suggest_categorical("dt_criterion", ["gini", "entropy"])
        dt_max_depth = trial.suggest_int("dt_max_depth", 2, 32, log=True)
        dt_max_features = trial.suggest_categorical("dt_max_features", ["auto", "sqrt", "log2"])

        classifier_obj = DecisionTreeClassifier(
            criterion=dt_criterion,
            max_depth=dt_max_depth,
            max_features=dt_max_features
        )
    elif classifier_name == "GaussianNB>":
        classifier_obj = GaussianNB()
    elif classifier_name == "HistGradientBoostingClassifier":
        hgb_loss = trial.suggest_categorical("hgb_loss", ["og_loss", "auto", "binary_crossentropy","categorical_crossentropy"])
        hgb_learning_rate = trial.suggest_loguniform("hgb_learning_rate", 0.01, 1)
        classifier_obj = HistGradientBoostingClassifier(
            criterion=hgb_loss,
            max_depth=hgb_learning_rate
        )
    elif classifier_name == "kNeighborsClassifier":
        knn_n_neighbors = trial.suggest_int("knn_n_neighbors", 1,20)
        knn_weights = trial.suggest_categorical("hnn_weights", ["uniform", "distance"])
        classifier_obj = KNeighborsClassifier(
            n_neighbors=knn_n_neighbors,
            weights=knn_weights
        )
    elif classifier_name == "LinearDiscriminantAnalysis":
        LDA_solver = trial.suggest_categorical("LDA_solver", ["svd", "lsqr","eigen"])
        LDA_tol = trial.suggest_loguniform('LDA_tol', 1e-5, 1e-1)
        classifier_obj = LinearDiscriminantAnalysis(
            solver=LDA_solver,
            tol=LDA_tol
        )
    elif classifier_name == "LinearSVC":
        scv_penalty = trial.suggest_categorical("svc_penalty", ["l1", "l2"])
        svc_loss = trial.suggest_categorical("scv_loss", ["hinge", "squared_hinge"])
        classifier_obj = LinearSVC(
            penanty=scv_penalty,
            loss=svc_loss
        )
    elif classifier_name =="MLPClassifier":
        MLP_activation = trial.suggest_categorical("MLP_activation", ["identity", "logistic", "tanh", "relu"])
        MLP_solver = trial.suggest_categorical("Mlp_solver", ["lbfgs", "sgd", "adam"])
        MLP_learning_rate = trial.suggest_categorical("MLP_learning_rate",["constant", "invscaling", "adaptive"])
        classifier_obj = MLPClassifier(
            activation=MLP_activation,
            solver=MLP_solver,
            learning_rate=MLP_learning_rate

        )
    elif classifier_name == "MultinomialNB":
        LNB_alpha = trial.suggest_float("LNB_alpha", ["identity", "logistic", "tanh", "relu"])
        MLP_solver = trial.suggest_categorical("Mlp_solver", ["lbfgs", "sgd", "adam"])
        MLP_learning_rate = trial.suggest_categorical("MLP_learning_rate", ["constant", "invscaling", "adaptive"])
        classifier_obj = MLPClassifier(
            alpha=LNB_alpha,
            solver=MLP_solver,
            learning_rate=MLP_learning_rate

        )










    score = sklearn.model_selection.cross_val_score(classifier_obj, X, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy





iris = sklearn.datasets.load_iris()
x, y = iris.data, iris.target
optimize = partial(objective, X=x, y=y, a=a)

study = optuna.create_study(direction="maximize")

study.optimize(optimize, n_trials=10)
print(study.best_trial)
