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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier





def objective(trial, X, y):

    classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest", "BernoulliNB", "DecisionTreeClassifier"])
    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
    elif classifier_name == "RandomForest":
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        rf_criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        #rf_n_estimators= trial.suggest_int("n_estimators", 10, 100)
        rf_max_features= trial.suggest_uniform("max_features", 0.01, 1.0)

        classifier_obj = sklearn.ensemble.RandomForestClassifier(
            n_estimators=10,
            max_depth=rf_max_depth,
            max_features=rf_max_features,
            criterion=rf_criterion)


    elif classifier_name == "BernoulliNB":
        alpha = trial.suggest_loguniform("alpha", 1e-2, 100)
        fit_prior = trial.suggest_categorical("fit_prior", [True, False])

        classifier_obj = BernoulliNB(
           alpha=alpha,
            fit_prior=fit_prior

        )
    elif classifier_name == "DecisionTreeClassifier":
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])

        classifier_obj = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            max_features=max_features
        )

    score = sklearn.model_selection.cross_val_score(classifier_obj, X, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy


iris = sklearn.datasets.load_iris()
x, y = iris.data, iris.target
optimize= partial(objective, X=x, y=y)

study = optuna.create_study(direction="maximize")

study.optimize(optimize, n_trials=10)
print(study.best_trial)