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



def objective(trial, X, y):

    classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
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
            criterion=rf_criterion
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



"""    elif classifier_name == "ExtraTreesClassifier":
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", "None"])
        self.min_samples_split = trial.suggest_int("min_samples_split", 2, 20, log=False)
        self.min_samples_leaf = trial.suggest_int(self.name + "min_samples_leaf", 1, 20, log=False)
        self.max_depth = None
        self.min_weight_fraction_leaf = 0.
        self.max_leaf_nodes = None
        self.min_impurity_decrease = 0.0
        self.bootstrap = trial.suggest_categorical(self.name + "bootstrap", [True, False])
        self.classes_ = np.unique(y.astype(int))
        self.n_jobs = 1
"""