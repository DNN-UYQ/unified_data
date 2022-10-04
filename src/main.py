from Preprocessing import data_download, unified_data
from sklearn.inspection import plot_partial_dependence, permutation_importance
import matplotlib.pyplot as plt
from autosklearn.metrics import balanced_accuracy, precision, recall, f1
import sklearn.model_selection
import autosklearn.classification
from Best_models import best_models
import warnings
warnings.simplefilter(action='ignore')
import pandas as pd
from Openml_task import datasets_filter
from sklearn.metrics import accuracy_score
import numpy as np


if __name__ == '__main__':
    data_id = [31, 1464]
    X_list, y_list = data_download(data_id)
    X_big, y_big, column = unified_data(X_list, y_list)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_big, y_big, random_state=1)
    automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=600, per_run_time_limit=10,
                                                              metric=f1)
    automl.fit(X_train, y_train)
    predictions = automl.predict(X_test)
    print("f1 score", sklearn.metrics.f1_score(y_test, predictions))
    print(f"Accuracy score: {accuracy_score(y_test, predictions): 6.3f}")
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))

    result = permutation_importance(automl, X_train, y_train, n_repeats=10, random_state=0)
    perm_sorted_idx = result.importances_mean.argsort()
    print (len(result))
    print (perm_sorted_idx)



    """r = permutation_importance(automl, X_big, y_big, n_repeats=10, random_state=0)
    sort_idx = r.importances_mean.argsort()[::-1]









    feature_selection = pd.DataFrame(r)
    feature_selection.to_excel("features_selection_new.xlsx")


    plt.boxplot(r.importances[sort_idx].T,
                labels=[X_big[i] for i in sort_idx])

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    for i in sort_idx[::-1]:
        print(f"{X_test[i]:10s}: {r.importances_mean[i]:.3f} +/- "
              f"{r.importances_std[i]:.3f}")"""

