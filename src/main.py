from Preprocessing import data_download, unified_data
from sklearn.inspection import plot_partial_dependence, permutation_importance
import matplotlib.pyplot as plt
from autosklearn.metrics import balanced_accuracy, precision, recall, f1
import sklearn.model_selection
import autosklearn.classification
from Adult_dataset import adult_dataset, featurize

if __name__ == '__main__':
    data_id = [31, 1464]
    X_list, y_list = data_download(data_id)
    X_big, y_big = unified_data(X_list, y_list)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_big, y_big, random_state=1)
    automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=600, per_run_time_limit=10,
                                                              metric=f1)

    automl.fit(X_train, y_train)

    r = permutation_importance(automl, X_test, y_test, n_repeats=10, random_state=0)
    sort_idx = r.importances_mean.argsort()[::-1]

    plt.boxplot(r.importances[sort_idx].T,
                labels=[X_big[i] for i in sort_idx])

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    for i in sort_idx[::-1]:
        print(f"{dataset.feature_names[i]:10s}: {r.importances_mean[i]:.3f} +/- "
              f"{r.importances_std[i]:.3f}")

    score = automl.score(X_train, y_train)
    print(f"Train score {score}")
    score = automl.score(X_test, y_test)
    print(f"Test score {score}")
    predictions = automl.predict(X_test)
    print("f1 score", sklearn.metrics.f1_score(y_test, predictions))
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))


    X_train_adult, X_test_adult, y_train_adult, y_test_adult = adult_dataset()
    X_big_adult = featurize(X_train_adult, y_train_adult, X_test_adult)
    predictions_adult = automl.predict(X_big_adult)
    print("f1 score (adult_dataset)", sklearn.metrics.f1_score(y_test_adult, predictions_adult))
    print("Accuracy score adult_dataset", sklearn.metrics.accuracy_score(y_test_adult, predictions_adult))





