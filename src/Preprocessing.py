import openml
import numpy as np
from Clf_list import random_molde
import pandas as pd
from Numeric_transformer import transformer
from Datasets_info import get_number_features, get_number_instance, f1_score_berechnen
from sklearn.metrics import f1_score
from openml.datasets import list_datasets



def data_download(data_id):
    """"Download the OpenML datasets:(ID of dataset list)"""

    X_list = []
    y_list = []
    for id in range(len(data_id)):
        dataset = openml.datasets.get_dataset(dataset_id=data_id[id])
        X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="array",
                                                                        target=dataset.default_target_attribute)
        preproceser= transformer(categorical_indicator)
        X_transformer=preproceser.fit_transform(X, y)
        """X_scaler = StandardScaler().fit_transform(X)

        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(X_scaler)
        X_scaler = imp.transform(X_scaler)
        min =np.amax(X_scaler)
        print(min)"""


        X_list.append(X_transformer)
        y_list.append(y)
        """: type X_list, y_list: list """
    return X_list, y_list
id=[31, 1464]
X_list,y_list =data_download(id)





def data_sampling(X_list, y_list):
    positive = np.where(y_list == 1)[0]
    negative = np.where(y_list == 0)[0]
    random_id_positive = np.random.permutation(positive)
    random_id_negative = np.random.permutation(negative)
    X_train_positive = []
    X_train_negative = []
    y_train = []
    for i in range(100):
        X_train_positive.append(X_list[random_id_positive[i]])
        y_train.append(y_list[random_id_positive[i]])
        X_train_negative.append(X_list[random_id_negative[i]])
        y_train.append(y_list[random_id_negative[i]])

    X_train = np.concatenate([X_train_positive, X_train_negative], axis=0)
    X_rest= X_list
    y_rest=y_list

    return X_train, np.array(y_train)



def unified_data(X_list, y_list):
    clf_list=[]
    name_clf=[]
    for model in range(50):
        name, clf = random_molde()
        clf_list.append(clf)
        name_clf.append(name)

    y_big = []
    X_big = []
    column= clf_list+ clf_list
    for i in range(len(X_list)):
        X_train, y_train = data_sampling(X_list[i], y_list[i])
        X_new = np.zeros((100, len(clf_list)))
        y_big.append(y_train[100:])
        #score_list = np.zeros((len(clf_list),))
        for clf_i in range(len(clf_list)):

            clf_list[clf_i].fit(X_train[:100], y_train[:100])
            X_new[:, clf_i] = clf_list[clf_i].predict_proba(X_train[100:])[:, 0]
            score = f1_score_berechnen(X_list[i], y_list[i], clf_list[clf_i])
            X_new = np.insert(X_new, X_new.shape[1], values=score, axis=1)
        number_features=get_number_features(X_list[i])
        number_instances=get_number_instance(X_list[i])

        X_new=np.insert(X_new, X_new.shape[1], values=number_features, axis=1)

        X_new=np.insert(X_new, X_new.shape[1], values=number_instances, axis=1)

        X_big.append(X_new)


    X_big = np.array(X_big).reshape((len(X_big) * 100, X_new.shape[1]))
    y_big = np.array(y_big).reshape((len(y_big)*100, ))
    column.append("number_features")
    column.append("number_instances")

    return (X_big, y_big, column)





