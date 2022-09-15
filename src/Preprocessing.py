import openml
import numpy as np
from sklearn.preprocessing import StandardScaler
from Clf_list import classfier, params
from sklearn.impute import SimpleImputer

from Feature_reduction import feture_reduction_pca, feture_reduction_tsne
from sklearn.metrics import accuracy_score



def data_download(data_id):
    """"Download the OpenML datasets:(ID of dataset list)"""

    X_list = []
    y_list = []
    for id in range(len(data_id)):
        dataset = openml.datasets.get_dataset(dataset_id=data_id[id])
        X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="array",
                                                                        target=dataset.default_target_attribute)
        X_scaler = StandardScaler().fit_transform(X)
        #X_tsne= feture_reduction_tsne(X_scaler)
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(X_scaler)
        X_scaler = imp.transform(X_scaler)
        #print("xscaler {X_scaler.shape}", X_scaler)
        #X_pca = feture_reduction_pca(X_scaler)
        #print("xpca{X_pca.shape}", X_pca)


        X_list.append(X_scaler)
        y_list.append(y)
        """: type X_list, y_list: list """
    return X_list, y_list




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

    return X_train, np.array(y_train)



def unified_data(X_list, y_list):
    clf_list = classfier()
    y_big = []
    X_big = []
    for i in range(len(X_list)):
        X_train, y_train = data_sampling(X_list[i], y_list[i])
        #print(type(X_train))
        X_new = np.zeros((100, len(clf_list)))
        y_big.append(y_train[100:])
        for clf_i in range(len(clf_list)):
            #print(clf_list[clf_i])
            clf_list[clf_i].fit(X_train[:100], y_train[:100])
            X_new[:, clf_i] = clf_list[clf_i].predict_proba(X_train[100:])[:, 0]
            #y_pred = clf_list[clf_i].predict_proba(X_train[100:])
            #acc = accuracy_score( y_big, y_pred)
            #print("accuracy clf ", acc, "clf:", clf_list[clf_i])
        X_big.append(X_new)


    X_big = np.array(X_big).reshape((len(X_big) * 100, 3))
    y_big = np.array(y_big).reshape((len(y_big*100, )))

    #X_big = np.array([np.array(x) for x in X_big])
    #y_big = np.array([np.array(x) for x in y_big])


    #X_big= X_big.reshape(X_big.shape[100:])
    #y_big= y_big.reshape(y_big.shape[100:])
    #X_big = np.squeeze(X_big).shape
    #y_big = np.squeeze(y_big).shape

    return (X_big, y_big)

