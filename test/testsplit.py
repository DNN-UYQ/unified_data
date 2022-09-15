from sklearn.model_selection import train_test_split
import numpy as np
from Preprocessing import data_sampling
from Clf_list import classfier

from Preprocessing import data_download
X_adult, y_adult = data_download(data_id=[179])



X_train_adult, X_test_adult, y_train_adult, y_test_adult = train_test_split(X_adult[0], y_adult[0], test_size=0.33,
                                                        random_state=42, stratify=y_adult[0])
X_sample, y_sample = data_sampling(X_train_adult, y_train_adult)
#print(X_sample.shape)
#print(y_sample.shape)
clf_list = classfier()
X_big = []
X_test_new = np.zeros((100, len(clf_list)))
for clf_i in range(len(clf_list)):
    clf_list[clf_i].fit(X_sample, y_sample)
    X_test_new[:, clf_i] = clf_list[clf_i].predict_proba(X_test_adult[:100])[:, 0]
X_big.append(X_test_new)
X_big = np.array(X_big).reshape((len(X_big) * 100, 3))
print(X_big)

