X_train_positive.append(X_list[random_id_positive[i]])
X_list = np.delete(X_list, random_id_positive[i], axis=0)
y_train.append(y_list[random_id_positive[i]])
y_list = np.delete(y_list, random_id_positive[i])
X_train_negative.append(X_list[random_id_negative[i]])

X_list = np.delete(X_list, random_id_negative[i], axis=0)
y_train.append(y_list[random_id_negative[i]])
y_list = np.delete(y_list, random_id_negative[i])
