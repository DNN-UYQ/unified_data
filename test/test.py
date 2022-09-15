from Preprocessing import data_download, unified_data, data_sampling

data_id = [31, 1464]
X_list, y_list = data_download(data_id)
X_train, y_train = data_sampling(X_list[0], y_list[0])
print(X_train.shape, y_train.shape)
X_big, y_big = unified_data(X_list, y_list)
print("xbig", X_big)
print("y_big", y_big)
#print(X_big)
#print(y_big)
