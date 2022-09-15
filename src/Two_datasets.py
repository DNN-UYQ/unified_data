from Preprocessing import data_download, unified_data

def two_datasets(data_id):
    X_list,y_list = data_download(data_id)
    X_big, y_big = unified_data(X_list, y_list)

    return X_big, y_big




    """X_train, y_train, X_test, y_test = train_test_split(X_list, y_list, test_size=0.33,
                                                                                random_state=42, stratify=y_list)

    X_big = unified_data(X_train, y_train, X_test)"""




