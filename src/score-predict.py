from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold

def trainModel(model, x_train, y_train, x_test, n_folds, seed):
    cv = KFold(n_splits= n_folds, random_state=seed)
    scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    y_pred = cross_val_predict(model, x_train, y_train, cv=cv, n_jobs=-1)
    return scores, y_pred