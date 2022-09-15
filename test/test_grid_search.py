from sklearn.model_selection import GridSearchCV
from Print_grid_cv_results import print_grid_cv_results
from sklearn.neighbors import KNeighborsClassifier
from Adult_dataset import adult_dataset
from Preprocessing import data_sampling
from Clf_list import classfier, params
X_train_adult, X_test_adult, y_train_adult, y_test_adult = adult_dataset()
print (X_train_adult.shape)
print(y_train_adult.shape)
print(X_test_adult.shape)
print(y_test_adult.shape)
clf_list = classfier()
param_list = params()
for i in range(len(clf_list)):
    grid = GridSearchCV(clf_list[i], param_list[i])
    grid_result = grid.fit(X_train_adult, y_train_adult)
    print_grid_cv_results(grid_result)



"""#def grid_search (clf, param, X_train, y_train):
    

    grid = GridSearchCV(clf, param, cv=3)
    grid_result = grid.fit(X_train, y_train)
    
    print_grid_cv_results(grid_result)
#    return grid_result
"""


