from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.stats import loguniform
from Adult_dataset import adult_dataset
from time import time
from Preprocessing import data_download, unified_data, data_sampling
from sklearn.model_selection import RandomizedSearchCV


data_id = [31, 1464]
X_list, y_list = data_download(data_id)
for i in range(len(X_list)):
    X_train, y_train = data_sampling(X_list[i], y_list[i])


names = [
    "Nearest Neighbors",
    "Linear SVM",
    "Decision Tree",
    "Random Forest",

]

classifiers = [
    KNeighborsClassifier(),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),

]



...
# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define search space
space_knn = dict()
space_knn['n_neighbors'] = ['1','2', '3','4', '5', '6', '7', '8']
space_knn['weights'] = ['unified', 'distance']
#svc
space_svc = dict()
space_svc['kernel'] = ['poly', 'rbf', 'sigmoid']
space_svc['c'] = ['50', '10', '1.0', '0.1', '0.01']
space_svc['gamma', 'auto'] = ['scale', ]
#decition tree
space_dt = dict()
space_dt['criterion'] = ['gini', 'entropy', 'log_loss']
space_dt['splitter'] = ['best', 'random']
#randomforest
space_rf = dict()
space_rf['n_estimators'] = ['10', '50', '100']
space_rf['criterion'] = ['gini', 'entropy', 'log_loss']
space = [space_knn, space_svc, space_dt, space_rf]
#GradientBoostingClassifier
"""model = GradientBoostingClassifier()
n_estimators = [10, 100, 1000]
learning_rate = [0.001, 0.01, 0.1]
subsample = [0.5, 0.7, 1.0]
max_depth = [3, 7, 9]
"""


"""space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = loguniform(1e-5, 100)"""




"""gamma = ['scale']
grid = dict(kernel=kernel,C=C,gamma=gamma)
list_hyperparam = [space_knn, space_svc, space_gp, ]"""



# define search
search = RandomizedSearchCV(classifiers, space, n_iter=10, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)

start = time()


...
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)




"""clf = []
for i in range(100):
    print(i)
    #clf.append(generate_random_model_and_hyperparameters())
"""