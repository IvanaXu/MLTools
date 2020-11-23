# -*-coding:utf-8-*-
# @auth ivan
# @time 2017年5月23日22:49:59
# @goal classifier
from sklearn.grid_search import GridSearchCV
from multiprocessing import cpu_count
from MLTools.code.product.config import min_cpu

__all__ = [
    # [Base Classifier]
    'naive_bayes_classifier',
    'knn_classifier',
    'decision_tree_classifier',
    'logistic_regression_classifier',
    'random_forest_classifier',
    'gradient_boosting_classifier',
    'svm_classifier',

    # [GridSearchCV]
    'logistic_regression_classifier_grid_search',
    'random_forest_classifier_grid_search',
    'gradient_boosting_classifier_grid_search',
    'svm_classifier_grid_search'
]
Num = 20
Job = min(cpu_count(), min_cpu)


# [Base Classifier]
# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10)
    model.fit(train_x, train_y)
    return model


# Gradient Boosting Decision Tree GBDTClassifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=10)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
# TODO: ERROR: ('score_x',) 核心引发出错的是SVM的输出概率
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(probability=True)
    model.fit(train_x, train_y)
    return model


# [GridSearchCV] Logistic Regression Classifier
def logistic_regression_classifier_grid_search(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()

    param_grid = {
        'solver': ['newton-cg', 'lbfgs', 'liblinear'],
        'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    }

    grid_search = GridSearchCV(model, param_grid)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)

    model = LogisticRegression(
        solver=best_parameters['solver'],
        C=best_parameters['C'],
        penalty='l2'
    )
    model.fit(train_x, train_y)
    return model


# [GridSearchCV] Random Forest Classifier
def random_forest_classifier_grid_search(train_x, train_y):
    """
    /data/python3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py:540:
    UserWarning: Multiprocessing backed parallel loops cannot be nested below threads, setting n_jobs=1
    """
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()

    t1 = []
    for i in range(1, Num):
        t1.append(i)
    param_grid = {
        'n_estimators': t1
    }

    grid_search = GridSearchCV(model, param_grid, n_jobs=Job, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)

    model = RandomForestClassifier(
        n_estimators=best_parameters['n_estimators']
    )
    model.fit(train_x, train_y)
    return model


# [GridSearchCV] Gradient Boosting Decision Tree GBDTClassifier
def gradient_boosting_classifier_grid_search(train_x, train_y):
    # TODO: n_estimators 在比较低的值也取得了最好模型 随后该值越大整体结果不变
    """
    D:\Python35\lib\site-packages\sklearn\grid_search.py:553:
    UserWarning: Multiprocessing backed parallel loops cannot be nested below threads,
    setting n_jobs=1 for parameters in parameter_iterable
    """
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier()

    t1, t2 = [], []
    for i in range(1, Num):
        if i < Num:
            t1.append(i)
        if i < Num/2:
            t2.append(i)
    param_grid = {
        'n_estimators': t1,
        'max_depth': t2,
    }

    grid_search = GridSearchCV(model, param_grid, n_jobs=Job, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)

    model = GradientBoostingClassifier(
        n_estimators=best_parameters['n_estimators'],
        max_depth=best_parameters['max_depth']
    )
    model.fit(train_x, train_y)
    return model


# [GridSearchCV] SVM Classifier
def svm_classifier_grid_search(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)

    param_grid = {
        'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
        'gamma': [0.001, 0.0001]
    }

    grid_search = GridSearchCV(model, param_grid, n_jobs=3, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)

    model = SVC(
        kernel='rbf',
        C=best_parameters['C'],
        gamma=best_parameters['gamma'],
        probability=True
    )
    model.fit(train_x, train_y)
    return model

