# -*-coding:utf-8-*-
# @auth ivan
# @time 2017年5月23日22:49:59
# @goal classifier


# 定义算法选择函数
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


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=10)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(probability=True)
    model.fit(train_x, train_y)
    return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model


# n_estimators 在比较低的值也取得了最好模型 随后该值越大整体结果不变
def gradient_boosting_classifier_good(train_x, train_y):
    a, b = [], []
    for i in range(1, 250):
        if i < 250:
            a.append(i)
        if i < 10:
            b.append(i)

    from sklearn.grid_search import GridSearchCV
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier()
    param_grid = {
        'n_estimators': a,
        'max_depth': b,
    }
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    """
    D:\Python35\lib\site-packages\sklearn\grid_search.py:553:
    UserWarning: Multiprocessing backed parallel loops cannot be nested below threads,
    setting n_jobs=1 for parameters in parameter_iterable
    """
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


# Logistic Regression Classifier
def logistic_regression_classifier_other(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
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


# Random Forest Classifier
def random_forest_classifier_other(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    a = []
    for i in range(1, 250):
        a.append(i)
    param_grid = {
        'n_estimators': a
    }
    grid_search = GridSearchCV(model, param_grid, n_jobs=100, verbose=1)
    # /data/python3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py:540:
    # UserWarning: Multiprocessing backed parallel loops cannot be nested below threads, setting n_jobs=1
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)

    model = RandomForestClassifier(
        n_estimators=best_parameters['n_estimators']
    )
    model.fit(train_x, train_y)
    return model

