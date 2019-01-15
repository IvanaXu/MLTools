# -*-coding:utf-8-*-
# @auth ivan
# @time 2017年5月27日19:01:25
# @goal classifier_run
import matplotlib
matplotlib.use('Agg')

from sklearn import metrics
import numpy as np
import pandas as pd
import time
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
import gevent
from gevent import monkey
from classifier import naive_bayes_classifier
from classifier import knn_classifier
from classifier import logistic_regression_classifier
from classifier import logistic_regression_classifier_other
from classifier import random_forest_classifier
from classifier import random_forest_classifier_other
from classifier import decision_tree_classifier
from classifier import gradient_boosting_classifier
from classifier import gradient_boosting_classifier_good
from classifier import svm_classifier
from classifier import svm_cross_validation
from cal_woe import Cal_WOE
from feature_preprocessing import FPreprocessing
from cal import KS_Cal
import utils
monkey.patch_all()


class Classifier(object):
    def __init__(self, data, deletions=None, typed='model'):
        """
        :param data:
        :param deletions:
        :param types:'model' or 'scorecards'
        """

        if not deletions:
            self.deletions = {}
        else:
            self.deletions = deletions
        self.F = FPreprocessing(data, self.deletions)
        self.F.type_num()
        self.F.type_str()
        self.X, self.Y = self.F.dataR, self.F.dataY
        self.path = utils.out_path + '/' + utils.get_time()

        self.types(typed)

        # JOIN CODE
        self.Y1 = self.Y[self.Y != -1]
        self.Y2 = self.Y[self.Y == -1]

        self.X1 = self.X[self.X.index.isin(self.Y1.index)]
        self.X2 = self.X[self.X.index.isin(self.Y2.index)]

        print('THE SELF.X2 %s SIZE: %s.' %
              (str(self.X2.shape), self.X2.size))
        # JOIN CODE

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X1, self.Y1, test_size=0.4)

        self.X2.to_csv(utils.mkdir(self.path)+'/X2.csv', index=False)

        self.X_train.to_csv(utils.mkdir(self.path)+'/X_train.csv')
        self.y_train.to_csv(utils.mkdir(self.path)+'/y_train.csv')
        self.X_test.to_csv(utils.mkdir(self.path)+'/X_test.csv')
        self.y_test.to_csv(utils.mkdir(self.path)+'/y_test.csv')

        # CHANGE NEED KEY
        self.X_train = self.X_train.\
            drop('instance_id', axis=1).\
            drop('G1_instance_id', axis=1).\
            drop('S1_instance_id', axis=1)
        self.X_test = self.X_test.\
            drop('instance_id', axis=1).\
            drop('G1_instance_id', axis=1).\
            drop('S1_instance_id', axis=1)

        print('y_train')
        self.info(self.y_train)
        print('y_test')
        self.info(self.y_test)

        self.classifiers = [
            naive_bayes_classifier,
            decision_tree_classifier,
            random_forest_classifier,
            # random_forest_classifier_other,
            # TODO: ERROR: ('score_x',) 核心引发出错的是SVM的输出概率
            gradient_boosting_classifier,
            logistic_regression_classifier,

            knn_classifier,
            svm_classifier,

            # logistic_regression_classifier_other,
            # gradient_boosting_classifier_good,
            # svm_cross_validation
        ]
        self.result = []

        self.run()

    def types(self, typed):
        if typed == 'model':
            pass
        elif typed == 'scorecards':
            woe = Cal_WOE(self.X, self.Y, times=5)
            self.X = woe.dataR
        else:
            pass

    def model(self, classifier):
        name = classifier.__name__
        accuracy, ks, train_time, test_time = 0, 0, 0, 0
        model = None
        try:
            print('******************* %s ********************' % name)
            time0 = time.time()
            # TODO: 定义最佳超参数
            model = classifier(self.X_train, self.y_train)
            train_time = time.time() - time0
            print('Training took %fs!' % train_time)

            time0 = time.time()
            predict = model.predict(self.X_test)

            predict_train = model.predict_proba(self.X_train)
            pd.DataFrame(predict_train).to_csv(utils.mkdir(self.path) + '/predict_train.csv')
            predict_tests = model.predict_proba(self.X_test)
            pd.DataFrame(predict_tests).to_csv(utils.mkdir(self.path) + '/predict_tests.csv')

            print('TRAIN KS______________________________________BEG')
            result_ks, ks = KS_Cal.cal_ks(self.y_train, predict_train[:, 0], draw=True)
            print(result_ks)
            print('KS值: %.2f%%' % (ks * 100))
            print('TRAIN KS______________________________________BEG')

            print('y_predict')
            self.info(predict)
            test_time = time.time() - time0
            print('Testing took %fs!' % test_time)

            # TODO：KS 丢进去预测为好的概率
            print('TESTS KS______________________________________BEG')
            result_ks, ks = KS_Cal.cal_ks(self.y_test, predict_tests[:, 0], draw=True)
            print(result_ks)
            print('KS值: %.2f%%' % (ks * 100))
            print('TESTS KS______________________________________END')

            accuracy = metrics.accuracy_score(self.y_test, predict)
            print('Accuracy: %.2f%%' % (100 * accuracy))

        except Exception as e:
            print('ERROR_____________________________ERROR')
            print(str(e))
            print('ERROR_____________________________ERROR')

        self.result.append({
            'name': name,
            'train_time': train_time,
            'test_time': test_time,
            'accuracy': accuracy * 100,
            'ks': ks * 100,
            'model': model
        })

    def run(self):
        spawnlist = []
        for i in self.classifiers:
            self.model(i)
            # spawnlist.append(gevent.spawn(self.model, i))
        # gevent.joinall(spawnlist)

        best_accuracy = 0
        best_model = None
        best_classifier = ''
        best_ks = 0
        best_score = 0

        for i in self.result:
            accuracy = i['accuracy']
            model = i['model']
            name = i['name']
            ks = i['ks']

            score = accuracy*0.25 + ks*0.75

            if score > best_score:
                best_accuracy = accuracy
                best_model = model
                best_classifier = name
                best_ks = ks
                best_score = score

        print('******************* %s ********************' % 'END!')
        print('Best_classifier', best_classifier)
        print('Best_accuracy: %.2f%%' % best_accuracy)
        print('Best_ks: %.2f%%' % best_ks)
        print('Best_score: %.2f%%' % best_score)
        print('Best_model', best_model)
        print('******************* %s ********************' % 'SAVE')
        print('Save path', self.path)
        joblib.dump(best_model, utils.mkdir(self.path) + '/best_model'+utils.randoms(8)+'.model')

    def info(self, data):
        print('0%%: %.2f%%' % (data[data == 0].shape[0]/data.shape[0]*100))


Classifier('/data/project/GitHubI/MLTools/data/test.csv')



