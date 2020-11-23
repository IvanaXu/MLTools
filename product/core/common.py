# -*-coding:utf-8-*-
# @auth ivan
# @time 2018-01-17 17:59
# @goal Base Common
import time
import pandas as pd
from sklearn import metrics
from sklearn.externals import joblib
from MLTools.code.product.arithmetic.classifier import *
from MLTools.code.product.calculate import KS_Cal, WOE_Cal
from MLTools.code.product.util.tools import cut_split, g1_do, s1_do, directing
from MLTools.code.product.core.base import BaseRun


class Run(BaseRun):
    def __init__(self, data, size=1, train_size=0.7, arith='2345', deletions=None, path=None, typed='model'):
        super(Run, self).__init__(data, size, train_size, arith, deletions, path, typed)

    def load(self):
        self.print_log('DATA: %s' % self.data)

        if self.data.endswith('.csv'):
            data_temp = pd.read_csv(self.data, encoding='gbk')

        elif self.data.endswith('.xls') \
                or self.data.endswith('.xlsx') \
                or self.data.endswith('xlsm'):
            data_temp = pd.read_excel(self.data)

        else:
            raise TypeError('Can Not Load The Data Type.')

        self.print_log('_______________dataB_______________')
        self.print_log('dataB Shape: %s' % str(data_temp.shape))

        data_temp = data_temp[(data_temp['target'] == 1) | (data_temp['target'] == 0)]

        if self.size == 1:
            self.dataX, self.dataY = data_temp.drop('target', axis=1), data_temp['target']
        else:
            self.dataX, _, self.dataY, _ = \
                cut_split(data_temp.drop('target', axis=1), data_temp['target'], self.size)

    def split(self):
        # 切分变量数据类型及是否特殊
        for i in self.dataX.describe():
            if i in self.deletions.keys():
                # 置变量 数值类型 有特殊值为-1
                self.fields.update({i: -1})
                self.fields_numS.append(i)
            else:
                # 置变量 数值类型 无特殊值为 1
                self.fields.update({i: 1})
                self.fields_numN.append(i)

        for i in self.dataX.columns:
            if i not in self.fields.keys():
                # 置变量 字符类型 为0
                self.fields.update({i: 0})
                self.fields_strA.append(i)

    def explorer(self):
        run_all = self.dataX.describe()

        # 给定变量的特殊值/缺失值，进行隔离处理后统计数据分布。
        for i in self.fields:
            if not self.fields[i]:
                # 字符类型
                self.result = pd.value_counts(self.dataX[i])
                # self.result = self.dataX[i].describe()

            elif self.fields[i] == 1:
                # 速度测试结果 数值类型 无特殊值
                # TODO：要不要考虑缺失值
                self.result = run_all[i]

            elif self.fields[i] == -1:
                # 数值类型 有特殊值/缺失值

                i_value = self.deletions[i]
                i_data = pd.DataFrame(self.dataX[i])

                # LYY BEGIN
                i_data1 = self.dataX[i][self.dataX[i].isin(i_value) |
                                        self.dataX[i].isnull()]
                result = pd.value_counts(i_data1, dropna=False)
                # LYY END

                # TODO: 加一句 - 负判断就可以搞定
                for j in i_value:
                    i_data = i_data[i_data[i] != j]
                i_data = i_data.dropna()
                self.result = result.append(i_data.describe()[i])

            else:
                self.result = Exception('ERROR')

            r = pd.Series(self.result)
            self.print_log(r.name + ' ' + str(r.dtype))
            for j in range(0, len(r)):
                self.print_log(str(r.keys()[j]) + ':' + str(r[j]))
            self.print_log('')

    def g1_change(self, name, string, con):
        # G1 归一化
        # con = 1(不含特殊值和缺失值)/con = 0(含特殊值和缺失值)
        if con:
            # 归一化G1
            # 数值类型 无特殊值 fields_numN
            self.dataT = self.dataX.loc[:, name]
            self.dataT = self.dataT.apply(g1_do)
            for j in self.dataT:
                self.dataR[string + j] = self.dataT[j]
        else:
            # 归一化G1
            # 数值类型 有特殊值 fields_numS
            self.dataT = self.dataX.loc[:, name]
            for i in name:
                i_value = self.deletions[i]

                self.changes_values(0, i, i_value)

                i_data = self.dataT[i].replace(i_value, [self.changes] * len(i_value))
                i_data = pd.DataFrame(i_data.fillna(float(self.changes)))
                data_r = i_data.apply(g1_do)

                for j in data_r:
                    self.dataR[string + j] = data_r[j]

    def s1_standard(self, name, string, con):
        # S1 标准化
        # con = 1(不含特殊值和缺失值)/con = 0(含特殊值和缺失值)
        if con:
            # 标准化S1
            # 数值类型 无特殊值 fields_numN
            self.dataT = self.dataX.loc[:, name]
            self.dataT = self.dataT.apply(s1_do)
            for j in self.dataT:
                self.dataR[string + j] = self.dataT[j]
        else:
            # 标准化S1
            # 数值类型 有特殊值 fields_numS
            self.dataT = self.dataX.loc[:, name]
            for i in name:
                i_value = self.deletions[i]

                self.changes_values(0, i, i_value)

                i_data = self.dataT[i].replace(i_value, [self.changes] * len(i_value))
                i_data = pd.DataFrame(i_data.fillna(float(self.changes)))
                data_r = i_data.apply(s1_do)

                for j in data_r:
                    self.dataR[string + j] = data_r[j]

    def o1_one_hot(self, name, string):
        # O1 ONE HOT
        self.dataT = self.dataX.loc[:, name]
        # 字符型若存在缺失值 取NAN作为一列
        self.dataT = pd.get_dummies(self.dataT, dummy_na=True)

        for i in self.dataT:
            self.dataT[string + self.dataT[i].name] = self.dataT[i]

        # ONE_HOT 之后删除字符型
        self.dataR = self.dataR.drop(name, axis=1)

    def x1_across(self, name, string):
        # X1 变量交叉
        self.dataT = self.dataX.loc[:, name]
        # 字符型若存在缺失值 取NAN作为一列
        self.dataT = pd.get_dummies(self.dataT, dummy_na=True)

        # {'A': ['A_1', 'A_2', 'A_3']}
        dict_str = {}
        for i in self.dataT.keys():
            for j in name:
                if j not in dict_str.keys():
                    dict_str.update({j: []})

                if i.startswith(j):
                    dict_str[j].append(i)
                    break

        for i in directing(name):
            t1 = dict_str[i[0]]
            t2 = dict_str[i[1]]

            for j in t1:
                a1 = self.dataT[j]
                a2 = self.dataT[t2]

                for k in a2:
                    name_n = string + str(a1.name) + str(a2[k].name)
                    self.dataR[name_n] = a1 * a2[k]

    def type_num(self):
        # 数值类型 处理逻辑控制
        # 区分数值类型是否有特殊值 进行归一化/标准化

        if self.fields_numN:
            # G1 归一化
            self.g1_change(self.fields_numN, self.G1, 1)
            # S1 标准化
            self.s1_standard(self.fields_numN, self.S1, 1)
        else:
            self.print_log('__无含正常值数值类型__', 'warning')

        if self.fields_numS:
            # G1 归一化
            self.g1_change(self.fields_numS, self.G1, 0)
            # S1 标准化
            self.s1_standard(self.fields_numS, self.S1, 0)
        else:
            self.print_log('__无含特殊值数值类型__', 'warning')

        # TODO：强制转换nan为0，待处理
        self.dataR = self.dataR.fillna(0)

    def type_str(self):
        # 字符类型 处理逻辑控制
        # 字符类型进行one_hot/变量交叉

        if self.fields_strA:
            # O1 ONE HOT
            self.o1_one_hot(self.fields_strA, self.O1)
            # X1 变量交叉
            self.x1_across(self.fields_strA, self.X1)
        else:
            self.print_log('__无字符类型__', 'warning')

    def changes_values(self, replace, i, i_value):
        if isinstance(replace, int):
            # 数值替换
            self.changes = replace
        elif replace == 'mean':
            # 平均值替换
            i_data = pd.DataFrame(self.dataT[i])
            for j in i_value:
                i_data = i_data[i_data[i] != j]
            i_data = i_data.dropna()
            self.changes = i_data.mean()
        else:
            self.changes = 0

    def cal_woe(self, runtime, spe):
        try:
            time0 = time.time()
            self.print_log('******************* %s ********************' % 'CAL WOE BEGIN')
            # w = WOE_Cal.IV(self.dataX[self.dataX.columns[1500:]], self.dataY, runtime, spe=spe)
            # w = WOE_Cal.IV(self.dataX, self.dataY, runtime, spe=spe)
            w = WOE_Cal.IV(self.dataX, self.dataY, runtime, spe=spe)
            self.dataR = w.x

            self.print_log(str(self.dataR.describe()))
            self.dataR.describe().to_csv(self.save_path+'woe_describe.csv')
            self.print_log(str(w.result))
            w.result.to_csv(self.save_path+'woe_result.csv')

            self.print_log('USE TIME: %.4f' % (time.time()-time0))
            self.print_log('******************* %s ********************' % 'CAL WOE END')
        except Exception as e:
            self.print_log(str(e.args))

    def model(self, classifier):
        name = classifier.__name__
        train_ks, test_ks = 0, 0
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score = 0, 0, 0, 0, 0
        model = None

        try:
            self.print_log('******************* %s ********************' % name)

            # TRAIN
            time0 = time.time()
            model = classifier(self.X_train, self.y_train)
            train_time = time.time() - time0
            self.print_log('TRAIN TOOK %.4fs!' % train_time)

            predict_train = model.predict_proba(self.X_train)
            self.print_log('TRAIN KS______________________________________BEG')
            result_ks, train_ks = \
                KS_Cal.cal_ks(self.y_train, predict_train[:, 0], self.pict_path+'train_'+name+'.jpg', draw=True)
            self.print_log(str(result_ks))
            self.print_log('TRAIN KS值: %.2f%%' % (train_ks * 100))
            self.print_log('TRAIN KS______________________________________END')

            # TEST
            time0 = time.time()
            predict = model.predict(self.X_test)
            predict_tests = model.predict_proba(self.X_test)

            self.print_log('Y_PREDICT')
            self.percent_zero(predict)
            test_time = time.time() - time0
            self.print_log('TEST TOOK %.4fs!' % test_time)

            self.print_log('TEST KS______________________________________BEG')
            result_ks, test_ks = \
                KS_Cal.cal_ks(self.y_test, predict_tests[:, 0], self.pict_path+'test_'+name+'.jpg', draw=True)
            self.print_log(str(result_ks))
            self.print_log('TEST KS值: %.2f%%' % (test_ks * 100))
            self.print_log('TEST KS______________________________________END')

            # 精度
            accuracy_score = metrics.accuracy_score(self.y_test, predict)
            self.print_log('ACCURACY SCORE: %.4f' % accuracy_score)

            # 准确率
            precision_score = metrics.precision_score(self.y_test, predict)
            self.print_log('PRECISION SCORE: %.4f' % precision_score)

            # 召回率
            recall_score = metrics.recall_score(self.y_test, predict)
            self.print_log('RECALL SCORE: %.4f' % recall_score)

            # F1
            f1_score = metrics.f1_score(self.y_test, predict)
            self.print_log('F1 SCORE: %.4f' % f1_score)

            # AUC
            roc_auc_score = metrics.roc_auc_score(self.y_test, predict)
            self.print_log('ROC AUC SCORE: %.4f\n' % roc_auc_score)

            # save
            self.print_log('SAVE TRAIN\TEST PREDICT CSV.')
            pd.DataFrame(predict_train).to_csv(self.save_path+'predict_train.csv')
            pd.DataFrame(predict_tests).to_csv(self.save_path+'predict_tests.csv')

        except Exception as e:
            self.print_log('ERROR_____________________________ERROR')
            self.print_log(str(e.args))
            self.print_log('ERROR_____________________________ERROR')

        self.result_list.append({
            'name': name,
            'train_ks': train_ks,
            'test_ks': test_ks,
            'accuracy_score': accuracy_score,
            'precision_score': precision_score,
            'recall_score': recall_score,
            'f1_score': f1_score,
            'roc_auc_score': roc_auc_score,
            'model': model
        })
        return self

    def select(self):
        all_classifiers = {
            '0': naive_bayes_classifier,
            '1': knn_classifier,
            '2': decision_tree_classifier,
            '3': logistic_regression_classifier,
            '4': random_forest_classifier,
            '5': gradient_boosting_classifier,
            '6': svm_classifier,
            #
            '7': logistic_regression_classifier_grid_search,
            '8': random_forest_classifier_grid_search,
            '9': gradient_boosting_classifier_grid_search,
            'A': svm_classifier_grid_search
        }
        self.classifiers = [all_classifiers[i] for i in self.arith]

    def classifier_run(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            cut_split(self.dataX, self.dataY, size=self.train_size)

        self.print_log('SAVE TRAIN\TEST CSV.')
        self.X_train.to_csv(self.save_path+'X_train.csv')
        self.y_train.to_csv(self.save_path+'y_train.csv')
        self.X_test.to_csv(self.save_path+'X_test.csv')
        self.y_test.to_csv(self.save_path+'y_test.csv')

        self.print_log('Y_TRAIN:')
        self.percent_zero(self.y_train)
        self.print_log('Y_TEST:')
        self.percent_zero(self.y_test)

        self.select()
        for i in self.classifiers:
            self.model(i)

        self.print_log('******************* %s ********************' % 'RESULT')

        best_precision_score, best_model, best_classifier, best_ks, best_score = \
            0, None, '', 0, 0

        self.print_log('|' + '-' * 50 + '|' + '-' * 64 + '|' + '-' * 8 + '|')
        self.print_log('|                    CLASSIFIER                    |'
                       ' TRAIN KS| TEST KS |ACCURACY|PRECISE | RECALL |   F1   |   AUC  |  SCORE |')
        self.print_log('|' + '-' * 50 + '|' + '-' * 64 + '|' + '-' * 8 + '|')
        for i in self.result_list:
            precision_score, model, name, test_ks = \
                i['precision_score'], i['model'], i['name'], i['test_ks']

            score = precision_score*0.3 + test_ks*0.7
            i['score'] = score

            if score > best_score:
                best_precision_score, best_model, best_classifier, best_ks, best_score = \
                    precision_score, model, name, test_ks, score

            self.print_log('|' + '{0: ^50}'.format(i['name'][:50]) +
                           '|%8.2f%%|%8.2f%%|%8.4f|%8.4f|%8.4f|%8.4f|%8.4f|%8.4f|'
                           % (i['train_ks'] * 100,
                              i['test_ks'] * 100,
                              i['accuracy_score'],
                              i['precision_score'],
                              i['recall_score'],
                              i['f1_score'],
                              i['roc_auc_score'],
                              i['score'] * 100)
                           )
            self.print_log('|' + '-' * 50 + '|' + '-' * 64 + '|' + '-' * 8 + '|')

        self.print_log('BEST CLASSIFIER: %s' % best_classifier)
        self.print_log('SAVE MODEL.')
        joblib.dump(best_model, self.save_path + '\\best_model.model')
        self.print_log('SAVE PATH: %s' % self.save_path)
        self.print_log('******************* %s ********************' % 'END!')

    def percent_zero(self, data):
        self.print_log('0%%: %.2f%%' % (data[data == 0].shape[0] / data.shape[0] * 100))

