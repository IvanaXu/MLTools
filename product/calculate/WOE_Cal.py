# -*-coding:utf-8-*-
# @auth LYY ivan
# @time 2017年6月12日20:58:19
# @goal test var_split
"""

    1.DecisionTreeClassifier
    2.if IV <  0.1: pass
      if IV >= 0.1: continue
    3.Sample N Times

"""
# TODO: N = 20; MayBe Can So More;
# TODO: A value is trying to be set on a copy of a slice from a DataFrame
# TODO: replace np.NAN
# ValueError: Length of values does not match length of index
# IndexError: list index out of range
# TypeError: list indices must be integers or slices, not str


import pandas as pd
import numpy as np
from sklearn import tree
from MLTools.code.develop.calculate.Cal import CAL
from MLTools.code.develop.drawing.Draw_Tree import draw_decision_tree as dt


class IV(CAL):
    def __init__(self, x, y, times=5, spe=None, min_portion=0.05, ivs=0.1):
        """
        #
        :param x: dataX
        :param y: dataY
        :param times: run_times
        :param spe: Special Value, Like [-99000792, -99000784, -99000776]
        :param min_portion: Min Potion
        [1, 2, 3] min_portion = 0.33 >> [1]
        :return: split_result
        """
        super(IV, self).__init__()
        self.x = x
        self.y = y

        self.times = times
        if not spe:
            self.spe = []
        else:
            self.spe = spe

        self.column_names = self.x.describe()
        self.min_portion = min_portion
        self.ivs = ivs
        self.cut = None

        self.result = pd.DataFrame()

        for i in self.column_names:
            result_each = self.single_var_split(column=i)
            self.result = pd.concat([self.result, result_each])

    def single_var_split(self, column):
        x_temp = pd.Series(self.x[column])

        ord_special = x_temp.isnull() | x_temp.isin(self.spe)

        x_special = x_temp[ord_special]
        y_special = self.y[ord_special]

        x_normal = x_temp[-ord_special]
        y_normal = self.y[-ord_special]

        # count |0|1|, [normal] concat [special]
        result = self.cross_table(x=x_normal, y=y_normal)
        result = pd.concat([
            result,
            pd.crosstab(x_special, y_special)
        ])

        # 整体IV值计算
        result, r_iv, woe = self.iv_cal(pd.DataFrame(result))

        if not self.cut:
            # cut == None
            pass
        elif self.cut == 'DROP':
            self.x.drop(column, axis=1, inplace=True)
        else:
            # TODO: CHANGED NOT ORDER BY !!!
            # split_x = pd.DataFrame(pd.cut(x_temp, bins=self.cut, include_lowest=True))
            # # 特殊值
            # split_x.update(x_special)
            # self.x.update(split_x)
            # self.x.replace(to_replace={column: dict(woe)}, inplace=True)
            pass

        result['Name'] = column
        result['IV'] = r_iv

        result = result[['Name', 'min', 'Count', 'Good', 'Bad',
                         'Good_Rate', 'Bad_Rate', 'WOE', 'IVi', 'IV']]
        # 排序，并替代原数据集
        result.sort_values(by='min', inplace=True)

        return result

    def cross_table(self, x, y):
        if np.unique(x).__len__() != 1:
            t_cross_table, t_split_x = \
                self.decision_tree(x, y, self.min_portion)
            # TODO:
            t_cross_table, t_iv, _ = self.iv_cal(t_cross_table)

            # 判断是否单调，且有区分能力
            spear_man = self.spear_man(t_cross_table[['min', 'Bad_Rate']])

            if spear_man == 1:
                print('X|%s, Run_Times=%d, Spear_Man=%.4f' % (x.name, 0, 1))
                cut = list(t_cross_table['min'])
                cut.sort()
                cut[0] = min(x)
                cut.append(max(x))
                self.cut = list(np.unique(cut))

            if spear_man != 1:
                if t_iv >= self.ivs:
                    # IV值大于阀值，重新抽样
                    self.cut = self.sample(x=x, y=y)
                    t_split_x = np.array(pd.cut(x, bins=self.cut, include_lowest=True))
                else:
                    self.cut = 'DROP'
            return pd.crosstab(t_split_x, y)
        else:
            # 整变量仅一个值，无需进行分箱
            self.cut = None
            return pd.crosstab(x, y)

    def sample(self, x, y):
        cross_table_all = []
        spear_man, spear_man_max, i = 0, 0, 0

        for i in range(0, self.times):
            data_1 = pd.DataFrame({'x': x, 'target': y})

            # 抽样
            data_1_0 = data_1[data_1['target'] == 0]
            data_1_1 = data_1[data_1['target'] == 1]
            min_sample = min(len(data_1_0), len(data_1_1))

            if min_sample == 0:
                # (pd.DataFrame a) a.sample, a must be greater than 0
                if len(data_1_0) == 0:
                    data_all = data_1_1
                elif len(data_1_1) == 0:
                    data_all = data_1_0
            else:
                data_all = pd.DataFrame(pd.concat(
                    [data_1_0.sample(n=min_sample, replace=True),
                     data_1_1.sample(n=min_sample, replace=True)]))

            # 分段
            x_temp = data_all['x']
            y_temp = data_all['target']

            cross_table_temp, split_x_temp = \
                self.decision_tree(x_temp, y_temp, self.min_portion)

            name1 = max(cross_table_temp.keys())

            cross_table_temp['Bad_Rate'] = \
                cross_table_temp[name1]/np.sum(cross_table_temp, axis=1)
            cross_table_temp['min'] = \
                [float(j.strip('()[]').split(',')[0])
                 if isinstance(j, str) else j for j in cross_table_temp.index]
            cross_table_temp.sort_values(by='min', inplace=True)

            # 秩相关，判断单调
            spear_man = self.spear_man(cross_table_temp[['min', 'Bad_Rate']])

            if spear_man > spear_man_max:
                spear_man_max = spear_man
                cross_table_all = cross_table_temp
                if spear_man_max == 1:
                    break

        print('X|%s, Run_Times=%d, Spear_Man=%.4f' % (x.name, i, spear_man_max))
        new_cuts = list(cross_table_all['min'])
        new_cuts[0] = min(x)
        new_cuts.append(max(x))

        return list(np.unique(new_cuts))

    @staticmethod
    def decision_tree(x, y, min_portion):
        """
        DecisionTree
        :param x: dataX
        :param y: dataY
        :param min_portion: int, Like 0.05
        :return:
        """
        tree_clf = tree.DecisionTreeClassifier(
            max_leaf_nodes=5,
            min_samples_split=max(round(3*min_portion * len(y)), 2),
            min_samples_leaf=max(round(min_portion * len(y)), 1),
            class_weight='balanced')
        tree_fit = tree_clf.fit(X=np.mat(x).T, y=y)

        t = tree_fit.tree_.threshold[tree_fit.tree_.feature != -2]
        cuts = np.unique(np.r_[t, [min(x), max(x)]])
        cuts = cuts[cuts >= min(x)]
        cuts.sort()

        if list(t) == [] and list(cuts) == [0]:
            split_x = x
        else:
            split_x = np.array(pd.cut(x, bins=cuts, include_lowest=True))
        cross_table = pd.crosstab(split_x, y)

        return cross_table, split_x

    @staticmethod
    def iv_cal(result):
        """
        # 计算IV值
        :param result: pd.crosstab(), Like Count |0|1|
        :return: result
        """
        # 避免抽样的时候 特殊值只存在于一种形式的y值
        result.fillna(0, inplace=True)

        key = np.sort(result.keys())
        name0 = key[0]
        r0 = result[name0]
        name1 = key[1]
        r1 = result[name1]

        # 若分段无数值, drop掉
        if sum((r0 + r1 == 0)) >= 1:
            zero_split = result.index[(r0 + r1 == 0)]
            result = result.drop(labels=zero_split[0], axis=0)

            key = np.sort(result.keys())
            name0 = key[0]
            r0 = result[name0]
            name1 = key[1]
            r1 = result[name1]

        good_self_rate = r0/np.sum(result)[name0]
        bad_self_rate = r1/np.sum(result)[name1]

        result['min'] = [float(j.strip('()[]').split(',')[0])
                         if isinstance(j, str) else j for j in result.index]
        result['Count'] = r0 + r1
        result['Good_Rate'] = r0/(r0 + r1)
        result['Bad_Rate'] = r1/(r0 + r1)

        result['WOE'] = [-np.inf if not j else np.log(j)
                         for j in good_self_rate / bad_self_rate]

        result['IVi'] = (good_self_rate - bad_self_rate) * result['WOE']
        r_iv = sum(result['IVi'].replace(-np.inf, 0).replace(np.inf, 0))

        result.rename(columns={name0: 'Good', name1: 'Bad'}, inplace=True)

        woe = result['WOE'].replace(-np.inf, 0).replace(np.inf, 0)
        return result, r_iv, woe

    @staticmethod
    def spear_man(cross_table):
        return abs(cross_table.corr(method='spearman').ix[0, 1])

