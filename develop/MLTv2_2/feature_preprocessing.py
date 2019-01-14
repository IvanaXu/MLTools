# -*-coding:utf-8-*-
# @auth ivan
# @time 2017年5月24日21:55:39
# @goal 特征预处理 feature_preprocessing
"""
1-预处理
 -给定了缺失比例，超过缺失比例的部分会直接被舍弃。

2-特征处理
 -接着进行的诸多特征性处理。
"""
import pandas as pd
import numpy as np

from feature_explore import FExplore


class Preprocessing(object):
    def __init__(self, data, deletions):
        # TODO：这样赋值比较累赘，后续修改
        # 初始化
        self.F = FExplore(data, deletions)
        self.F.split()
        self.F.explorer()

        # 获取数据集
        self.dataX = self.F.dataX
        self.dataY = self.F.dataY
        # 缓存数据集
        self.dataT = pd.DataFrame()
        # 保存结果集
        self.dataR = self.dataX.copy()

        self.deletions = self.F.deletions
        # 变量 数值类型 有特殊值
        self.fields_numS = self.F.fields_numS
        # 变量 数值类型 无特殊值
        self.fields_numN = self.F.fields_numN
        # 变量 字符类型
        self.fields_strA = self.F.fields_strA

        self.changes = 0

        # 标签
        self.G1 = 'G1_'
        self.S1 = 'S1_'
        self.O1 = 'O1_'
        self.X1 = 'X1_'

    def del_vari(self, data, vari):
        # TODO:剔除指定变量/列
        return data.drop(vari, axis=1)

    def del_miss(self, per_miss):
        # TODO:缺失值 缺失值比例过大 直接剔除变量
        print(per_miss)

    def g1_do(self, x):
        # 归一化G1 TODO: g1_do
        if np.max(x) == np.min(x):
            # Loss_Contact9_Cnt 均是0
            return x
        else:
            return (x - np.min(x)) / (np.max(x) - np.min(x))

    def s1_do(self, x):
        # 标准化S1 TODO: s1_do
        if not np.std(x):
            # Loss_Contact9_Cnt 均是0
            return x
        else:
            return (x - np.mean(x)) / (np.std(x))

    def g1_change(self, name, string, con):
        # G1 归一化
        # con = 1(不含特殊值和缺失值)/con = 0(含特殊值和缺失值)
        if con:
            # 归一化G1
            # 数值类型 无特殊值 fields_numN
            self.dataT = self.dataX.loc[:, name]
            self.dataT = self.dataT.apply(self.g1_do)
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
                dataR = i_data.apply(self.g1_do)

                for j in dataR:
                    self.dataR[string + j] = dataR[j]

    def s1_standard(self, name, string, con):
        # S1 标准化
        # con = 1(不含特殊值和缺失值)/con = 0(含特殊值和缺失值)
        if con:
            # 标准化S1
            # 数值类型 无特殊值 fields_numN
            self.dataT = self.dataX.loc[:, name]
            self.dataT = self.dataT.apply(self.s1_do)
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
                dataR = i_data.apply(self.s1_do)

                for j in dataR:
                    self.dataR[string + j] = dataR[j]

    def o1_one_hot(self, name, string):
        # O1 ONE HOT
        self.dataT = self.dataX.loc[:, name]
        # 字符型若存在缺失值 取NAN作为一列
        self.dataT = pd.get_dummies(self.dataT, dummy_na=True)
        # SOMETIMES dataT be DataFrame

        for i in self.dataT:
            try:
                self.dataT[string + self.dataT[i].name] = self.dataT[i]
            except:
                print(i)

        # ONE_HOT 之后删除字符型
        self.dataR = self.dataR.drop(name, axis=1)

    def x1_across(self, name, string):
        # X1 变量交叉
        self.dataT = self.dataX.loc[:, name]
        # 字符型若存在缺失值 取NAN作为一列
        self.dataT = pd.get_dummies(self.dataT, dummy_na=True)

        # {'A': ['A_1', 'A_2', 'A_3']}
        dict_strA = {}
        for i in self.dataT.keys():
            for j in name:
                if j not in dict_strA.keys():
                    dict_strA.update({j: []})

                if i.startswith(j):
                    dict_strA[j].append(i)
                    break

        for i in self.directing(name):
            t1 = dict_strA[i[0]]
            t2 = dict_strA[i[1]]

            for j in t1:
                a1 = self.dataT[j]
                a2 = self.dataT[t2]

                for k in a2:
                    name_n = string + str(a1.name) + str(a2[k].name)
                    self.dataR[name_n] = a1 * a2[k]

    def type_num(self):
        # 数值类型 处理逻辑控制
        pass

    def type_str(self):
        # 字符类型 处理逻辑控制
        pass

    def changes_values(self, replace, i, i_values):
        # 替换
        pass

    def directing(self, lists):
        # 遍历出一个交叉指向说明
        # IN : ['A','B','C']
        # OUT: [('A','B'),('A','C'),('B','C')]
        dn = len(lists)
        result = []
        for di in range(0, dn - 1):
            for dj in range(di + 1, dn):
                result.append((lists[di], lists[dj]))
        return result

    def shape(self, data):
        print(data.shape)


class FPreprocessing(Preprocessing):
    def __init__(self, data, deletions):
        super(FPreprocessing, self).__init__(data, deletions)

    def type_num(self):
        # 数值类型 处理逻辑控制
        # 区分数值类型是否有特殊值 进行归一化/标准化

        if self.fields_numN:
            # G1 归一化
            self.g1_change(self.fields_numN, self.G1, 1)
            # S1 标准化
            self.s1_standard(self.fields_numN, self.S1, 1)
        else:
            print('__无含正常值数值类型__')

        if self.fields_numS:
            # G1 归一化
            self.g1_change(self.fields_numS, self.G1, 0)
            # S1 标准化
            self.s1_standard(self.fields_numS, self.S1, 0)
        else:
            print('__无含特殊值数值类型__')

        # TODO：强制转换nan为0，待处理
        self.dataR = self.dataR.fillna(0)

    def type_str(self):
        # 字符类型 处理逻辑控制
        # TODO：剔除部分不需要进行处理的字符类型
        # 字符类型进行one_hot/变量交叉

        if self.fields_strA:
            # O1 ONE HOT
            self.o1_one_hot(self.fields_strA, self.O1)
            # X1 变量交叉
            self.x1_across(self.fields_strA, self.X1)
        else:
            print('__无字符类型__')

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

