# -*-coding:utf-8-*-
# @auth ivan
# @time 2017年5月11日20:31:10
# @goal load_data
import pandas
import numpy as np
from sklearn.cross_validation import train_test_split


class Load(object):
    def __init__(self, data):
        # data/data_T
        self.data = data
        # 随机矩阵作为监督初始化
        self.dataN = pandas.DataFrame(np.random.randn(1, 2))
        self.dataT, self.dataX, self.dataY = self.dataN, self.dataN, self.dataN

        self.read()

    def read(self):
        print('DATA:', self.data)
        # TODO：待解决数量过大的时候 导入报错
        # TODO：统一编码,补充其他文本数据类型
        # TODO：定义dtype的类型来实现一些重要字段 0000000038 NOT 38

        if self.data.endswith('.csv'):
            self.dataT = pandas.read_csv(self.data, encoding='gbk')

        elif self.data.endswith('.xls') \
                or self.data.endswith('.xlsx') \
                or self.data.endswith('xlsm'):
            self.dataT = pandas.read_excel(self.data)

        # TODO：SAS数据集导入异常
        # elif self.data.endswith('.sas7bdat'):
        #     self.dataT = pandas.read_sas(self.data)

        else:
            self.dataT = self.dataN

        # 多分类
        # self.dataT = self.dataT[self.dataT['target'] != 2]

        self.dataX = self.dataT.iloc[:, :-1]
        # FOR TEST
        # self.dataX = self.dataT.iloc[:, :-1][
        #     [
        #         'MONTH_NBR',
        #         'R_Pay_A_Curr',
        #         'R_PRODUCT',
        #         'R_Bal_16_Pct_Avg_Bal_1N',
        #         'R_MS_16_Pct_Avg_MS_1N'
        #     ]
        # ]
        self.dataY = self.dataT.iloc[:, -1]

        # 切割
        # self.cut_run(size=0.99)

        self.print()

    def test(self):
        print(self.dataT.dtypes)
        # 测试浮点型数据被识别的类型
        print('\n_______________Tests_______________')
        print('The Float Str To:', self.dataT.iloc[:, 2].dtype)

    def print(self):
        print('\n_______________dataT_______________')
        print(self.dataT.shape)
        print('\n_______________dataX_______________')
        print(self.dataX.shape)
        # print(self.dataX)
        print('\n_______________dataY_______________')
        print(self.dataY.shape)
        # print(self.dataY)

    def cut_run(self, size):
        self.dataX, x, self.dataY, y = \
            train_test_split(self.dataX, self.dataY, train_size=size)

# TODO: 对外开放切割的参数

