# -*-coding:utf-8-*-
# @auth ivan
# @time 2017年5月20日10:58:18
# @goal feature_explore
# TODO: 后续将在缺失比例给定的情况下进行删除该变量的处理。

# 变量   R_POS_CNT_16_Pct_Avg_POS_CNT_1N
# 特殊值 -99000792.0(真特殊值) 和 85(假特殊值)

# 变量1   R_POS_CNT_16_Pct_Avg_POS_CNT_1N
# 变量2  R_Con_Incs_in_INC_Pay_P_BAL
# 缺失值 变量1(6个缺失值)|变量2(2个缺失值)
import pandas as pd
import matplotlib.pyplot as plt

import utils
from load_data import Load

N = 0


class Explore(object):
    def __init__(self, data, deletions):
        self.deletions = deletions
        self.L = Load(data)
        self.dataX = self.L.dataX
        self.dataY = self.L.dataY
        self.fields = {}
        self.run_all = self.dataX.describe()
        self.result = pd.Series()

        self.path = utils.out_path + '/' + utils.get_time() + '/'
        self.log = utils.mkdir(self.path) + utils.out_Name + \
                   '_' + utils.randoms(4)+utils.out_type

    def info(self):
        pass

    def split(self):
        pass

    def explorer(self):
        pass

    def writer(self, result):
        pass

    def draw_box(self, name):
        pass


class FExplore(Explore):
    def __init__(self, data, deletions):
        # Explore.__init__(self)
        super(FExplore, self).__init__(data, deletions)

        self.fields_numS = []
        self.fields_numN = []
        self.fields_strA = []

    def info(self):
        # 输出基本信息
        col = self.dataX.shape[1]
        return 'NUM:', col

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
        print('LOG :', self.log)
        # 给定变量的特殊值/缺失值，进行隔离处理后统计数据分布。
        for i in self.fields:
            if not self.fields[i]:
                # 字符类型
                self.result = pd.value_counts(self.dataX[i])
                # self.result = self.dataX[i].describe() # 在字符类型上有区别

            elif self.fields[i] == 1:
                # 速度测试结果 数值类型 无特殊值
                # TODO：要不要考虑缺失值
                self.result = self.run_all[i]

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

            self.write(pd.Series(self.result))

    def write(self, result):
        self.writer(result.name + ' ' + str(result.dtype), write=N)
        for j in range(0, len(result)):
            self.writer(str(result.keys()[j]) + ':' + str(result[j]), write=N)
        self.writer('', write=N)

    def draw_box(self, name):
        try:
            pd.DataFrame(self.dataX[name]).boxplot(return_type='axes')
            plt.savefig(self.path + '/' + name + '.jpg')
        except Exception as e:
            print(str(e))

    def writer(self, string, **kwargs):
        # write 1 输出并打印日志;0 输出;不赋值 不做任何操作
        if 'write' in kwargs:
            if kwargs['write']:
                print(string)
                with open(self.log, 'a+', encoding='utf-8') as f:
                    f.write(string + '\n')
            else:
                print(string)
        else:
            pass

