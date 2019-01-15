# -*-coding:utf-8-*-
# @auth ivan
# @time 2017年5月11日20:43:42
# @goal FRO THE UTILS
# TODO:写得有点累赘了的，要把功能函数和基本配置的东西在最后区分出来。
import datetime
import random
import os


def get_days():
    return datetime.datetime.now().strftime('%Y%m%d')


def get_time():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def randoms(n):
    if n > 0:
        return str(random.randint(10**n, 10**(n+1)-1))
    else:
        return str(random.randint(100000, 999999))

# csv xls sas7bdat
data_path = u'/data/project/GitHubI/MLTools/data/'
out_path = u'/data/project/GitHubI/MLTools/out/'

data_Name = u'm1m2_sample'
data_Tame = u'm1m2_sample_T200(MISS)'
# data_Tame = u'm1m2_sample_T200'
data_type = u'.csv'

# 文本型输出使用log模式
out_Name = u'fields'
out_type = u'.log'

data = data_path + data_Name + data_type
data_T = data_path + data_Tame + data_type

# 创建今天的输出文件夹
out_path += get_days()
if os.path.isdir(out_path):
    pass
else:
    print('MAKE THE PATH')
    os.system('mkdir ' + out_path)

# log = out_path+'\\'+out_Name+get_time()+'_'+randoms(4)+out_type

# 定义特殊值
# 若无特殊值，仅需处理缺失值，可赋空[]
deletions = {
    'R_POS_CNT_16_Pct_Avg_POS_CNT_1N': [-99000792.0, 85],
    'R_Con_Incs_in_INC_Pay_P_BAL': []
}


def mkdir(path):
    # 创建目录
    if os.path.isdir(path):
        pass
    else:
        os.system('mkdir ' + path)
    return path

