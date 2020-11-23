# -*-coding:utf-8-*-
# @auth ivan
# @time 2018-01-16 14:29
# @goal config
from logging import NOTSET

__all__ = [
    'main_path_data',
    'main_path_out',
    'main_path_test',
    'data_Name',
    'data_Tame',
    'data_type',
    'out_Name',
    'out_type',
    'size',
    'times',
    'deletions',
    'data',
    'data_T',
    'pro_gra_dot',
    'min_cpu',
    'log_mode',
    'log_level',
    'log_format',
    'log_encode'
]

main_path_data = u'G:\\OUT\\07.AI\\MLTools\\data\\'

main_path_out = u'G:\\OUT\\07.AI\\MLTools\\out\\'

main_path_test = u'G:\\OUT\\07.AI\\MLTools\\out\\test\\'

data_Name = u'm1m2_sample'

data_Tame = u'm1m2_sample_T200(MISS)'

data_type = u'.csv'

out_Name = u'fields'

out_type = u'.log'

size = 0.05

times = 5

deletions = {
    'R_POS_CNT_16_Pct_Avg_POS_CNT_1N': [-99000792.0, 85],
    'R_Con_Incs_in_INC_Pay_P_BAL': []
}

special = [-99000792, -99000784, -99000776]

data = main_path_data + data_Name + data_type

data_T = main_path_data + data_Tame + data_type

pro_gra_dot = 'C:\\Program Files (x86)\\Graphviz2.38\\bin\\dot.exe'

min_cpu = 2

log_mode = 'w'

log_level = NOTSET

log_format = '%(levelname)s| %(message)s'

log_encode = 'utf-8'

