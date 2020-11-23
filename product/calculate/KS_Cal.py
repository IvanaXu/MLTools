# -*-coding:utf-8-*-
# @auth LYY ivan
# @time 2017年6月3日17:13:29
# @goal ks_cal
"""
V0.1 DEMO
V1.0 CHANGE 修改程序
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def cal_ks(y, score, path, draw=False):
    # TODO： 存疑 切出来的效果怎么样
    per = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    cut0 = []
    for i in per:
        cut0.append(np.percentile(score, i))
    cut0 = np.unique(cut0)

    cuts = pd.cut(score, bins=cut0, include_lowest=True)
    ks_tab = pd.DataFrame({"target": y, "Score_Ranks": cuts, "score": score})
    groups = pd.pivot_table(data=ks_tab,
                            index='Score_Ranks',
                            columns='target',
                            aggfunc='count')
    groups.fillna(0, inplace=True)
    # print('groups', groups)

    result_ks = pd.merge(left=groups,
                         right=groups.cumsum() / groups.sum(),
                         how='inner',
                         left_index=True,
                         right_index=True)
    # print('result_ks', result_ks)

    # TODO：假定1是好的
    result_ks['number'] = result_ks['score_x'][0] + result_ks['score_x'][1]
    result_ks['Bad_Rate'] = result_ks['score_x'][0]/result_ks['number']
    result_ks['KSi'] = result_ks['score_y'][1] - result_ks['score_y'][0]
    ks = max(result_ks['KSi'])

    # 画KS曲线
    good_temp = list(result_ks['score_y'][1])
    bad_temp = list(result_ks['score_y'][0])
    good_temp.insert(0, 0)
    bad_temp.insert(0, 0)
    all_temp = pd.DataFrame({0: bad_temp, 1: good_temp})

    if draw:
        # 标出最大的KS位置
        max_ks = all_temp[1] - all_temp[0]
        result = max_ks[max_ks == max_ks.max()]

        r_index = result.index[0]
        r_min = all_temp[0][r_index]
        r_max = all_temp[1][r_index]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(all_temp[0], 'ko-', label='target =  Bad', color='blue')
        ax.plot(all_temp[1], 'ko-', label='target = Good', color='red')

        ax.set_title(label='KS')
        ax.set_xlabel('Groups')
        ax.set_ylabel('CumSum')
        ax.legend(loc='best')

        ax.text(r_index, (r_max + r_min) / 2, str(round(max(max_ks), 4) * 100) + '%',
                color='green',
                ha='left',
                fontsize=13)
        ax.annotate('',
                    xytext=(r_index, r_min),
                    xy=(r_index, r_max),
                    arrowprops=dict(arrowstyle='<->', facecolor='black'),
                    horizontalalignment='left',
                    verticalalignment='top',
                    fontsize='x-large'
                    )

    plt.savefig(path)
    return result_ks, ks

