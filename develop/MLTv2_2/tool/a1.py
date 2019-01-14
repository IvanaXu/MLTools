# -*-coding:utf-8-*-
# @auth ivan
# @time 2017年6月21日08:40:02
# @goal test a1
import pandas as pd
import matplotlib.pyplot as plot
log = 'G:\\OUT\\07.AI\\MachineLearning\\out\\20170619\\output.log'
run_times = []

with open(log, 'r', encoding='utf-8') as f:
    a = f.readline()
    while a:
        if 'Run_Times' in a:
            t = a.strip('\n').split(' = ')
            run_times.append(t[1])
        a = f.readline()

run_times = pd.Series(run_times, dtype=int)
run_times.sort_values(inplace=True)
# run_times.sort_values(by='times', inplace=True)
# print(run_times)
run_times0 = run_times[run_times < 9999]
print(run_times0)
plot.draw(run_times0)
plot.show()

