# -*-coding:utf-8-*-
# @auth ivan
# @time 2017年6月8日08:41:01
# @goal cal_woe
import numpy as np
import pandas as pd
from sklearn import tree
import time
import utils
from load_data import Load
from cal.WOE_Cal import IV as woe


class Cal_WOE:
    def __init__(self, x, y, times=5):
        self.dataX, self.dataY = x, y
        self.dataR = None
        self.times = times
        self.run()

    def run(self):
        try:
            time0 = time.time()
            print('BEGIN _________________________')
            w = woe(self.dataX, self.dataY, self.times)
            self.dataR = w.x

            print(self.dataR.describe())
            print(w.result)
            print('_USE_TIME_', time.time() - time0)
            print('_END_ _________________________')
        except Exception as e:
            print(e.args)

