# -*-coding:utf-8-*-
# @auth ivan
# @time 2018-01-17 15:42
# @goal Base Run
import logging
import pandas as pd
import os
from MLTools.code.product.util.tools import get_days, get_time, mk_dir, randoms
from MLTools.code.product.config \
    import main_path_out, log_mode, log_level, log_format, log_encode, times, special


class BaseRun(object):
    def __init__(self, data, size=1, train_size=0.66, arith='2345', deletions=None, path=None, typed='model'):
        """
        Use the DataFrame.describe(), code in <pandas.core.generic.NDFrame.describe>.
        RuntimeWarning: Invalid value encountered in percentile.
        :param data:
        :param size:
        :param deletions:
        :param path:
        :param typed: model, scorecards.
        """
        self.data = data
        self.size = size
        self.train_size = train_size
        self.arith = arith

        if not deletions:
            self.deletions = {}
        else:
            self.deletions = deletions

        if not path:
            self.path = mk_dir(main_path_out+get_days()+os.sep)
        else:
            self.path = path

        self.id = 'T' + randoms(16)
        self.logs_path = mk_dir(self.path+self.id+os.sep+'logs'+os.sep)
        self.pict_path = mk_dir(self.path+self.id+os.sep+'pict'+os.sep)
        self.save_path = mk_dir(self.path+self.id+os.sep+'save'+os.sep)

        self.typed = typed

        self.dataX, self.dataY = pd.DataFrame([]), pd.DataFrame([])
        self.dataT, self.dataR = pd.DataFrame([]), pd.DataFrame([])
        self.logger = None
        self.fields = {}
        self.result = pd.Series()
        self.fields_numS = []
        self.fields_numN = []
        self.fields_strA = []
        self.changes = 0
        self.G1 = 'G1_'
        self.S1 = 'S1_'
        self.O1 = 'O1_'
        self.X1 = 'X1_'
        self.X_train, self.X_test = pd.DataFrame([]), pd.DataFrame([])
        self.y_train, self.y_test = pd.DataFrame([]), pd.DataFrame([])
        self.classifiers = []
        self.result_list = []

        self.run()

    def run(self):
        self.initlog()
        self.start()

        self.load_data()
        self.feature_explore()
        self.type_change()
        self.classifier_run()

    def start(self):
        self.print_log('WorkID: %s' % self.id)
        self.print_log('CreateTime: %s\n' % get_time())

    def initlog(self):
        formatter = logging.Formatter(log_format)

        handler = logging.FileHandler(self.logs_path+'main.log', log_mode, encoding=log_encode)
        handler.setFormatter(formatter)

        self.logger = logging.getLogger()
        self.logger.addHandler(handler)
        self.logger.setLevel(log_level)

    def print_log(self, details, types='info', log=True):
        """
        Print the log details and save to logfile.
        :param details: LogRecord details.
        :param types: str, logging types, default 'info'.
        In (critical, error, exception, warning, info, debug).
        :param log: the switch to log, default True.
        """
        if log:
            if types == 'critical':
                self.logger.critical(details)
            elif types == 'error':
                self.logger.error(details)
            elif types == 'exception':
                self.logger.exception(details, exc_info=False)
            elif types == 'warning':
                self.logger.warning(details)
            elif types == 'info':
                self.logger.info(details)
            else:
                self.logger.debug(details)
        print(details)

    def info(self, task):
        """
        Print the status, dataX.shape, dataY.shape.
        :param task: task name.
        :return:
        """
        self.print_log('_______________dataX_______________')
        self.print_log('dataX Shape: %s' % str(self.dataX.shape))
        self.print_log('_______________dataY_______________')
        self.print_log('dataY Shape: %s' % str(self.dataY.shape))

        self.print_log('***************The %s Finished.****************\n' % task)

    def load(self):
        pass

    def load_data(self):
        self.load()
        self.info('Load Data')

    def split(self):
        pass

    def explorer(self):
        pass

    def feature_explore(self):
        self.split()
        self.explorer()

        self.info('Feature Explore')

    def g1_change(self, name, string, con):
        pass

    def s1_standard(self, name, string, con):
        pass

    def o1_one_hot(self, name, string):
        pass

    def x1_across(self, name, string):
        pass

    def type_num(self):
        pass

    def type_str(self):
        pass

    def changes_values(self, replace, i, i_values):
        pass

    def type_change(self):
        self.dataR = self.dataX.copy()
        self.type_num()
        self.type_str()
        self.dataX = self.dataR

        if self.typed == 'model':
            pass
        elif self.typed == 'scorecards':
            self.cal_woe(runtime=times, spe=special)
            self.dataX = self.dataR
            self.dataX.to_csv(self.save_path+'dataX.csv')
        else:
            pass

        self.info('Feature Processing')

    def cal_woe(self, runtime, spe):
        pass

    def model(self, classifier):
        pass

    def select(self):
        pass

    def classifier_run(self):
        pass

