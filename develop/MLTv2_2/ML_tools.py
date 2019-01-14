# -*-coding:utf-8-*-
# @auth ivan
# @time 2017年5月26日21:31:17
# @goal V1.0


class ML(object):
    def __init__(self, data, deletions):
        from classifier_run import Classifier
        Classifier(data, deletions)

