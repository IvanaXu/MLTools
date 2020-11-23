# -*-coding:utf-8-*-
# @auth ivan
# @time 2017年5月11日20:43:42
# @goal tools

__all__ = [
    'get_days',
    'get_time',
    'randoms',
    'mk_dir',
    'cut_split',
    'g1_do',
    's1_do',
    'directing'
]


def get_days():
    """
    :return: date like 20180101.
    """
    import datetime
    return datetime.datetime.now().strftime('%Y%m%d')


def get_time():
    """
    :return: date like 20180101_093001.
    """
    import datetime
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def randoms(n):
    """
    If n > 0 then random(n) else random(6).
    :param n: int.
    :return: randoms number by n.
    """
    import random
    return str(random.randint(10**(n-1) if n > 0 else 100000, 10**n-1 if n > 0 else 999999))


def mk_dir(path):
    """
    If path is not exist then create it.
    :param path: absolute path.
    :return: path.
    """
    import os
    if os.path.isdir(path):
        pass
    else:
        os.system('mkdir ' + path)
    return path


def cut_split(x, y, size):
    # TODO: When version 0.20.
    """
    The cross_validation.train_test_split module will be removed in 0.20.
    Use model_selection.train_test_split will instead.
    :param x: dataX.
    :param y: dataY.
    :param size: float, int, or None (default is None).
    If float, should be between 0.0 and 1.0. If size is also None, size is set to 0.75.
    :return: dataX_train, dataX_test, dataY_train, dataY_test.
    """
    from sklearn import __version__ as sk_ver
    if float(sk_ver[:-2]) >= 0.18:
        from sklearn.model_selection import train_test_split
    else:
        from sklearn.cross_validation import train_test_split
    x1, x2, y1, y2 = \
        train_test_split(x, y, train_size=size)
    return x1, x2, y1, y2


def g1_do(x):
    """
    G1 归一化 normalization.
    :param x: DataFrame.
    :return: lambda x: x if np.max(x) == np.min(x) else (x - np.min(x)) / (np.max(x) - np.min(x)).
    """
    import numpy as np
    if np.max(x) == np.min(x):
        # Loss_Contact9_Cnt 均是0
        return x
    else:
        return (x - np.min(x)) / (np.max(x) - np.min(x))


def s1_do(x):
    """
    S1 标准化 standardizing.
    :param x: DataFrame.
    :return: lambda x: x if not np.std(x) else (x - np.mean(x)) / (np.std(x)).
    """
    import numpy as np
    if not np.std(x):
        # Loss_Contact9_Cnt 均是0
        return x
    else:
        return (x - np.mean(x)) / (np.std(x))


def directing(name):
    """
    Cross directing 遍历出一个交叉指向说明.
    :param name: list, ['A','B','C'].
    :return: list, [('A','B'),('A','C'),('B','C')].
    """
    dn = len(name)
    result = []
    for di in range(0, dn - 1):
        for dj in range(di + 1, dn):
            result.append((name[di], name[dj]))
    return result

