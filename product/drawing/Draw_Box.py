# -*-coding:utf-8-*-
# @auth ivan
# @time 2018-01-17 14:52
# @goal Draw Box.


def draw_box(data, name, path):
    """
    Draw the data box.
    :param data: DataFrame.
    :param name: DataFrame columns.
    :param path: save path.
    """
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        pd.DataFrame(data[name]).boxplot(return_type='axes')
        plt.savefig(path + '\\' + name + '.jpg')
        plt.close()
    except Exception as e:
        print(str(e))

