# -*-coding:utf-8-*-
# @auth ivan LYY
# @time 2017年4月23日13:02:14
# @goal Draw Tree.


def draw_decision_tree(path, d_tree, name):
    """
    Draw the decision tree.
    :param path: save path.
    :param d_tree: decision tree.
    :param name: The filename(.png) of the picture.
    """
    import os
    from sklearn import tree
    from MLTools.code.product.config import pro_gra_dot

    with open(path + '\\' + name + ".dot", 'w') as f:
        tree.export_graphviz(d_tree, out_file=f)

    if os.path.exists(pro_gra_dot):
        save = path + '\\' + name
        run_cmd = '"' + pro_gra_dot + '" -Tpng ' + save + '.dot -o ' + save + '.png'
        os.system(run_cmd)
    else:
        raise EnvironmentError(pro_gra_dot + ' is not exist.')

