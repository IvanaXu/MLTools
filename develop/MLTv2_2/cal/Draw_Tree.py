# -*-coding:utf-8-*-
# @auth ivan LYY
# @time 2017年4月23日13:02:14
# @goal Draw_Tree


def draw_tree(path, d_tree, name):
    """
    Draw Decision Tree
    :param path: Save Path
    :param d_tree: Tree
    :param name: Filename (.png) For The Name of The Picture(Tree)
    :return:
    """
    import os
    from sklearn import tree
    with open(path + '\\' + name + ".dot", 'w') as f:
        tree.export_graphviz(d_tree, out_file=f)
    graphviz = 'C:\\Program Files (x86)\\Graphviz2.38\\bin\\dot.exe'
    if os.path.exists(graphviz):
        save = path + '\\' + name
        run_cmd = '"' + graphviz + '" -Tpng ' + save + '.dot -o ' + save + '.png'
        os.system(run_cmd)
    else:
        raise Exception('PATH ERROR: '+graphviz + ' IS NOT EXISTS!')

