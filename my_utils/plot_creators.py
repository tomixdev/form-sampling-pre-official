import copy
import numpy as np
import matplotlib.pyplot as plt

from . import data_structure_helpers
from . import file_and_dir_interaction_helpers
from . import misc_helpers as mh
import seaborn
seaborn.set_style('darkgrid')


def plot_one_x_y_graph(X=None,
                       Y=None,
                       x_axis_label='',
                       y_axis_label='',
                       graph_title='',
                       width_height_tuple_of_each_fig=(8, 3)):
    """
    X can be a list or ndarray
    Y can be a list or ndarray
    """

    plt.close('all')

    if Y is None:
        raise Exception("Y must be designated!!!! Otherwise, a graph cannot be plotted!!!")
    else:
        Y = data_structure_helpers.convert_to_ndarray_if_list(Y)
        if Y.size == 0:
            raise Exception("Y must be designated!!!! Otherwise, a graph cannot be plotted!!!")

    assert isinstance(Y, (list, np.ndarray))
    assert isinstance(width_height_tuple_of_each_fig, tuple)

    if X is None or (isinstance(X, list) and len(X) == 0):
        X = np.arange(0, Y.size)
        x_axis_label = 'counter'
    else:
        X = data_structure_helpers.convert_to_ndarray_if_list(X)

    mh.assert_value_equality(X.size, Y.size)

    assert isinstance(Y, np.ndarray)
    assert isinstance(X, np.ndarray)

    X = np.array(X, copy=True)
    Y = np.array(Y, copy=True)

    plot_multiple_x_y_graphs(list_of_Xs=[X],
                             list_of_Ys=[Y],
                             list_of_x_axis_labels=[x_axis_label],
                             list_of_y_axis_labels=[y_axis_label],
                             list_of_graph_titles=[graph_title],
                             width_height_tuple_of_each_fig=width_height_tuple_of_each_fig,
                             display_plot_in_run_time=True)

    # X.clear()
    # Y.clear()


def plot_multiple_x_y_graphs(list_of_Xs=None,
                             list_of_Ys=None,
                             list_of_x_axis_labels=None,
                             list_of_y_axis_labels=None,
                             list_of_graph_titles=None,
                             number_of_columns=None,
                             width_height_tuple_of_each_fig=(4, 1.5),
                             save_fig=False,
                             save_fig_path=None,
                             big_title_for_all_plots=None,
                             share_vertical_plot_scaling=True,
                             display_plot_in_run_time=False):

    plt.close('all')

    assert isinstance(share_vertical_plot_scaling, bool), type(share_vertical_plot_scaling)
    assert isinstance(display_plot_in_run_time, bool), type(display_plot_in_run_time)

    # ---------------------------------------------------------------------------------------
    # Step 1: Ensure all arguments are correct. Ensure all arguments are valid.
    #        If there is any invalid arguments, give valid arguments or simply raise errors
    # ----------------------------------------------------------------------------------------
    # ensure type of width_height_tuple
    mh.assert_class(width_height_tuple_of_each_fig, tuple)

    # ensure Y is correctly formatted--------------
    if list_of_Ys is None or (isinstance(list_of_Ys, list) and len(list_of_Ys) == 0):
        raise Exception("List of Ys must be designated!!! Otherwise, graphs cannot be plotted!!!")
    if not isinstance(list_of_Ys, list):
        raise Exception("List of Ys need to be a list!!!!")

    list_of_Ys = copy.deepcopy(list_of_Ys)

    for i in range(0, len(list_of_Ys)):
        if isinstance(list_of_Ys[i], list):
            list_of_Ys[i] = data_structure_helpers.convert_to_ndarray_if_list(list_of_Ys[i])

        mh.assert_class(list_of_Ys[i], np.ndarray)

    # ensure X is correctly formatted-----------------
    # if there is no X, make x axis labels "counter"

    if list_of_Xs is None or (isinstance(list_of_Xs, list) and len(list_of_Xs) == 0):
        list_of_Xs = []
        if list_of_x_axis_labels is None:
            list_of_x_axis_labels = []
        for i in range(0, len(list_of_Ys)):
            a_X = np.arange(0, list_of_Ys[i].size)
            list_of_Xs.append(a_X)
            if list_of_x_axis_labels is None:
                list_of_x_axis_labels.append('counter')  # TODO: Noneだったらappend操作などできないはずだけど、いじるのが怖いので放置。

    if isinstance(list_of_Xs, list):
        for a_X in list_of_Xs:
            a_X = data_structure_helpers.convert_to_ndarray_if_list(a_X)
    else:
        raise Exception("List of Xs need to be a list!!!!")

    # ensure the lengths of X and Y are the same length-------------------------------
    mh.assert_value_equality(len(list_of_Xs), len(list_of_Ys))

    # ensure x axis labels--------------
    if list_of_x_axis_labels is None or len(list_of_x_axis_labels) == 0:
        # if no x asis labels are given, generate an empty label for each graph
        list_of_x_axis_labels = []
        for i in range(0, len(list_of_Xs)):
            list_of_x_axis_labels.append('')  # append empty label
    else:
        # if x axis labels are given, ensure the length is correct
        mh.assert_class(list_of_x_axis_labels, list)
        mh.assert_value_equality(len(list_of_Xs), len(list_of_x_axis_labels))

    for i in range(0, len(list_of_x_axis_labels)):
        list_of_x_axis_labels[i] = list_of_x_axis_labels[i] + "\n" + "\n" + _get_vector_statistic_info_string_from_ndarray(list_of_Ys[i])

    # ensure y axis labels--------------
    if list_of_y_axis_labels is None or len(list_of_y_axis_labels) == 0:
        list_of_y_axis_labels = []
        # if no y asis labels are given, generate an empty label for each graph
        for i in range(0, len(list_of_Ys)):
            list_of_y_axis_labels.append('')
    else:
        # if y axis labels are given, ensure the length is correct
        mh.assert_class(list_of_y_axis_labels, list)
        mh.assert_value_equality(len(list_of_Ys), len(list_of_x_axis_labels))

    # ensure graph titles----------------
    if list_of_graph_titles is None or len(list_of_graph_titles) == 0:
        list_of_graph_titles = []
        for i in range(0, len(list_of_Xs)):
            list_of_graph_titles.append('Graph ' + str(i))
    else:
        mh.assert_class(list_of_graph_titles, list)
        mh.assert_value_equality(len(list_of_Xs), len(list_of_graph_titles))

    # raise errors if the lengths of all the arguments are not the same:
    assert len(list_of_Xs) == len(list_of_Ys) == len(list_of_x_axis_labels) == len(list_of_y_axis_labels) == len(list_of_graph_titles)

    assert isinstance(save_fig, bool)
    if save_fig == True:
        assert save_fig_path is not None
    if save_fig_path is not None:
        assert isinstance(save_fig_path, str)
        assert save_fig_path.endswith(('.png', '.jpeg', '.jpg', '.pdf'))
        a_str = mh.get_a_str_before_last_slash(save_fig_path) + '/'
        file_and_dir_interaction_helpers.raise_exception_if_path_to_a_folder_is_in_wrong_format(a_str)
        assert file_and_dir_interaction_helpers.does_a_dir_exist(a_str)

    # ---------------------------------------------------------------------------------------
    # Step 2: Plot a graph for each element in lists
    # ----------------------------------------------------------------------------------------

    if number_of_columns is None:
        number_of_rows = 1
        number_of_columns = len(list_of_Xs)
    else:
        assert isinstance(number_of_columns, int)
        assert number_of_columns >= 1
        assert len(list_of_Ys) % number_of_columns == 0
        number_of_rows = int(len(list_of_Ys) / number_of_columns)

    width_of_each_graph = list(width_height_tuple_of_each_fig)[0]
    height_of_each_graph = list(width_height_tuple_of_each_fig)[1]

    # fig=大きな領域、axes=それぞれのグラフ #行数がnumber_of_rowsつ、列数がnumber_of_column個
    fig, axes = plt.subplots(number_of_rows,
                             number_of_columns,
                             figsize=(width_of_each_graph*number_of_columns, height_of_each_graph*number_of_rows),
                             squeeze=False,
                             sharey=share_vertical_plot_scaling)

    # いちいち、二次元で[1][0] など指定してグラフを呼び出すのはめんどくさいので、通し番号をつけることできる。それがravel() method
    for i in range(0, number_of_rows*number_of_columns):
        a_X = list_of_Xs[i]
        a_Y = list_of_Ys[i]
        a_x_axis_label = list_of_x_axis_labels[i]
        a_y_axis_label = list_of_y_axis_labels[i]
        a_graph_title = list_of_graph_titles[i]

        axes.ravel()[i].set_xlabel(a_x_axis_label, fontsize=10)
        axes.ravel()[i].set_ylabel(a_y_axis_label, fontsize=10)
        axes.ravel()[i].set_title(a_graph_title)
        axes.ravel()[i].plot(a_X, a_Y)

    if big_title_for_all_plots is not None:
        mh.assert_class(big_title_for_all_plots, str)
        fig.suptitle(big_title_for_all_plots)

    plt.tight_layout()

    if save_fig == True:
        plt.savefig(save_fig_path)

    if display_plot_in_run_time:
        plt.show()

    # list_of_Xs.clear()
    # list_of_Ys.clear()
    # list_of_x_axis_labels.clear()
    # list_of_y_axis_labels.clear()
    # list_of_graph_titles.clear()


def stringfy_a_dict_for_graph_description(a_dict):
    assert isinstance(a_dict, dict)

    str_to_return = ''
    for a_key, a_value in a_dict.items():
        if a_key == 'time_point_tuple':  # TODO!!!! ものすごくダメな付け焼き刃な対策！！！本当はDBに入れるString自体をminutes' seconds''にするべき。
            a_value = mh.convert_float_num_tuple_str_to_tuple(a_value)
            a_value = mh.convert_time_point_tuple_in_s_to_tuple_in_minutes_and_seconds(a_value)
        str_to_return += f"{a_key} | {a_value} \n"

    return str_to_return


def _get_vector_statistic_info_string_from_ndarray(a_ndarray):
    string_to_return = ''
    if (a_ndarray.dtype == 'bool'):
        string_to_return = 'boolean'
    else:
        string_to_return = "n = " + str(a_ndarray.size) + " (" + str(np.sum(np.isnan(a_ndarray))) + " nan vals)" + "\n" + \
            "mean = " + str(round(np.mean(a_ndarray), 3)) + "\n" + \
            "variance = " + str(round(np.var(a_ndarray), 3)) + "\n" + \
            "st.dev. = " + str(round(np.std(a_ndarray), 3)) + '\n' + \
            "max = " + str(round(np.max(a_ndarray), 3)) + "\n" + \
            "min = " + str(round(np.min(a_ndarray), 3))

    return string_to_return


# driver code for testing
'''
#driver code to test
X1 = np.sort(np.random.random(20))
Y1 = np.sort(np.random.random(20))
X2 = np.sort(np.random.random(20))
Y2 = np.sort(np.random.random(20))
X3 = np.sort(np.random.random(20))
Y3 = np.sort(np.random.random(20))
x_labels = ['minato x label 1', 'minato x label 2', 'minato x label 3']
y_labels = ['minato y label 1', 'minato y label 2', 'minato y label 3']
graph_titles = ['minato graph title 1', 'minato graph title 2', 'minato graph title 3']

graph_generators.plot_one_x_y_graph(X=X1, Y=Y1, x_axis_label='adfaf', y_axis_label='sdsd', graph_title='this is title')
graph_generators.plot_multiple_x_y_graphs (list_of_Xs=[X1, X2], list_of_Ys=[Y1, Y2], list_of_graph_titles=['sfasd', 'asfsas'], list_of_x_axis_labels=['x1', 'x2'], list_of_y_axis_labels=['y1', 'y2'])

'''
