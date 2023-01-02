import copy
import math
import os.path

import numpy as np
from sklearn.preprocessing import minmax_scale
from . import misc_helpers


def convert_values_of_absolute_path_strings_in_a_dict_into_relative_path_strings(a_dict):
    assert isinstance(a_dict, dict)
    assert not is_a_dict_nested(a_dict)

    dict_to_return = copy.deepcopy(a_dict)

    for a_key, a_value in dict_to_return.items():
        if isinstance(a_value, str) and os.path.isabs(a_value):
            dict_to_return[a_key] = misc_helpers.to_relative_path(a_value)

    return dict_to_return


def is_a_dict_nested(a_dict):
    assert isinstance(a_dict, dict)
    for a_key, a_value in a_dict.items():
        if isinstance(a_value, dict):
            return True
    return False


def are_keys_of_two_dicts_identical(dict1, dict2):
    assert isinstance(dict1, dict)
    assert isinstance(dict2, dict)

    if len(dict1.keys()) != len(dict2.keys()):
        return False
    if set(dict1.keys()) != set(dict2.keys()):
        return False

    return True


def flatten_dict(a_nested_dict, check_key_duplicates=True):
    assert isinstance(a_nested_dict, dict)
    dict_to_return = {}
    for k, v in a_nested_dict.items():
        if isinstance(v, dict):
            if check_key_duplicates and k in dict_to_return.keys():
                raise Exception('A duplicate key is found when flattening a nested dict!')
            else:
                dict_to_return.update(flatten_dict(v))
        else:
            dict_to_return[k] = v
    return dict_to_return


def are_all_elements_in_a_list_unique(a_list):
    assert isinstance(a_list, list), type(a_list)
    return len(a_list) == len(set(a_list))


def do_two_sets_intersect(setA, setB):
    assert isinstance(setA, set)
    assert isinstance(setB, set)
    return bool(setA & setB)


def assert_nd_array_content_is_starting_from_zero_and_ascending_by_one(a_ndarray):
    assert isinstance(a_ndarray, np.ndarray)
    assert a_ndarray[0] == 0
    assert a_ndarray[-1] == a_ndarray.size - 1
    assert np.all(np.diff(a_ndarray) == 1)


def assert_all_elements_in_list_are_tuple(list_of_tuples):
    assert isinstance(list_of_tuples, list)
    for a_tuple in list_of_tuples:
        assert isinstance(a_tuple, tuple)


def expand_ndarray_to_size_of_at_least_x(ndarray, x=1000):
    '''This function does NOT intepolate array, but EXPAND to a certain length.'''
    '''Resulting array's size is most likely more than x, unless x is a multiple of ndarray size'''
    assert isinstance(ndarray, np.ndarray)
    assert len(ndarray.shape) == 1

    if ndarray.size >= x:
        return ndarray
    else:
        how_many_times_to_duplicate_each_element_in_ndarray = math.floor(x/ndarray.size) + 1
        return np.repeat(ndarray, how_many_times_to_duplicate_each_element_in_ndarray)


def assert_all_elements_in_a_list_are_different(a_list):
    assert isinstance(a_list, list)
    assert len(a_list) > 0
    try:
        assert len(a_list) == len(set(a_list))  # if the list includes an element like ndarray, set(a_list) will not work.
    except:
        assert_all_elements_in_a_list_of_ndarray_are_different(a_list)


def assert_all_elements_in_a_list_of_ndarray_are_different(a_list_of_ndarray):
    assert isinstance(a_list_of_ndarray, list)
    assert len(a_list_of_ndarray) > 0
    assert all(isinstance(elem, np.ndarray) for elem in a_list_of_ndarray)

    for i in range(len(a_list_of_ndarray)):
        for j in range(i+1, len(a_list_of_ndarray)):
            assert_all_elements_in_two_ndarreys_are_different(a_list_of_ndarray[i], a_list_of_ndarray[j])


def assert_all_elements_in_two_ndarreys_are_different(ndarray1, ndarray2):
    assert isinstance(ndarray1, np.ndarray)
    assert isinstance(ndarray2, np.ndarray)
    assert len(ndarray1.shape) == 1
    assert len(ndarray2.shape) == 1
    if ndarray1.size != ndarray2.size:
        return  # two arrays are different!!! So, no need to do further assertion
    else:
        assert not np.array_equiv(ndarray1, ndarray2),\
            f"\nndarray1: ------------------------ \n {ndarray1} \nndarray2: ------------------------ \n {ndarray2}"
        assert not np.allclose(ndarray1, ndarray2), \
            f"\n ndarray1: ------------------------ \n {ndarray1} \n ndarray2: ------------------------ \n {ndarray2}"


def clip_a_ndarray(a_ndarray, lower_thresh_percentile, upper_thresh_percentile):
    misc_helpers.ensure_a_value_is_a_number(lower_thresh_percentile)
    misc_helpers.ensure_a_value_is_a_number(upper_thresh_percentile)

    a_ndarray = convert_to_ndarray_if_list(a_ndarray)
    is_valid_vector_ndarray_or_list(a_ndarray)
    assert 0 <= lower_thresh_percentile <= 100
    assert 0 <= upper_thresh_percentile <= 100
    assert lower_thresh_percentile < 10.0, 'I think....the lower threshold should be lower than 10 to make sense...but I might be wrong'
    assert upper_thresh_percentile > 90.0, 'I think....the upper threshold should be higehr than 90 to make sense...but I might be wrong'

    lower_thresh_value = np.percentile(a_ndarray, lower_thresh_percentile)
    upper_thresh_value = np.percentile(a_ndarray, upper_thresh_percentile)
    clipped_array = np.clip(a=a_ndarray, a_min=lower_thresh_value, a_max=upper_thresh_value)
    return clipped_array


def assert_list1_contains_all_elements_of_list2(list1, list2):
    assert isinstance(list1, list)
    assert isinstance(list2, list)
    assert all(elem in list1 for elem in list2)


def assert_lengths_of_all_tuples_in_list_are_the_same(list_of_tuples):
    assert isinstance(list_of_tuples, list)
    assert len(list_of_tuples) > 0

    len_of_first_tuple = len(list_of_tuples[0])
    for a_tuple in list_of_tuples:
        assert len(a_tuple) == len_of_first_tuple


def assert_all_elements_in_ndarray_are_number(a_ndarray):
    assert isinstance(a_ndarray, np.ndarray), type(a_ndarray)
    assert not np.isnan(a_ndarray).any(), a_ndarray
    assert np.isfinite(a_ndarray).all(), a_ndarray
    # If the above evaluates to True, then myarray contains none of numpy.nan, numpy.inf or -numpy.inf.

    '''
    for element in a_ndarray:
        if not is_number(element):
            return False
    return True
    '''


def assert_all_values_in_one_dimentional_ndarray_are_different(a_ndarray):
    assert isinstance(a_ndarray, np.ndarray)
    assert len(a_ndarray.shape) == 1
    assert_all_elements_in_ndarray_are_number(a_ndarray)

    assert np.max(a_ndarray) != np.min(a_ndarray)
    assert not np.isclose(np.max(a_ndarray), np.min(a_ndarray))


def convert_all_nan_values_to_min_value_in_ndarray(a_ndarray, copy_ndarray=True):
    assert isinstance(a_ndarray, np.ndarray)
    a_ndarray[np.isnan(a_ndarray)] = np.nanmin(a_ndarray)
    return np.array(a_ndarray, copy=copy_ndarray)


def convert_to_ndarray_if_list(a_ndarray_or_list):
    if isinstance(a_ndarray_or_list, list):
        return np.array(a_ndarray_or_list)
    elif isinstance(a_ndarray_or_list, np.ndarray):
        return a_ndarray_or_list
    else:
        raise Exception("Neither a list nor ndarray!!!!!")


def interpolate_or_shrink_ndarray_to_length_n(a_ndarray, n):
    '''
    Below is a good test code to see the behavior of this function

    a_ndarray = np.array([8.4, 2.3, 9.9])
    print (a_ndarray)
    new = gh.interpolate_or_shrink_ndarray_to_length_n(a_ndarray, 10)
    print (new)
    print (new[5])

    a_ndarray = np.linspace(3, 8, 6)
    print (a_ndarray)
    new = gh.interpolate_or_shrink_ndarray_to_length_n(a_ndarray, 4)
    print (new)
    '''
    assert isinstance(a_ndarray, np.ndarray)
    assert len(a_ndarray.shape) == 1
    assert isinstance(n, int)
    assert n > 0
    if len(a_ndarray) == n:
        return a_ndarray
    else:
        counter_array_X = np.linspace(1, a_ndarray.size, a_ndarray.size)
        X_scaled_to_between_1_and_n = minmax_scale(counter_array_X, feature_range=(0, n), axis=0)
        xp = X_scaled_to_between_1_and_n
        assert np.all(np.diff(xp) > 0)
        fp = a_ndarray
        array_from_1_to_n = np.linspace(0, n, n)
        return np.interp(array_from_1_to_n, xp, fp)


def average_a_ndarray_over_n_elements(a_ndarray, n):
    assert isinstance(a_ndarray, np.ndarray)
    assert len(a_ndarray.shape) == 1
    assert isinstance(n, int)
    assert n > 0

    if a_ndarray.size % n != 0:
        length_to_which_an_array_is_set = a_ndarray.size + (n - a_ndarray.size % n)
        a_ndarray = interpolate_or_shrink_ndarray_to_length_n(a_ndarray=a_ndarray, n=length_to_which_an_array_is_set)

    return np.average(a_ndarray.reshape(-1, n), axis=1)


def randomly_drop_n_elements_from_ndarray(a_ndarray, n):
    assert isinstance(a_ndarray, np.ndarray)
    assert len(a_ndarray.shape) == 1
    assert isinstance(n, int)
    assert n > 0
    assert a_ndarray.size > n

    number_of_elements_to_drop = n
    indices_to_drop = np.random.choice(a_ndarray.size, number_of_elements_to_drop, replace=False)
    return np.delete(a_ndarray, indices_to_drop)


def convert_to_list_if_ndarray(a_ndarray_or_list):
    if isinstance(a_ndarray_or_list, list):
        return a_ndarray_or_list
    elif isinstance(a_ndarray_or_list, np.ndarray):
        return a_ndarray_or_list.tolist()
    else:
        raise Exception("Neither a list nor ndarray!!!!!")


def drop_x_elements_from_a_ndarray_randomly_and_return(a_ndarray, x):
    assert isinstance(a_ndarray, np.ndarray)
    assert len(a_ndarray.shape) == 1
    assert isinstance(x, int)

    number_of_elements_to_choose = a_ndarray.size - x
    # numbers below will never contain repeated numbers (replace=False)
    return np.random.choice(a_ndarray, number_of_elements_to_choose, replace=False)


def set_two_ndarrays_to_the_length_of_smaller_ndarray_randomly_and_return(ndarray1, ndarray2):
    ndarray1 = ndarray1
    ndarray2 = ndarray2
    if ndarray1.size > ndarray2.size:
        ndarray_length_difference = ndarray1.size - ndarray2.size
        ndarray1 = drop_x_elements_from_a_ndarray_randomly_and_return(ndarray1, ndarray_length_difference)
    else:
        ndarray_length_difference = ndarray2.size - ndarray1.size
        ndarray2 = drop_x_elements_from_a_ndarray_randomly_and_return(ndarray2, ndarray_length_difference)
    return ndarray1, ndarray2


def raise_exception_if_not_a_valid_vector_ndarray_or_list(a_value):
    if not is_valid_vector_ndarray_or_list(a_value):
        raise Exception("Not a valid array or list!!!!!")


def is_valid_vector_ndarray_or_list(a_value):
    assert a_value is not None

    if misc_helpers.is_number(a_value):
        return False
    if isinstance(a_value, str):
        return False
    if isinstance(a_value, list):
        if len(a_value) == 0 or len(a_value) == 1:
            return False
        else:
            return True
    if isinstance(a_value, np.ndarray):
        if a_value.size == 0 or a_value.size == 1:
            return False
        if (a_value.dtype == 'bool'):
            raise Exception('this is a boolean ndarray!!!!!')
        else:
            return True

    return False


def convert_all_numbers_in_ndarray_to_floating_point_numbers(a_ndarray):
    return np.asarray(a_ndarray).astype('float32')


def assert_all_values_in_a_dict_are_all_none_or_all_non_none_if_a_dict_is_not_empty(a_dict):
    misc_helpers.assert_class(a_dict, dict)

    if bool(dict) == True:  # if a dict is not empty
        assert all(a_value is None for a_value in a_dict.values()) or \
            all(a_value is not None for a_value in a_dict.values())


def normalize_and_set_values_below_i_percentile_to_zero(X, a_percentile):
    from my_utils import data_scalers

    assert isinstance(X, np.ndarray)
    assert isinstance(a_percentile, (int, float))
    X = data_scalers.z_score_normalization(X)
    the_i_percentile = np.percentile(X, a_percentile)
    X[X < the_i_percentile] = 0
    return X


def upside_down_a_ndarray(a_ndarray):
    """
    TESTED on 20220903
    """

    misc_helpers.assert_class(a_ndarray, np.ndarray)

    if np.max(a_ndarray) == np.min(a_ndarray) or np.isclose(np.max(a_ndarray), np.min(a_ndarray)):
        to_return = copy.deepcopy(a_ndarray)  # Not sure if I need to deeocopy
    else:
        new_min = np.max(a_ndarray)
        new_max = np.min(a_ndarray)

        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html#sklearn.preprocessing.minmax_scale
        scale = (new_max - new_min) / (a_ndarray.max(axis=0) - a_ndarray.min(axis=0))
        to_return = copy.deepcopy(scale * a_ndarray + new_min - a_ndarray.min(axis=0) * scale)  # TODO:　ここのdeepcopy必要なのかどうかイマイチわからない。

    return to_return
