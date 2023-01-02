import math
import numpy as np
import scipy.stats
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import robust_scale

from . import data_structure_helpers


def no_scaling(a_ndarray):
    return a_ndarray


def z_score_normalization(a_ndarray):
    assert isinstance(a_ndarray, np.ndarray)
    assert a_ndarray.size >= 2, 'z_score_normalization() needs at least 2 numbers'
    assert a_ndarray.dtype != 'bool', 'z-score normalization has to be applied for number array'

    return scipy.stats.zscore(a_ndarray)


def min_max_normalization(a_ndarray, min_to_scale=0.0, max_to_scale=1.0):
    return minmax_scale(a_ndarray, feature_range=(min_to_scale, max_to_scale), axis=0)


def robust_scaler(a_ndarray):
    assert isinstance(a_ndarray, np.ndarray)
    assert a_ndarray.size >= 2
    assert a_ndarray.dtype != 'bool'
    return robust_scale(a_ndarray)


def scale_two_ndarrays_so_the_starting_num_and_maximum_num_become_the_same(ndarray1, ndarray2):
    assert isinstance(ndarray1, np.ndarray)
    assert isinstance(ndarray2, np.ndarray)
    assert ndarray1.size >= 2
    assert ndarray2.size >= 2
    assert len(ndarray1.shape) == 1
    assert len(ndarray2.shape) == 1
    assert ndarray1.dtype != 'bool'
    assert ndarray2.dtype != 'bool'
    data_structure_helpers.assert_all_elements_in_ndarray_are_number(ndarray1)
    data_structure_helpers.assert_all_elements_in_ndarray_are_number(ndarray2)

    ndarray1 = robust_scaler(ndarray1)
    ndarray2 = robust_scaler(ndarray2)
    # ndarray1_min_max_normalized = min_max_normalization(ndarray1, min_to_scale=0.0, max_to_scale=1000.0)
    # ndarray2_min_max_normalized = min_max_normalization(ndarray2, min_to_scale=0.0, max_to_scale=1000.0)

    ndarray1_with_first_number_moved_to_zero = ndarray1 - ndarray1[0]
    ndarray2_with_first_number_moved_to_zero = ndarray2 - ndarray2[0]

    a_small_constant = 0.01
    if ndarray1_with_first_number_moved_to_zero.max() == 0 or math.isclose(ndarray1_with_first_number_moved_to_zero.max(), 0.0):
        if not (ndarray2_with_first_number_moved_to_zero.max() == 0 or math.isclose(ndarray2_with_first_number_moved_to_zero.max(), 0.0)):
            print('read1')
            ndarray1_with_first_number_moved_to_zero = np.insert(ndarray1_with_first_number_moved_to_zero, 0, a_small_constant)
    elif ndarray2_with_first_number_moved_to_zero.max() == 0 or math.isclose(ndarray2_with_first_number_moved_to_zero.max(), 0.0):
        if not (ndarray1_with_first_number_moved_to_zero.max() == 0 or math.isclose(ndarray1_with_first_number_moved_to_zero.max(), 0.0)):
            print('read2')
            ndarray2_with_first_number_moved_to_zero = np.insert(ndarray2_with_first_number_moved_to_zero, 0, a_small_constant)
    else:
        pass

    scaled_ndarray1 = ndarray1_with_first_number_moved_to_zero * ndarray2_with_first_number_moved_to_zero.max()
    scaled_ndarray2 = ndarray2_with_first_number_moved_to_zero * ndarray1_with_first_number_moved_to_zero.max()

    # scaled_ndarray1 = minmax_scale(scaled_ndarray1)
    # scaled_ndarray2 = minmax_scale(scaled_ndarray2)

    return scaled_ndarray1, scaled_ndarray2
