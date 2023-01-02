from tslearn import metrics
from tslearn.metrics import lcss
from tslearn.metrics import lcss_path
from . import dtw_and_ddtw_copied_from_a_website
from . import JoeZhao_github_z2e2_fastddtw as fastddtw
import numpy as np

import my_utils.data_structure_helpers
import my_utils.file_and_dir_interaction_helpers
from numpy.linalg import norm
import scipy.stats
import statistics
from my_utils import misc_helpers as gh
from sklearn.metrics.pairwise import cosine_similarity

def _common_argument_assertions_for_similarity_calculation_between_two_ndarrays(ndarray1, ndarray2):
    assert isinstance(ndarray1, np.ndarray) and isinstance(ndarray2, np.ndarray)
    assert len(ndarray1.shape) == 1 and len(ndarray2.shape) == 1
    assert ndarray1.size == ndarray2.size
    assert ndarray1.size >= 2 and ndarray2.size >= 2
    my_utils.data_structure_helpers.assert_all_elements_in_ndarray_are_number(ndarray1)
    my_utils.data_structure_helpers.assert_all_elements_in_ndarray_are_number(ndarray2)
    my_utils.data_structure_helpers.assert_all_values_in_one_dimentional_ndarray_are_different(ndarray1)
    my_utils.data_structure_helpers.assert_all_values_in_one_dimentional_ndarray_are_different(ndarray2)


def something(ndarray1, ndarray2):
    _common_argument_assertions_for_similarity_calculation_between_two_ndarrays(ndarray1, ndarray2)
    raise NotImplementedError("TODO: Not Implemented Yet")


def difference_direction_similarity(ndarray1, ndarray2):
    _common_argument_assertions_for_similarity_calculation_between_two_ndarrays(ndarray1, ndarray2)
    raise NotImplementedError("TODO: Not Implemented Yet")


def compute_L1_norm(ndarray1, ndarray2):
    _common_argument_assertions_for_similarity_calculation_between_two_ndarrays(ndarray1, ndarray2)
    raise NotImplementedError('TODO')
    return np.linalg.norm((ndarray1 - ndarray2), ord=1)


def compute_L2_norm(ndarray1, ndarray2):
    _common_argument_assertions_for_similarity_calculation_between_two_ndarrays(ndarray1, ndarray2)
    raise NotImplementedError('TODO')
    return np.linalg.norm(ndarray1-ndarray2)



def compute_cosine_similarity(ndarray1, ndarray2):
    '''
    compute_cosine_similarity.__dict__['read_num'] += 1
    print (compute_cosine_similarity.__dict__['read_num'])
    '''
    _common_argument_assertions_for_similarity_calculation_between_two_ndarrays(ndarray1, ndarray2)

    return cosine_similarity([ndarray1], [ndarray2])[0][0]

'''
compute_cosine_similarity.__dict__['read_num'] = 0
'''
def compute_absolute_value_of_pearson_correlation_coefficient(ndarray1, ndarray2):
    _common_argument_assertions_for_similarity_calculation_between_two_ndarrays(ndarray1, ndarray2)

    corr, __ = scipy.stats.pearsonr(ndarray1, ndarray2)
    assert isinstance(corr, (int, float)), str(type(corr))
    assert not np.isinf(corr)
    # gh.debuginfo(ndarray1)
    # gh.debuginfo(ndarray2)
    assert not np.isnan(corr)
    return abs(corr)


def compute_mean_deviation_cosine_similarity(ndarray1, ndarray2):
    def compute_mean_deviations(a_ndarray):
        res = []
        # getting mean
        mean_val = statistics.mean(a_ndarray)
        # list comprehension used for 1 liner
        res = [abs(ele - mean_val) for ele in a_ndarray]
        return res

    raise NotImplementedError('TODO: Not Implemented')

    _common_argument_assertions_for_similarity_calculation_between_two_ndarrays(ndarray1, ndarray2)

    mean_deviations_of_ndarray1 = np.array(compute_mean_deviations(ndarray1))
    mean_deviations_of_ndarray2 = np.array(compute_mean_deviations(ndarray2))
    return compute_cosine_similarity(mean_deviations_of_ndarray1, mean_deviations_of_ndarray2)


def dynamic_time_warping(ndarray1, ndarray2):
    assert isinstance(ndarray1, np.ndarray)
    assert isinstance(ndarray2, np.ndarray)
    assert ndarray1.size >= 2
    assert ndarray2.size >= 2
    assert len(ndarray1.shape) == 1
    assert len(ndarray2.shape) == 1
    assert ndarray1.dtype != 'bool'
    assert ndarray2.dtype != 'bool'
    my_utils.data_structure_helpers.assert_all_elements_in_ndarray_are_number(ndarray1)
    my_utils.data_structure_helpers.assert_all_elements_in_ndarray_are_number(ndarray2)

    alignment, sim = metrics.soft_dtw_alignment(ndarray1, ndarray2, gamma=1.0)
    return sim


def derivative_dynamic_time_warping(ndarray1, ndarray2, return_negative_distance_for_maximum_similarity_calculation=True):

    # TODO: Not sure if this is correct. Also, this is very slow.

    assert isinstance(ndarray1, np.ndarray), type(ndarray1)
    assert isinstance(ndarray2, np.ndarray), type(ndarray2)
    assert len(ndarray1.shape) == 1, ndarray1.shape
    assert len(ndarray2.shape) == 1, ndarray2.shape

    distance, path = dtw_and_ddtw_copied_from_a_website.get_DDWT_results(ndarray1, ndarray2)

    if return_negative_distance_for_maximum_similarity_calculation:
        return distance * -1.
    else:
        return distance

    '''
    ndarray1 = data_scalers.z_score_normalization(ndarray1)
    ndarray2 = data_scalers.z_score_normalization(ndarray2)

    mean_deviations_of_ndarray1 = np.array(compute_mean_deviations(ndarray1))
    mean_deviations_of_ndarray2 = np.array(compute_mean_deviations(ndarray2))

    ndarray1 = mean_deviations_of_ndarray1
    ndarray2 = mean_deviations_of_ndarray2

    distance, path = fastddtw.fast_ddtw(ndarray1, ndarray2, K=10) # K is the Sakoe-Chuba Band width used to constrain the search space of dynamic programming.

    if return_negative_distance_for_maximum_similarity_calculation:
        return distance * -1.
    else:
        return distance
    '''


def delta_derivative_dynamic_time_warping():
    raise Exception('To Implement')


def longest_common_subsequence_similarity_measure_no_constraint(ndarray1, ndarray2):
    assert isinstance(ndarray1, np.ndarray)
    assert isinstance(ndarray2, np.ndarray)
    assert ndarray1.size >= 2
    assert ndarray2.size >= 2
    assert len(ndarray1.shape) == 1
    assert len(ndarray2.shape) == 1
    assert ndarray1.dtype != 'bool'
    assert ndarray2.dtype != 'bool'
    my_utils.data_structure_helpers.assert_all_elements_in_ndarray_are_number(ndarray1)
    my_utils.data_structure_helpers.assert_all_elements_in_ndarray_are_number(ndarray2)

    # path, sim = lcss_path(ndarray1, ndarray2)
    # cost = lcss(ndarray1, ndarray2, global_constraint="sakoe_chiba", sakoe_chiba_radius=3)
    cost = lcss(ndarray1, ndarray2)  # global_constraint="itakura", itakura_max_slope=2.)

    return cost


def calculate_similarity_value_between_two_ndarrays_with_different_length(ndarray1,
                                                                          ndarray2,
                                                                          similarity_calculation_func_obj=None,
                                                                          show_graph_of_how_similarity_value_changes_over_randomization_trials=False,
                                                                          assert_all_array_elements_are_numbers=True,
                                                                          ndarray_length_to_which_two_ndarrays_are_converted=None,
                                                                          set_two_arrays_to_same_length_and_calculate_similarity_value=True):  # TODO: 本当はAudio Data ExtractionのときにSame Length Arrayにしておけば、同じAudio Extracted Arrayにたいして、何回も何回も同じintepolationをしなくてよくなる。
    ''''''

    from src_for_form_sampling.parameters_package import PrmsContainer  # TODO:　毎回parameters をimportするなんて馬鹿げているし計算を遅くしてしまいそう。
    
    assert isinstance(ndarray1, np.ndarray) and isinstance(ndarray2, np.ndarray)
    assert len(ndarray1.shape) == 1 and len(ndarray2.shape) == 1
    my_utils.data_structure_helpers.assert_all_elements_in_ndarray_are_number(ndarray1)
    my_utils.data_structure_helpers.assert_all_elements_in_ndarray_are_number(ndarray2)
    my_utils.data_structure_helpers.assert_all_values_in_one_dimentional_ndarray_are_different(ndarray1)
    my_utils.data_structure_helpers.assert_all_values_in_one_dimentional_ndarray_are_different(ndarray2)
    gh.assert_class(show_graph_of_how_similarity_value_changes_over_randomization_trials, bool)
    if assert_all_array_elements_are_numbers:
        my_utils.data_structure_helpers.assert_all_elements_in_ndarray_are_number(ndarray1)
        my_utils.data_structure_helpers.assert_all_elements_in_ndarray_are_number(ndarray2)
    if similarity_calculation_func_obj is None:
        try:
            similarity_calculation_func_obj = PrmsContainer.vector_similarity_calculation_func_obj
        except:
            raise Exception('similarity_calculation_func_obj is None and parameters.vector_similarity_calculation_func_obj is None too. Please set one of them.')
    if ndarray_length_to_which_two_ndarrays_are_converted is None:
        try:
            ndarray_length_to_which_two_ndarrays_are_converted = PrmsContainer.ndarray_length_to_which_two_ndarrays_are_converted_in_vector_comparison
        except:
            raise Exception(
                'ndarray_length_to_which_two_ndarrays_are_converted is None and parameters.ndarray_length_to_which_two_ndarrays_are_converted_in_vector_comparison is None too. Please set one of them.')

    if set_two_arrays_to_same_length_and_calculate_similarity_value:
        # same_length_ndarray1 = ndarray1
        # same_length_ndarray2 = ndarray2
        same_length_ndarray1 = my_utils.data_structure_helpers.interpolate_or_shrink_ndarray_to_length_n(
            a_ndarray=ndarray1, n=ndarray_length_to_which_two_ndarrays_are_converted)
        same_length_ndarray2 = my_utils.data_structure_helpers.interpolate_or_shrink_ndarray_to_length_n(
            a_ndarray=ndarray2, n=ndarray_length_to_which_two_ndarrays_are_converted)

        # print (f"{hash_funcs.hash_a_ndarray(same_length_ndarray1)=}")
        # print (f"{hash_funcs.hash_a_ndarray(same_length_ndarray2)=}")

        a_similarity_value = similarity_calculation_func_obj(same_length_ndarray1, same_length_ndarray2)
        # print (f"{a_similarity_value=}")
    else:
        a_similarity_value = similarity_calculation_func_obj(ndarray1, ndarray2)

    if PrmsContainer.does_vector_similarity_calculation_func_return_distance:
        a_similarity_value = a_similarity_value * (-1)

    return a_similarity_value


''' ====================================================================================================================
Test Code
'''
if __name__ == '__main__':
    sample_nd_array_short1 = np.array([2, 1, 2, 3, 2, 9])
    sample_nd_array_short2 = np.array([20, 10, 20, 30, 20, 90])

    sample_nd_array_long1 = np.array(
        my_utils.file_and_dir_interaction_helpers.get_a_list_from_text_file("zzz-numberSequenceDataFiles/00-HandDrawn1.txt"))  # length = 1000
    sample_nd_array_long2 = np.array(
        my_utils.file_and_dir_interaction_helpers.get_a_list_from_text_file("zzz-numberSequenceDataFiles/00-HandDrawn4.txt"))  # length = 1000

    sample_nd_array_long3 = np.array(
        my_utils.file_and_dir_interaction_helpers.get_a_list_from_text_file("zzz-numberSequenceDataFiles/00-HandDrawn1.txt"))  # length = 1000
    sample_nd_array_long4 = np.array(
        my_utils.file_and_dir_interaction_helpers.get_a_list_from_text_file("zzz-numberSequenceDataFiles/00-HandDrawn3.txt"))  # length = 60
    sample_nd_array_long5 = np.array(
        my_utils.file_and_dir_interaction_helpers.get_a_list_from_text_file("zzz-numberSequenceDataFiles/00-HandDrawn5.txt"))  # length = 200
    sample_nd_array_long6 = np.array(
        my_utils.file_and_dir_interaction_helpers.get_a_list_from_text_file("zzz-numberSequenceDataFiles/00-HandDrawn6.txt"))  # length = 7

    print(compute_L2_norm(sample_nd_array_short1, sample_nd_array_short2))
    print(compute_cosine_similarity(sample_nd_array_short1, sample_nd_array_short2))
    print(compute_absolute_value_of_pearson_correlation_coefficient(sample_nd_array_short1, sample_nd_array_short2))
    print(compute_mean_deviation_cosine_similarity(sample_nd_array_short1, sample_nd_array_short2))
    print('---------------------------')
    print(compute_L2_norm(sample_nd_array_long1, sample_nd_array_long2))
    print(compute_cosine_similarity(sample_nd_array_long1, sample_nd_array_long2))
    print(compute_absolute_value_of_pearson_correlation_coefficient(sample_nd_array_long1, sample_nd_array_long2))
    print(compute_mean_deviation_cosine_similarity(sample_nd_array_long1, sample_nd_array_long2))
