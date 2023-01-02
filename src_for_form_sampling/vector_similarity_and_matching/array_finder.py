import tssearch

import my_utils.data_structure_helpers
from my_utils import misc_helpers as gh
from my_utils import plot_creators
import numpy as np


def find_an_array_segment_like_ndarray1_in_ndarray2_and_plot(ndarray1,
                                                             ndarray2,
                                                             distance_measurement_str="Dynamic Time Warping",
                                                             array_scaling_method_obj=None):
    # assert arguments-------------------------------------------------------------------------------------------------
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
    # assert ndarray1.size <= ndarray2.size, 'Do you mean to find ndarray2 in ndarray1, rather than finding ndarray1 in ndarray2'

    possible_distance_measurement_str_list = [
        "Short Time Series Distance",
        "Pearson Correlation Distance",
        "Dynamic Time Warping",
        "Longest Common Subsequence",
        "Time Warp Edit Distance"
    ]
    assert distance_measurement_str in possible_distance_measurement_str_list

    # Scale arrays ----------------------------------------------------------------------------------------------------
    if array_scaling_method_obj is None:
        from my_utils import data_scalers
        array_scaling_method_obj = data_scalers.min_max_normalization

    ndarray1 = array_scaling_method_obj(ndarray1)
    ndarray2 = array_scaling_method_obj(ndarray2)

    # array query (core logic of this function)-------------------------------------------------------------------------
    query_array = ndarray1
    inquired_array = ndarray2  # array for which

    cfg = tssearch.get_distance_dict([distance_measurement_str])
    if distance_measurement_str in ['Dynamic Time Warping', 'Longest Common Subsequence']:
        cfg['elastic'][distance_measurement_str]['parameters']['alpha'] = 0.5

    out = tssearch.time_series_search(cfg, query_array, inquired_array)

    # print (out[distance_measurement_str].keys())
    # print (out[distance_measurement_str]['path_dist'])
    # print (out[distance_measurement_str]['distance'])
    # print (out[distance_measurement_str]['start'])
    # print (out[distance_measurement_str]['end'])
    # print (out[distance_measurement_str]['path'])

    query_arr_indexes = out[distance_measurement_str]['path'][0][0]
    # print (query_arr_indexes)
    sequence_arr_indexes = out[distance_measurement_str]['path'][0][1]
    # print (sequence_arr_indexes)

    a_segment_from_query_array = query_array[query_arr_indexes[0]: query_arr_indexes[-1]]
    a_segment_from_inquired_array = inquired_array[sequence_arr_indexes[0]: sequence_arr_indexes[-1]]

    query_array_graph_title = f'Ndarray1\n' \
        '(the last three elements are not included...perhaps, this is a bug in tssearch library. \n ' \
        'Or, I can just add three meaningless values to the end of ndarray1 at the beginning of this function)'
    a_found_segment_graph_title = f'Found Segment from Ndarray2 \n' \
        f'Number of data points in original ndarray2 = {ndarray2.size} \n' \
        f'Found array is from index {sequence_arr_indexes[0]} to index {sequence_arr_indexes[-1]}'

    # Show Matched Portions of Array (left=array shape that I wnat to find, right=array shape that is found)------------
    plot_creators.plot_multiple_x_y_graphs(
        list_of_Ys=[a_segment_from_query_array, a_segment_from_inquired_array],
        share_vertical_plot_scaling=False,
        list_of_graph_titles=[query_array_graph_title, a_found_segment_graph_title]
    )


if __name__ == "__main__":
    raise Exception('WARNING！！！:以下のコードとParametersはそのまま論文に使えるので変えないようにする。')

    sample_ndarray1, sample_ndarray2 = gh.return_two_example_ndarrays()

    a_chopin_ndarray = json_handlers.JsonReader(
        'zzz-numberSequenceDataFiles_escaped/12-chopin_pianosonatano2_4thMovement_highestNotesOfEachChord.json').read_as_ndarray()
    a_ravel_ndarray = json_handlers.JsonReader(
        'zzz-numberSequenceDataFiles_escaped/12-highestNotesOfEachChord-RavelPrelude.json').read_as_ndarray()

    a_score_extracted_ndarray = a_ravel_ndarray

    # tanigami_ade = gsaade('zz-audioFiles_honban/Tanigami-STE-006.wav')
    # hanakuma_ade = gsaade('zz-audioFiles_excaped/hankyu_hanakuma_field_recording.wav')
    okamoao_ade = gsaade('zz-audioFiles_excaped/8000_series_Shukuugawa_to_Okamoto.wav')
    # gto_ade = gsaade('zz-audioFiles_excaped/Hankyu-8000-1.wav')
    an_audio_extracted_ndarray = okamoao_ade.compute_rmse_enegy()[0]
    an_audio_extracted_ndarray = my_utils.data_structure_helpers.average_a_ndarray_over_n_elements(an_audio_extracted_ndarray, 50)

    ndarray1 = a_score_extracted_ndarray
    ndarray2 = an_audio_extracted_ndarray
    graph_generators.plot_one_x_y_graph(Y=ndarray1, graph_title='I need to find this array')
    graph_generators.plot_one_x_y_graph(Y=ndarray2, graph_title='Find in this array')

    find_an_array_segment_like_ndarray1_in_ndarray2_and_plot(ndarray1, ndarray2)
