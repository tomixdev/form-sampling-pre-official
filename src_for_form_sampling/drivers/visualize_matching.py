"""
Standlone Visualising Matching Driver
"""

import sys
import random

from src_for_form_sampling.vector_similarity_and_matching import similarity_matching_db_manager
from src_for_form_sampling.preprocessing_audio_feature_extraction import extracted_audio_features_db_manager
from my_utils import plot_creators
from my_utils import misc_helpers as gh
import argparse
from src_for_form_sampling.parameters_package import PrmsContainer
import numpy as np
from src_for_form_sampling.vector_similarity_and_matching import vector_similarity_calculators
from src_for_form_sampling.vector_similarity_and_matching.matching_between_audio_extacted_data_and_a_set_of_target_shapes import _TargetShapesJsonFolder, _SimilarityMatrix


def create_an_matching_image_file_and_return_the_path_to_it(nth, show_scaled_data_rather_than_original_data_when_plotting=False, print_debug_info_to_terminal=False):
    def create_dict_of_audio_extracted_array_parameters_to_show_in_plot_from_a_parameter_combo_hash_value(a_parameter_combo_hash_value):
        a_parameter_combo_as_dict = extracted_audio_features_db_manager.Table_From_ParameterComboHashValue_To_ParameterCombo.get_a_parameter_combo_dict_from_parameter_combo_hash_value(
            a_parameter_combo_hash_value)
        an_audio_hash_value = a_parameter_combo_as_dict['audio_hash_value']
        an_audio_file_path = extracted_audio_features_db_manager.Table_From_AudioHashValue_To_AudioFileRelativePath.get_value_from_key(
            audio_hash_value=an_audio_hash_value)
        del a_parameter_combo_as_dict['audio_hash_value']
        dict_of_audio_extracted_array_parameters_to_show_in_plot = {'audio': an_audio_file_path, **a_parameter_combo_as_dict}
        return dict_of_audio_extracted_array_parameters_to_show_in_plot

    assert isinstance(nth, int)
    assert nth >= 1, 'nth counts from 1, not 0'
    random.seed(PrmsContainer.rand_seed)

    similarity_matching_db_manager.View_ListOfMaximumSimilarityValues_Between_AudioSegment_And_SetOfTargetShapes.create_or_replace_the_table_ordered_by_similarity_value()

    list_of_audio_extracted_arrays = []
    list_of_dict_of_audio_extracted_array_parameters_to_show_in_plots = []
    list_of_target_shape_arrays = []
    list_of_json_relative_paths = []
    list_of_similarity_values = []

    matching_tuples = similarity_matching_db_manager.View_ListOfMaximumSimilarityValues_Between_AudioSegment_And_SetOfTargetShapes.\
        get_matching_list_of_tuples_of_parameter_combo_hash_value_and_target_shape_hash_value_of_nth_row(nth=nth)

    if print_debug_info_to_terminal:
        print(f"{matching_tuples = }")

    for a_matching_tuple in matching_tuples:
        a_parameter_combo_hash_value = a_matching_tuple[0]
        dict_of_audio_extracted_array_parameters = \
            create_dict_of_audio_extracted_array_parameters_to_show_in_plot_from_a_parameter_combo_hash_value(a_parameter_combo_hash_value)
        a_target_shape_hash_value = a_matching_tuple[1]

        if print_debug_info_to_terminal:
            print(f"{a_parameter_combo_hash_value = }")
            print(f"{a_target_shape_hash_value = }")

        an_audio_extracted_array = extracted_audio_features_db_manager.Table_From_ParameterComboHashValue_To_FinalExtractedData.get_extracted_array_from_parameter_combo_hash_value(
            parameter_combo_hash_value=a_parameter_combo_hash_value)
        a_target_shape_as_list = similarity_matching_db_manager.Table_From_JsonSavedListHashValue_To_JsonSavedList.get_json_saved_list_from_json_saved_list_hash_value(
            a_target_shape_hash_value)
        a_target_shape_as_ndarray = np.array(a_target_shape_as_list)

        # an_audio_extracted_array = parameters.audio_extracted_array_scaling_method(an_audio_extracted_array)
        # a_target_shape_as_ndarray = parameters.json_stored_target_shape_scaling_method(a_target_shape_as_ndarray)

        an_audio_extracted_array = PrmsContainer.scaling_method_for_array_comparison(an_audio_extracted_array)
        a_target_shape_as_ndarray = PrmsContainer.scaling_method_for_array_comparison(a_target_shape_as_ndarray)

        # an_audio_extracted_array, a_target_shape_as_ndarray = \
        #    data_scalers.scale_two_ndarrays_so_the_starting_num_and_maximum_num_become_the_same(an_audio_extracted_array, a_target_shape_as_ndarray)

        if print_debug_info_to_terminal:
            print(f"{an_audio_extracted_array.tolist()[0:5] = }")
            print(f"{a_target_shape_as_ndarray.tolist()[0:5] = }")

        a_json_relative_path = _TargetShapesJsonFolder(
            relative_path_to_a_folder_of_list_shaped_json_files=PrmsContainer.relative_path_to_target_shape_arrays_json_files).dict_from_json_hash_value_to_relative_path[a_target_shape_hash_value]
        a_similarity_value = vector_similarity_calculators.calculate_similarity_value_between_two_ndarrays_with_different_length(
            ndarray1=a_target_shape_as_ndarray,
            ndarray2=an_audio_extracted_array,
            similarity_calculation_func_obj=_SimilarityMatrix.similarity_calculation_func_obj
        )

        if print_debug_info_to_terminal:
            print(f"{a_similarity_value=}")

        '''
        if show_scaled_data_rather_than_original_data_when_plotting:
            an_audio_extracted_array = parameters.scaling_method_for_array_comparison(an_audio_extracted_array)
            a_target_shape_as_ndarray = parameters.scaling_method_for_array_comparison(a_target_shape_as_ndarray)
        '''

        list_of_audio_extracted_arrays.append(an_audio_extracted_array)
        list_of_dict_of_audio_extracted_array_parameters_to_show_in_plots.append(dict_of_audio_extracted_array_parameters)
        list_of_target_shape_arrays.append(a_target_shape_as_ndarray)
        list_of_json_relative_paths.append(a_json_relative_path)
        list_of_similarity_values.append(a_similarity_value)

    assert len(list_of_audio_extracted_arrays) == len(list_of_dict_of_audio_extracted_array_parameters_to_show_in_plots) \
        == len(list_of_target_shape_arrays) == len(list_of_json_relative_paths) == len(list_of_similarity_values)

    list_of_arrays_to_plot = []
    list_of_graph_descriptions = []
    for i in range(len(list_of_audio_extracted_arrays)):
        list_of_arrays_to_plot.append(list_of_audio_extracted_arrays[i])
        list_of_graph_descriptions.append(plot_creators.stringfy_a_dict_for_graph_description(
            list_of_dict_of_audio_extracted_array_parameters_to_show_in_plots[i]))

        list_of_arrays_to_plot.append(list_of_target_shape_arrays[i])
        list_of_graph_descriptions.append(list_of_json_relative_paths[i] +
                                          f"\n similarity = {list_of_similarity_values[i]}" +
                                          f"\n (note that this similarity value might be different from \n"
                                          f"the value used during the calculation if the similarity calculation method uses randomization process. This similarity value \n"
                                          f"is not read from the calculation result, but is newly calculated for this plot generation)")

    the_maximum_similarity_value = similarity_matching_db_manager.View_ListOfMaximumSimilarityValues_Between_AudioSegment_And_SetOfTargetShapes.get_maximum_similarity_value_of_nth_row(
        nth=nth)

    path_to_a_matching_image_file = PrmsContainer.path_to_dir_to_save_matching_figure_images + \
        gh.get_current_time_ns_str() + f"_RankingNo{nth}.pdf"

    plot_creators.plot_multiple_x_y_graphs(
        list_of_Ys=list_of_arrays_to_plot,
        list_of_graph_titles=list_of_graph_descriptions,
        number_of_columns=2,
        width_height_tuple_of_each_fig=(10, 6),
        save_fig=True,
        save_fig_path=path_to_a_matching_image_file,
        big_title_for_all_plots="similarity_sum = " + str(the_maximum_similarity_value),
        share_vertical_plot_scaling=False,
    )

    return path_to_a_matching_image_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('nth', type=int, help='show nth in the ranking')
    args = parser.parse_args()

    assert isinstance(args.nth, int)
    assert args.nth >= 1, 'nth counts from 1, not 0'
    matching_image_file_path = create_an_matching_image_file_and_return_the_path_to_it(args.nth)
    print(f"An Image is created at {gh.to_absolute_path(matching_image_file_path)}")
    sys.exit()
