import datetime
import my_utils.file_and_dir_interaction_helpers
import my_utils.terminal_interaction_helpers
from src_for_form_sampling.vector_similarity_and_matching import matching_between_audio_extacted_data_and_a_set_of_target_shapes
from src_for_form_sampling.parameters_package import PrmsContainer
import random
from my_utils import misc_helpers as mh
from src_for_form_sampling import parameters_package
import mlflow
from src_for_form_sampling.vector_similarity_and_matching import similarity_matching_db_manager
import visualize_matching
from src_for_form_sampling.vector_similarity_and_matching.matching_between_audio_extacted_data_and_a_set_of_target_shapes import _SimilarityMatrix
from my_utils import my_db_utils
import my_utils as mu


def one_similarity_computation_common():
    ####################################################################################################################
    # ------------------------------------- STEP1: Execute Program ------------------------------------------
    ####################################################################################################################
    execution_start_time = datetime.datetime.now()

    random.seed(PrmsContainer.rand_seed)

    similarity_matching_db_manager._this_db_instance = my_db_utils.DbOperators(db_type='sqlite', sqlite_db_path=PrmsContainer.path_to_similarity_matching_db)

    _SimilarityMatrix.similarity_calculation_func_obj = PrmsContainer.vector_similarity_calculation_func_obj
    _SimilarityMatrix.similarity_calculation_func_name_str = PrmsContainer.vector_similarity_calculation_func_obj.__name__
    _SimilarityMatrix.similarity_matrix_json_folder_path = PrmsContainer.path_to_similarity_matrix_json_cache_directory
    _SimilarityMatrix.common_vector_length_to_which_all_arrays_are_converted = PrmsContainer.ndarray_length_to_which_two_ndarrays_are_converted_in_vector_comparison
    _SimilarityMatrix.array_preprocessing_scaling_method_obj = PrmsContainer.scaling_method_for_array_comparison
    _SimilarityMatrix.array_preprocessing_scaling_method_str = PrmsContainer.scaling_method_for_array_comparison.__name__

    matching_between_audio_extacted_data_and_a_set_of_target_shapes.RankAndMatchAudioSegmentsAccordingToASetOfTargetShapes(
        path_to_extracted_audio_features_db=PrmsContainer.path_to_db_that_stores_audio_feature_extraction_results,
        target_shape_json_folder_path=PrmsContainer.relative_path_to_target_shape_arrays_json_files
    )

    total_execution_time = str(datetime.datetime.now() - execution_start_time)

    ####################################################################################################################
    # ------------------------------------- STEP2: Log Parameters and Results ------------------------------------------
    ####################################################################################################################
    mlflow.set_tracking_uri(uri=f'file://{mh.to_absolute_path(PrmsContainer.path_to_dir_of_mlflow_run_results_for_similarity_computation)}')
    mlflow.set_experiment(experiment_name=mh.get_module_python_file_name(__name__))
    with mlflow.start_run(run_name=''):
        mlflow.log_text(total_execution_time, artifact_file=mh.varnameof(total_execution_time)+'.txt')

        mlflow.log_artifact(local_path=PrmsContainer.path_to_similarity_matching_db, artifact_path='')

        num_of_rows = similarity_matching_db_manager.View_ListOfMaximumSimilarityValues_Between_AudioSegment_And_SetOfTargetShapes.get_number_of_rows_currently_in_db()
        for i in range(1, num_of_rows+1):
            an_image_pdf_path = visualize_matching.create_an_matching_image_file_and_return_the_path_to_it(nth=i)
            mlflow.log_artifact(local_path=an_image_pdf_path, artifact_path='similarity_matching_pdf_images')
            my_utils.delete_a_file(an_image_pdf_path, confirmation_needed=False)

        audio_files_list = similarity_matching_db_manager.View_ListOfMaximumSimilarityValues_Between_AudioSegment_And_SetOfTargetShapes.get_list_of_paths_to_audios_used_for_matching()
        mlflow.log_dict(audio_files_list, artifact_file='list_of_audio_files_used_for_matching.yaml')
        # mlflow.log_text(text='\n'.join(audio_files_list), artifact_file='list_of_audio_files_used_for_matching.txt')

        mlflow.log_artifacts(local_dir=PrmsContainer.relative_path_to_target_shape_arrays_json_files, artifact_path='target_shape_arrays_json_files')

        prm_dict = mh.get_dict_of_user_defined_public_class_attributes_excluding_methods(PrmsContainer)
        prm_dict = mu.convert_values_of_absolute_path_strings_in_a_dict_into_relative_path_strings(prm_dict)
        mlflow.log_params(prm_dict)

        mlflow.log_artifact(PrmsContainer.PATH_TO_A_CONFIG_FILE_THAT_WAS_USED_TO_SET_UP_SPECIFIC_PRMS, artifact_path='')

        # TODO: 本当はSimilarity MatrixのJsonファイルもそのままコピーして保存したい。Cacheフォルダーに入れちゃったらあとでどうしたらよいかわからなくなる。


def one_similarity_driver__with_user_confirmation__set_prms_from_python_parameters_file():
    my_utils.confirm_dangerous_operation_with_kboard_input_or_exit_from_sys(
        "Are you using 'time' command?"
    )

    if my_utils.ask_for_input_on_terminal_and_get_true_or_false(f'Deleting all json files in {PrmsContainer.path_to_similarity_matrix_json_cache_directory}?'):
        my_utils.delete_all_jsons_in_a_folder(PrmsContainer.path_to_similarity_matrix_json_cache_directory)
        my_utils.terminal_message_start_print()
        print('similarity matrix jsons are now deleted!!!')
        my_utils.terminal_message_end_print()

    my_utils.terminal_interaction_helpers.ask_for_input_on_terminal_and_get_true_or_false(
        f"Proceed with Current Parameter Configuration? \n {[a_cls.__name__ for a_cls in parameters_package.prms_stack]}"
    )

    parameters_package.set_up_params_for_a_single_run()
    one_similarity_computation_common()

    my_utils.delete_a_file(PrmsContainer.path_to_similarity_matching_db, confirmation_needed=True)
    my_utils.delete_a_file(PrmsContainer.PATH_TO_A_CONFIG_FILE_THAT_WAS_USED_TO_SET_UP_SPECIFIC_PRMS, confirmation_needed=True)


def one_similarity_driver__without_user_confirmation__set_prms_from_yaml_config_file():
    """
    This func is intended to be used in parallel executions of the same python script...So, no user interaction via terminal is provided.
    """
    path_to_yaml_config_file = _config_path
    PrmsContainer.set_values_from_structured_yaml_config_file(path_to_yaml_conifg=path_to_yaml_config_file)
    one_similarity_computation_common()


'''
#######################################################################################################################
######################################## if __name__ == '__main__': ###################################################
#######################################################################################################################
'''
if __name__ == "__main__":
    # Get and Validate Command Line Argument ----------------------------------
    positional_args = my_utils.terminal_interaction_helpers.parse_list_of_positional_arguments_from_terminal(
        ["user_confirmation_needed_or_not (y_or_n)", "config_yaml_path_or_n"])
    assert isinstance(positional_args, list) and len(positional_args) == 2

    if positional_args[0] in ('y', 'yes'):
        _user_confirmation_needed: bool = True
    elif positional_args[0] in ('n', 'no'):
        _user_confirmation_needed: bool = False
    else:
        raise Exception("The first positional argument must be either 'y' , 'n', 'yes', or 'no")

    if positional_args[1].endswith('.yaml') or positional_args[1].endswith('.yml'):
        _config_path = positional_args[1]
    elif positional_args[1] in ('n', 'no'):
        _config_path = None
    else:
        raise Exception("The second positional argument must be either 'n' or a path to a yaml file.")

    # run a driver -------------------------------------------------------------
    if _user_confirmation_needed and _config_path is None:
        one_similarity_driver__with_user_confirmation__set_prms_from_python_parameters_file()
    elif not _user_confirmation_needed and _config_path is not None:
        one_similarity_driver__without_user_confirmation__set_prms_from_yaml_config_file()
    else:
        raise Exception("The combination of the two positional arguments is not valid.")
