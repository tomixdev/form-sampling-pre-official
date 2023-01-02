import pprint
import sys
import copy
import tssearch
import time
import inspect

import my_utils as mu
from my_utils import yaml_handlers
import itertools
from my_utils import data_scalers
from src_for_form_sampling.vector_similarity_and_matching import vector_similarity_calculators

'''
########################################################################################################################
######################################  Below are UNstructured parameters  #############################################
########################################################################################################################
'''

class PrmsContainer:
    PATH_TO_A_CONFIG_FILE_THAT_WAS_USED_TO_SET_UP_SPECIFIC_PRMS = 'dummy'
    PATH_TO_COMPUTED_DATA_DIR = mu.to_absolute_path('./../../data_computed/')
    PATH_TO_UNPROCESSED_DATA_DIR = mu.to_absolute_path('./../../data_unprocessed/')

    scaling_method_for_array_comparison = None
    vector_similarity_calculation_func_obj = None
    does_vector_similarity_calculation_func_return_distance = None
    ndarray_length_to_which_two_ndarrays_are_converted_in_vector_comparison = None

    ambience_reducer_list_of_possible_percentiles_below_which_numbers_are_set_to_zero = None

    warning_with_traceback: bool = None
    completely_supress_all_warnings: bool = None
    supress_printing_my_debug_info: bool = None
    enable_profiling: bool = None

    analyze_audio_even_if_audio_hash_value_is_already_in_db = None  # If the program has not changed from previous run, I can change this to False.

    # ##################################################################################################################
    # Path to Directories and Files ------------------------------------------------------------------------------------
    path_to_audio_folder = None

    path_to_dir_of_mlflow_run_results_for_audio_feature_extraction = PATH_TO_COMPUTED_DATA_DIR + 'extracted_audio_features_data/mlruns/'
    path_to_audio_extracted_features_cache_directory = PATH_TO_COMPUTED_DATA_DIR + 'extracted_audio_features_data/cached_pickles/'

    path_to_dir_that_stores_audio_feature_extraction_db_temporarily = PATH_TO_COMPUTED_DATA_DIR + 'temp_files_before_organized_into_mlruns/'

    # path_to_db_that_stores_audio_feature_extraction_results = path_to_dir_that_stores_audio_feature_extraction_db_temporarily \
    #                                                          + 'extracted_audio_feature_' + mu.get_current_time_ns_str() + '.sqlite'
    path_to_db_that_stores_audio_feature_extraction_results = path_to_dir_that_stores_audio_feature_extraction_db_temporarily \
        + 'extracted_audio_feature_20221119_005748_157526793387708_2260629.sqlite'

    relative_path_to_target_shape_arrays_json_files = None

    path_to_dir_of_mlflow_run_results_for_similarity_computation = PATH_TO_COMPUTED_DATA_DIR + 'similarity_matching_data/mlruns/'
    path_to_similarity_matrix_json_cache_directory = PATH_TO_COMPUTED_DATA_DIR + 'similarity_matching_data/cached_similarity_matrixes/'
    path_to_dir_that_stores_similarity_matching_db_temporarily = PATH_TO_COMPUTED_DATA_DIR + 'temp_files_before_organized_into_mlruns/'
    path_to_similarity_matching_db = path_to_dir_that_stores_similarity_matching_db_temporarily + 'similarity_matching_' + mu.get_current_time_ns_str() + '.sqlite'

    path_to_dir_to_save_matching_figure_images = PATH_TO_COMPUTED_DATA_DIR + 'temp_files_before_organized_into_mlruns/'


    path_to_dir_of_parameter_config_files_generated_dynamically_at_run_time = PATH_TO_COMPUTED_DATA_DIR \
        + 'temp_files_before_organized_into_mlruns/' \
        + 'parameter_config_files_generated_dynamically_at_run_time/'

    # ##################################################################################################################
    # Audio Data Extraction Param ------------------------------------------------------------------------------------
    minimum_required_audio_duration_of_audio_folder_in_s = None

    # For the meaning of the following three parameters, look at behavior1.jpeg
    minimum_audio_segment_duration_in_s = None
    audio_segment_duration_increase_in_s = None
    audio_segment_interval_in_s = None
    maximum_audio_segment_duration_in_s = None

    extracted_data_rounding_decimals = None

    # ##################################################################################################################
    # Parameters that I rarely need to change..... ---------------------------------------------------------------------
    rand_seed = 9999
    # If I set this true, strings like 'prm1: 20, prm2:40" will be save to sql. If I turn this to false, strings like "func_name: over_tone_variety, a_small_constant:20" wiil be saved to sql.
    save_intelligible_ids_for_audio_extraction_functions = False
    convert_nan_to_min_value_of_in_results_of_audio_data_extraction_functions = True

    how_many_times_to_retry_db_access = None
    interval_in_s_between_db_access_retries = 0.01

    ''' ################################################################################################################
    ################################## Class Methods of Prms Container #################################################
    #################################################################################################################'''
    @classmethod
    def validate_values(cls):
        if not cls.minimum_audio_segment_duration_in_s <= cls.maximum_audio_segment_duration_in_s:
            raise Exception(f"{cls.minimum_audio_segment_duration_in_s=}, {cls.maximum_audio_segment_duration_in_s=}")

        if not (cls.minimum_audio_segment_duration_in_s > 0 and cls.audio_segment_duration_increase_in_s > 0 and cls.audio_segment_interval_in_s > 0):
            raise Exception(f"{cls.minimum_audio_segment_duration_in_s=}, {cls.audio_segment_duration_increase_in_s=}, {cls.audio_segment_interval_in_s=}")

        # ----------------------------------------------------------------------------------------------------------------------
        cls.print_all_none_variables_to_terminal_and_terminate_the_program_if_a_none_variable_exists()
        # datatypes assertions-----------------------------------------------------------------------------------------
        mu.assert_class(cls.warning_with_traceback, bool)
        mu.assert_class(cls.completely_supress_all_warnings, bool)
        mu.assert_class(cls.supress_printing_my_debug_info, bool)
        mu.assert_class(cls.enable_profiling, bool)
        mu.assert_class(cls.how_many_times_to_retry_db_access, int)
        mu.assert_class(cls.interval_in_s_between_db_access_retries, (int, float))
        assert 0.0 < cls.interval_in_s_between_db_access_retries < 1.0

        mu.raise_exception_if_path_to_a_folder_is_in_wrong_format(cls.PATH_TO_COMPUTED_DATA_DIR)
        assert mu.does_a_dir_exist(cls.PATH_TO_COMPUTED_DATA_DIR)
        mu.raise_exception_if_path_to_a_folder_is_in_wrong_format(cls.PATH_TO_UNPROCESSED_DATA_DIR)
        assert mu.does_a_dir_exist(cls.PATH_TO_UNPROCESSED_DATA_DIR)

        a_str = mu.get_a_str_before_last_slash(cls.path_to_db_that_stores_audio_feature_extraction_results) + '/'
        assert mu.does_a_dir_exist(a_str), cls.path_to_db_that_stores_audio_feature_extraction_results
        a_str = mu.get_a_str_before_last_slash(cls.path_to_similarity_matrix_json_cache_directory) + '/'
        assert mu.does_a_dir_exist(a_str), cls.path_to_similarity_matrix_json_cache_directory

        mu.does_a_dir_exist(cls.path_to_dir_of_parameter_config_files_generated_dynamically_at_run_time)

        mu.raise_exception_if_path_to_a_folder_is_in_wrong_format(cls.path_to_dir_of_parameter_config_files_generated_dynamically_at_run_time)
        mu.raise_exception_if_path_to_a_folder_is_in_wrong_format(cls.path_to_audio_folder)
        mu.raise_exception_if_path_to_a_folder_is_in_wrong_format(cls.path_to_audio_extracted_features_cache_directory)
        mu.raise_exception_if_path_to_a_folder_is_in_wrong_format(cls.relative_path_to_target_shape_arrays_json_files)

        mu.does_a_dir_exist(cls.path_to_dir_to_save_matching_figure_images)

        # ----------------------------------------------------------------------------------------------------------------------
        list_of_similarity_calculation_func_strings = ['compute_L1_norm',
                                                       'compute_L2_norm',
                                                       'compute_cosine_similarity',
                                                       'compute_absolute_value_of_pearson_correlation_coefficient',
                                                       'derivative_dynamic_time_warping',
                                                       'compute_mean_deviation_cosine_similarity',
                                                       'dynamic_time_warping',
                                                       'longest_common_subsequence_similarity_measure',
                                                       'longest_common_subsequence_similarity_measure_no_constraint',
                                                       'short_time_series_distance',
                                                       'twed',
                                                       'dlp']

        assert cls.vector_similarity_calculation_func_obj.__name__ in list_of_similarity_calculation_func_strings, '''
        DO NOT Change these vector similarity calculation func names without precaution!!! These strings are used when calculating hash value of similarity matrix between
        audio segment and a set of target shapes
        '''

        # ----------------------------------------------------------------------------------------------------------------------
        list_of_data_scaler_func_strings = ['no_scaling',
                                            'z_score_normalization',
                                            'min_max_normalization',
                                            'robust_scaler']
        assert cls.scaling_method_for_array_comparison.__name__ in list_of_data_scaler_func_strings, '''
        DO NOT Change these scaling function names without precaution!!! These strings are used when calculating hash value of similarity matrix between
        audio'''

        # Value Validations ---------------------------------------------------------------------------------------------------------------------
        assert isinstance(cls.extracted_data_rounding_decimals, int)

        if None in mu.get_dict_of_user_defined_public_class_attributes_excluding_methods(cls).values():
            raise Exception(
                f'{cls.set_values_from_structured_yaml_config_file.__name__} failed '
                f'because some of the variables in {cls.__name__} are still None. '
                f'Please check config file.'
            )

    @classmethod
    def get_variable_names_as_set(cls):
        return set(mu.get_dict_of_user_defined_public_class_attributes_excluding_methods(cls).keys())

    @classmethod
    def get_dict_of_non_none_variables(cls):
        cls_user_defined_variables = mu.get_dict_of_user_defined_public_class_attributes_excluding_methods(cls)
        return {k: v for k, v in cls_user_defined_variables.items() if v is not None}

    @classmethod
    def print_all_none_variables_to_terminal_and_terminate_the_program_if_a_none_variable_exists(cls):
        list_of_none_variable_names = []
        for a_key, a_value in mu.get_dict_of_user_defined_public_class_attributes_excluding_methods(cls).items():
            if a_value is None:
                list_of_none_variable_names.append(a_key)

        if len(list_of_none_variable_names) == 0:
            mu.debuginfo(f"Parameters look good! No None values exist in {cls}. Parameters are set as below: ")
            pprint.pprint(mu.get_dict_of_user_defined_public_class_attributes_excluding_methods(cls))
            mu.terminal_message_end_print()
        else:
            raise Exception(f"List of None Values in {cls} at the point when this function is called: \n"
                            f"{pprint.pformat(list_of_none_variable_names)}")

    @classmethod
    def set_values_from_dict_and_validate_and_change_warning_print_setting(cls, flattened_prm_dict):
        assert isinstance(flattened_prm_dict, dict)
        assert len(mu.get_dict_of_user_defined_public_class_attributes_excluding_methods(cls)) == len(flattened_prm_dict.keys())
        assert not mu.is_a_dict_nested(flattened_prm_dict)

        setA = set(flattened_prm_dict)
        setB = set(mu.get_dict_of_user_defined_public_class_attributes_excluding_methods(cls))
        assert setA.issubset(setB)
        assert mu.are_keys_of_two_dicts_identical(flattened_prm_dict, mu.get_dict_of_user_defined_public_class_attributes_excluding_methods(cls))

        for a_dict_key, a_dict_value in flattened_prm_dict.items():
            if a_dict_key in cls.__dict__.keys():
                setattr(cls, a_dict_key, a_dict_value)
            else:
                raise Exception(f"{a_dict_key} does not exist in the class '{cls.__name__}'")

        cls.validate_values()

        mu.SUPRESS_DEBUG_PRINTING = cls.supress_printing_my_debug_info
        mu.set_terminal_warnings_print_setting(warning_with_traceback=cls.warning_with_traceback,
                                               completely_supress_all_warnings=cls.completely_supress_all_warnings)

    @classmethod
    def set_values_from_structured_yaml_config_file(cls, path_to_yaml_conifg):

        assert mu.does_a_file_exist(path_to_yaml_conifg)

        dict_loaded_from_config = yaml_handlers.load_dict_from_a_yaml(path_to_yaml_conifg)
        a_flattened_prm_dict = mu.flatten_dict(dict_loaded_from_config, check_key_duplicates=True)

        cls.set_values_from_dict_and_validate_and_change_warning_print_setting(a_flattened_prm_dict)



class StructuredPrms:
    @classmethod
    def get_prms_stack(cls):
        return [
            cls.PrmsNotYetSoSure,
            cls.Configs.PrmsForRealExperiments
        ]

    class PrmsNotYetSoSure(PrmsContainer):
        scaling_method_for_array_comparison = (
            data_scalers.z_score_normalization,
            data_scalers.robust_scaler,
            data_scalers.min_max_normalization
        )
        vector_similarity_calculation_func_obj = (
            tssearch.dlp,
            tssearch.short_time_series_distance,
            vector_similarity_calculators.derivative_dynamic_time_warping,
            vector_similarity_calculators.longest_common_subsequence_similarity_measure_no_constraint,
            vector_similarity_calculators.dynamic_time_warping,
            vector_similarity_calculators.compute_absolute_value_of_pearson_correlation_coefficient,
            vector_similarity_calculators.compute_cosine_similarity
        )
        does_vector_similarity_calculation_func_return_distance = True, False

        ndarray_length_to_which_two_ndarrays_are_converted_in_vector_comparison = 100, 1000

        ambience_reducer_list_of_possible_percentiles_below_which_numbers_are_set_to_zero = [0]

    class Configs:
        class PrmsForDebugging(PrmsContainer):
            warning_with_traceback = False
            completely_supress_all_warnings = False
            supress_printing_my_debug_info = False
            enable_profiling = False

            analyze_audio_even_if_audio_hash_value_is_already_in_db = True  # If the program has not changed from previous run, I can change this to False.

            path_to_audio_folder = PrmsContainer.PATH_TO_UNPROCESSED_DATA_DIR + "audio/small_toy_data_audio_files/"
            relative_path_to_target_shape_arrays_json_files = PrmsContainer.PATH_TO_UNPROCESSED_DATA_DIR + 'target_shape_arrays/test_num_arrays/'

            # TODO: After finishing the experiment stage of this program, change this value to 30.0 or so.
            minimum_required_audio_duration_of_audio_folder_in_s = 0.0

            # For the meaning of the following three parameters, look at behavior1.jpeg
            minimum_audio_segment_duration_in_s = 120.0  # TODO: Later Change This to 3 minutes or 4 minutes
            audio_segment_duration_increase_in_s = 120.0  # TODO: Later change this value
            audio_segment_interval_in_s = 30.0  # TODO: Later change this value
            maximum_audio_segment_duration_in_s = 180.0

            extracted_data_rounding_decimals = 4  # TODO: これを大きくすると、小さなSegmentで音量が弱いけど、弱い音量の中での面白い動きなどを誇張して捉えられそう。そういう値ってroundすると、計算機上0になっちゃって、roundすると出てこないことがあり得ると思われる。

            how_many_times_to_retry_db_access = 10000

        class PrmsForRealExperiments(PrmsContainer):
            warning_with_traceback = False
            completely_supress_all_warnings = True
            supress_printing_my_debug_info = True
            enable_profiling = False

            analyze_audio_even_if_audio_hash_value_is_already_in_db = True  # If the program has not changed from previous run, I can change this to False.

            path_to_audio_folder = PrmsContainer.PATH_TO_UNPROCESSED_DATA_DIR + "audio/20221024_Hangzhou_Xifu/"
            relative_path_to_target_shape_arrays_json_files = PrmsContainer.PATH_TO_UNPROCESSED_DATA_DIR + 'target_shape_arrays/honban_num_arrays/'

            minimum_required_audio_duration_of_audio_folder_in_s = 120.0
            minimum_audio_segment_duration_in_s = 120.0  # TODO: Later Change This to 3 minutes or 4 minutes
            audio_segment_duration_increase_in_s = 20.0  # TODO: Later change this value
            audio_segment_interval_in_s = 20.0  # interval between starting points of time point tuple TODO: Later change this value
            maximum_audio_segment_duration_in_s = 360.0  # 6 minutes

            extracted_data_rounding_decimals = 10  # TODO: これを大きくすると、小さなSegmentで音量が弱いけど、弱い音量の中での面白い動きなどを誇張して捉えられそう。そういう値ってroundすると、計算機上0になっちゃって、roundすると出てこないことがあり得ると思われる。

            how_many_times_to_retry_db_access = 10000

    '''
    ====================================================================================================================
    --------------------------------------Static and Class Methods Below------------------------------------------------
    ====================================================================================================================
    '''
    @classmethod
    def save_a_single_prms_combo_to_config_file_and_return_the_config_file_path(cls):
        """
        Note that this function does two things:
            (1) save parameters to a config file
            (2) return the path to the config file

        Flattenされたデータでなく、構造化されたデータをConfigに保存する。
        """
        entire_relative_path_to_config_file = PrmsContainer.path_to_dir_of_parameter_config_files_generated_dynamically_at_run_time \
            + mu.get_current_time_ns_str() + '.yaml'

        PrmsContainer.PATH_TO_A_CONFIG_FILE_THAT_WAS_USED_TO_SET_UP_SPECIFIC_PRMS = entire_relative_path_to_config_file

        cls._validate_variable_names_in_parameter_stack()

        entire_dict_to_save_to_yaml = {}
        for APrmsClass in cls.get_prms_stack():
            assert inspect.isclass(APrmsClass)

            APrmsClass_variables_as_dict = mu.get_dict_of_user_defined_public_class_attributes_excluding_methods(APrmsClass)

            # もし　a valueがtupleなら最初のElementをValueとする。
            for k, v in APrmsClass_variables_as_dict.items():
                if isinstance(v, tuple):
                    v = copy.deepcopy(v[0])
                else:
                    v = copy.deepcopy(v)
                APrmsClass_variables_as_dict[k] = v

            entire_dict_to_save_to_yaml.update({APrmsClass.__name__: APrmsClass_variables_as_dict})

        prms_container_dict = PrmsContainer.get_dict_of_non_none_variables()
        entire_dict_to_save_to_yaml.update({PrmsContainer.__name__: prms_container_dict})

        yaml_handlers.save_dict_to_yaml(a_dict=entire_dict_to_save_to_yaml,
                                        filepath=entire_relative_path_to_config_file)

        return entire_relative_path_to_config_file

    @classmethod
    def save_multiple_parameter_combos_to_config_files_and_return_the_path_to_config_files_dir(cls):

        cls._validate_variable_names_in_parameter_stack()

        flattened_all_prms_dict = {}
        flattened_all_prms_dict.update(PrmsContainer.get_dict_of_non_none_variables())
        for APrmClass in (cls.get_prms_stack()):
            variables_dict_of_APrmClass = mu.get_dict_of_user_defined_public_class_attributes_excluding_methods(APrmClass)
            flattened_all_prms_dict.update(variables_dict_of_APrmClass)

        list_of_all_prms_variable_names = []
        list_of_all_values = []
        for k, v in flattened_all_prms_dict.items():
            list_of_all_prms_variable_names.append(k)
            list_of_all_values.append(v)

        for i in range(len(list_of_all_values)):
            if isinstance(list_of_all_values[i], tuple):
                list_of_all_values[i] = list(list_of_all_values[i])  # Transform tuple into a list for the easiness of itertools.product later
            else:
                # Transform primitive values into a list of one element for the easiness of itertools.product later
                list_of_all_values[i] = [list_of_all_values[i]]

        directory_to_store_parameter_config_files = PrmsContainer.path_to_dir_of_parameter_config_files_generated_dynamically_at_run_time \
            + mu.get_current_time_ns_str() + '/'

        mu.create_a_dir_if_not_exists(directory_to_store_parameter_config_files)

        for a_list_of_prm_values in itertools.product(*list_of_all_values):
            assert len(list_of_all_prms_variable_names) == len(
                a_list_of_prm_values), f"{len(list_of_all_prms_variable_names)} is not {len(a_list_of_prm_values)}"

            a_flattened_prm_dict_in_which_there_is_no_tuple_values = {}
            for i in range(len(list_of_all_prms_variable_names)):
                k = list_of_all_prms_variable_names[i]
                v = a_list_of_prm_values[i]
                a_flattened_prm_dict_in_which_there_is_no_tuple_values[k] = v

            a_specific_config_file_name = mu.get_current_time_ns_str() + '.yaml'
            an_entire_path_to_a_specific_prm_config = directory_to_store_parameter_config_files + a_specific_config_file_name
            a_flattened_prm_dict_in_which_there_is_no_tuple_values[mu.varnameof(PrmsContainer.PATH_TO_A_CONFIG_FILE_THAT_WAS_USED_TO_SET_UP_SPECIFIC_PRMS)] \
                = an_entire_path_to_a_specific_prm_config

            a_flattened_prm_dict_in_which_there_is_no_tuple_values[mu.varnameof(PrmsContainer.path_to_similarity_matching_db)] \
                = PrmsContainer.path_to_dir_that_stores_similarity_matching_db_temporarily + 'similarity_matching_' + mu.get_current_time_ns_str() + '.sqlite'

            yaml_handlers.save_dict_to_yaml(a_dict=a_flattened_prm_dict_in_which_there_is_no_tuple_values,
                                            filepath=an_entire_path_to_a_specific_prm_config)

        return directory_to_store_parameter_config_files

    @classmethod
    def _validate_variable_names_in_parameter_stack(cls):
        assert isinstance(cls.get_prms_stack(), list), cls.get_prms_stack()

        flattened_parameters_dict = {}
        flattened_parameters_dict.update(PrmsContainer.get_dict_of_non_none_variables())

        for APrmsClass in cls.get_prms_stack():
            assert inspect.isclass(APrmsClass)  # TODO: Not sure if this is the right way to check whether an element is a class

            APrmsClass_variables_as_dict = mu.get_dict_of_user_defined_public_class_attributes_excluding_methods(APrmsClass)
            APrmsClass_user_defined_class_variable_names_as_set = set(APrmsClass_variables_as_dict.keys())

            assert len(APrmsClass_user_defined_class_variable_names_as_set) >= 1
            assert mu.do_all_class_attributes_of_classA_inherit_some_class_attributes_of_classB(classA=APrmsClass, classB=PrmsContainer)
            assert not mu.do_two_sets_intersect(set(flattened_parameters_dict.keys()), APrmsClass_user_defined_class_variable_names_as_set)

            flattened_parameters_dict.update(APrmsClass_variables_as_dict)

        if len(list(flattened_parameters_dict.keys())) == len(list(PrmsContainer.get_variable_names_as_set())) and \
                set(flattened_parameters_dict.keys()) == PrmsContainer.get_variable_names_as_set():
            pass
        else:
            mu.debuginfo(f"{PrmsContainer.__name__} has the following variable names:")
            pprint.pprint(PrmsContainer.get_dict_of_non_none_variables(), indent=4)
            mu.debuginfo(f"Below are the values that I defined directly in {PrmsContainer} and {StructuredPrms}")
            pprint.pprint(flattened_parameters_dict, indent=4)
            if len(list(flattened_parameters_dict.keys())) < len(list(PrmsContainer.get_variable_names_as_set())):
                mu.debuginfo(f"{PrmsContainer.__name__} contains more parameters than {mu.varnameof(flattened_parameters_dict)}:")
                pprint.pprint(PrmsContainer.get_variable_names_as_set().difference(set(flattened_parameters_dict.keys())), indent=4)
            else:
                mu.debuginfo(f"{mu.varnameof(flattened_parameters_dict)} contains more parameters than {PrmsContainer.__name__}:")
                pprint.pprint(set(flattened_parameters_dict.keys()).difference(PrmsContainer.get_variable_names_as_set()))
            mu.terminal_message_end_print()
            raise Exception

        if all([a_value is not None for a_value in flattened_parameters_dict.values()]):
            pass
        else:
            mu.terminal_message_start_print()
            print(f"None value(s) exist in {mu.varnameof(flattened_parameters_dict)}:")
            for k, v in flattened_parameters_dict.items():
                if v is None:
                    print(f"{k} is None")
            mu.terminal_message_end_print()

    @staticmethod
    def _get_list_of_prm_combos_from_a_cls_variables_dict(cls_variables_as_a_dict):
        assert isinstance(cls_variables_as_a_dict, dict), type(cls_variables_as_a_dict)

        cls_variables_as_a_dict = copy.deepcopy(cls_variables_as_a_dict)

        for a_key in cls_variables_as_a_dict.keys():
            if not isinstance(cls_variables_as_a_dict[a_key], tuple):
                cls_variables_as_a_dict[a_key] = (cls_variables_as_a_dict[a_key],)

        keys = cls_variables_as_a_dict.keys()
        vals = cls_variables_as_a_dict.values()

        return [dict(zip(keys, instance)) for instance in itertools.product(*vals)]
