import numpy as np

import my_utils.data_structure_helpers
import my_utils.file_and_dir_interaction_helpers
import my_utils.terminal_interaction_helpers
from my_utils import misc_helpers as gh
import json
import os
from src_for_form_sampling.parameters_package import PrmsContainer
from . import vector_similarity_calculators
from scipy.optimize import linear_sum_assignment
from . import similarity_matching_db_manager
from src_for_form_sampling.preprocessing_audio_feature_extraction import extracted_audio_features_db_manager
from my_utils import json_handlers
import copy
from my_utils import hash_funcs

'''
TODO: いくつかcopy.deepcopy()をしているところがある。おそらく、大した計算量を取らないところばかりだと思うけど、もし計算量かかり過ぎの疑いが出るなら、
きちっときちっと、本当に必要かどうか検討する。
'''

# ======================================================================================================================

def compute_similarity_matrix_hash_value(a_SimilarityMatrix_obj):
    assert isinstance(a_SimilarityMatrix_obj, _SimilarityMatrix)
    '''
    Compute some hashvalue from vector_similarity_calculation_method, json_folder_hash_value, audio_hash_value, audio_segment_tuple
    '''
    assert isinstance(a_SimilarityMatrix_obj.similarity_calculation_func_name_str, str)
    assert isinstance(a_SimilarityMatrix_obj.json_folder_hash_value, str)
    assert isinstance(a_SimilarityMatrix_obj.audio_hash_value, str)
    assert isinstance(a_SimilarityMatrix_obj.audio_segment_tuple, tuple)
    assert isinstance(a_SimilarityMatrix_obj.common_vector_length_to_which_all_arrays_are_converted, int)
    assert isinstance(a_SimilarityMatrix_obj.array_preprocessing_scaling_method_str, str)

    an_obj = a_SimilarityMatrix_obj
    a_list_as_multiset = [an_obj.similarity_calculation_func_name_str,
                          an_obj.json_folder_hash_value,
                          an_obj.audio_hash_value,
                          an_obj.audio_segment_tuple,
                          an_obj.common_vector_length_to_which_all_arrays_are_converted,
                          an_obj.array_preprocessing_scaling_method_str]
    return hash_funcs.hash_a_list_as_multiset(a_list_as_multiset)

# ======================================================================================================================


class RankAndMatchAudioSegmentsAccordingToASetOfTargetShapes:
    is_AudioExtractedData_View_For_SimilarityMatrixCalculation_created = False

    def __init__(self, path_to_extracted_audio_features_db, target_shape_json_folder_path):
        # Argument1 of this __init__() function before 20221113 refactoring: path_to_dir_of_audio_set_to_be_matched
        # self.list_of_hash_values_of_audios_whose_audio_segments_are_to_be_matched = \
        #    hash_funcs.compute_list_of_audio_hash_values_in_audio_folder(path_to_dir_of_audio_set_to_be_matched)
        assert isinstance(path_to_extracted_audio_features_db, str), type(path_to_extracted_audio_features_db)
        assert my_utils.does_a_file_exist(path_to_extracted_audio_features_db), path_to_extracted_audio_features_db

        extracted_audio_features_db_manager.AudioExtractedData_View_For_SimilarityMatrixCalculation.create_or_replace_the_table()

        if path_to_extracted_audio_features_db != PrmsContainer.path_to_db_that_stores_audio_feature_extraction_results:
            my_utils.terminal_interaction_helpers.confirm_dangerous_operation_with_kboard_input_or_exit_from_sys(
                f"20221113の自分からの謝罪 (preprocessingのparametersファイルとexperimentsのparametersファイルを分けないからこういうことになる。。。): \n "
                f"{PrmsContainer} の {PrmsContainer.path_to_db_that_stores_audio_feature_extraction_results =} は"
                f"audio feature extraction (preprocessing)のときに使えば、extracted audio featuresを saveする先ということになる。"
                f"同じvariableだけど、Similarty matchingのときにつかうと、それは、similarity matchimgのときに、target shapesをmatchingが"
                f"行われるaudio features dataのデータベースという意味になる。\n"
                f"したがって、一応この{RankAndMatchAudioSegmentsAccordingToASetOfTargetShapes.__name__}クラスには、{gh.varnameof(path_to_extracted_audio_features_db)}"
                f"という引数があるけど、意味をなしていない。実際にtarget_shape_json_folder_pathにあるtarget shapesとmatchingされるのは"
                f"{PrmsContainer.__name__} クラスの、{gh.varnameof(PrmsContainer.path_to_db_that_stores_audio_feature_extraction_results)}のDBである。"
            )

        assert isinstance(target_shape_json_folder_path, str), type(target_shape_json_folder_path)
        self.a_target_shapes_folder_obj = _TargetShapesJsonFolder(relative_path_to_a_folder_of_list_shaped_json_files=target_shape_json_folder_path)
        self.hash_value_of_set_of_jsons = self.a_target_shapes_folder_obj.hash_value_of_json_folder

        # extracted_audio_features_db_manager.AudioExtractedData_View_For_SimilarityMatrixCalculation.create_or_replace_the_table(
        #    list_of_audio_hash_values_to_include=self.list_of_hash_values_of_audios_whose_audio_segments_are_to_be_matched
        # )
        similarity_matching_db_manager.ListOfMaximumSimilarityValues_Between_AudioSegment_And_SetOfTargetShapes.create_the_table_if_not_exists()

        self._compute_maximum_similarity_value_and_insert_into_db()
        '''
        # TODO: 本当はMultiple RowsをinsertmanyによりいっきにSQLに入れたほうが速いのだろうけど、いったんpythonの中でlistにいれるときに、
        # どれくらいメモリが必要なのかわからないので、一つのRowずつSQLに足すようにしている。
        # 今は_compute_maximum_similarity_value_and_insert_into_dbと名付けているやつ、もともとは_compute_maximum_similarity_value_to_insert_into_dbという名前。
        # 一旦Python Listにtuplesを保存して、それらを一気にSQLに入れるようにしていた。
        self.list_of_new_row_tuples_to_insert_into_db = []
        self._compute_maximum_similarity_value_to_insert_into_db()
        if len(self.list_of_new_row_tuples_to_insert_into_db) >= 1:
            similarity_matching_db_manager.ListOfMaximumSimilarityValues_Between_AudioSegment_And_SetOfTargetShapes.\
                add_multiple_rows_to_the_table(list_of_new_row_tuples=self.list_of_new_row_tuples_to_insert_into_db)
        '''

        similarity_matching_db_manager.View_ListOfMaximumSimilarityValues_Between_AudioSegment_And_SetOfTargetShapes.create_or_replace_the_table_ordered_by_similarity_value()

    def _compute_maximum_similarity_value_and_insert_into_db(self):
        for an_audio_hash_value, a_time_point_tuple\
                in extracted_audio_features_db_manager.AudioExtractedData_View_For_SimilarityMatrixCalculation.get_list_of_tuples_of_audio_hash_value_and_time_point_tuple():

            an_audio_segment_info_obj = _AudioSegmentInfo(audio_hash_value=an_audio_hash_value, time_point_tuple=a_time_point_tuple)

            a_matching_between_an_audio_segment_and_a_set_of_target_shapes = _SimilarityMatrix(
                a_TargetShapesJsonFolder_obj=self.a_target_shapes_folder_obj,
                an_AudioSegmentInfo_obj=an_audio_segment_info_obj
            )

            matching = a_matching_between_an_audio_segment_and_a_set_of_target_shapes.matching_list_of_tuples_of_parameter_combo_hash_value_and_target_shape_hash_value
            maximum_similarity_value = a_matching_between_an_audio_segment_and_a_set_of_target_shapes.maximum_similarity_value

            a_row = (an_audio_hash_value, a_time_point_tuple, self.hash_value_of_set_of_jsons, matching, maximum_similarity_value)
            similarity_matching_db_manager.ListOfMaximumSimilarityValues_Between_AudioSegment_And_SetOfTargetShapes.add_or_replace_a_row(a_row)
            '''
            Row Tuple:
            (audio_hash_value, time_point_tuple, target_shape_hash_value, maximum_similarity_value between audio segment and target shapes)
            '''
            # self.list_of_new_row_tuples_to_insert_into_db.append(a_row)


class _SimilarityMatrix:
    # The class attributes below are meant to be set by outside driver codes!!!
    similarity_calculation_func_obj = None
    similarity_calculation_func_name_str = None
    similarity_matrix_json_folder_path = None
    common_vector_length_to_which_all_arrays_are_converted = None
    array_preprocessing_scaling_method_obj = None
    array_preprocessing_scaling_method_str = None

    def __init__(self, a_TargetShapesJsonFolder_obj, an_AudioSegmentInfo_obj):
        assert isinstance(a_TargetShapesJsonFolder_obj, _TargetShapesJsonFolder)
        assert isinstance(an_AudioSegmentInfo_obj, _AudioSegmentInfo)

        self.json_folder_hash_value = a_TargetShapesJsonFolder_obj.hash_value_of_json_folder
        self.audio_hash_value, self.audio_segment_tuple = an_AudioSegmentInfo_obj.audio_hash_value, an_AudioSegmentInfo_obj.time_point_tuple

        self.similarity_matrix_hash_value = compute_similarity_matrix_hash_value(self)

        similarity_matching_db_manager.Table_From_JsonSavedListHashValue_To_JsonSavedList.create_the_table_if_not_exists()

        self.matching_list_of_tuples_of_parameter_combo_hash_value_and_target_shape_hash_value = None
        self.maximum_similarity_value = None
        path_to_json_cache = self.__class__.similarity_matrix_json_folder_path + self.similarity_matrix_hash_value + '.json'
        if my_utils.file_and_dir_interaction_helpers.does_a_file_exist(path_to_json_cache):
            a_json = json_handlers.JsonReader(path_to_json_file=path_to_json_cache)
            self.matching_list_of_tuples_of_parameter_combo_hash_value_and_target_shape_hash_value = a_json.get_value_from_json_dict(
                key_str=gh.varnameof(self.matching_list_of_tuples_of_parameter_combo_hash_value_and_target_shape_hash_value))
            self.maximum_similarity_value = a_json.get_value_from_json_dict(key_str=gh.varnameof(self.maximum_similarity_value))
        else:
            self.list_of_target_shape_json_hash_values = copy.deepcopy(a_TargetShapesJsonFolder_obj.list_of_json_hash_values)
            self.list_of_target_shape_arrays = copy.deepcopy(a_TargetShapesJsonFolder_obj.list_of_ndarrays)  # Row of Similarity Matrix

            self.list_of_audio_segment_data_parameter_combo_hash_values = an_AudioSegmentInfo_obj.list_of_parameter_combo_hash_values
            self.list_of_audio_segment_data_arrays = an_AudioSegmentInfo_obj.list_of_ndarray_data  # Column of Similarity Matrix

            assert len(self.list_of_audio_segment_data_parameter_combo_hash_values) == len(self.list_of_audio_segment_data_arrays)
            assert len(self.list_of_audio_segment_data_parameter_combo_hash_values) >= len(self.list_of_target_shape_arrays)

            self.similarity_matrix_as_ndarray = None  # Look at "SimilarityMatrix_Explanation.jpeg" for explanation
            self.compute_similarity_matrix()

            '''
            json_handlers.JsonWriter(value_to_save_to_json=self.similarity_matrix_as_ndarray,
                                     path_to_json_file=str(random.uniform(0., 100.)) + 'guaguasomething.json',
                                     delete_json_if_fails=True)
            '''
            # 最終的には以下の値を求めたい!
            self.matching_list_of_tuples_of_parameter_combo_hash_value_and_target_shape_hash_value = []
            self.maximum_similarity_value = None
            self.assign_audio_segment_data_to_target_shapes()

            # delete a few class instance attributes that are big. Deleting these will probably significantly reduce json write time.
            del self.list_of_audio_segment_data_arrays
            del self.list_of_target_shape_arrays
            del self.similarity_matrix_as_ndarray

            json_handlers.JsonWriter(
                value_to_save_to_json=self.__dict__,
                path_to_json_file=path_to_json_cache,
                auto_convert_json_unserializable_datatypes=True,
                delete_json_if_fails=True
            )

    def compute_similarity_matrix(self):
        # TODO: 以下のLogicをpython のProductをつかってかいたほうが、nested for loopsより速いかもしれないけど、
        # ネットをみると、そうでもないぽっかたり、あと、そんなにnested for loopと差がなかったり、そもそもコードが煩雑になるなどの影響があるので、
        # とりあえずよほど、以下のコードが遅いということがなければ、様子見でいく。
        similarity_matrix_as_list_of_list = []
        # TODO: deque is a linked list, which means that we don't have to reallocate memory for the whole container once new elements appear.というふうな
        #  記述を見つけたけど、数千くらいのElementなら、Performance Differenceはほとんどないとも書いてあって、もしかしたらcollection_dequeを
        #  使ったほうがはやくいくかもしれない。collections.dequeを使う場合は、np.arrayを作るときに注意が必要。list of listなら
        #  np.ndarray(list_of_list)で行けるが、deque_of_listなら list(deque_of_list)とlistに変換してから、np.ndarray()のargumentに入れる必要がある。
        #  このlistへの変換はコストがかかるかもしれない。

        for targed_shape_index in range(len(self.list_of_target_shape_arrays)):
            a_target_shape_array = self.list_of_target_shape_arrays[targed_shape_index]
            a_scaled_target_shape_array = self.__class__.array_preprocessing_scaling_method_obj(a_target_shape_array)
            # a_scaled_target_shape_array = a_target_shape_array
            a_row_of_similarity_values = []
            for audio_segment_data_index in range(len(self.list_of_audio_segment_data_arrays)):
                an_audio_segment_data_array = self.list_of_audio_segment_data_arrays[audio_segment_data_index]
                a_scaled_audio_segment_data_array = self.__class__.array_preprocessing_scaling_method_obj(an_audio_segment_data_array)

                a_similarity_value = \
                    vector_similarity_calculators.calculate_similarity_value_between_two_ndarrays_with_different_length(
                        ndarray1=a_scaled_target_shape_array,
                        ndarray2=a_scaled_audio_segment_data_array,
                        similarity_calculation_func_obj=self.__class__.similarity_calculation_func_obj
                    )

                a_row_of_similarity_values.append(a_similarity_value)

            similarity_matrix_as_list_of_list.append(a_row_of_similarity_values)

        self.similarity_matrix_as_ndarray = np.array(similarity_matrix_as_list_of_list, dtype=np.float32)  # TODO: np.float64のほうが良いかもしれない。わからん。

    def assign_audio_segment_data_to_target_shapes(self):
        target_shape_indexes, audio_segment_data_indexes = linear_sum_assignment(self.similarity_matrix_as_ndarray, maximize=True)

        assert isinstance(target_shape_indexes, np.ndarray) and isinstance(audio_segment_data_indexes, np.ndarray)
        assert target_shape_indexes.size == audio_segment_data_indexes.size
        my_utils.data_structure_helpers.assert_nd_array_content_is_starting_from_zero_and_ascending_by_one(target_shape_indexes)

        for i in range(target_shape_indexes.size):
            a_target_shape_index = target_shape_indexes[i]
            a_target_shape_hash_value = self.list_of_target_shape_json_hash_values[a_target_shape_index]

            an_audio_segment_data_index = audio_segment_data_indexes[i]
            an_audio_segment_data_parameter_combo_hash_value = self.list_of_audio_segment_data_parameter_combo_hash_values[an_audio_segment_data_index]

            '''
            a_target_shape_as_array = similarity_matching_db_manager.Table_From_JsonSavedListHashValue_To_JsonSavedList.get_json_saved_list_as_ndarray_from_json_saved_list_hash_value(
                a_target_shape_hash_value)
            an_audio_segment_data_array = extracted_audio_features_db_manager.Table_From_ParameterComboHashValue_To_FinalExtractedData.get_extracted_array_from_parameter_combo_hash_value(
                an_audio_segment_data_parameter_combo_hash_value)

            a_scaled_target_shape_array = parameters.scaling_method_for_array_comparison(a_target_shape_as_array)
            a_scaled_audio_segment_data_array = parameters.scaling_method_for_array_comparison(an_audio_segment_data_array)

            newly_calc_similarity_value = vector_similarity_calculators.calculate_similarity_value_between_two_ndarrays_with_different_length(
                ndarray1=a_scaled_target_shape_array,
                ndarray2=a_scaled_audio_segment_data_array,
                similarity_calculation_func_obj=_SimilarityMatrix.similarity_calculation_func_obj
            )
            '''

            tuple_of_matched_parameter_combo_hash_value_and_target_shape_hash_value = (
                an_audio_segment_data_parameter_combo_hash_value, a_target_shape_hash_value)
            self.matching_list_of_tuples_of_parameter_combo_hash_value_and_target_shape_hash_value.append(
                tuple_of_matched_parameter_combo_hash_value_and_target_shape_hash_value)

        # print (f"{self.similarity_matrix_as_ndarray[target_shape_indexes, audio_segment_data_indexes]=}")

        self.maximum_similarity_value = self.similarity_matrix_as_ndarray[target_shape_indexes, audio_segment_data_indexes].sum()

        # print (f"{self.maximum_similarity_value=}")


class _TargetShapesJsonFolder:
    # json_data_normalization_method = parameters.json_stored_target_shape_scaling_method

    def __init__(self, relative_path_to_a_folder_of_list_shaped_json_files):
        my_utils.file_and_dir_interaction_helpers.raise_exception_if_path_to_a_folder_is_in_wrong_format(relative_path_to_a_folder_of_list_shaped_json_files)

        self.json_folder_path = relative_path_to_a_folder_of_list_shaped_json_files

        self.list_of_paths_to_json_files = []
        self.list_of_ndarrays = []

        self.list_of_json_hash_values = []
        self.list_of_tuples_of_json_hash_value_and_json_relative_path_and_ndarray = []

        self.dict_from_json_hash_value_to_relative_path = {}

        self._set_up_lists_and_dicts_in_this_instance()
        self._save_mapping_from_json_hash_value_to_json_list_into_db()

        assert len(self.list_of_paths_to_json_files) == len(self.list_of_ndarrays) == \
            len(self.list_of_json_hash_values) == len(self.list_of_tuples_of_json_hash_value_and_json_relative_path_and_ndarray)

        '''
        WARNING: Do NOT sort these lists. Each list item corresponds to each other
        '''

        self.hash_value_of_json_folder = hash_funcs.compute_hash_value_of_a_set_of_json_files_from_hash_value_str_list(self.list_of_json_hash_values)

    def _set_up_lists_and_dicts_in_this_instance(self):
        my_utils.file_and_dir_interaction_helpers.raise_exception_if_path_to_a_folder_is_in_wrong_format(self.json_folder_path)

        for filename in os.listdir(self.json_folder_path):
            if filename.startswith('.'):
                continue  # Ignore hidden files like ".DS_store"

            a_relative_path = os.path.join(self.json_folder_path, filename)
            assert os.path.isfile(a_relative_path), f"All files in {self.json_folder_path} needs to be a file of json. Please do not put folders"
            assert a_relative_path[-5:] == '.json'

            with open(a_relative_path) as f:
                data = json.load(f)

            a_relative_path = a_relative_path
            a_ndarray = np.asarray(data)

            a_hash_value = hash_funcs.compute_hash_value_from_json_file(a_relative_path)
            self.dict_from_json_hash_value_to_relative_path[a_hash_value] = a_relative_path

            self.list_of_paths_to_json_files.append(a_relative_path)
            self.list_of_ndarrays.append(copy.deepcopy(a_ndarray))
            self.list_of_json_hash_values.append(a_hash_value)
            self.list_of_tuples_of_json_hash_value_and_json_relative_path_and_ndarray.append((a_hash_value, a_relative_path, a_ndarray))

    def _save_mapping_from_json_hash_value_to_json_list_into_db(self):
        assert len(self.list_of_ndarrays) == len(self.list_of_json_hash_values)

        for i in range(len(self.list_of_json_hash_values)):
            assert isinstance(self.list_of_json_hash_values[i], str)
            assert isinstance(self.list_of_ndarrays[i], np.ndarray)

            similarity_matching_db_manager.Table_From_JsonSavedListHashValue_To_JsonSavedList.add_a_row(
                a_json_saved_list_hash_value=copy.deepcopy(self.list_of_json_hash_values[i]),
                a_json_saved_list=copy.deepcopy(self.list_of_ndarrays[i]).tolist()
            )


class _AudioSegmentInfo:
    def __init__(self, audio_hash_value, time_point_tuple):
        self.audio_hash_value = audio_hash_value
        self.time_point_tuple = time_point_tuple

        self.list_of_parameter_combo_hash_values = []
        self.list_of_ndarray_data = []  # TODO: 将来データ数が大きくなったときにメモリに全部乗るかな？ 分割して計算みたいなこと必要かも。

        self._set_up_lists_in_this_instance()

        assert len(self.list_of_parameter_combo_hash_values) == len(self.list_of_ndarray_data)

        '''
        WARNING: Do NOT sort these lists. Each list item corresponds to each other
        '''

    def _set_up_lists_in_this_instance(self):
        a_list_of_tuples = extracted_audio_features_db_manager.AudioExtractedData_View_For_SimilarityMatrixCalculation.\
            get_list_of_tuples_of_parameter_combo_hash_values_and_corresponding_array_for_an_audio_segment(
                audio_hash_value=self.audio_hash_value,
                time_point_tuple=self.time_point_tuple
            )

        for a_tuple in a_list_of_tuples:
            a_parameter_combo_hash_value = a_tuple[0]
            a_ndarray = a_tuple[1]
            self.list_of_parameter_combo_hash_values.append(a_parameter_combo_hash_value)
            self.list_of_ndarray_data.append(a_ndarray)
