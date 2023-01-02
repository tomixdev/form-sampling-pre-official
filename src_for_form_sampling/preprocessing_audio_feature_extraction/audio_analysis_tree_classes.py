import inspect
import pickle
import functools
import numpy as np
from sklearn import preprocessing
import librosa.display

import my_utils.data_structure_helpers
import my_utils.file_and_dir_interaction_helpers
import my_utils.terminal_interaction_helpers
from my_utils import misc_helpers as gh
import librosa
from . import extracted_audio_features_db_manager
import copy
import itertools
from .audio_analysis_tree_helpers import TreeHelpers as TH
from my_utils import my_db_utils
from src_for_form_sampling.parameters_package import PrmsContainer
from my_utils import hash_funcs
from my_utils import data_scalers
import statistics
import warnings


def get_stand_alone_audio_data_extractors_only_for_debugging_purpose(path_to_audio_file):
    my_utils.terminal_interaction_helpers.warninginfo('Use this method only for debugging purpose!! This class disables audio_analysis_tree. ')

    # Disable Tree Structure Temporarily
    SingleAudioInfo.create_next_node_upon_instantiation_of_this_node = False
    AudioDataExtractors.create_next_node_upon_instantiation_of_this_node = False

    audio_hash_value = hash_funcs.compute_hash_value_of_a_file(path_to_file=path_to_audio_file)

    extracted_audio_features_db_manager.Table_From_AudioHashValue_To_AudioFileRelativePath.add_or_replace_a_row(
        audio_hash_value=audio_hash_value,
        relative_path_to_audio=path_to_audio_file
    )

    single_audio_info_obj = SingleAudioInfo(audio_hash_value=audio_hash_value)
    return AudioDataExtractors(parent_SingleAudioInfo_node=single_audio_info_obj)


def get_stand_alone_data_transformers_only_for_debugging_purpose(ndarray):
    my_utils.terminal_interaction_helpers.warninginfo('Use this method only for debugging purpose!! This class disables audio_analysis_tree. ')
    gh.assert_class(ndarray, np.ndarray)
    isinstance(ndarray, np.ndarray)

    FlippedArray.create_next_node_upon_instantiation_of_this_node = False
    DataTransformers.create_next_node_upon_instantiation_of_this_node = False

    a_flipped_array_obj = FlippedArray(flipped_or_nonflipped_array=ndarray)
    return DataTransformers(parent_FlippedArray_node=a_flipped_array_obj)


class TreeClassesCommon:
    create_next_node_upon_instantiation_of_this_node = True
    class_intelligible_id = None
    previous_class_name = None
    my_utils.terminal_interaction_helpers.warninginfo(
        f"NEVER serialize or stringfy a dict or a list for values to save in relational database in {inspect.stack()[0][3]} class."
        f"In the current implementation as of Sep 8, 2022, there are special mechamism through hashing to"
        f"save dict as a parameter. Especially, once I stringfy dict, it becomes difficult to evaluate the equality of dict"
        f"whose element is in different order. I also built the the mechanism to serialize list in my_db_utils.py."
        f"So do NOT serialize list or dict")

    def __init__(self,
                 parent_node=None,
                 value_to_save_to_relational_db=None,
                 column_name_of_relational_db=None,
                 previous_column_name_of_relational_db=None,
                 array_before_computing_this_node=None,
                 array_after_computing_this_node=None):

        if value_to_save_to_relational_db is not None:
            assert column_name_of_relational_db is not None
        if column_name_of_relational_db is not None:
            assert value_to_save_to_relational_db is not None
        if previous_column_name_of_relational_db is not None:
            assert column_name_of_relational_db is not None
            assert value_to_save_to_relational_db is not None

        self.parent_node = parent_node
        self.value_to_save_to_relational_db = value_to_save_to_relational_db
        self.column_name_of_relational_db = column_name_of_relational_db
        self.previous_column_name_of_relational_db = previous_column_name_of_relational_db
        self.attribute_alias19482 = self.previous_column_name_of_relational_db  # This variable is referred in other files. So, do not easily delete!!!!
        self.array_before_computing_this_node = array_before_computing_this_node
        self.array_after_computing_this_node = array_after_computing_this_node


class AudioFolderAnalysis(TreeClassesCommon):
    class_intelligible_id = 'AudioAnalysisNodeClass1'

    def __init__(self, path_to_audio_folder, path_to_audio_extracted_data_cache_folder=None):
        my_utils.file_and_dir_interaction_helpers.raise_exception_if_path_to_a_folder_is_in_wrong_format(path_to_audio_folder)

        if path_to_audio_extracted_data_cache_folder is not None:
            my_utils.file_and_dir_interaction_helpers.raise_exception_if_path_to_a_folder_is_in_wrong_format(path_to_audio_extracted_data_cache_folder)
            TH.path_to_audio_extracted_data_cached_folder = path_to_audio_extracted_data_cache_folder

        # self.parent_node = None
        self.path_to_audio_folder = path_to_audio_folder
        # self.children_nodes = []

        '''
        TODO: Question
        たぶん、このフォルダーレベルのノードにおいて、このchildren_nodesはいらないと思う。なぜなら、children_nodesが必要な唯一の理由は、
        cached_instance_variablesを呼び出すために、cached_nodesを下っていくときに必要なものだから。このFolder Levelのデータは、cacheされずに、
        Singel Audioのレベルからデータがcacheされるわけだから、children nodesを記録する意味はない。
        もしchildren nodesを作るにしても、ほかのNodeとあわせるために、ListではなくDictを使うべき。
        
        おそらく、parent_nodeもいらないはずで、最後に結果計算結果が出たときに、このNodeまでさかのぼってくる必要ないから。
        '''

        extracted_audio_features_db_manager.Table_From_AudioHashValue_To_AudioFileRelativePath.create_the_table_if_not_exists()

        super().__init__()
        if self.create_next_node_upon_instantiation_of_this_node == True:
            self.create_and_execute_SingleAudio_node()

    def create_and_execute_SingleAudio_node(self):
        ''''''
        '''
        TODO: Rewrite the inside of for loop. 今はpickleを使ったImplementationだが、将来はjson baseのものに書き換える。
        TODO: AudioフォルダからAudioを削除したら、次にAudio Analysis Programを走らせるときに、DataBaseからそれに対応するDataが削除される仕様にする。
        TODO: 将来的に大きなアプリにするときは、ZOTEROみたいに各AudioFileの上にフォルダーを作って、そのフォルダーの名前を、Audioに対応するHashValueみたいにすれば
              データベースシステムに頼らなくても、フォルダーだけで、HashvalueとAudio Fileの関係を維持できる。
        '''

        list_of_relative_paths_to_audio_files = my_utils.get_list_of_entire_relative_path_to_all_audio_files_in_a_folder(self.path_to_audio_folder)
        for a_relative_path_to_audio_file in list_of_relative_paths_to_audio_files:
            a_relative_path_to_audio_file = my_utils.to_relative_path(a_relative_path_to_audio_file)

            # Step 0: Check Minimum Required Audio Duration requirement
            if gh.get_audio_length_in_s(a_relative_path_to_audio_file) < PrmsContainer.minimum_required_audio_duration_of_audio_folder_in_s:
                continue

            # Step1: Put the audio hash value and relative path to database
            the_hash_value_of_an_audio_file = hash_funcs.compute_hash_value_of_a_file(path_to_file=a_relative_path_to_audio_file)
            extracted_audio_features_db_manager.Table_From_AudioHashValue_To_AudioFileRelativePath.add_a_row(audio_hash_value=the_hash_value_of_an_audio_file,
                                                                                                             relative_path_to_audio=a_relative_path_to_audio_file)

            # Step2: Set the path name of the cached data of Root Node of Single Audio Analysis
            path_to_data_cache_of_root_of_single_audio_analysis = TH.get_path_to_audio_data_cache(audio_hash_value=the_hash_value_of_an_audio_file)

            if my_utils.file_and_dir_interaction_helpers.does_a_file_exist(path_to_data_cache_of_root_of_single_audio_analysis) and \
                    PrmsContainer.analyze_audio_even_if_audio_hash_value_is_already_in_db == False:
                continue

            try:
                # Step3: Open Cached Root of Single Audio Analysis
                f_open_for_old_pickled_node = open(path_to_data_cache_of_root_of_single_audio_analysis, 'rb')
                cached_instance_variables_of_next_SingleAudio_node = pickle.load(f_open_for_old_pickled_node)
                f_open_for_old_pickled_node.close()  # TODO これってここでclose()しちゃっても大丈夫なのかな？このコードの時点では、cached_instance_variablesは使われていないので、使われる前に、f_openを閉じちゃって大丈夫なのか?
            except:
                cached_instance_variables_of_next_SingleAudio_node = None

            # -----------Takes a lot of time!!!--------------------------------------------------------------------
            # Step4: Analyze...This single line of code executes depth-first search. So it takes time!!!
            node_for_single_audio_analysis = SingleAudioInfo(audio_hash_value=the_hash_value_of_an_audio_file,
                                                             parent_AudioFolderAnalysis_node=self,
                                                             cached_instance_variables_of_this_node=cached_instance_variables_of_next_SingleAudio_node)

            # Step5: Add Node for Single Audio Analysis to Children of This Class
            # self.children_nodes.append(node_for_single_audio_analysis)

            # Step 6: Clear Cache(もしCacheまでDatabaseに保存してしまうと、雪だるま式にdatabaseの容量が増大することになる)

            # Step7: Overwrite the old pickled node with the new pickled node
            f_open_for_new_pickled_node = open(path_to_data_cache_of_root_of_single_audio_analysis, 'wb')
            pickle.dump(node_for_single_audio_analysis, f_open_for_new_pickled_node)
            f_open_for_new_pickled_node.close()

        '''TODO: Impelement "does_the_same_value_exists" later
        if extracted_audio_features_db_manager.Table_From_AudioHashValue_To_AudioFileRelativePath.does_the_same_value_exists(column_name_to_search_the_same_value='audio_hash_value') == True:
        if extracted_audio_features_db_manager.Table_From_AudioHashValue_To_AudioFileRelativePath.does_the_same_value_exists(column_name_to_search_the_same_value='audio_hash_value') == True:
            raise Exception ("Audio hash value conflict happened!!!Some of the audio data might be unintentionally deleted!!!! It is 99.99999...% program bug!!!")
        '''


class SingleAudioInfo (TreeClassesCommon):
    class_intelligible_id = 'AudioAnalysisNodeClass2'

    def __init__(self, audio_hash_value, parent_AudioFolderAnalysis_node=None, cached_instance_variables_of_this_node=None):
        assert isinstance(audio_hash_value, str)
        assert parent_AudioFolderAnalysis_node is None or isinstance(parent_AudioFolderAnalysis_node, AudioFolderAnalysis)
        # TODO: Once I change pickle based caching into dict based caching, then I need the assertion below:
        #  assert cached_instance_variables_of_this_node is None or isinstance(cached_instance_variables_of_this_node, dict)

        # --------------------------------------------------------------------------------------------------------------
        self.audio_hash_value = audio_hash_value
        super().__init__(value_to_save_to_relational_db=self.audio_hash_value, column_name_of_relational_db=gh.varnameof(audio_hash_value))

        try:
            self.relative_path_to_audio = extracted_audio_features_db_manager.Table_From_AudioHashValue_To_AudioFileRelativePath.get_value_from_key(
                audio_hash_value=audio_hash_value)
        except:
            raise Exception(
                f'The audio file relative path corresponding to the audio hash value {self.audio_hash_value} is Not in relational database. Put the pair of audio hash value and relative path to audio in database first!!')

        # --------------------------------------------------------------------------------------------------------------
        # TODO: 以下の一連のやつ、絶対にこの部分は読み込まれるのに、gh.varnameofを使っていて、inspect moduleでやたら時間がかかっていそうな雰囲気。
        self.y_sr_tuple = None
        self.y_sr_tuple = TH.get_value_from_cached_variables_or_compute_value(
            cached_variables=cached_instance_variables_of_this_node,
            varname_str_as_dict_key=gh.varnameof(self.y_sr_tuple),
            the_previous_varname_str=None,
            func_to_compute_value=lambda: librosa.load(self.relative_path_to_audio)
        )

        y = self.y_sr_tuple[0]
        sr = self.y_sr_tuple[1]

        audio_length_in_s = float(len(y)/sr)
        self.attribute_alias1 = audio_length_in_s

        self.S = None
        self.S = TH.get_value_from_cached_variables_or_compute_value(
            cached_variables=cached_instance_variables_of_this_node,
            varname_str_as_dict_key=gh.varnameof(self.S),
            the_previous_varname_str=None,
            func_to_compute_value=lambda: np.abs(librosa.stft(y))
        )

        self.times = None
        self.times = TH.get_value_from_cached_variables_or_compute_value(
            cached_variables=cached_instance_variables_of_this_node,
            varname_str_as_dict_key=gh.varnameof(self.times),
            func_to_compute_value=lambda: librosa.times_like(self.S)
        )

        self.fft_freqs = None
        self.fft_freqs = TH.get_value_from_cached_variables_or_compute_value(
            cached_variables=cached_instance_variables_of_this_node,
            varname_str_as_dict_key=gh.varnameof(self.fft_freqs),
            func_to_compute_value=lambda: librosa.fft_frequencies(sr=sr)
        )

        self.onset_time_points_in_s = None
        self.onset_time_points_in_s = TH.get_value_from_cached_variables_or_compute_value(
            cached_variables=cached_instance_variables_of_this_node,
            varname_str_as_dict_key=gh.varnameof(self.onset_time_points_in_s),
            func_to_compute_value=lambda: self.compute_onset_time_points_in_s(y=y, sr=sr)
        )

        self.spectral_contrast = None
        self.spectral_contrast = TH.get_value_from_cached_variables_or_compute_value(
            cached_variables=cached_instance_variables_of_this_node,
            varname_str_as_dict_key=gh.varnameof(self.spectral_contrast),
            func_to_compute_value=lambda: librosa.feature.spectral_contrast(y=y, sr=sr)
        )

        # --------------------------------------------------------------------------------------------------------------
        self.child_AudioDataExtractors_node = None
        cached_child_AudioDataExtractors_instance_variables = TH.get_value_from_cached_variables_or_compute_value(
            cached_variables=cached_instance_variables_of_this_node,
            varname_str_as_dict_key=gh.varnameof(self.child_AudioDataExtractors_node),
        )
        self.child_AudioDataExtractors_node = AudioDataExtractors(
            parent_SingleAudioInfo_node=self,
            cached_instance_variables_of_this_node=cached_child_AudioDataExtractors_instance_variables
        )

    @staticmethod
    def compute_onset_time_points_in_s(y, sr):
        ''''''
        '''
        TODO: 
        # onset_detect returns frame indices
        #   onset_detectのいろいろなParameterをどのようにいじれば、onset_framesがどれくらい増えるのかとかはわからない家ド、
        #   post_avgをふやすと、onset_framesの数が増える傾向にあることがわかったので、とりあえず10000にしている。
        #   ほんとうは、wait, pre_avg, post_avg, pre_max, post_maxの値などを0.1, 0.5, 1, 10, 100, 1000, 10000暗い用意して、
        #   総当りで組み合わせを試していって、onset_timesの数が最大となるところを探るとかしても良いとは思う。(Noted on July, 13, 2022)
        '''
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, wait=1, pre_avg=1, post_avg=10000, pre_max=1,
                                                  post_max=1)  # backtrack=True, Backtrackパラメータの説明は、4-Rhythm-Tempo-and_Beat_Tracking-05-onset_segmentationlipynbにある。まあ、このParameterは別にTrueにしなくても動く。Trueにする必要があるのかすらわからない。
        onset_time_points = librosa.frames_to_time(onset_frames)
        return onset_time_points


class AudioDataExtractors(TreeClassesCommon):
    class_intelligible_id = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    plot_graphs = False

    def __init__(self, parent_SingleAudioInfo_node=None, cached_instance_variables_of_this_node=None):
        assert parent_SingleAudioInfo_node is None or isinstance(parent_SingleAudioInfo_node, SingleAudioInfo)
        # TODO: Once I change pickle based caching into dict based caching, then I need the assertaion below:
        #  assert cached_instance_variables_of_this_node is None or isinstance(cached_instance_variables_of_this_node, dict)

        self.parent_SingleAudioInfo_node = parent_SingleAudioInfo_node
        self.audio_info = self.parent_SingleAudioInfo_node

        self.hop_length = 256
        self.frame_length = 512

        self.cached_children_ExtractedData_instance_variables_stored_as_dict = None
        self.children_ExtractedData_nodes = None
        '''  { general_helpers.hash_a_dict ({parameters_dict})             : children_nodes_instances}
        e.g. { hash value of {func_id: 'function1', 'prm1': 20, 'prm2':10} : children_node_instance1,
               hash value of {func_id: 'function2', 'prm1':-15}            : children_node_instance2 }
        '''
        self.cached_children_ExtractedData_instance_variables_stored_as_dict = TH.get_value_from_cached_variables_or_compute_value(
            cached_variables=cached_instance_variables_of_this_node,
            varname_str_as_dict_key=gh.varnameof(self.children_ExtractedData_nodes)
        )
        self.children_ExtractedData_nodes = {}

        super().__init__()
        if self.create_next_node_upon_instantiation_of_this_node:
            TH.call_all_public_methods_in_a_class_instance(a_class_instance=self)

        self.cached_children_ExtractedData_instance_variables_stored_as_dict = None  # TODO : Cache Clearしているけどこれでうまくいくのかな?


    def _data_extraction_methods_common_procedures(undecorated_func):
        '''This is the dirtiest function in my code. It is so difficult to follow the logic. どうにかしろ。TODO:'''
        '''
         0. add Function name and parameters to super class so that they can be read from the final node of the tree and stored to relational database
         1. run method according to all combinations of possible values if no parameter is provided
         2. get value from cache if available. If not available, compute value.
         3. cut off long floating point numbers 
         4. create next AudioExtractedArray node and append to children node dict of this node
         5. return func result from either cache or computation
        '''
        @functools.wraps(undecorated_func)
        def wrapper(self, *args, **kwargs):
            assert isinstance(self, AudioDataExtractors)

            func_arguments_dict = TH.get_func_arguments_dict_from_within_wrapper_method(undecorated_func, self, *args, **kwargs)
            list_of_argument_tuples_in_this_method_call = list(func_arguments_dict.items())  # <-- returns something like [('param1', 100), ('param2', 200)]
            dict_from_normal_varnames_to_parameter_values, \
                dict_from_intelligible_id_varnames_to_parameter_values, \
                list_of_normal_varnames, \
                list_of_intelligible_id_varnames, \
                list_of_list_of_possible_values \
                = TH.extract_relevant_dicts_and_lists_from_function_arguments_tuples(list_of_argument_tuples_in_this_method_call)

            # gh.debuginfo(func_arguments_dict, varname_str=gh.varnameof(func_arguments_dict))
            # gh.debuginfo(dict_from_normal_varnames_to_parameter_values, varname_str=gh.varnameof(dict_from_normal_varnames_to_parameter_values))
            # gh.debuginfo(dict_from_intelligible_id_varnames_to_parameter_values, varname_str=gh.varnameof(dict_from_intelligible_id_varnames_to_parameter_values))
            # gh.debuginfo(list_of_normal_varnames, varname_str=gh.varnameof(list_of_normal_varnames))
            # gh.debuginfo(list_of_intelligible_id_varnames,  varname_str=gh.varnameof(list_of_intelligible_id_varnames))
            # gh.debuginfo(list_of_list_of_possible_values, varname_str=gh.varnameof(list_of_list_of_possible_values))

            # if two dicts are empty...then, read from possible values list
            if (bool(dict_from_normal_varnames_to_parameter_values) == False and
                bool(dict_from_intelligible_id_varnames_to_parameter_values) == False and
                    len(list_of_normal_varnames) > 0 and len(list_of_intelligible_id_varnames) > 0 and len(list_of_list_of_possible_values) > 0):

                decorated_func = getattr(self, undecorated_func.__name__)

                list_of_possible_parameter_combination_tuples = list(itertools.product(*list_of_list_of_possible_values))
                for a_parameter_combination_tuple in list_of_possible_parameter_combination_tuples:
                    dict_from_normal_varnames_to_parameter_values = {}
                    dict_from_intelligible_id_varnames_to_parameter_values = {}
                    for i in range(0, len(a_parameter_combination_tuple)):
                        dict_from_normal_varnames_to_parameter_values[list_of_normal_varnames[i]] = a_parameter_combination_tuple[i]
                        dict_from_intelligible_id_varnames_to_parameter_values[list_of_intelligible_id_varnames[i]] = a_parameter_combination_tuple[i]

                    decorated_func(**{**dict_from_normal_varnames_to_parameter_values, **dict_from_intelligible_id_varnames_to_parameter_values})

                return  # TODO このReturnひつようか?必要だとしたら何をReturnするのか?
            else:  # if one or two of two dicts contain a value, then retreieve value from cache or compute value according to parameter values in dict
                '''TODO: いちおう以下にCacheから、Function Execution Resultをよみこむように書いているけど、Sとかsst_freqとか、
                計算が重そうな値はは、すでにCachingしてあるので、Function ResultをCachingする効果はほとんどないかもしれない。'''
                assert 'func_intelligible_id' in func_arguments_dict.keys()
                dict_of_data_extraction_func_id_and_parameters_under_intelligible_id_varname = {
                    **{'func_intelligible_id': func_arguments_dict['func_intelligible_id']}, **(dict_from_intelligible_id_varnames_to_parameter_values)}
                '''e.g. {'func_intelligible_id':'audio_data_extraction_func1, 'prm1':20, 'prm2':30}'''
                dict_of_data_extraction_func_name_and_parameters_under_normal_varname = {
                    **{'func_name': undecorated_func.__name__}, **dict_from_normal_varnames_to_parameter_values}
                '''e.g. {'func_name':'extract_overtone_data', 'a_small_constant': 20, 'number_of_overtone_divisions': 30}'''

                hashed_dict_as_the_key_to_store_a_cached_child_node = hash_funcs.hash_a_dict(
                    dict_of_data_extraction_func_id_and_parameters_under_intelligible_id_varname)

                if PrmsContainer.save_intelligible_ids_for_audio_extraction_functions:
                    dict_value_to_save_to_db = dict_of_data_extraction_func_id_and_parameters_under_intelligible_id_varname
                else:
                    dict_value_to_save_to_db = dict_of_data_extraction_func_name_and_parameters_under_normal_varname
                super().__init__(
                    parent_node=self.parent_SingleAudioInfo_node,
                    value_to_save_to_relational_db=dict_value_to_save_to_db,
                    column_name_of_relational_db=gh.varnameof(AudioDataExtractors)
                )

                tuple_of_extracted_ndarray_and_T = None
                cached_instance_variables_of_next_node = None
                if (self.cached_children_ExtractedData_instance_variables_stored_as_dict is not None and
                        hashed_dict_as_the_key_to_store_a_cached_child_node in self.cached_children_ExtractedData_instance_variables_stored_as_dict.keys()):
                    # TODO: Debugのときに 'attribute_alias2'というparameter nameがAudioExtractedArray Classに存在することを確認せよ。
                    tuple_of_extracted_ndarray_and_T = self.cached_children_ExtractedData_instance_variables_stored_as_dict[
                        hashed_dict_as_the_key_to_store_a_cached_child_node].attribute_alias2
                    cached_instance_variables_of_next_node = self.cached_children_ExtractedData_instance_variables_stored_as_dict[
                        hashed_dict_as_the_key_to_store_a_cached_child_node]
                    my_utils.terminal_interaction_helpers.debuginfo(f"AudioDataExtractors: Value Read from cache (i.e. NOT computed)!")
                else:
                    tuple_of_extracted_ndarray_and_T = undecorated_func(**{**{'self': self},
                                                                           **dict_from_normal_varnames_to_parameter_values,
                                                                           **dict_from_intelligible_id_varnames_to_parameter_values})

                    assert not np.isnan(tuple_of_extracted_ndarray_and_T[0]).any(), \
                        TH.get_two_lists_about_upper_nodes_in_the_tree_structure___return_as_str___I_randomly_wrote_this_function_for_debugging(self)

                    cached_instance_variables_of_next_node = None
                    my_utils.terminal_interaction_helpers.debuginfo(f"AudioDataExtractors: Value Computed (i.e. NOT read from cache)!")

                # eliminate decimal points in extracted data
                assert len(tuple_of_extracted_ndarray_and_T) == 2
                gh.assert_class(tuple_of_extracted_ndarray_and_T[0], np.ndarray)
                if tuple_of_extracted_ndarray_and_T[1] is not None:
                    gh.assert_class(tuple_of_extracted_ndarray_and_T[1], np.ndarray)
                my_utils.data_structure_helpers.assert_all_values_in_one_dimentional_ndarray_are_different(tuple_of_extracted_ndarray_and_T[0])

                # rounded_extracted_ndarray = np.around(tuple_of_extracted_ndarray_and_T[0], self.__class__.class_attribute_alias3917)
                # T = tuple_of_extracted_ndarray_and_T[1]
                # tuple_of_extracted_ndarray_and_T = (rounded_extracted_ndarray, T)

                if self.create_next_node_upon_instantiation_of_this_node:
                    a_child_AudioExtractedArray_node = AudioExtractedArray(
                        parent_AudioDataExtractors_node=self,
                        dict_of_data_extraction_func_id_and_parameters_under_intelligible_id_varname=dict_of_data_extraction_func_id_and_parameters_under_intelligible_id_varname,
                        dict_of_data_extraction_func_name_and_parameters_under_normal_varname=dict_of_data_extraction_func_name_and_parameters_under_normal_varname,
                        tuple_of_extracted_ndarray_and_T=tuple_of_extracted_ndarray_and_T,
                        cached_instance_variables_of_this_node=cached_instance_variables_of_next_node
                    )

                    if hashed_dict_as_the_key_to_store_a_cached_child_node in self.children_ExtractedData_nodes.keys():
                        raise Exception(
                            f"Perhaps hashvalue conflict is happening in the dict '{gh.varnameof(hashed_dict_as_the_key_to_store_a_cached_child_node)}'?????")
                    else:
                        self.children_ExtractedData_nodes[hashed_dict_as_the_key_to_store_a_cached_child_node] = a_child_AudioExtractedArray_node

                return tuple_of_extracted_ndarray_and_T

        return wrapper

    @_data_extraction_methods_common_procedures
    def compute_spectral_centroids(self,
                                   a_small_constant=None, prm1=None, prm1_possible_values=[0.01],  # 0.1, 1.0
                                   func_intelligible_id='data_extraction_func1',
                                   **kwargs):

        y, sr = self.parent_SingleAudioInfo_node.y_sr_tuple
        # The spectral centroid indicate at which frequency the enegy of a spectrum is centered upon.
        # librosa.feature.spectral_centroid computes the spectral centroid for each time in signal

        spectral_centroids = librosa.feature.spectral_centroid(y=y+a_small_constant, sr=sr)[0]

        # Compute the time variable for visualization:
        frames = range(len(spectral_centroids))
        T = librosa.frames_to_time(frames)

        return spectral_centroids, T

    # This function is based on:
    #  https://colab.research.google.com/github/stevetjoa/musicinformationretrieval.com/blob/gh-pages/energy.ipynb

    @_data_extraction_methods_common_procedures
    def compute_rmse_enegy(self, func_intelligible_id='data_extraction_func2', **kwargs):
        y, sr = self.parent_SingleAudioInfo_node.y_sr_tuple
        hop_length = self.hop_length
        frame_length = self.frame_length

        energy = np.array([
            sum(abs(y[i:i + frame_length] ** 2))
            for i in range(0, len(y), hop_length)
        ])

        rmse = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True)
        rmse = rmse[0]

        frames = range(len(energy))
        T = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

        rmse_averaged_over_5_datapoints = my_utils.data_structure_helpers.average_a_ndarray_over_n_elements(a_ndarray=rmse, n=5)

        return rmse_averaged_over_5_datapoints, None
        # return rmse, T

    @_data_extraction_methods_common_procedures
    def compute_overtone_variety_stft(self, func_intelligible_id='data_extraction_func_3', **kwargs):
        y, sr = self.parent_SingleAudioInfo_node.y_sr_tuple
        y_harmonic = librosa.effects.harmonic(y)

        # ------------------------------------------------------------------
        # librosa.interp_harmonics()----------------------------------------
        harmonics = [1]  # TODO: このList, 1, 2, 3, 4, 5, みたいに他の数字をListに入れることができて、Audioによっては、1と4ではピークの場所が違ったりする。まあ大きな違いは出ないと判断して、パラメータ削減のため、とりあえず１だけにしている。
        S = np.abs(librosa.stft(y))
        fft_freqs = self.parent_SingleAudioInfo_node.fft_freqs

        # print (fft_freqs.shape) #FrequencyのSubdivisionを表していると思われる。すべてのFrequencyをsampling rateに基づいて、1025分割している
        # 一番上のFrequencyは11025 だと思われる。
        S_harm = librosa.interp_harmonics(S, freqs=fft_freqs, harmonics=harmonics, axis=0)
        # S_harm1つ目は、グラフの番号。
        # S_harm2つ目は、おそらくFrequencyのSubdivisionの個数(縦軸のsubdivisionの個数)。
        # S_harm3つ目が横軸の個数(つまりAudio lengthによってかわる)。
        # では疑問点として、各Frequency(たとえば４４０HZにおける)における、今日どのへんを見ればどうすればよいのか。

        # calculate the overtone variety at each moment----------------------
        # 自分のoverton varietyの定義: 各time frameにおける、すべてのfrequencyにおける値を0と１の間にscaleして、その和をとる
        # overtone_variety_my_definition.jpegも見よ。
        time_and_frequencyMagnitude_ndarray = S_harm[0].transpose()

        overtone_variety_list = []
        number_of_nonzero_values_list = []
        for i in range(0, time_and_frequencyMagnitude_ndarray.shape[0]):
            frequency_magnitude_ndarray = time_and_frequencyMagnitude_ndarray[i]
            frequency_magnitude_ndarray_scaled_to_zero_and_one = data_scalers.min_max_normalization(frequency_magnitude_ndarray, 0., 1.)

            a_overtone_variety = frequency_magnitude_ndarray_scaled_to_zero_and_one.sum()
            overtone_variety_list.append(a_overtone_variety)
            number_of_nonzero_values_list.append(np.count_nonzero(frequency_magnitude_ndarray))

        overtone_variety_ndarray = np.array(overtone_variety_list)
        non_zero_values = np.array(number_of_nonzero_values_list)
        # TODO このnon_zero_ valuesって何? 何のために計算しているんだっけ? 自分で書いたコードだから何らかの意図があったはずだけど。
        T = None

        return overtone_variety_ndarray, T


    @_data_extraction_methods_common_procedures
    def compute_the_order_p_spectral_bandwidth(self,
                                               func_intelligible_id='data_extraction_func5',
                                               **kwargs):
        a_small_constant = 0.02
        y, sr = self.parent_SingleAudioInfo_node.y_sr_tuple
        spectral_bandwidth_p = librosa.feature.spectral_bandwidth(y=y + a_small_constant, sr=sr)[0]
        # Compute the time variable for visualization:
        frames = range(len(spectral_bandwidth_p))
        T = librosa.frames_to_time(frames)

        return spectral_bandwidth_p, None

    '''
    Spectral contrast sonciders the spectral peak, the spectral valley, and their difference in each frequency subband.
        cf. Jiang, Dan-Ning, Lie Lu, Hong-Jiang Zhang, Jian-Hua Tao, and Lian-Hong Cai. “Music type classification by spectral contrast feature.” In Multimedia and Expo, 2002. ICME‘02. Proceedings. 2002 IEEE International Conference on, vol. 1, pp. 113-116. IEEE, 2002.
    librosa.feature_spectral_contrast computes the specral contrast for six subbands for each time frame
    音源やiの価によって、大した動きのないGraphが得られるときもあれば、いい感じのShapeが得られるときもあるので、
    ちょっとBehaviorの予測ができない。
    
    TODO:もしprm1_possible_valuesが0, 1, 2, 3, 4, 5, 6,だと多すぎる場合は、1と5だけでも十分かも。
    '''
    @_data_extraction_methods_common_procedures
    def compute_spectral_contrast_of_i_th_subband(self,
                                                  i=None, prm1=None, prm1_possible_values=[1, 2, 3, 4, 5],  # 1,2, 3, 4, 5, 6
                                                  func_intelligible_id='data_extraction_func6',
                                                  **kwargs):
        gh.assert_class(i, int)
        assert isinstance(i, int)
        assert 0 <= i <= 6

        # i is counting from zero!!!

        spectral_contrast = self.parent_SingleAudioInfo_node.spectral_contrast
        i_th_spectral_contrast = spectral_contrast[i]
        T = None

        i_th_spectral_contrast_averaged_over_10_datapoints = my_utils.data_structure_helpers.average_a_ndarray_over_n_elements(
            a_ndarray=i_th_spectral_contrast,
            n=10
        )

        return i_th_spectral_contrast_averaged_over_10_datapoints, T
        # return i_th_spectral_contrast, T

    @_data_extraction_methods_common_procedures
    def compute_normalized_variances_of_all_seven_spectral_bands(self, func_intelligible_id='data_extraction_func6', **kwargs):
        seven_spectral_contrasts = self.parent_SingleAudioInfo_node.spectral_contrast
        seven_spectral_contrasts_transposed = seven_spectral_contrasts.transpose()
        list_of_variances_of_all_seven_spectral_bands = []
        for i in range(0, seven_spectral_contrasts_transposed.shape[0]):
            the_seven_spectral_energy_values_in_the_frame_i = seven_spectral_contrasts_transposed[i]
            scaled_to_between_zero_and_one = data_scalers.min_max_normalization(
                the_seven_spectral_energy_values_in_the_frame_i, min_to_scale=0.0, max_to_scale=1.0)
            the_variance = statistics.variance(scaled_to_between_zero_and_one)
            list_of_variances_of_all_seven_spectral_bands.append(the_variance)

        # Compute the time variable for visualization:
        frames = range(len(seven_spectral_contrasts_transposed))
        T = librosa.frames_to_time(frames)

        ndarray_of_variances_of_all_seven_spectral_bands_averaged_over_20_datapoints = my_utils.data_structure_helpers.average_a_ndarray_over_n_elements(
            a_ndarray=np.array(list_of_variances_of_all_seven_spectral_bands), n=20
        )
        return ndarray_of_variances_of_all_seven_spectral_bands_averaged_over_20_datapoints, None
        # return np.array(list_of_variances_of_all_seven_spectral_bands), T

    # TODO: Computationの時間に余力があるなら、prm1_possioble_valuesを0.1, 0.2, 0.3, 0.4, 0.5,と0.1ごとにしたいくらい。それくらい形は変わる。
    # とりあえず今の実装は、のちのちのData Transformerががんばっていろいろな形にしてくれると信じて、prm1 possible valuesとして３つのみ指定している。

    @_data_extraction_methods_common_procedures
    def compute_zero_crossing_rate(self,
                                   a_small_constant=None, prm1=None, prm1_possible_values=[0.0001],  # [0.1, 0.5]も入れても面白い。
                                   func_intelligible_id='data_extraction_func7', **kwargs):
        # based on 3-SignalAnalysisAndFeatureExtraction-04-zero_crossing_rate.ipynb

        gh.assert_class(a_small_constant, float)
        assert 0.0 <= a_small_constant <= 1.0

        y, sr = self.parent_SingleAudioInfo_node.y_sr_tuple
        zcrs = librosa.feature.zero_crossing_rate(y+a_small_constant)
        # without a_small_constant, the high rate appears near the beginning, because the silence oscillates quietly around zero.
        # A simple hack around this is to add a small constant before computing the zero crossing rate
        # 本当は、0.0001がmusicinformationretrieval.comにかかれたpatchでつかわれている値だけど、この値を変えることで、得られる形がけっこうかわるので、
        # やっていいかどうかはわからないけど、a_small_constantを柔軟に変更できるようなコードの書き方にしている。

        T = None
        return zcrs[0], T

    @_data_extraction_methods_common_procedures
    def compute_energy_novelty(self,
                               # ambience reduction percentileとして他に50や95が考えられるけど、なんかvector similarity calculationがうまく行かないので、やめた。
                               ambience_reduction_clip_percentile=None, prm1=None, prm1_possible_values=[0],
                               func_intelligible_id='data_extraction_func8', **kwargs):
        hop_length = self.hop_length
        frame_length = self.frame_length
        y, sr = self.parent_SingleAudioInfo_node.y_sr_tuple

        rmse = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length).flatten()
        rmse_diff = np.zeros_like(rmse)
        rmse_diff[1:] = np.diff(rmse)

        energy_novelty = np.max([np.zeros_like(rmse_diff), rmse_diff], axis=0)
        energy_novelty = my_utils.data_structure_helpers.normalize_and_set_values_below_i_percentile_to_zero(
            X=energy_novelty, a_percentile=ambience_reduction_clip_percentile)

        frames = np.arange(len(rmse))
        T = librosa.frames_to_time(frames, sr=sr)

        energy_novelty_averaged_over_25_datapoints = my_utils.data_structure_helpers.average_a_ndarray_over_n_elements(
            a_ndarray=energy_novelty, n=25
        )

        return energy_novelty_averaged_over_25_datapoints, None
        # return energy_novelty, T

    @_data_extraction_methods_common_procedures
    def compute_spectral_novelty(self, func_intelligible_id='data_extraction_func9', **kwargs):
        y, sr = self.parent_SingleAudioInfo_node.y_sr_tuple
        spectral_novelty = librosa.onset.onset_strength(y=y, sr=sr)

        frames = np.arange(len(spectral_novelty))
        T = librosa.frames_to_time(frames, sr=sr)

        spectral_novelty_averaged_over_25_datapoints = my_utils.data_structure_helpers.average_a_ndarray_over_n_elements(a_ndarray=spectral_novelty, n=25)
        # T_averaged_over_25_datapoints = gh.average_a_ndarray_over_n_elements(a_ndarray=T, n=25)

        return spectral_novelty_averaged_over_25_datapoints, None
        # return spectral_novelty, T

    @_data_extraction_methods_common_procedures
    def compute_tempo_detection_intensity(self, func_intelligible_id='data_extraction_func10', **kwargs):
        # TODO: 本当はこれをBPMの時間変化が分かるFunctionにしたい。とりあえず今は、tempogramに現れる、書くsegmentでの値の平均値(つまり、どれくらいtempoが強くDetectされるか）という値を返している。
        # 今のImplementationだと、単にNovelty　Functionなどから書くSegmentの強さを計算しているのと変わらない気がする。
        # あるいは、Attackの個数を数えているだけとかと同じ感じがする。もっともBPMというのは、Attackの数から計算しているだけかもしれないけど。
        # This function is based on 4-Rhythm-Tempo-and_Beat_Tracking-05-tempo_estimation.ipynb

        y, sr = self.parent_SingleAudioInfo_node.y_sr_tuple
        hop_length = 200  # samples per frame
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, n_fft=2048)

        frames = range(len(onset_env))
        t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

        S = librosa.stft(onset_env, hop_length=1, n_fft=512)
        fourier_tempogram = np.absolute(S)

        # librosa.display.specshow(fourier_tempogram, sr=sr, hop_length=hop_length, x_axis='time')

        n0 = 100
        n1 = 500

        tmp = np.log1p(onset_env[n0:n1])
        r = librosa.autocorrelate(tmp)

        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length, win_length=400)
        array_of_bpm_intensities_at_every_bpm_from_bpm0_to_bpm400 = tempogram.transpose()
        list_of_bpm_intensity_averages = []
        for an_array in array_of_bpm_intensities_at_every_bpm_from_bpm0_to_bpm400:
            # the_25_percentile = np.percentile(an_array, 25)
            # the_75_percentile = np.percentile(an_array, 75)
            list_of_bpm_intensity_averages.append(np.average(an_array))

        T = None
        return np.array(list_of_bpm_intensity_averages), T


class AudioExtractedArray (TreeClassesCommon):
    def __init__(self,
                 parent_AudioDataExtractors_node,
                 dict_of_data_extraction_func_id_and_parameters_under_intelligible_id_varname,
                 dict_of_data_extraction_func_name_and_parameters_under_normal_varname,
                 tuple_of_extracted_ndarray_and_T,
                 cached_instance_variables_of_this_node=None):
        assert isinstance(parent_AudioDataExtractors_node, AudioDataExtractors)
        assert isinstance(dict_of_data_extraction_func_id_and_parameters_under_intelligible_id_varname, dict)
        assert isinstance(dict_of_data_extraction_func_name_and_parameters_under_normal_varname, dict)
        assert isinstance(tuple_of_extracted_ndarray_and_T, tuple)
        assert isinstance(tuple_of_extracted_ndarray_and_T[0], np.ndarray), f"{type(tuple_of_extracted_ndarray_and_T[0])} is detected..."
        # TODO: Once I change pickle based caching into dict based caching, then I need the assertaion below:
        #  assert cached_instance_variables_of_this_node is None or isinstance(cached_instance_variables_of_this_node, dict)

        if PrmsContainer.convert_nan_to_min_value_of_in_results_of_audio_data_extraction_functions:
            extracted_ndarray_with_nans_converted_to_min_value = my_utils.data_structure_helpers.convert_all_nan_values_to_min_value_in_ndarray(
                tuple_of_extracted_ndarray_and_T[0])
            tuple_of_extracted_ndarray_and_T = (extracted_ndarray_with_nans_converted_to_min_value, tuple_of_extracted_ndarray_and_T[1])

        my_utils.data_structure_helpers.assert_all_elements_in_ndarray_are_number(tuple_of_extracted_ndarray_and_T[0])

        self.parent_AudioDataExtractors_node = parent_AudioDataExtractors_node
        super().__init__(parent_node=self.parent_AudioDataExtractors_node)
        self.dict_of_data_extraction_func_id_and_parameters_under_intelligible_id_varname = dict_of_data_extraction_func_id_and_parameters_under_intelligible_id_varname
        self.dict_of_data_extraction_func_name_and_parameters_under_normal_varname = dict_of_data_extraction_func_name_and_parameters_under_normal_varname

        self.tuple_of_extracted_ndrray_and_T = tuple_of_extracted_ndarray_and_T
        self.attribute_alias2 = self.tuple_of_extracted_ndrray_and_T

        extracted_nd_array, T = tuple_of_extracted_ndarray_and_T
        gh.assert_class(extracted_nd_array, np.ndarray)

        # print ('extracted ndarray ====================')
        # print (extracted_nd_array)
        # print(TH.get_two_lists_about_upper_nodes_in_the_tree_structure___return_as_str___I_randomly_wrote_this_function_for_debugging(self))

        if T is not None:
            gh.assert_class(extracted_nd_array, np.ndarray)
            gh.assert_value_equality(extracted_nd_array.size, T.size)

        if self.create_next_node_upon_instantiation_of_this_node == False:
            return

        self.child_ArraySegmenter_node = None
        cached_child_ArraySegmenter_instance_variables = TH.get_value_from_cached_variables_or_compute_value(
            cached_variables=cached_instance_variables_of_this_node,
            varname_str_as_dict_key=gh.varnameof(self.child_ArraySegmenter_node)
        )
        self.child_ArraySegmenter_node = ArraySegmenter(
            parent_AudioExtractedArray_node=self,
            cached_instance_variables_of_this_node=cached_child_ArraySegmenter_instance_variables
        )


class ArraySegmenter(TreeClassesCommon):
    class_intelligible_id = 'YYYYYYYYYYYYYYYYYYYYYYYYYYY'  # TODO: Write this some time when necessary

    def __init__(self, parent_AudioExtractedArray_node, cached_instance_variables_of_this_node=None):
        assert isinstance(parent_AudioExtractedArray_node, AudioExtractedArray)
        # TODO: Once I change pickle based caching into dict based caching, then I need the assertaion below:
        #  assert cached_instance_variables_of_this_node is None or isinstance(cached_instance_variables_of_this_node, dict)

        self.parent_AudioExtractedArray_node = parent_AudioExtractedArray_node
        super().__init__(parent_node=self.parent_AudioExtractedArray_node)
        self.entire_audio_length_in_s = self.parent_AudioExtractedArray_node.parent_AudioDataExtractors_node.parent_SingleAudioInfo_node.attribute_alias1

        list_of_time_point_tuples = self._compute_list_of_all_possible_audio_segment_time_point_tuples_in_s_and_return()
        # del list_of_time_point_tuples[0]  #TODO: これいらない
        # TODO: 多分ここはうまくCacheから読み込む方法があると思うけど、どれくらい計算のはやさに貢献するかわからないので、とりあえずCacheからは読みこまない実装にしている。

        self.cached_children_SegmentedArray_instance_variables_stored_as_dict = None
        self.children_SegmentedArray_nodes = {}
        '''  {time_point_tuple : a_child_SegmentedArray_node
        e.g. {(0.0, 60.0) : SegmentedArray instance 1,
              (30.0, 90.0): SegmentedArray instance 2}
        '''
        self.cached_children_SegmentedArray_instance_variables_stored_as_dict = TH.get_value_from_cached_variables_or_compute_value(
            cached_variables=cached_instance_variables_of_this_node,
            varname_str_as_dict_key=gh.varnameof(self.children_SegmentedArray_nodes)
        )

        list_of_ndarrays = []  # TODO: これいらない

        for a_time_point_tuple in list_of_time_point_tuples:
            a_segmented_ndarray = self._retrieve_the_ndarray_segment_corresponding_to_an_original_audio_section_between_two_time_points(
                a_time_point_tuple_in_s=a_time_point_tuple)

            assert not np.isnan(a_segmented_ndarray).any(), \
                f"{self.parent_AudioExtractedArray_node.parent_AudioDataExtractors_node.parent_SingleAudioInfo_node.relative_path_to_audio} \n" \
                f"{a_time_point_tuple} \n" \
                f"{a_segmented_ndarray}"
            assert np.isfinite(a_segmented_ndarray).all(),  \
                f"{self.parent_AudioExtractedArray_node.parent_AudioDataExtractors_node.parent_SingleAudioInfo_node.relative_path_to_audio} \n" \
                f"{a_time_point_tuple} \n" \
                f"{a_segmented_ndarray}"
            my_utils.data_structure_helpers.assert_all_elements_in_ndarray_are_number(a_segmented_ndarray)

            a_cached_child_SegmentedArray_node_instance_variables = None
            if self.cached_children_SegmentedArray_instance_variables_stored_as_dict is not None and \
                    a_time_point_tuple in self.cached_children_SegmentedArray_instance_variables_stored_as_dict.keys():
                a_cached_child_SegmentedArray_node_instance_variables = self.cached_children_SegmentedArray_instance_variables_stored_as_dict[
                    a_time_point_tuple]

            a_SegmentedArray_node = SegmentedArray(
                parent_ArraySegmenter_node=self,
                time_point_tuple=a_time_point_tuple,
                segmented_ndarray=a_segmented_ndarray,
                cached_instance_variables_of_this_node=a_cached_child_SegmentedArray_node_instance_variables
            )

            self.children_SegmentedArray_nodes[a_time_point_tuple] = a_SegmentedArray_node
            list_of_ndarrays.append(a_segmented_ndarray)  # TODO: これいらない

        # graph_generators.plot_multiple_x_y_graphs(list_of_Ys=list_of_ndarrays) #TODO: これいらない

        my_utils.assert_all_elements_in_a_list_of_ndarray_are_different(list_of_ndarrays)  # TODO:これいらない

        self.cached_children_SegmentedArray_instance_variables_stored_as_dict = None  # Clear Cache

    ''' TESTED on Aug 21, 2022: Behavior of this method is noted in 'method_behavior1.jpg' '''

    def _compute_list_of_all_possible_audio_segment_time_point_tuples_in_s_and_return(self):
        entire_audio_length_in_s = self.entire_audio_length_in_s
        gh.assert_class(entire_audio_length_in_s, float)

        list_of_time_point_tuples_to_return = []
        time_point_tuple_representing_entire_audio = (0, entire_audio_length_in_s)
        list_of_time_point_tuples_to_return.append(time_point_tuple_representing_entire_audio)

        if entire_audio_length_in_s <= PrmsContainer.minimum_audio_segment_duration_in_s:
            list_of_time_point_tuples_to_return.append((0, entire_audio_length_in_s))
            return list_of_time_point_tuples_to_return

        segment_duration_in_s = PrmsContainer.minimum_audio_segment_duration_in_s
        while segment_duration_in_s < entire_audio_length_in_s and segment_duration_in_s < PrmsContainer.maximum_audio_segment_duration_in_s:
            start_time_point_in_s = 0.0
            while start_time_point_in_s + segment_duration_in_s < entire_audio_length_in_s and segment_duration_in_s < PrmsContainer.maximum_audio_segment_duration_in_s:
                # TODO:本当は、このifの２つ目のandはいらない。外側のWhileループで確認済。でも、もし将来Inner While Lopopをいじったときに、
                # 間違ってmaximum_segment_durationをこえてしまわないように二つ目のif conditionを入れておいた。
                a_time_point_tuple = (start_time_point_in_s, start_time_point_in_s + segment_duration_in_s)
                list_of_time_point_tuples_to_return.append(a_time_point_tuple)
                start_time_point_in_s = start_time_point_in_s + PrmsContainer.audio_segment_interval_in_s
            segment_duration_in_s = segment_duration_in_s + PrmsContainer.audio_segment_duration_increase_in_s

        return list_of_time_point_tuples_to_return

    ''' This is very rough implementation. This implementation ignores issues around floating point number divisions, floating point number roundings (切り捨て、切り上げ)
     なんか、list index out of rangeのerorrが怖いから、floating numbersはすべて切り捨てにしている。
     だから、多分　list_to_be_segmentedの最後の数字とかが、反映されなかったり、もともとのaudio の sectionと、このmethodから得られるarrayが微妙にずれている、
     などの問題が考えられる。'''
    '''TODO: NOT TESTED as of Aug 21, 2022'''

    def _retrieve_the_ndarray_segment_corresponding_to_an_original_audio_section_between_two_time_points(self, a_time_point_tuple_in_s):
        entire_audio_length_in_s = self.entire_audio_length_in_s
        gh.assert_class(entire_audio_length_in_s, float)

        ndarray_to_be_segmented = self.parent_AudioExtractedArray_node.attribute_alias2[0]
        gh.assert_class(ndarray_to_be_segmented, np.ndarray)

        assert isinstance(a_time_point_tuple_in_s, tuple) and len(a_time_point_tuple_in_s) == 2
        ndarray_to_be_segmented = my_utils.data_structure_helpers.convert_to_ndarray_if_list(ndarray_to_be_segmented)

        ndarray_index_start = int(ndarray_to_be_segmented.size * (a_time_point_tuple_in_s[0] / entire_audio_length_in_s))
        ndarray_index_end = int(ndarray_to_be_segmented.size * (a_time_point_tuple_in_s[1] / entire_audio_length_in_s))

        return ndarray_to_be_segmented[ndarray_index_start:ndarray_index_end]


class SegmentedArray(TreeClassesCommon):
    def __init__(self, parent_ArraySegmenter_node=None, time_point_tuple=None, segmented_ndarray=None, cached_instance_variables_of_this_node=None):
        # TODO: Once I change pickle based caching into dict based caching, then I need the assertaion below:
        #  assert cached_instance_variables_of_this_node is None or isinstance(cached_instance_variables_of_this_node, dict)
        if parent_ArraySegmenter_node is not None:
            assert isinstance(parent_ArraySegmenter_node, ArraySegmenter)
            assert time_point_tuple is not None
            assert segmented_ndarray is not None
        if time_point_tuple is not None:
            assert isinstance(time_point_tuple, tuple) and len(time_point_tuple) == 2
            assert parent_ArraySegmenter_node is not None
            assert segmented_ndarray is not None
        if segmented_ndarray is not None:
            assert isinstance(segmented_ndarray, np.ndarray)
            pass  # No need to make sure other variables are not none. This is because I might be using this class as something that is not part of the tree.

        self.parent_ArraySegmenter_node = parent_ArraySegmenter_node
        self.time_point_tuple = time_point_tuple
        my_utils.data_structure_helpers.assert_all_elements_in_ndarray_are_number(segmented_ndarray)
        self.segmented_ndarray = segmented_ndarray
        self.attribute_alias1385 = self.segmented_ndarray

        super().__init__(parent_node=self.parent_ArraySegmenter_node,
                         value_to_save_to_relational_db=self.time_point_tuple,
                         column_name_of_relational_db=gh.varnameof(self.time_point_tuple))

        # print ('Segmented Array------------------------------')
        # print (self.segmented_ndarray)
        # print (TH.get_two_lists_about_upper_nodes_in_the_tree_structure___return_as_str___I_randomly_wrote_this_function_for_debugging(self))

        self.child_OutsiderDataClipper_node = None
        cached_child_OutsiderDataClipper_node = TH.get_value_from_cached_variables_or_compute_value(
            cached_variables=cached_instance_variables_of_this_node,
            varname_str_as_dict_key=gh.varnameof(self.child_OutsiderDataClipper_node)
        )
        if self.create_next_node_upon_instantiation_of_this_node == True:
            self.child_OutsiderDataClipper_node = OutsiderDataClipper(
                parent_SegmentedArray_node=self,
                cached_instance_variables_of_this_node=cached_child_OutsiderDataClipper_node
            )


class OutsiderDataClipper(TreeClassesCommon):
    possible_clip_thresh_percentiles = [0.0, 0.05, 0.1, 1.0]
    '''
    percentile takes value between 0 and 100
    If a clip_thresh_percentile is 0.05, the numbers below 0.05 percentile and the number below 0.05 percentaile wil be taken.
    This means that if there are 1000 elements in an array, 1 element will be clipped.
    0.0 means NO
    '''

    def __init__(self, parent_SegmentedArray_node, cached_instance_variables_of_this_node=None):
        assert isinstance(parent_SegmentedArray_node, SegmentedArray)

        self.parent_SegmentedArray_node = parent_SegmentedArray_node
        super().__init__(parent_node=self.parent_SegmentedArray_node)

        if self.create_next_node_upon_instantiation_of_this_node == False:
            return

        self.children_ClippedArray_nodes = None
        '''  {clip_thresh_percentile: a_child_node_instance}
        e.g. {0.05                  : child_node_instance1,
              0.1                   : child_node_instance2}'''
        cached_children_ClippedArray_nodes_stored_as_dict = TH.get_value_from_cached_variables_or_compute_value(
            cached_variables=cached_instance_variables_of_this_node,
            varname_str_as_dict_key=gh.varnameof(self.children_ClippedArray_nodes)
        )
        self.children_ClippedArray_nodes = {}

        ndarray_before_clipped = self.parent_SegmentedArray_node.attribute_alias1385

        # ------------Below is the logic to clip array------------------------------------------------------------
        for a_clip_thresh_percentile in self.__class__.possible_clip_thresh_percentiles:
            clipped_array = None
            cached_instance_variables_of_next_node = None
            if cached_children_ClippedArray_nodes_stored_as_dict is not None and \
                    a_clip_thresh_percentile in cached_children_ClippedArray_nodes_stored_as_dict:
                clipped_array = cached_children_ClippedArray_nodes_stored_as_dict[a_clip_thresh_percentile].attribute_alias1
                cached_instance_variables_of_next_node = cached_children_ClippedArray_nodes_stored_as_dict[a_clip_thresh_percentile]
            else:
                clipped_array = my_utils.data_structure_helpers.clip_a_ndarray(a_ndarray=ndarray_before_clipped,
                                                                               lower_thresh_percentile=a_clip_thresh_percentile,
                                                                               upper_thresh_percentile=100-a_clip_thresh_percentile)
                cached_instance_variables_of_next_node = None

            a_child_ClippedArray_node = ClippedArray(
                parent_OutsiderDataClipper_node=self,
                clipped_array=clipped_array,
                clip_thresh_percentile=a_clip_thresh_percentile,
                cached_instance_variables_of_this_node=cached_instance_variables_of_next_node
            )

            self.children_ClippedArray_nodes[a_clip_thresh_percentile] = a_child_ClippedArray_node


class ClippedArray(TreeClassesCommon):
    def __init__(self, parent_OutsiderDataClipper_node, clipped_array, clip_thresh_percentile, cached_instance_variables_of_this_node=None):
        assert isinstance(parent_OutsiderDataClipper_node, OutsiderDataClipper)
        assert isinstance(clipped_array, np.ndarray)
        my_utils.data_structure_helpers.assert_all_elements_in_ndarray_are_number(clipped_array)
        gh.ensure_a_value_is_a_number(clip_thresh_percentile)
        assert 0.0 <= clip_thresh_percentile <= 100.0
        # TODO: Once I change pickle based caching into dict based caching, then I need the assertaion below:
        #  assert cached_instance_variables_of_this_node is None or isinstance(cached_instance_variables_of_this_node, dict)

        self.parent_OutsiderDataClipper_node = parent_OutsiderDataClipper_node
        self.clipped_array = clipped_array
        self.attribute_alias1 = self.clipped_array
        self.clip_thresh_percentile = clip_thresh_percentile
        super().__init__(parent_node=parent_OutsiderDataClipper_node,
                         value_to_save_to_relational_db=self.clip_thresh_percentile,
                         column_name_of_relational_db=gh.varnameof(self.clip_thresh_percentile))

        # print ('ClippedArray-----------------')
        # print (self.clipped_array)
        # print( TH.get_two_lists_about_upper_nodes_in_the_tree_structure___return_as_str___I_randomly_wrote_this_function_for_debugging(self))

        self.child_AmbienceReducer_node = None
        cached_child_AmbienceReducer_node = TH.get_value_from_cached_variables_or_compute_value(
            cached_variables=cached_instance_variables_of_this_node,
            varname_str_as_dict_key=gh.varnameof(self.child_AmbienceReducer_node)
        )

        if self.create_next_node_upon_instantiation_of_this_node == True:
            self.child_AmbienceReducer_node = AmbienceReducer(
                parent_ClippedArray_node=self,
                cached_instance_variables_of_this_node=cached_child_AmbienceReducer_node
            )


class AmbienceReducer(TreeClassesCommon):
    # TODO: コメントアウトしたAmbience Reduced Arrayを復活させないと、このList は意味ない。
    def __init__(self, parent_ClippedArray_node=None, cached_instance_variables_of_this_node=None):
        if parent_ClippedArray_node is not None:
            gh.assert_class(parent_ClippedArray_node, ClippedArray)
            assert isinstance(parent_ClippedArray_node, ClippedArray)

        self.parent_ClippedArray_node = parent_ClippedArray_node
        super().__init__(parent_node=self.parent_ClippedArray_node)

        self.children_AmbienceReducedArray_nodes = None
        '''  {a_percentile: a_child_node_instance}
        e.g. {75          : child_node_instance1,
              80          : child_node_instance2}'''
        cached_children_AmbienceReducedArray_nodes_stored_as_dict = TH.get_value_from_cached_variables_or_compute_value(
            cached_variables=cached_instance_variables_of_this_node,
            varname_str_as_dict_key=gh.varnameof(self.children_AmbienceReducedArray_nodes)
        )
        assert cached_children_AmbienceReducedArray_nodes_stored_as_dict is None or \
            isinstance(cached_children_AmbienceReducedArray_nodes_stored_as_dict, dict)
        self.children_AmbienceReducedArray_nodes = {}

        if self.create_next_node_upon_instantiation_of_this_node == False:
            return

        array_before_reducing_ambience = self.parent_ClippedArray_node.clipped_array
        for a_percentile in PrmsContainer.ambience_reducer_list_of_possible_percentiles_below_which_numbers_are_set_to_zero:
            # ambience_reduced_array = \
            #    gh.normalize_and_set_values_below_i_percentile_to_zero(X=array_before_reducing_ambience,
            #                                                           a_percentile=a_percentile)
            # gh.assert_all_elements_in_ndarray_are_number(ambience_reduced_array)

            cached_instance_variables_of_next_node = None
            if cached_children_AmbienceReducedArray_nodes_stored_as_dict is not None and a_percentile in cached_children_AmbienceReducedArray_nodes_stored_as_dict.keys():
                cached_instance_variables_of_next_node = cached_children_AmbienceReducedArray_nodes_stored_as_dict[a_percentile]

            an_ambience_reduced_array_obj = AmbienceReducedArray(
                parent_AmbienceReducer_node=self,
                ambience_reduced_ndarray=array_before_reducing_ambience,  # ambience_reduced_ndarray=ambience_reduced_array,
                reduced_ambience_percentile=a_percentile,
                cached_instance_variables_of_this_node=cached_instance_variables_of_next_node
            )

            self.children_AmbienceReducedArray_nodes[a_percentile] = an_ambience_reduced_array_obj


class AmbienceReducedArray(TreeClassesCommon):
    def __init__(self,
                 parent_AmbienceReducer_node=None,
                 ambience_reduced_ndarray=None,
                 reduced_ambience_percentile=None,
                 cached_instance_variables_of_this_node=None):
        if parent_AmbienceReducer_node is not None:
            gh.assert_class(parent_AmbienceReducer_node, AmbienceReducer)
            assert isinstance(parent_AmbienceReducer_node, AmbienceReducer)
        if ambience_reduced_ndarray is not None:
            gh.assert_class(ambience_reduced_ndarray, np.ndarray)
            assert isinstance(ambience_reduced_ndarray, np.ndarray)
        if reduced_ambience_percentile is not None:
            gh.is_number(reduced_ambience_percentile)

        self.parent_AmbienceReducer_node = parent_AmbienceReducer_node
        self.ambience_reduced_ndarray = ambience_reduced_ndarray

        self.reduced_ambience_percentile = reduced_ambience_percentile
        super().__init__(parent_node=parent_AmbienceReducer_node,
                         value_to_save_to_relational_db=self.reduced_ambience_percentile,
                         column_name_of_relational_db=gh.varnameof(self.reduced_ambience_percentile))

        # print ('AmbienceReducedArray--------------------------------------')
        # print (ambience_reduced_ndarray)
        # print(TH.get_two_lists_about_upper_nodes_in_the_tree_structure___return_as_str___I_randomly_wrote_this_function_for_debugging(self))
        # assert not np.isnan(ambience_reduced_ndarray).any(), \
        #   str(ambience_reduced_ndarray) + TH.get_two_lists_about_upper_nodes_in_the_tree_structure___return_as_str___I_randomly_wrote_this_function_for_debugging(self)
        # assert np.isfinite(ambience_reduced_ndarray).all(), \
        #    str(ambience_reduced_ndarray) + TH.get_two_lists_about_upper_nodes_in_the_tree_structure___return_as_str___I_randomly_wrote_this_function_for_debugging(
        #       self)

        if self.create_next_node_upon_instantiation_of_this_node == False:
            return

        self.child_ArrayFlipper_node = None
        cached_child_ArrayFlipper_node = TH.get_value_from_cached_variables_or_compute_value(
            cached_variables=cached_instance_variables_of_this_node,
            varname_str_as_dict_key=gh.varnameof(self.child_ArrayFlipper_node)
        )
        if self.create_next_node_upon_instantiation_of_this_node == True:
            self.child_ArrayFlipper_node = ArrayFlipper(
                parent_AmbienceReducedArray_node=self,
                cached_instance_variables_of_this_node=cached_child_ArrayFlipper_node
            )


class ArrayFlipper(TreeClassesCommon):
    def __init__(self, parent_AmbienceReducedArray_node, cached_instance_variables_of_this_node=None):
        assert isinstance(parent_AmbienceReducedArray_node, AmbienceReducedArray)
        # TODO: Once I change pickle based caching into dict based caching, then I need the assertaion below:
        #  assert cached_instance_variables_of_this_node is None or isinstance(cached_instance_variables_of_this_node, dict)

        self.parent_AmbienceReducedArray_node = parent_AmbienceReducedArray_node
        super().__init__(parent_node=self.parent_AmbienceReducedArray_node)

        self.children_FlippedArray_nodes = None
        '''  {a_boolean_val(flip or not) : a_child_node_instance}
        e.g. {True                       : child_node_instance1,
              False                      : child_node_instance2}'''

        cached_children_FlippedArray_nodes_stored_as_dict = TH.get_value_from_cached_variables_or_compute_value(
            cached_variables=cached_instance_variables_of_this_node,
            varname_str_as_dict_key=gh.varnameof(self.children_FlippedArray_nodes)
        )
        self.children_FlippedArray_nodes = {}

        ndarray_before_flipped = self.parent_AmbienceReducedArray_node.ambience_reduced_ndarray

        # Below is the logic to flip array-------------------------------------------------------------------------------
        for upside_down in [True, False]:
            resulting_array = None
            cached_instance_variables_of_next_node = None
            if cached_children_FlippedArray_nodes_stored_as_dict is not None and \
                    upside_down in cached_children_FlippedArray_nodes_stored_as_dict.keys():
                resulting_array = cached_children_FlippedArray_nodes_stored_as_dict[upside_down].attribute_alias1
                cached_instance_variables_of_next_node = cached_children_FlippedArray_nodes_stored_as_dict[upside_down]
            else:
                my_utils.data_structure_helpers.assert_all_elements_in_ndarray_are_number(ndarray_before_flipped)
                if upside_down:
                    resulting_array = my_utils.data_structure_helpers.upside_down_a_ndarray(ndarray_before_flipped)
                else:
                    resulting_array = ndarray_before_flipped
                cached_instance_variables_of_next_node = None

                my_utils.data_structure_helpers.assert_all_elements_in_ndarray_are_number(resulting_array)

            a_child_FlippedArray_node = FlippedArray(
                parent_ArrayFlipper_node=self,
                flipped_or_nonflipped_array=resulting_array,
                upside_down=upside_down,
                cached_instance_variables_of_this_node=cached_instance_variables_of_next_node
            )

            self.children_FlippedArray_nodes[upside_down] = a_child_FlippedArray_node


class FlippedArray(TreeClassesCommon):
    def __init__(self, parent_ArrayFlipper_node=None, flipped_or_nonflipped_array=None, upside_down=None, cached_instance_variables_of_this_node=None):
        # TODO: Once I change pickle based caching into dict based caching, then I need the assertaion below:
        #  assert cached_instance_variables_of_this_node is None or isinstance(cached_instance_variables_of_this_node, dict)
        if parent_ArrayFlipper_node is not None:
            assert isinstance(parent_ArrayFlipper_node, ArrayFlipper)
            assert flipped_or_nonflipped_array is not None
            assert upside_down is not None
        if upside_down is not None:
            assert isinstance(upside_down, bool)
            assert parent_ArrayFlipper_node is not None
            assert flipped_or_nonflipped_array is not None
        if flipped_or_nonflipped_array is not None:
            assert isinstance(flipped_or_nonflipped_array, np.ndarray)
            assert not np.isnan(flipped_or_nonflipped_array).any(), \
                str(flipped_or_nonflipped_array) + TH.get_two_lists_about_upper_nodes_in_the_tree_structure___return_as_str___I_randomly_wrote_this_function_for_debugging(
                    parent_ArrayFlipper_node) + f" | {upside_down}"
            assert np.isfinite(flipped_or_nonflipped_array).all(),\
                str(flipped_or_nonflipped_array) + TH.get_two_lists_about_upper_nodes_in_the_tree_structure___return_as_str___I_randomly_wrote_this_function_for_debugging(
                    parent_ArrayFlipper_node) + f" | {upside_down}"
            my_utils.data_structure_helpers.assert_all_elements_in_ndarray_are_number(flipped_or_nonflipped_array)
            pass  # No need to make sure other variables are not none. This is because I might be using this class as something that is not part of the tree.

        self.parent_ArrayFlipper_node = parent_ArrayFlipper_node
        self.flipped_or_nonflipped_array = flipped_or_nonflipped_array
        self.attribute_alias1 = self.flipped_or_nonflipped_array
        self.upside_down = upside_down

        self.child_DataTransformers_node = None
        cached_child_DataTransformers_node = TH.get_value_from_cached_variables_or_compute_value(
            cached_variables=cached_instance_variables_of_this_node,
            varname_str_as_dict_key=gh.varnameof(self.child_DataTransformers_node)
        )

        if self.create_next_node_upon_instantiation_of_this_node == True:
            super().__init__(parent_node=self.parent_ArrayFlipper_node,
                             value_to_save_to_relational_db=self.upside_down,
                             column_name_of_relational_db=gh.varnameof(self.upside_down))

            # print ('FlippedArray-----------------------------')
            # print (self.flipped_or_nonflipped_array)
            # print (TH.get_two_lists_about_upper_nodes_in_the_tree_structure___return_as_str___I_randomly_wrote_this_function_for_debugging(self))

            self.child_DataTransformers_node = DataTransformers(
                parent_FlippedArray_node=self,
                cached_instance_variables_of_this_node=cached_child_DataTransformers_node
            )


class DataTransformers(TreeClassesCommon):
    def __init__(self, parent_FlippedArray_node, cached_instance_variables_of_this_node=None):
        assert isinstance(parent_FlippedArray_node, FlippedArray)
        # TODO: Once I change pickle based caching into dict based caching, then I need the assertaion below:
        #  assert cached_instance_variables_of_this_node is None or isinstance(cached_instance_variables_of_this_node, dict)

        self.parent_FlippedArray_node = parent_FlippedArray_node
        self.ndarray_before_transformed = my_utils.data_structure_helpers.convert_to_ndarray_if_list(parent_FlippedArray_node.attribute_alias1)
        self.attribute_alias11 = self.ndarray_before_transformed

        self.cached_instance_variables_of_this_node = cached_instance_variables_of_this_node

        self.cached_children_TransformedData_nodes_stored_as_dict = None
        self.children_TransformedData_nodes = None
        ''' { name_of_data_transformer : children_node_instance_that_stores_ndarray_after_transformed}
        e.g.{'log_transformer'         : child_node_instance1, 
             'uniform_transformer'     : child_node_instance2}   '''

        self.cached_children_TransformedData_nodes_stored_as_dict = TH.get_value_from_cached_variables_or_compute_value(
            cached_variables=cached_instance_variables_of_this_node,
            varname_str_as_dict_key=gh.varnameof(self.children_TransformedData_nodes)
        )
        self.children_TransformedData_nodes = {}
        assert self.cached_children_TransformedData_nodes_stored_as_dict is None or \
            isinstance(self.cached_children_TransformedData_nodes_stored_as_dict, dict)

        super().__init__()
        if self.create_next_node_upon_instantiation_of_this_node:
            TH.call_all_public_methods_in_a_class_instance(a_class_instance=self)

        self.cached_instance_variables_of_this_node = None  # Clear Cache
        self.cached_children_TransformedData_nodes_stored_as_dict = None  # Clear Cache

    def _common_func_wrapper_to_create_next_TransformedData_node(undecorated_func):
        @functools.wraps(undecorated_func)
        def wrapper(self, *args, **kwargs):
            assert isinstance(self, DataTransformers)

            super().__init__(parent_node=self.parent_FlippedArray_node,
                             value_to_save_to_relational_db=undecorated_func.__name__,
                             column_name_of_relational_db=gh.varnameof(DataTransformers))

            transformed_array_to_return = None
            cached_instance_variables_of_next_node = None
            if self.cached_children_TransformedData_nodes_stored_as_dict is not None and \
                    undecorated_func.__name__ in self.cached_children_TransformedData_nodes_stored_as_dict.keys():
                transformed_array_to_return = self.cached_children_TransformedData_nodes_stored_as_dict[undecorated_func.__name__].attribute_alias1
                cached_instance_variables_of_next_node = self.cached_children_TransformedData_nodes_stored_as_dict[undecorated_func.__name__]
                # gh.debuginfo('DataTransformers: read from cache')
            else:
                transformed_array_to_return = undecorated_func(self, *args, **kwargs)
                # transformed_array_to_return = np.around(transformed_array_to_return, decimals=self.__class__.class_attribute_alias1905830)
                cached_instance_variables_of_next_node = None
                # gh.debuginfo("DataTransformers: computed (i.e. NOT read from cached)")

            assert isinstance(transformed_array_to_return, np.ndarray) and transformed_array_to_return.size > 1

            if self.create_next_node_upon_instantiation_of_this_node:
                a_child_TransformedData_node = TransformedData(
                    parent_DataTransformers_node=self,
                    data_transformation_func_name=undecorated_func.__name__,
                    transformed_ndarray=transformed_array_to_return,
                    cached_instance_variables_of_this_node=cached_instance_variables_of_next_node)

                self.children_TransformedData_nodes[undecorated_func.__name__] = a_child_TransformedData_node

            return transformed_array_to_return

        return wrapper

    @_common_func_wrapper_to_create_next_TransformedData_node
    def no_transformation(self):
        # TODO: Not sure if I really need to do deep copy...But after performing test, deepcopy does not seem to consume a lot of run time.
        return copy.deepcopy(self.ndarray_before_transformed)

    # @_common_func_wrapper_to_create_next_TransformedData_node
    # def power_transformer_yeo_johnson(self):
    # TODO:このコードだとエラーになるので、scikitlearn用にreshapeされていない、もともとの1-dimentional arrayを用いて scipy.stats.yeojohnsonを使う。もっともあまりもとのShapeに対して大きな影響を与えていなさそうな気もする。
    #    return sklearn.preprocessing.PowerTransformer(method='yeo-johnson').fit_transform(self.formatted_y_for_sklearn_data_transformation)

    @_common_func_wrapper_to_create_next_TransformedData_node
    def quantile_transformer_uniform(self):
        a_ndarray = self.ndarray_before_transformed.reshape(-1, 1)
        transformed_ndarray = preprocessing.QuantileTransformer(output_distribution='uniform').fit_transform(a_ndarray)
        transformed_ndarray = transformed_ndarray.reshape(-1)
        return transformed_ndarray

    @_common_func_wrapper_to_create_next_TransformedData_node
    def quantile_transformer_normal(self):
        a_ndarray = self.ndarray_before_transformed.reshape(-1, 1)
        transformed_ndarray = preprocessing.QuantileTransformer(output_distribution='normal').fit_transform(a_ndarray)
        transformed_ndarray = transformed_ndarray.reshape(-1)
        return transformed_ndarray

    @_common_func_wrapper_to_create_next_TransformedData_node
    def log_transformer(self):

        # move all the data in ndarray to upper side so that the minumim value of the array becomes 1
        min_value_of_ndarray_before_transfrmed = np.min(self.ndarray_before_transformed)
        difference = 1.0 - min_value_of_ndarray_before_transfrmed
        if difference >= 0:
            return np.log(self.ndarray_before_transformed + difference)  # '+100' is to avoid causing -inf value
        else:
            return np.log(self.ndarray_before_transformed)

    @_common_func_wrapper_to_create_next_TransformedData_node
    def softmax(self):
        # TODO: なんか、ndarrayのままだと計算がうまく行かないから、一回Listにへんかんして、その後ndarrayに変換し直している。非効率なので改善が必要。
        X = self.ndarray_before_transformed
        X = data_scalers.z_score_normalization(X)
        # return return 1 / (abs(X) + 1)**2

        return np.exp(X) / np.exp(X).sum(axis=0)


class TransformedData(TreeClassesCommon):
    def __init__(self, parent_DataTransformers_node, data_transformation_func_name, transformed_ndarray, cached_instance_variables_of_this_node):

        # print ('TransformedData-----------------------')
        # print(transformed_ndarray)
        # print(TH.get_two_lists_about_upper_nodes_in_the_tree_structure___return_as_str___I_randomly_wrote_this_function_for_debugging(parent_DataTransformers_node))
        # print (data_transformation_func_name)

        my_utils.data_structure_helpers.assert_all_elements_in_ndarray_are_number(transformed_ndarray)
        assert isinstance(parent_DataTransformers_node, DataTransformers)
        assert isinstance(data_transformation_func_name, str)
        assert isinstance(transformed_ndarray, np.ndarray)
        # TODO: Once I change pickle based caching into dict based caching, then I need the assertaion below:
        #  assert cached_instance_variables_of_this_node is None or isinstance(cached_instance_variables_of_this_node, dict)

        self.parent_DataTransformers_node = parent_DataTransformers_node
        super().__init__(parent_node=self.parent_DataTransformers_node)
        self.data_transformation_func_name = data_transformation_func_name
        self.transformed_ndarray = transformed_ndarray
        self.attribute_alias1 = self.transformed_ndarray

        if self.create_next_node_upon_instantiation_of_this_node == True:
            LinearDataScalersAndNormalizers(self)


class LinearDataScalersAndNormalizers (TreeClassesCommon):
    warnings.warn('Deprecated. This module just passes the array to next node without scaling!!!!!On 20221016, '
                  'I decided NOT to do scaling when I extract features from audio. All linear scalings are to be done when'
                  'I compare vectors ', DeprecationWarning, stacklevel=2)

    # TODO: このクラスのコードがひどすぎるのでどうにかする。
    # array_scaling_method = PrmsContainer.audio_extracted_array_scaling_method

    def __init__(self, parent_TransformedData_node, an_array_to_scale=None):
        assert isinstance(parent_TransformedData_node, TransformedData)
        assert an_array_to_scale is None or isinstance(an_array_to_scale, np.ndarray)

        self.parent_TransformedData_node = parent_TransformedData_node

        if an_array_to_scale is None:
            an_array_to_scale = parent_TransformedData_node.attribute_alias1

        self.data_to_pass_to_next_node = an_array_to_scale
        # self.data_to_pass_to_next_node = self.__class__.array_scaling_method(an_array_to_scale)
        # self.data_to_pass_to_next_node = np.around(self.data_to_pass_to_next_node, decimals=self.__class__.decimal_points_for_scaled_data)
        self.attribute_alias1 = self.data_to_pass_to_next_node

        super().__init__(
            parent_node=parent_TransformedData_node,
            column_name_of_relational_db=None,  # gh.varnameof(self.array_scaling_method),
            value_to_save_to_relational_db=None  # self.__class__.array_scaling_method.__name__ (get function name of array_scaling_method)
        )

        if self.create_next_node_upon_instantiation_of_this_node == True:
            SaveFinalArrayToDB(parent=self)


class SaveFinalArrayToDB(TreeClassesCommon):
    # TODO: Get Current Git Version
    # TODO: Get the current time!!! Record the current time to the database!!!このNodeからSQLにデータがSAVEされた瞬間のTimingをかく。

    already_updated_sql_columns_according_to_the_current_tree_structure = False
    instance_count = 0

    def __init__(self, parent):
        self.__class__.instance_count += 1

        extracted_array_to_save = parent.attribute_alias1
        extracted_array_to_save = data_scalers.z_score_normalization(extracted_array_to_save)
        my_utils.data_structure_helpers.assert_all_values_in_one_dimentional_ndarray_are_different(extracted_array_to_save)

        super().__init__(parent_node=parent,
                         value_to_save_to_relational_db=extracted_array_to_save,
                         column_name_of_relational_db='array_data')

        dict_from_parameter_combo_column_name_to_corresponding_value = {}
        # TODO: If I need to really optimzie, I can use stack (queue.LifoQueue(maxsize=30))
        list_of_tuples_of_column_name_and_previous_column_name_and_datatype = []
        list_of_column_names_of_relational_db = []
        list_of_values_to_save_to_relational_db = []

        current = parent
        while True:
            column_name = current.column_name_of_relational_db
            if column_name is not None:
                value = current.value_to_save_to_relational_db
                assert value is not None  # This assertation is actually not needed. The same assertaion exists in the __init__ of TreeClassesCommon as of Sep 10, 2022
                sql_datatype = my_db_utils.get_sql_datatype_for_a_python_value(value)  # "REAL", "TEXT
                previous_column_name = current.attribute_alias19482

                dict_from_parameter_combo_column_name_to_corresponding_value[column_name] = value
                list_of_tuples_of_column_name_and_previous_column_name_and_datatype.append((column_name, previous_column_name, sql_datatype))
                list_of_column_names_of_relational_db.append(column_name)
                list_of_values_to_save_to_relational_db.append(value)

            assert hasattr(current, gh.varnameof(current.parent_node)), f"all tree nodes need to have {gh.varnameof(TreeClassesCommon)} as super class"
            if current.parent_node is None:
                break
            else:
                current = current.parent_node

        list_of_tuples_of_column_name_and_previous_column_name_and_datatype.reverse()
        list_of_column_names_of_relational_db.reverse()
        list_of_values_to_save_to_relational_db.reverse()

        if self.__class__.instance_count == 1:
            extracted_audio_features_db_manager.Table_From_ParameterComboHashValue_To_ParameterCombo.create_the_table_if_not_exists()
            extracted_audio_features_db_manager.Table_From_ParameterComboHashValue_To_FinalExtractedData.create_the_table_if_not_exists()

        if self.__class__.instance_count == 1:
            extracted_audio_features_db_manager.AudioDataViewTable_HumanComprehensible.delete_the_table_if_exists()

        if self.__class__.instance_count == 1:
            my_utils.terminal_interaction_helpers.warninginfo("Updating sql columns according to the current tree structure. If this message appears more than one time in the same run time,"
                                                              "then something wrong is happening...")

            extracted_audio_features_db_manager.Table_From_ParameterComboHashValue_To_ParameterCombo.change_columns_according_to_the_newest_tree_structure(

                list_of_newest_tuples_of_column_name_and_previous_column_name_and_datatype=list_of_tuples_of_column_name_and_previous_column_name_and_datatype
            )

        parameter_combo_hash_value = hash_funcs.hash_an_audio_extraction_parameter_combo_list(list_of_values_to_save_to_relational_db)

        extracted_audio_features_db_manager.Table_From_ParameterComboHashValue_To_FinalExtractedData.add_a_row(
            a_parameter_combo_hash_value=parameter_combo_hash_value,
            audio_extracted_data_as_ndarray=extracted_array_to_save
        )

        extracted_audio_features_db_manager.Table_From_ParameterComboHashValue_To_ParameterCombo.add_a_row(
            parameter_combo_hash_value=parameter_combo_hash_value,
            parameter_combo_columns_names_and_values_as_dict=dict_from_parameter_combo_column_name_to_corresponding_value
        )

        if self.__class__.instance_count == 1:
            extracted_audio_features_db_manager.AudioDataViewTable_HumanComprehensible.create_or_replace_the_table()

        return

    def _delete_an_element_from_two_database(self):
        raise Exception('not implemented yet')
