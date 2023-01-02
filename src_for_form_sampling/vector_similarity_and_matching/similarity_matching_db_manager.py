import copy
import sys

import numpy as np

import my_utils.data_structure_helpers
from my_utils import misc_helpers as gh
from src_for_form_sampling.parameters_package import PrmsContainer
import ast
from my_utils import my_db_utils

# ======================================================================================================================
# ----------------------------Change This Db Configuration if I use other database--------------------------------------
my_db_utils.INTERVAL_IN_S_BETWEEN_DB_ACCESS_RETRIES = PrmsContainer.interval_in_s_between_db_access_retries
my_db_utils.HOW_MANY_TIMES_TO_RETRY_DB_ACCESS = PrmsContainer.how_many_times_to_retry_db_access


# _this_db_instanceは_db_operators_instance_of_this_moduleとしたほうが名前的にわかりやすいけど、名前の短さを優先して、この名前になった。
_this_db_instance = None
# Set this from outside this module: e.g.
#  _this_db_instance = my_db_utils.DbOperators (db_type='sqlite', sqlite_db_path=PrmsContainer.path_to_similarity_matching_db
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================


class Table_From_JsonSavedListHashValue_To_JsonSavedList:
    table_name = "From_JsonSavedListHashValue_To_JsonSavedList"  # inspect.stack()[0][3] can get class name
    list_of_tuples_of_column_name_and_sql_datatype = [
        ('json_saved_list_hash_value', 'TEXT'),
        ('json_saved_list', 'TEXT')
    ]
    primary_key_column_names = ['json_saved_list_hash_value']

    @classmethod
    def create_the_table_if_not_exists(cls):
        _this_db_instance.create_a_table_if_not_exists(
            table_name=cls.table_name,
            list_of_tuples_of_column_name_and_sql_datatype=cls.list_of_tuples_of_column_name_and_sql_datatype,
            list_of_primary_key_column_names=cls.primary_key_column_names
        )

    @classmethod
    def add_a_row(cls, a_json_saved_list_hash_value, a_json_saved_list):
        assert isinstance(a_json_saved_list_hash_value, str)
        assert isinstance(a_json_saved_list, list)

        cls.create_the_table_if_not_exists()

        _this_db_instance.add_a_row_to_a_table(
            table_name=cls.table_name,
            row_values_as_dict_from_column_name_to_corresponding_value={'json_saved_list_hash_value': a_json_saved_list_hash_value,
                                                                        'json_saved_list': my_db_utils.serialize_a_value_to_store_in_sql_db(a_json_saved_list)},
            action_for_duplicates="REPLACE"
        )

    @classmethod
    def does_a_column_exist(cls, column_name):
        return _this_db_instance.does_a_column_exist_in_a_table(table_name=cls.table_name, column_name=column_name)

    @classmethod
    def get_json_saved_list_from_json_saved_list_hash_value(cls, json_saved_list_hash_value):
        assert isinstance(json_saved_list_hash_value, str)
        json_saved_list_as_str = _this_db_instance.get_value_from_key(table_name=cls.table_name,
                                                                      primary_key_as_dict_from_column_name_to_corresponding_value={'json_saved_list_hash_value': json_saved_list_hash_value})
        json_saved_list = my_db_utils.get_python_value_from_sql_text_str(json_saved_list_as_str, list)
        return json_saved_list

    @classmethod
    def get_json_saved_list_as_ndarray_from_json_saved_list_hash_value(cls, json_saved_list_hash_value):
        assert isinstance(json_saved_list_hash_value, str)
        json_saved_list_as_str = _this_db_instance.get_value_from_key(
            table_name=cls.table_name,
            primary_key_as_dict_from_column_name_to_corresponding_value={'json_saved_list_hash_value': json_saved_list_hash_value}
        )
        json_saved_list = my_db_utils.get_python_value_from_sql_text_str(json_saved_list_as_str, list)
        return np.array(json_saved_list)


class ListOfMaximumSimilarityValues_Between_AudioSegment_And_SetOfTargetShapes:
    table_name = 'ListOfMaximumSimilarityValues_Between_AudioSegment_And_SetOfTargetShapes'
    column_names = ['audio_hash_value',
                    'time_point_tuple',
                    'hash_value_of_a_set_of_target_shapes',
                    'matching_list_between_audio_parameter_combo_hash_value_and_target_shape_hash_value',
                    'maximum_similarity_value']
    sql_datatypes = ['TEXT', 'TEXT', 'TEXT', 'TEXT', 'REAL']
    primary_key_column_names = ['audio_hash_value', 'time_point_tuple', 'hash_value_of_a_set_of_target_shapes']
    list_of_tuples_of_column_name_and_sql_datatype = []
    assert len(column_names) == len(sql_datatypes)
    for i in range(0, len(column_names)):
        list_of_tuples_of_column_name_and_sql_datatype.append((column_names[i], sql_datatypes[i]))

    @classmethod
    def create_the_table_if_not_exists(cls):
        _this_db_instance.create_a_table_if_not_exists(
            table_name=cls.table_name,
            list_of_tuples_of_column_name_and_sql_datatype=cls.list_of_tuples_of_column_name_and_sql_datatype,
            list_of_primary_key_column_names=cls.primary_key_column_names
        )

    @classmethod
    def does_a_column_exist(cls, column_name):
        return _this_db_instance.does_a_column_exist_in_a_table(table_name=cls.table_name, column_name=column_name)

    @classmethod
    def does_a_row_exist(cls, audio_hash_value, time_point_tuple, hash_value_of_a_set_of_target_shapes):
        return _this_db_instance.does_a_row_exist_in_a_table(
            table_name=cls.table_name,
            row_values_as_dict_from_primary_key_column_name_to_corresponding_value={
                'audio_hash_value': audio_hash_value,
                'time_point_tuple': time_point_tuple,
                'hash_value_of_a_set_of_target_shapes': hash_value_of_a_set_of_target_shapes
            }
        )

    @classmethod
    def add_or_replace_a_row(cls, a_row_as_tuple):
        assert isinstance(a_row_as_tuple, tuple)
        assert len(a_row_as_tuple) == len(cls.column_names)

        _this_db_instance.add_a_row_to_a_table(
            table_name=cls.table_name,
            list_of_column_names_for_new_values=cls.column_names,
            list_of_new_values=list(a_row_as_tuple),
            action_for_duplicates='REPLACE'
        )

    @classmethod
    def add_multiple_rows_to_the_table(cls, list_of_new_row_tuples):
        _this_db_instance.add_multiple_rows_to_a_table(table_name=cls.table_name, list_of_new_row_tuples=list_of_new_row_tuples)


class From_AudioHashValue_To_AudioFileRelativePath:
    """
    TODO: このクラスは、Extracted Audio Features Databaseにある同名のTable"From_AudioHashValue_To_AudioFileRelativePath"を
        そのまま書き写したようなクラスである。本当は、2つのdatabase間でReference関係を作れるみたいなことができればいいけど、そんなに簡単ではないみたい。
    """
    table_name = "From_AudioHashValue_To_AudioFileRelativePath"
    column_names = ['audio_hash_value', 'relative_path_to_audio_file']
    primary_key_column_names = ['audio_hash_value']
    sql_datatypes = ['TEXT', 'TEXT']
    dict_from_column_names_to_sql_datatypes = dict(zip(column_names, sql_datatypes))

    @classmethod
    def create_or_replace_the_table(cls):
        _this_db_instance.create_a_table_if_not_exists(
            table_name=cls.table_name,
            dict_from_column_names_to_sql_datatypes=cls.dict_from_column_names_to_sql_datatypes,
            list_of_primary_key_column_names=cls.primary_key_column_names
        )

    @classmethod
    def add_a_row_to_the_table(cls, audio_hash_value, relative_path_to_audio_file):
        assert isinstance(audio_hash_value, str), type(audio_hash_value)
        assert isinstance(relative_path_to_audio_file, str), type(relative_path_to_audio_file)

        _this_db_instance.add_a_row_to_a_table(
            table_name=cls.table_name,
            list_of_column_names_for_new_values=cls.column_names,
            list_of_new_values=[audio_hash_value, relative_path_to_audio_file],
            action_for_duplicates='REPLACE'
        )


class View_ListOfMaximumSimilarityValues_Between_AudioSegment_And_SetOfTargetShapes:
    table_name = "View_ListOfMaximumSimilarityValues_Between_AudioSegment_And_SetOfTargetShapes"
    column_names = ['relative_path_to_audio_file',
                    'time_point_tuple',
                    'hash_value_of_a_set_of_target_shapes',
                    'matching_list_between_audio_parameter_combo_hash_value_and_target_shape_hash_value',
                    'maximum_similarity_value']
    sql_datatypes = copy.deepcopy(ListOfMaximumSimilarityValues_Between_AudioSegment_And_SetOfTargetShapes.sql_datatypes)
    inner_join_column_name1 = 'audio_hash_value'
    column_name_to_order_by = column_names[4]
    assert column_name_to_order_by == 'maximum_similarity_value'

    @classmethod
    def create_or_replace_the_table_ordered_by_similarity_value(cls):
        """
        TODO: This function is just so dirty!!!! おそらく、Table_From_AudioHashValue_To_AudioFileRelativePathクラスの中に、いろいろな
            Assertionsを埋め込んでしまえば、スッキリ書けるはず。本来はextracted audio features dbの方でやるべきことを、こちらでやってしまっている。
        """

        from src_for_form_sampling.preprocessing_audio_feature_extraction import extracted_audio_features_db_manager

        cls._assert_inner_join_is_possible()
        cls.delete_the_table_if_exists()

        list_of_tuples_of_audio_hash_value_and_relative_path_to_audio_file = \
            extracted_audio_features_db_manager.Table_From_AudioHashValue_To_AudioFileRelativePath.get_all_rows_of_the_table_as_a_list_of_row_value_tuples()

        From_AudioHashValue_To_AudioFileRelativePath.create_or_replace_the_table()

        for an_element in list_of_tuples_of_audio_hash_value_and_relative_path_to_audio_file:
            audio_hash_value = an_element[0]
            relative_path_to_audio_file = an_element[1]
            From_AudioHashValue_To_AudioFileRelativePath.add_a_row_to_the_table(
                audio_hash_value=audio_hash_value, relative_path_to_audio_file=relative_path_to_audio_file)

        sql_statement = f"""
            CREATE VIEW {cls.table_name} AS
                SELECT {my_db_utils.convert_list_to_comma_separated_str(cls.column_names)}
                FROM {ListOfMaximumSimilarityValues_Between_AudioSegment_And_SetOfTargetShapes.table_name}
                INNER JOIN {extracted_audio_features_db_manager.Table_From_AudioHashValue_To_AudioFileRelativePath.table_name}
                    ON {ListOfMaximumSimilarityValues_Between_AudioSegment_And_SetOfTargetShapes.table_name}.{cls.inner_join_column_name1}
                    = {extracted_audio_features_db_manager.Table_From_AudioHashValue_To_AudioFileRelativePath.table_name}.{cls.inner_join_column_name1}                
                ORDER BY {cls.column_name_to_order_by} DESC       
        """

        _this_db_instance._execute_sql_statement(sql_statement)

    @classmethod
    def _assert_inner_join_is_possible(cls):
        """
        TODO: This function is just so dirty!!!! おそらく、Table_From_AudioHashValue_To_AudioFileRelativePathクラスの中に、いろいろな
            Assertionsを埋め込んでしまえば、スッキリ書けるはず。本来はextracted audio features dbの方でやるべきことを、こちらでやってしまっている。
        """

        from src_for_form_sampling.preprocessing_audio_feature_extraction import extracted_audio_features_db_manager

        # TODO: 特に以下の2つのAssertionがめちゃくちゃわかりにくい。extracted_audio_featrues_db_managerを使っているassertionはすべて
        #   extracted_audio-features_db_managersの方に持っていくべき。
        assert extracted_audio_features_db_manager._this_db_instance.does_a_table_exist(
            extracted_audio_features_db_manager.Table_From_AudioHashValue_To_AudioFileRelativePath.table_name)
        assert _this_db_instance.does_a_table_exist(ListOfMaximumSimilarityValues_Between_AudioSegment_And_SetOfTargetShapes.table_name)

        assert extracted_audio_features_db_manager.Table_From_AudioHashValue_To_AudioFileRelativePath.does_a_column_exist(cls.inner_join_column_name1)
        assert ListOfMaximumSimilarityValues_Between_AudioSegment_And_SetOfTargetShapes.does_a_column_exist(cls.inner_join_column_name1)

        my_utils.data_structure_helpers.assert_list1_contains_all_elements_of_list2(
            list1=ListOfMaximumSimilarityValues_Between_AudioSegment_And_SetOfTargetShapes.column_names,
            list2=cls.column_names[1:]
        )

    '''
    TODO: To Implement
    @classmethod
    def get_first_x_rows_as_list_of_tuples (cls, x):
        return _this_db_instance.get_first_x_rows_from_a_table_as_list_of_tuples_of_string(table_name=cls.table_name, x=x)
    '''

    @classmethod
    def get_nth_row_as_a_tuple(cls, nth):
        cls.create_or_replace_the_table_ordered_by_similarity_value()  # TODO: いちいちCreate_or_replace_tableするのは、非効率かもしれない。Performance Optimizaeするとき注意。

        column_names = _this_db_instance.get_list_of_columns_of_a_table(table_name=cls.table_name)

        assert len(column_names) == 5
        assert cls.column_names == column_names

        a_tuple_of_strings = _this_db_instance.get_nth_row_from_a_table_as_a_tuple_of_strings(table_name=cls.table_name, nth=nth)

        assert isinstance(a_tuple_of_strings[0], str)
        assert isinstance(ast.literal_eval(a_tuple_of_strings[1]), tuple)
        assert isinstance(a_tuple_of_strings[2], str)
        assert isinstance(ast.literal_eval(a_tuple_of_strings[3]), list)
        assert isinstance(a_tuple_of_strings[4], float)

        tuple_to_return = (
            a_tuple_of_strings[0],
            my_db_utils.get_python_value_from_sql_text_str(a_tuple_of_strings[1], tuple),
            a_tuple_of_strings[2],
            ast.literal_eval(a_tuple_of_strings[3]),
            a_tuple_of_strings[4]
        )

        return tuple_to_return

    @classmethod
    def get_matching_list_of_tuples_of_parameter_combo_hash_value_and_target_shape_hash_value_of_nth_row(cls, nth):
        assert cls.column_names[3] == 'matching_list_between_audio_parameter_combo_hash_value_and_target_shape_hash_value'

        a_list_of_lists = cls.get_nth_row_as_a_tuple(nth=nth)[3]
        assert isinstance(a_list_of_lists, list)
        assert len(a_list_of_lists) > 0
        list_of_tuples_to_return = []
        for element in a_list_of_lists:
            assert isinstance(element, list)
            assert len(element) == 2
            assert isinstance(element[0], str) and isinstance(element[1], str)
            list_of_tuples_to_return.append((element[0], element[1]))

        return list_of_tuples_to_return

    @classmethod
    def get_maximum_similarity_value_of_nth_row(cls, nth):
        assert cls.column_names[4] == 'maximum_similarity_value'
        a_similarity_value = cls.get_nth_row_as_a_tuple(nth=nth)[4]
        gh.is_number(a_similarity_value)
        return a_similarity_value

    @classmethod
    def delete_the_table_if_exists(cls):
        sql_statement = f"""
            DROP VIEW IF EXISTS {cls.table_name}
        """
        _this_db_instance._execute_sql_statement(sql_statement)

    @classmethod
    def get_number_of_rows_currently_in_db(cls):
        cls.create_or_replace_the_table_ordered_by_similarity_value()

        sql_statement = f"""
            SELECT COUNT(*) FROM {cls.table_name}
        """
        result = _this_db_instance._execute_sql_statement(sql_statement)
        assert len(result) == 1
        assert len(result[0]) == 1
        assert isinstance(result[0][0], int)
        return result[0][0]

    @classmethod
    def get_list_of_paths_to_audios_used_for_matching(cls):
        assert cls.column_names[0] == 'relative_path_to_audio_file'
        cls.create_or_replace_the_table_ordered_by_similarity_value()

        sql_statement = f"""
            SELECT DISTINCT {cls.column_names[0]} 
            FROM {cls.table_name}
        """
        result = _this_db_instance._execute_sql_statement(sql_statement)
        result = list(map(lambda x: x[0], result))
        return result

        # return _this_db_instance.get_list_of_unique_values_of_a_column_in_a_table(a_table_name=cls.table_name, a_column_name=cls.column_names[0])
