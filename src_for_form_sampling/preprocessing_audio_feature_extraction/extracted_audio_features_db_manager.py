import copy
import numpy as np

import my_utils.data_structure_helpers
import my_utils.terminal_interaction_helpers
from my_utils import misc_helpers as gh
from src_for_form_sampling.parameters_package import PrmsContainer
import warnings
from my_utils import my_db_utils


# ======================================================================================================================
# ----------------------------Change This Db Configuration if I use other database--------------------------------------
my_db_utils.INTERVAL_IN_S_BETWEEN_DB_ACCESS_RETRIES = PrmsContainer.interval_in_s_between_db_access_retries
my_db_utils.HOW_MANY_TIMES_TO_RETRY_DB_ACCESS = PrmsContainer.how_many_times_to_retry_db_access

_this_db_instance = my_db_utils.DbOperators(db_type='sqlite', sqlite_db_path=PrmsContainer.path_to_db_that_stores_audio_feature_extraction_results)
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================


class Table_From_AudioHashValue_To_AudioFileRelativePath:
    table_name = "From_AudioHashValue_To_AudioFileRelativePath"  # inspect.stack()[0][3] can get class name
    dict_from_column_names_to_sql_datatypes = {
        'audio_hash_value': 'TEXT',
        'relative_path_to_audio_file': 'TEXT'
    }
    list_of_primary_key_column_names = ['audio_hash_value']

    @classmethod
    def create_the_table_if_not_exists(cls):
        _this_db_instance.create_a_table_if_not_exists(
            table_name=cls.table_name,
            dict_from_column_names_to_sql_datatypes=cls.dict_from_column_names_to_sql_datatypes,
            list_of_primary_key_column_names=cls.list_of_primary_key_column_names
        )

    @classmethod
    def delete_the_table(cls):
        _this_db_instance.delete_a_table(table_name=cls.table_name)

    @classmethod
    def add_a_row(cls, audio_hash_value, relative_path_to_audio):
        warnings.warn(f'Deprecated!! Use {gh.varnameof(cls.add_or_replace_a_row)} for more descriptive function name!!!', DeprecationWarning, stacklevel=2)
        _this_db_instance.add_a_row_to_a_table(
            table_name=cls.table_name,
            row_values_as_dict_from_column_name_to_corresponding_value={'audio_hash_value': audio_hash_value,
                                                                        'relative_path_to_audio_file': relative_path_to_audio},
            action_for_duplicates="REPLACE"
        )

    @classmethod
    def add_or_replace_a_row(cls, audio_hash_value, relative_path_to_audio):
        _this_db_instance.add_a_row_to_a_table(
            table_name=cls.table_name,
            row_values_as_dict_from_column_name_to_corresponding_value={'audio_hash_value': audio_hash_value,
                                                                        'relative_path_to_audio_file': relative_path_to_audio},
            action_for_duplicates="REPLACE"
        )

    @classmethod
    def get_value_from_key(cls, audio_hash_value):
        relative_path_to_audio = _this_db_instance.get_value_from_key(table_name=cls.table_name,
                                                                      primary_key_as_dict_from_column_name_to_corresponding_value={'audio_hash_value': audio_hash_value})
        return relative_path_to_audio

    @classmethod
    def does_a_column_exist(cls, column_name):
        return _this_db_instance.does_a_column_exist_in_a_table(table_name=cls.table_name, column_name=column_name)

    @classmethod
    def get_all_rows_of_the_table_as_a_list_of_row_value_tuples(cls):
        return _this_db_instance.get_all_rows_of_a_table_as_a_list_of_row_value_tuples(table_name=cls.table_name)


class Table_From_ParameterComboHashValue_To_FinalExtractedData:
    table_name = "From_ParameterComboHashValue_To_FinalExtractedData"  # inspect.stack()[0][3] can get class name
    list_of_tuples_of_column_name_and_sql_datatype = [
        ('parameter_combo_hash_value', 'TEXT'),
        ('final_extracted_array', 'TEXT')
    ]
    primary_key_column_names = ['parameter_combo_hash_value']

    @classmethod
    def create_the_table_if_not_exists(cls):
        _this_db_instance.create_a_table_if_not_exists(
            table_name=cls.table_name,
            list_of_tuples_of_column_name_and_sql_datatype=cls.list_of_tuples_of_column_name_and_sql_datatype,
            list_of_primary_key_column_names=cls.primary_key_column_names
        )

    @classmethod
    def add_a_row(cls, a_parameter_combo_hash_value, audio_extracted_data_as_ndarray):
        assert isinstance(a_parameter_combo_hash_value, str)
        gh.assert_class(audio_extracted_data_as_ndarray, np.ndarray)

        _this_db_instance.add_a_row_to_a_table(
            table_name=cls.table_name,
            row_values_as_dict_from_column_name_to_corresponding_value={'parameter_combo_hash_value': a_parameter_combo_hash_value,
                                                                        'final_extracted_array': my_db_utils.serialize_a_value_to_store_in_sql_db(audio_extracted_data_as_ndarray)},
            action_for_duplicates="REPLACE"
        )

    @classmethod
    def does_a_column_exist(cls, column_name):
        return _this_db_instance.does_a_column_exist_in_a_table(table_name=cls.table_name, column_name=column_name)

    @classmethod
    def get_extracted_array_from_parameter_combo_hash_value(cls, parameter_combo_hash_value):
        assert isinstance(parameter_combo_hash_value, str)
        extracted_array_as_str = _this_db_instance.get_value_from_key(table_name=cls.table_name,
                                                                      primary_key_as_dict_from_column_name_to_corresponding_value={'parameter_combo_hash_value': parameter_combo_hash_value})
        extracted_array_as_list = my_db_utils.get_python_value_from_sql_text_str(extracted_array_as_str, list)
        extracted_array_as_ndarray = np.array(extracted_array_as_list)
        return extracted_array_as_ndarray


class Table_From_ParameterComboHashValue_To_ParameterCombo:
    table_name = "From_ParameterComboHashValue_To_ParameterCombo"
    column_name_for_parameter_combo_hash_value = 'parameter_combo_hash_value'

    list_of_tuples_of_default_column_name_and_datatype = [
        ('parameter_combo_hash_value', 'TEXT'),
        ('audio_hash_value', 'TEXT')
    ]
    list_of_primary_key_column_names = ['parameter_combo_hash_value']
    foreign_key_column_names = ['parameter_combo_hash_value', 'audio_hash_value']
    list_of_tuples_of_table_name_and_column_name_referenced_by_foreign_keys = [
        (Table_From_ParameterComboHashValue_To_FinalExtractedData.table_name, 'parameter_combo_hash_value'),
        (Table_From_AudioHashValue_To_AudioFileRelativePath.table_name, 'audio_hash_value')
    ]

    print_create_the_table_warning = True

    @classmethod
    def create_the_table_if_not_exists(cls):
        if cls.print_create_the_table_warning == True:
            my_utils.terminal_interaction_helpers.warninginfo(f"As of Sep 9, 2022, I assume that {gh.varnameof(_this_db_instance.create_a_table_if_not_exists)} in {gh.varnameof(Table_From_ParameterComboHashValue_To_ParameterCombo)}"
                                                              f"is used with {gh.varnameof(cls.change_columns_according_to_the_newest_tree_structure)}. This function"
                                                              f"only creates a table with one column...")
            cls.print_create_the_table_warning = False

        _this_db_instance.create_a_table_if_not_exists(
            table_name=cls.table_name,
            list_of_tuples_of_column_name_and_sql_datatype=cls.list_of_tuples_of_default_column_name_and_datatype,
            list_of_primary_key_column_names=cls.list_of_primary_key_column_names,
            foreign_key_column_names=cls.foreign_key_column_names,
            list_of_tuples_of_table_name_and_column_name_referenced_by_foreign_keys=cls.list_of_tuples_of_table_name_and_column_name_referenced_by_foreign_keys
        )

    @classmethod
    def does_the_table_exist(cls):
        _this_db_instance.does_a_table_exist(table_name=cls.table_name)

    @classmethod
    def add_a_row(cls, parameter_combo_hash_value, parameter_combo_columns_names_and_values_as_dict):
        a_dict = copy.deepcopy(parameter_combo_columns_names_and_values_as_dict)
        a_dict[cls.column_name_for_parameter_combo_hash_value] = parameter_combo_hash_value
        _this_db_instance.add_a_row_to_a_table(table_name=cls.table_name,
                                               row_values_as_dict_from_column_name_to_corresponding_value=a_dict,
                                               action_for_duplicates="REPLACE")

    @classmethod
    def change_columns_according_to_the_newest_tree_structure(cls,
                                                              list_of_newest_tuples_of_column_name_and_previous_column_name_and_datatype):
        gh.assert_class(list_of_newest_tuples_of_column_name_and_previous_column_name_and_datatype, list)
        my_utils.terminal_interaction_helpers.warninginfo(f"The function {gh.varnameof(cls.change_columns_according_to_the_newest_tree_structure)} is NOT supposed to be used frequently. \n"
                                                          f"If you see this message frequently, then something is probably wrong!!")

        list_of_newest_column_names = []
        for a_tuple in list_of_newest_tuples_of_column_name_and_previous_column_name_and_datatype:
            column_name = a_tuple[0]
            list_of_newest_column_names.append(column_name)
            previous_column_name = a_tuple[1]
            datatype = a_tuple[2]

            if not _this_db_instance.does_a_column_exist_in_a_table(table_name=cls.table_name, column_name=column_name):
                if (previous_column_name is not None and
                        _this_db_instance.does_a_column_exist_in_a_table(table_name=cls.table_name, column_name=previous_column_name)):
                    _this_db_instance.change_a_column_name(table_name=cls.table_name,
                                                           old_column_name=previous_column_name,
                                                           new_column_name=column_name)
                else:
                    _this_db_instance.add_a_column_to_a_table(table_name=cls.table_name,
                                                              column_name=column_name,
                                                              column_sql_datatype=datatype)

        old_column_names_plus_newly_added_or_changed_column_names = _this_db_instance.get_list_of_columns_of_a_table(table_name=cls.table_name)
        for a_column_name in old_column_names_plus_newly_added_or_changed_column_names:
            if a_column_name != cls.column_name_for_parameter_combo_hash_value and a_column_name not in list_of_newest_column_names:
                _this_db_instance.delete_a_column_in_a_table(table_name=cls.table_name, column_name=a_column_name)

    @classmethod
    def does_a_column_exist(cls, column_name):
        return _this_db_instance.does_a_column_exist_in_a_table(table_name=cls.table_name, column_name=column_name)

    @classmethod
    def get_a_parameter_combo_dict_from_parameter_combo_hash_value(cls, parameter_combo_hash_value):
        list_of_column_names = _this_db_instance.get_list_of_columns_of_a_table(table_name=cls.table_name)
        list_of_column_names.remove('parameter_combo_hash_value')

        sql_statement = f"""
            SELECT {my_db_utils.convert_list_to_comma_separated_str(list_of_column_names)}
            FROM {cls.table_name}
            WHERE "parameter_combo_hash_value" = "{parameter_combo_hash_value}"
        """

        tuple_of_parameters = _this_db_instance._execute_sql_statement(sql_statement)[0]

        assert isinstance(list_of_column_names, list)
        assert isinstance(tuple_of_parameters, tuple)
        assert len(list_of_column_names) == len(tuple_of_parameters)

        dict_to_return = {}
        for i in range(len(list_of_column_names)):
            dict_to_return[list_of_column_names[i]] = tuple_of_parameters[i]

        return dict_to_return


class AudioDataViewTable_HumanComprehensible():
    table_name = "AudioDataViewTable_HumanComprehensible"  # inspect.stack()[0][3] can get class name
    dict_from_column_names_to_sql_datatypes = {}
    '''
    SQLite view is read only. It means you cannot use INSERT, DELETE, and  UPDATE statements to update data in the base tables through the view.
    '''
    inner_join_column_name1 = 'parameter_combo_hash_value'
    inner_join_column_name2 = 'audio_hash_value'
    _already_made_sure_inner_join_is_possible = False

    list_of_column_names = []
    most_left_column_name = None

    @classmethod
    def create_or_replace_the_table(cls):
        if cls._already_made_sure_inner_join_is_possible == False:
            cls._assert_inner_join_is_possible()
            cls._already_made_sure_inner_join_is_possible = True

        cls._set_up_list_of_column_names()
        cls.delete_the_table_if_exists()

        sql_statement = f"""    
            CREATE VIEW {cls.table_name} AS
                SELECT {my_db_utils.convert_list_to_comma_separated_str(cls.list_of_column_names)}
                FROM {Table_From_ParameterComboHashValue_To_ParameterCombo.table_name}
                INNER JOIN {Table_From_ParameterComboHashValue_To_FinalExtractedData.table_name} 
                    ON {Table_From_ParameterComboHashValue_To_ParameterCombo.table_name}.{cls.inner_join_column_name1} = {Table_From_ParameterComboHashValue_To_FinalExtractedData.table_name}.{cls.inner_join_column_name1}
                INNER JOIN {Table_From_AudioHashValue_To_AudioFileRelativePath.table_name}
                    ON {Table_From_ParameterComboHashValue_To_ParameterCombo.table_name}.{cls.inner_join_column_name2} = {Table_From_AudioHashValue_To_AudioFileRelativePath.table_name}.{cls.inner_join_column_name2}
                ORDER BY 
                    {cls.most_left_column_name} ASC;
        """

        _this_db_instance._execute_sql_statement(sql_statement)
        # _this_db_instance.delete_a_column_in_a_table(table_name=cls.table_name, column_name=cls.inner_join_column_name1)
        # _this_db_instance.delete_a_column_in_a_table(table_name=cls.table_name, column_name=cls.inner_join_column_name2)
        # TODO: There may be a way to put these "Drop Table" statements into the sql statement when I create the table

    @classmethod
    def _assert_inner_join_is_possible(cls):
        assert Table_From_ParameterComboHashValue_To_ParameterCombo.does_a_column_exist(cls.inner_join_column_name1)
        assert Table_From_ParameterComboHashValue_To_FinalExtractedData.does_a_column_exist(cls.inner_join_column_name1)
        assert Table_From_ParameterComboHashValue_To_ParameterCombo.does_a_column_exist(cls.inner_join_column_name2)
        assert Table_From_AudioHashValue_To_AudioFileRelativePath.does_a_column_exist(cls.inner_join_column_name2)

    @classmethod
    def _set_up_list_of_column_names(cls):
        cls.list_of_column_names += _this_db_instance.get_list_of_columns_of_a_table(table_name=Table_From_ParameterComboHashValue_To_ParameterCombo.table_name)
        cls.list_of_column_names += _this_db_instance.get_list_of_columns_of_a_table(
            table_name=Table_From_ParameterComboHashValue_To_FinalExtractedData.table_name)
        cls.list_of_column_names = list(filter(lambda a: a != 'audio_hash_value', cls.list_of_column_names))
        cls.list_of_column_names = list(filter(lambda a: a != 'parameter_combo_hash_value', cls.list_of_column_names))
        cls.most_left_column_name = _this_db_instance.get_list_of_columns_of_a_table(
            table_name=Table_From_AudioHashValue_To_AudioFileRelativePath.table_name)[1]
        assert cls.most_left_column_name in _this_db_instance.get_list_of_columns_of_a_table(
            table_name=Table_From_AudioHashValue_To_AudioFileRelativePath.table_name)
        cls.list_of_column_names.insert(0, cls.most_left_column_name)

    '''
    @classmethod
    def delete_a_column (cls, column_name):
        raise Exception("This Method Somehow Does Not Work for View Table!!!!!!!!!!!!!!")
        _this_db_instance.delete_a_column_in_a_table(table_name=cls.table_name, column_name=column_name)
    '''
    @classmethod
    def get_list_of_columns(cls):
        return _this_db_instance.get_list_of_columns_of_a_table(table_name=cls.table_name)

    @classmethod
    def delete_the_table_if_exists(cls):
        sql_statement = f"""
            DROP VIEW IF EXISTS {cls.table_name}
        """
        _this_db_instance._execute_sql_statement(sql_statement)


class AudioExtractedData_View_For_SimilarityMatrixCalculation:
    table_name = "AudioExtractedData_View_For_SimilarityMatrixCalculation"
    column_names = ['audio_hash_value', 'time_point_tuple', 'parameter_combo_hash_value', 'final_extracted_array']
    inner_join_column_name1 = 'parameter_combo_hash_value'
    assert inner_join_column_name1 in column_names
    assert column_names[0] == "audio_hash_value"
    assert column_names[1] == "time_point_tuple"
    assert column_names[2] == "parameter_combo_hash_value"

    _already_made_sure_inner_join_is_possible = False

    @classmethod
    def create_or_replace_the_table(cls, list_of_audio_hash_values_to_include=None):
        if cls._already_made_sure_inner_join_is_possible == False:
            cls._assert_inner_join_is_possible()
            cls._already_made_sure_inner_join_is_possible = True

        if _this_db_instance.does_a_table_exist(cls.table_name):
            return

        # cls.delete_the_table_if_exists()

        sql_statement = f"""
            CREATE VIEW IF NOT EXISTS {cls.table_name} AS
                SELECT 
                    {cls.column_names[0]}, 
                    {cls.column_names[1]}, 
                    {Table_From_ParameterComboHashValue_To_FinalExtractedData.table_name}.{cls.column_names[2]},
                    {cls.column_names[3]}
                FROM {Table_From_ParameterComboHashValue_To_ParameterCombo.table_name}
                INNER JOIN {Table_From_ParameterComboHashValue_To_FinalExtractedData.table_name}
                    ON {Table_From_ParameterComboHashValue_To_FinalExtractedData.table_name}.{cls.inner_join_column_name1}
                    = {Table_From_ParameterComboHashValue_To_ParameterCombo.table_name}.{cls.inner_join_column_name1}  
        """
        #       WHERE {cls.column_names[0]} IN ({my_db_utils.convert_list_to_comma_separated_str_with_single_quotes(list_of_audio_hash_values_to_include)})

        _this_db_instance._execute_sql_statement(sql_statement)

    @classmethod
    def _assert_inner_join_is_possible(cls):
        assert Table_From_ParameterComboHashValue_To_ParameterCombo.does_a_column_exist(cls.column_names[0])
        assert Table_From_ParameterComboHashValue_To_ParameterCombo.does_a_column_exist(cls.column_names[1])
        assert Table_From_ParameterComboHashValue_To_ParameterCombo.does_a_column_exist(cls.column_names[2])

        assert Table_From_ParameterComboHashValue_To_FinalExtractedData.does_a_column_exist(cls.column_names[2])
        assert Table_From_ParameterComboHashValue_To_FinalExtractedData.does_a_column_exist(cls.column_names[3])

    @classmethod
    def delete_the_table_if_exists(cls):
        sql_statement = f"""
            DROP VIEW IF EXISTS {cls.table_name}
        """
        _this_db_instance._execute_sql_statement(sql_statement)

    @classmethod
    def get_list_of_tuples_of_audio_hash_value_and_time_point_tuple(cls):
        sql_statement = f"""
            SELECT DISTINCT {cls.column_names[0]}, {cls.column_names[1]} 
            FROM {cls.table_name}
        """

        list_of_tuples_of_audio_hash_value_str_and_time_point_tuple_str = _this_db_instance._execute_sql_statement(sql_statement)

        list_of_tuples_of_audio_hash_value_str_and_time_point_tuple = []
        for a_tuple in list_of_tuples_of_audio_hash_value_str_and_time_point_tuple_str:
            a_new_tuple = (a_tuple[0], my_db_utils.get_python_value_from_sql_text_str(a_tuple[1], tuple))
            list_of_tuples_of_audio_hash_value_str_and_time_point_tuple.append(a_new_tuple)

        return list_of_tuples_of_audio_hash_value_str_and_time_point_tuple

    @classmethod
    def get_list_of_tuples_of_parameter_combo_hash_values_and_corresponding_array_for_an_audio_segment(cls,
                                                                                                       audio_hash_value,
                                                                                                       time_point_tuple):
        sql_statement = f"""
            SELECT {cls.column_names[2]}, {cls.column_names[3]} 
            FROM {cls.table_name}
            WHERE {cls.column_names[0]} = "{audio_hash_value}" AND {cls.column_names[1]} = "{my_db_utils.serialize_a_value_to_store_in_sql_db(time_point_tuple)}"
        """

        list_of_tuples = _this_db_instance._execute_sql_statement(sql_statement)

        assert isinstance(list_of_tuples, list)
        my_utils.data_structure_helpers.assert_all_elements_in_list_are_tuple(list_of_tuples)
        my_utils.data_structure_helpers.assert_lengths_of_all_tuples_in_list_are_the_same(list_of_tuples)
        assert len(list_of_tuples[0]) == 2
        assert isinstance(list_of_tuples[0][0], str) and isinstance(list_of_tuples[0][1], str)

        list_to_return = []
        for a_tuple in list_of_tuples:
            a_new_tuple = (a_tuple[0], my_db_utils.get_python_value_from_sql_text_str(a_tuple[1], np.ndarray))
            list_to_return.append(a_new_tuple)

        assert isinstance(list_to_return[0][0], str) and isinstance(list_to_return[0][1], np.ndarray)

        return list_to_return
