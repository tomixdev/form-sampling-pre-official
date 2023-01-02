import sqlite3
import numpy as np

from . import data_structure_helpers
from . import terminal_interaction_helpers
from . import misc_helpers as mh
import inspect
import ast
import json
import warnings
import retry

POSSIBLE_SQL_DATATYPES = ['NULL', 'INTEGER', 'REAL', 'TEXT', 'BLOB']
POSSIBLE_ACTIONS_FOR_DUPLICATES = ['ABORT', 'FAIL', 'IGNORE', 'REPLACE', 'ROLLBACK']
HOW_MANY_TIMES_TO_RETRY_DB_ACCESS = 100
INTERVAL_IN_S_BETWEEN_DB_ACCESS_RETRIES = 0.01


# ===================================================================================================================
# ==================================Db Manager Related Utility Functions ============================================
# -------------------------------------------------------------------------------------------------------------------
def get_sql_datatype_for_a_python_value(a_python_value):
    # TODO: For now (Sep 4, 2022), I assume sqlite3 datatypes. In the future, I can change this functino
    assert a_python_value is not None

    if mh.is_number(a_python_value):
        return "REAL"
    else:
        return "TEXT"


def is_a_string_valid_sql_datatype(a_string):
    warnings.warn("The meaning of this function's name is not very clear...", DeprecationWarning, stacklevel=2)
    if a_string in POSSIBLE_SQL_DATATYPES:
        return True
    else:
        return False


def serialize_a_value_to_store_in_sql_db(a_value):
    if isinstance(a_value, dict):
        return json.dumps(a_value)
    elif isinstance(a_value, list):
        return json.dumps(a_value)
    elif isinstance(a_value, tuple):
        assert len(a_value) >= 2, "if a tuple length is 1, then str(a_tuple) will stringfy a_tuple in an awkward, nonintuitive way in python. So, I need to provide my own method to stringfy a tuple"
        return str(a_value)
    elif isinstance(a_value, bool):
        return str(a_value)
    elif isinstance(a_value, np.ndarray):
        return json.dumps(a_value.tolist())
    elif isinstance(a_value, str):
        return a_value
    elif mh.is_number(a_value):
        if isinstance(a_value, int):
            return a_value
        elif isinstance(a_value, float):
            return a_value
        elif isinstance(a_value, np.int64):
            return int(a_value)
        elif isinstance(a_value, np.float64):
            return float(a_value)
        elif isinstance(a_value, np.bool_):
            return bool(a_value)
        elif isinstance(a_value, np.float32):
            return float(a_value)
        elif isinstance(a_value, np.int32):
            return int(a_value)
        elif isinstance(a_value, np.float16):
            return float(a_value)
        elif isinstance(a_value, np.int16):
            return int(a_value)
        elif isinstance(a_value, np.uint64):
            return int(a_value)
        elif isinstance(a_value, np.uint32):
            return int(a_value)
        elif isinstance(a_value, np.uint16):
            return int(a_value)
        elif isinstance(a_value, np.uint8):
            return int(a_value)
        elif isinstance(a_value, np.int8):
            return int(a_value)
        else:
            raise Exception(f"{a_value.__class__} is not serializable to sql database.")
    else:
        raise Exception('Not Defined')


def get_python_value_from_sql_text_str(sql_text_str, python_class):
    if python_class == dict:
        a_dict = json.loads(sql_text_str)
        assert isinstance(a_dict, dict)
        return a_dict
    elif python_class == list:
        a_list = json.loads(sql_text_str)
        assert isinstance(a_list, list)
        return a_list
    elif python_class == tuple:
        a_tuple = ast.literal_eval(sql_text_str)
        assert isinstance(a_tuple, tuple)
        return a_tuple
    elif python_class == bool:
        if sql_text_str == 'True':
            return True
        elif sql_text_str == "False":
            return False
        else:
            raise Exception('sql string is not a boolean type')
    elif python_class == np.ndarray:
        a_list = json.loads(sql_text_str)
        assert isinstance(a_list, list)
        return np.array(a_list)
    elif python_class == str:
        return sql_text_str
    else:
        try:
            a_val = json.loads(sql_text_str)
            if mh.is_number(a_val):
                return a_val
            else:
                raise Exception('Not Defined')
        except:
            raise Exception('Not Defined')


def convert_list_to_comma_separated_str(a_list):
    ''''''
    '''if a list has only one element, this function retuns that one element as string
       For example, if the list is [3], then the function returns '3' as string    '''
    mh.assert_class(a_list, list)
    return ','.join(map(str, a_list))


def convert_list_to_comma_separated_str_with_single_quotes(a_list):
    mh.assert_class(a_list, list)
    return str(a_list)[1:-1]


'''
########################################################################################################################
##########################################  Class DbManager ############################################################
########################################################################################################################
'''


class DbOperators:
    def __init__(self, db_type='sqlite', sqlite_db_path=None):
        self.db_type = db_type
        self.sqlite_db_path = sqlite_db_path

    @retry.retry(Exception, tries=HOW_MANY_TIMES_TO_RETRY_DB_ACCESS, delay=INTERVAL_IN_S_BETWEEN_DB_ACCESS_RETRIES)
    def _connect_to_db_and_return_connection_and_cursor(self):
        if self.db_type == 'sqlite':
            assert self.sqlite_db_path is not None
            connection = sqlite3.connect(self.sqlite_db_path)
            cursor = connection.cursor()
            return connection, cursor
        elif self.db_type == 'mysql':
            raise NotImplementedError('Not Implemented Yet')
        elif self.db_type == 'postgresql':
            raise NotImplementedError('Not Implemented Yet')
        elif self.db_type == 'mssql':
            raise NotImplementedError('Not Implemented Yet')
        elif self.db_type == 'oracle':
            raise NotImplementedError('Not Implemented Yet')
        elif self.db_type == 'firebird':
            raise NotImplementedError('Not Implemented Yet')
        else:
            raise NotImplementedError("Not implemented yet")

    @retry.retry(Exception, tries=HOW_MANY_TIMES_TO_RETRY_DB_ACCESS, delay=INTERVAL_IN_S_BETWEEN_DB_ACCESS_RETRIES)
    def _finish_db_operation_and_close_connection_to_db(self, connection, cursor):
        if connection:
            connection.commit()
            cursor.close()
            connection.close()
        else:
            raise Exception('Something is Wrong!!!, cannot close connection....')

    @retry.retry(Exception, tries=HOW_MANY_TIMES_TO_RETRY_DB_ACCESS, delay=INTERVAL_IN_S_BETWEEN_DB_ACCESS_RETRIES)
    def _execute_sql_statement(self, sql_statement):
        connection, cursor = self._connect_to_db_and_return_connection_and_cursor()
        result = cursor.execute(sql_statement).fetchall()
        # 上のコードは、sql_statementを実行することと、その結果を得ることの２つの意味が込められているので、debugのときは注意する
        self._finish_db_operation_and_close_connection_to_db(connection, cursor)
        return result

    already_printed_create_a_table_warning = False

    def create_a_table_if_not_exists(self,
                                     table_name,
                                     dict_from_column_names_to_sql_datatypes=None,
                                     list_of_tuples_of_column_name_and_sql_datatype=None,
                                     list_of_primary_key_column_names=None,
                                     foreign_key_column_names=None,
                                     list_of_tuples_of_table_name_and_column_name_referenced_by_foreign_keys=None):

        if self.already_printed_create_a_table_warning == False:
            terminal_interaction_helpers.warninginfo(f"BE CAREFUL about difference between {mh.varnameof(self.create_a_table_if_not_exists)} and 'Table_From_AudioHashValue_To_AudioFileRelativePath.create_the_table_if_not_exists'. \n"
                                                     f" {mh.varnameof(self.create_a_table_if_not_exists)} is meant to be read by inheriting children classes."
                                                     f" Be careful if I use this method directly without going through other classes in db_manager.py")
            # TODO: 上のwarning messageにおける'AudioDataViewTable_HumanComprehensible' は db_utils.pyには存在するべきではない。
            #  このpythonファイルは、いろいろなProjectにおいて使用可能な、ファイルであり、feel the field projectにspecificなものではない。
            self.already_printed_create_a_table_warning = True

        # Format Arguments, Ensure they are correct----------------------------------------------------------------------
        if dict_from_column_names_to_sql_datatypes is not None:
            mh.assert_class(dict_from_column_names_to_sql_datatypes, dict)
        if list_of_tuples_of_column_name_and_sql_datatype is not None:
            mh.assert_class(list_of_tuples_of_column_name_and_sql_datatype, list)
        if list_of_primary_key_column_names is not None:
            mh.assert_class(list_of_primary_key_column_names, list)

        if dict_from_column_names_to_sql_datatypes is None and list_of_tuples_of_column_name_and_sql_datatype is None:
            raise Exception(
                f'Either {mh.varnameof(dict_from_column_names_to_sql_datatypes)} or {mh.varnameof(list_of_tuples_of_column_name_and_sql_datatype)} must be specifdied!!')
        if list_of_tuples_of_column_name_and_sql_datatype is None:
            assert dict_from_column_names_to_sql_datatypes is not None
            list_of_tuples_of_column_name_and_sql_datatype = []
            for a_column_name, a_sql_datatype in dict_from_column_names_to_sql_datatypes.items():
                list_of_tuples_of_column_name_and_sql_datatype.append((a_column_name, a_sql_datatype))

        list_of_column_names = [a_tuple[0] for a_tuple in list_of_tuples_of_column_name_and_sql_datatype]
        for a_primary_key_column_name in list_of_primary_key_column_names:
            assert a_primary_key_column_name in list_of_column_names, \
                f"a primary key is not in the list of column names.\n" \
                f"{mh.varnameof(list_of_column_names)} = {list_of_column_names} \n" \
                f"{mh.varnameof(list_of_primary_key_column_names)} = {list_of_primary_key_column_names}"

        # Create SQL Statement-------------------------------------------------------------------------------------------
        sql_statement = ''
        sql_statement += f"CREATE TABLE IF NOT EXISTS {table_name}"
        sql_statement += "( \n"
        for a_column_name, a_sql_datatype in list_of_tuples_of_column_name_and_sql_datatype:
            sql_statement += f"{a_column_name} {a_sql_datatype} ,\n"

        if list_of_primary_key_column_names is not None:
            data_structure_helpers.assert_list1_contains_all_elements_of_list2(list1=list_of_column_names, list2=list_of_primary_key_column_names)
            sql_statement += f"PRIMARY KEY ({convert_list_to_comma_separated_str(list_of_primary_key_column_names)})"

        if foreign_key_column_names is not None:
            assert list_of_tuples_of_table_name_and_column_name_referenced_by_foreign_keys is not None
            mh.assert_class(foreign_key_column_names, list)
            mh.assert_class(list_of_tuples_of_table_name_and_column_name_referenced_by_foreign_keys, list)
            data_structure_helpers.assert_list1_contains_all_elements_of_list2(list1=list_of_column_names, list2=foreign_key_column_names)
            assert len(foreign_key_column_names) == len(list_of_tuples_of_table_name_and_column_name_referenced_by_foreign_keys)

            sql_statement += ","
            for i in range(0, len(foreign_key_column_names)):
                mh.assert_class(list_of_tuples_of_table_name_and_column_name_referenced_by_foreign_keys[i], tuple)
                sql_statement += f" FOREIGN KEY ({foreign_key_column_names[i]}) REFERENCES" \
                                 f" {list_of_tuples_of_table_name_and_column_name_referenced_by_foreign_keys[i][0]}" \
                                 f"({list_of_tuples_of_table_name_and_column_name_referenced_by_foreign_keys[i][1]}) \n"

        sql_statement += ")"

        self._execute_sql_statement(sql_statement)

        '''
        is_table_created = False
        for a_tuple in list_of_tuples_of_column_name_and_sql_datatype:
            a_column_name = a_tuple[0]
            a_sql_datatype = a_tuple[1]
            assert is_a_string_valid_sql_datatype(a_sql_datatype)
            if is_table_created == False:
                sql_statement = f"CREATE TABLE {table_name} ({a_column_name} {a_sql_datatype})"
                cursor.execute(sql_statement)
                is_table_created = True
            else:
                sql_statement = f"ALTER TABLE {table_name} ADD {a_column_name} {a_sql_datatype}"
                cursor.execute(sql_statement)

        cls._finish_db_operation_and_close_connection_to_db(connection, cursor)
        '''

    def delete_a_table(self, table_name):
        terminal_interaction_helpers.confirm_dangerous_operation_with_kboard_input_or_exit_from_sys(f"Are you sure to delete the table {table_name}?")
        sql_statement = f"DROP TABLE {table_name}"
        self._execute_sql_statement(sql_statement)

    def does_a_row_exist_in_a_table(self, table_name, row_values_as_dict_from_primary_key_column_name_to_corresponding_value):
        assert self.does_a_table_exist(table_name=table_name), table_name

        data_structure_helpers.assert_list1_contains_all_elements_of_list2(
            list1=self.get_list_of_columns_of_a_table(table_name=table_name),
            list2=list(row_values_as_dict_from_primary_key_column_name_to_corresponding_value.keys())
        )

        sql_statement = ""
        sql_statement += "SELECT EXISTS \n"
        sql_statement += f"  (SELECT * FROM {table_name} \n"
        sql_statement += "    WHERE "
        condition_sql_statement = ""
        dict_element_count = 0
        for a_column_name, a_corresponding_value in row_values_as_dict_from_primary_key_column_name_to_corresponding_value.items():
            if dict_element_count >= 1:
                condition_sql_statement += "AND "
            condition_sql_statement += str(a_column_name) + " = '" + str(a_corresponding_value) + "' "
            dict_element_count += 1
        sql_statement += condition_sql_statement
        sql_statement += ")\n"

        sql_execution_result = self._execute_sql_statement(sql_statement)
        assert isinstance(sql_execution_result, list) and len(sql_execution_result) == 1
        assert len(sql_execution_result[0]) == 1 and (sql_execution_result[0][0] == 0) or (sql_execution_result[0][0] == 1)
        return sql_execution_result[0][0]

    def add_a_row_to_a_table(self,
                             table_name,
                             row_values_as_dict_from_column_name_to_corresponding_value=None,
                             list_of_column_names_for_new_values=None,
                             list_of_new_values=None,
                             action_for_duplicates="REPLACE"  # TODO: "IGNORE" might be better
                             ):
        # TODO: This implementation opens and closes connection/cursor at least two times!!!!
        #  This is for code clarity. But there should be a better implementation! Maybe, write a logic purely within SQL?
        assert isinstance(table_name, str)
        assert isinstance(action_for_duplicates, str)
        assert action_for_duplicates in POSSIBLE_ACTIONS_FOR_DUPLICATES

        assert row_values_as_dict_from_column_name_to_corresponding_value is not None or \
            (list_of_column_names_for_new_values is not None and list_of_new_values is not None)

        column_names_of_the_table = self.get_list_of_columns_of_a_table(table_name=table_name)
        if row_values_as_dict_from_column_name_to_corresponding_value is not None:
            mh.assert_class(row_values_as_dict_from_column_name_to_corresponding_value, dict)
            for a_column_name_of_the_table in column_names_of_the_table:
                assert a_column_name_of_the_table in row_values_as_dict_from_column_name_to_corresponding_value.keys(), \
                    'Perhaps, column name is misspelled????'
        else:  # row_values_as_dict_from_column_name_to_corresponding_value is None
            mh.assert_class(list_of_column_names_for_new_values, list)
            mh.assert_class(list_of_new_values, list)
            for a_column_name_of_the_table in column_names_of_the_table:
                assert a_column_name_of_the_table in list_of_column_names_for_new_values, "Perhaps, column name misspelled????"

        if list_of_column_names_for_new_values is None:
            assert list_of_new_values is None
            list_of_column_names_for_new_values = []
            list_of_new_values = []
            for a_column_name, a_value in row_values_as_dict_from_column_name_to_corresponding_value.items():
                list_of_column_names_for_new_values.append(a_column_name)
                a_value = serialize_a_value_to_store_in_sql_db(a_value=a_value)
                list_of_new_values.append(a_value)

        list_of_new_values_serialized = [serialize_a_value_to_store_in_sql_db(a_value=a_value) for a_value in list_of_new_values]

        sql_statement = ""
        sql_statement += f"INSERT OR {action_for_duplicates} " \
            f"INTO {table_name} ( {convert_list_to_comma_separated_str(list_of_column_names_for_new_values)}) \n"
        sql_statement += f"VALUES ( {convert_list_to_comma_separated_str_with_single_quotes(list_of_new_values_serialized)});"

        self._execute_sql_statement(sql_statement)

    @retry.retry(Exception, tries=HOW_MANY_TIMES_TO_RETRY_DB_ACCESS, delay=INTERVAL_IN_S_BETWEEN_DB_ACCESS_RETRIES)
    def add_multiple_rows_to_a_table(self, table_name, list_of_new_row_tuples, action_for_duplicates="REPLACE"):
        assert isinstance(table_name, str)
        assert isinstance(action_for_duplicates, str)
        assert action_for_duplicates in POSSIBLE_ACTIONS_FOR_DUPLICATES

        assert isinstance(list_of_new_row_tuples, list)
        assert len(list_of_new_row_tuples) > 0
        data_structure_helpers.assert_all_elements_in_list_are_tuple(list_of_new_row_tuples)
        data_structure_helpers.assert_lengths_of_all_tuples_in_list_are_the_same(list_of_new_row_tuples)
        assert len(list_of_new_row_tuples[0]) == len(self.get_list_of_columns_of_a_table(table_name=table_name))

        # たとえば各row を示すlistの中に、tupleがあった場合、そのtupleをsqlに保存できる形でSerializeしてから出ないと、executemanyを使えない。
        # たとえば、('a_string', (0, 30))というのをinsertしようと思ったら、(0, 30)を先にSerializeしないといけない。
        list_of_new_row_tuples_serialized = []
        for a_tuple in list_of_new_row_tuples:
            list_of_new_row_tuples_serialized.append(tuple([serialize_a_value_to_store_in_sql_db(a_value=a_value) for a_value in a_tuple]))
        '''
        for a_new_row_tuple in list_of_new_row_tuples:
            cls.add_a_row_to_a_table(
                table_name=table_name,
                list_of_column_names_for_new_values=cls.get_list_of_columns_of_a_table(table_name=table_name),
                list_of_new_values=a_new_row_tuple
        )
        '''
        connection, cursor = self._connect_to_db_and_return_connection_and_cursor()

        # insert many elements (execute many)
        sql_statement = ''
        sql_statement += f"INSERT OR {action_for_duplicates} INTO {table_name} VALUES (?"
        for i in range(0, len(list_of_new_row_tuples_serialized[0])-1):
            sql_statement += ",?"
        sql_statement += ")"

        cursor.executemany(sql_statement, list_of_new_row_tuples_serialized)
        '''
        SQL statement and sqlite cursor code should look like below:
        cursor.executemany("INSERT INTO customers4 VALUES (?,?,?)", many_elements)
        '''
        # cursor.close()
        self._finish_db_operation_and_close_connection_to_db(connection, cursor)

    def add_a_column_to_a_table(self, table_name, column_name, column_sql_datatype, is_part_of_primary_key=False):
        sql_statement = f"""
            ALTER TABLE {table_name}
            ADD {column_name} {column_sql_datatype}
        """
        self._execute_sql_statement(sql_statement)

    def delete_a_column_in_a_table(self, table_name, column_name):
        terminal_interaction_helpers.confirm_dangerous_operation_with_kboard_input_or_exit_from_sys(
            f"The program is trying to delete the column '{column_name}' from '{table_name}' in {self.sqlite_db_path}. "
            f"This means that the program might be deleting a column name that already has data. "
            f"If table name is 'AudioDataViewTable_HumanComprehensible', this problem might be fixed by setting 'attribute_alias19482' variable in TreeClassCommon Class in audio_analysis_tree_classes.py"
            f" Are you sure to delete {column_name}?"
        )
        # TODO: 上のwarning messageにおける'AudioDataViewTable_HumanComprehensible' は db_utils.pyには存在するべきではない。
        #  このpythonファイルは、いろいろなProjectにおいて使用可能な、ファイルであり、feel the field projectにspecificなものではない。

        sql_statement = f"""
             ALTER TABLE {table_name}
             DROP COLUMN {column_name}
         """
        self._execute_sql_statement(sql_statement)

    def does_a_column_exist_in_a_table(self, table_name, column_name):
        # TODO: There should be a sql statement to check this, rather than getting all column names and cheking whether a column exists from application program side
        # TODO: This Function can easily beconme a bottleneck...This is a bad implementation
        assert self.does_a_table_exist(table_name=table_name), table_name

        column_names = self.get_list_of_columns_of_a_table(table_name=table_name)

        if column_name in column_names:
            return True
        else:
            return False

        '''
        connection, cursor = cls._connect_to_db_and_return_connection_and_cursor()
        sql_statement = f"""
            SELECT {column_name} FROM {table_name}
        """
        try_result = None
        try:
            cursor.execute(sql_statement)
            try_result = 'success'
        except:
            try_result = 'failure'

        if try_result == 'success':
            return True
        else:
            return False

        '''
        '''
        sql_statement = f"""
            SELECT COUNT(*) 
            AS CNTREC FROM pragma_table_info({cls.table_name}) 
            WHERE name={column_name}
        """ これはうまくいかない。。。。。
        result = cls._execute_sql_statement(sql_statement)
        print (result)
        raise Exception('not implemented yet')
        '''

    def change_a_column_name(self, table_name, old_column_name, new_column_name):
        # SQLite added support for renaming column since version 3.25.0
        sql_statement = f"""
            ALTER TABLE {table_name}
            RENAME COLUMN {old_column_name} TO {new_column_name};
        """
        self._execute_sql_statement(sql_statement)

    def does_a_table_exist(self, table_name):
        """
        Loosely tested on 20220905
        """

        assert isinstance(table_name, str)

        sql_statement = f"""
               SELECT EXISTS(
                   SELECT name 
                   FROM sqlite_master 
                   WHERE type='table' AND name='{table_name}'
                   );
           """
        result = self._execute_sql_statement(sql_statement)

        return bool(result[0][0])  # 1 or 0

    def get_value_from_key(self, table_name, primary_key_as_dict_from_column_name_to_corresponding_value):
        mh.assert_class(table_name, str)
        mh.assert_class(primary_key_as_dict_from_column_name_to_corresponding_value, dict)

        condition_sql_statement = ""
        column_names_for_key = []
        dict_element_count = 0
        for a_column_name, a_corresponding_value in primary_key_as_dict_from_column_name_to_corresponding_value.items():
            if dict_element_count >= 1:
                condition_sql_statement += " AND "
            condition_sql_statement += str(a_column_name) + " = '" + str(a_corresponding_value) + "' "
            dict_element_count += 1
            column_names_for_key.append(a_column_name)

        all_column_names = self.get_list_of_columns_of_a_table(table_name=table_name)
        assert len(all_column_names) - 1 == len(column_names_for_key), "!! The number of keys has to be exactly 1 less than the number of all column names"
        column_name_for_value = list(set(all_column_names) - set(column_names_for_key))[0]

        sql_statement = f"""
            SELECT {column_name_for_value}
            FROM {table_name}
            WHERE {condition_sql_statement}
        """
        sql_execution_result = self._execute_sql_statement(sql_statement)
        assert isinstance(sql_execution_result, list)
        if len(sql_execution_result) != 1:
            terminal_interaction_helpers.confirm_dangerous_operation_with_kboard_input_or_exit_from_sys(f"Function '{inspect.stack()[0][3]}' returned more than one value for the key."
                                                                                                        f"This is most likely an error. \n"
                                                                                                        f"{mh.varnameof(table_name)} =  {table_name}, {mh.varnameof(primary_key_as_dict_from_column_name_to_corresponding_value)} = {primary_key_as_dict_from_column_name_to_corresponding_value} \n"
                                                                                                        f"Results from sql execution: {sql_execution_result} \n"
                                                                                                        f"Are you sure to go ahead?")
        assert len(sql_execution_result[0]) == 1

        return sql_execution_result[0][0]


    def get_list_of_table_names_in_db(self):
        '''Loosely Tested on 20220907'''

        sql_statement = f"""
            SELECT name FROM sqlite_master where type='table';
        """
        result = self._execute_sql_statement(sql_statement)
        return list(result[0])

    @retry.retry(Exception, tries=HOW_MANY_TIMES_TO_RETRY_DB_ACCESS, delay=INTERVAL_IN_S_BETWEEN_DB_ACCESS_RETRIES)
    def get_list_of_columns_of_a_table(self, table_name):
        connection, cursor = self._connect_to_db_and_return_connection_and_cursor()
        cursor = connection.execute(f'SELECT * FROM {table_name}')
        column_names = list(map(lambda x: x[0], cursor.description))
        self._finish_db_operation_and_close_connection_to_db(connection, cursor)
        return column_names



    def change_a_table_name(self, old_table_name, new_table_name):
        '''Not Tested'''
        mh.assert_class(old_table_name, str)
        mh.assert_class(new_table_name, str)
        assert self.does_a_table_exist(old_table_name), old_table_name

        sql_statement = f"""
            ALTER TABLE {old_table_name}
            RENAME TO {new_table_name}
        """
        self._execute_sql_statement(sql_statement)

    def get_all_rows_of_a_table_as_a_list_of_row_value_tuples(self, table_name):
        mh.assert_class(table_name, str)
        assert self.does_a_table_exist(table_name), table_name

        sql_statement = f"""
            SELECT * FROM {table_name}
        """

        result = self._execute_sql_statement(sql_statement)

        assert isinstance(result, list)
        assert isinstance(result[0], tuple)
        assert len(result[0]) == len(self.get_list_of_columns_of_a_table(table_name))

        return result

    def get_first_x_rows_from_a_table_as_list_of_tuples_of_strings(self, table_name, x):
        assert isinstance(table_name, str)
        assert self.does_a_table_exist(table_name), table_name
        assert isinstance(x, int)
        assert x >= 1
        '''
        TODO: Not really checked if this works or not
        '''
        sql_statement = f"""
            SELECT * FROM {table_name} LIMIT {x}
        """
        result = self._execute_sql_statement(sql_statement)

        assert isinstance(result, list)
        assert len(result) == x
        assert isinstance(result[0], tuple)
        assert len(result[0]) == len(self.get_list_of_columns_of_a_table(table_name))

        return result

    def get_nth_row_from_a_table_as_a_tuple_of_strings(self, table_name, nth):
        # assert cls.does_a_table_exist(table_name) TODO: 本当はこのAssertion入れたいけど、table_nameがview tableだった場合、このfunctionは常にFalseを返すので、assertion errorになる. how to check whethe a view table exists????
        assert isinstance(nth, int)
        assert nth >= 1, "nth counts from 1, not 0!!"

        sql_statement = f"""
            SELECT * 
            FROM {table_name} 
            LIMIT 1 OFFSET {nth-1}
        """

        result = self._execute_sql_statement(sql_statement)

        assert len(result) == 1
        assert isinstance(result[0], tuple)
        assert len(result[0]) == len(self.get_list_of_columns_of_a_table(table_name))

        return result[0]

    def get_number_of_rows_currently_in_a_table(self, table_name):
        assert isinstance(table_name, str)
        assert self.does_a_table_exist(table_name), f"{table_name} in {self.sqlite_db_path}"

        sql_statement = f"""
            SELECT COUNT(*) FROM {table_name}
        """
        result = self._execute_sql_statement(sql_statement)
        assert len(result) == 1
        assert len(result[0]) == 1
        assert isinstance(result[0][0], int)
        return result[0][0]

    def get_list_of_unique_values_of_a_column_in_a_table(self, a_table_name, a_column_name):
        assert isinstance(a_table_name, str)
        assert isinstance(a_column_name, str)
        assert self.does_a_table_exist(a_table_name), a_table_name
        assert a_column_name in self.get_list_of_columns_of_a_table(a_table_name)

        sql_statement = f"""
            SELECT DISTINCT {a_column_name} 
            FROM {a_table_name}
        """
        result = self._execute_sql_statement(sql_statement)
        result = list(map(lambda x: x[0], result))
        return result
