import my_utils.data_structure_helpers
from my_utils import misc_helpers as gh
import typing
import inspect


class TreeHelpers:
    path_to_audio_extracted_data_cached_folder = None

    @classmethod
    def get_path_to_audio_data_cache(cls, audio_hash_value):
        return cls.path_to_audio_extracted_data_cached_folder + audio_hash_value + ".pickle"

    @staticmethod
    def get_value_from_cached_variables_or_compute_value(cached_variables,
                                                         # TODO: As of the experiment stage, this is a python pickle object. This needs to be a json-based dict in the future.
                                                         varname_str_as_dict_key,
                                                         the_previous_varname_str=None,
                                                         func_to_compute_value=None):
        gh.assert_class(varname_str_as_dict_key, str)

        if gh.object_exists_and_has_attribute(cached_variables, varname_str_as_dict_key):
            return cached_variables.__dict__[varname_str_as_dict_key]
        elif the_previous_varname_str is not None and \
                gh.object_exists_and_has_attribute(cached_variables, the_previous_varname_str):
            gh.assert_class(the_previous_varname_str, str)
            return cached_variables.__dict__[the_previous_varname_str]
        elif func_to_compute_value is not None:
            gh.assert_class(func_to_compute_value, typing.Callable)
            return func_to_compute_value()
        elif func_to_compute_value is None:
            return None  # varname_str is not cached, and there is no function that calculates the varname
        else:
            raise Exception()

    @staticmethod
    def get_func_arguments_dict_from_within_wrapper_method(func, self, *args, **kwargs):
        bound = inspect.signature(func).bind(self, *args, **kwargs)
        bound.apply_defaults()
        func_arguments_dict = bound.arguments
        func_arguments_dict.pop('self')
        return func_arguments_dict

    @staticmethod
    def call_all_public_methods_in_a_class_instance(a_class_instance):
        attrs = (getattr(a_class_instance, name) for name in dir(a_class_instance) if not name.startswith("__") and not name.startswith('_'))
        methods = filter(inspect.ismethod, attrs)
        for a_method in methods:
            try:
                a_method()
            except Exception as e:
                raise Exception(f"The method {a_method.__name__} execution failed. ")
                # raise Exception(f"The method {a_method.__name__} execution failed. "
                #                f"This function {inspect.stack()[0][3]} is intended for audio data extraction class and data transformer class.!!!"
                #                f"The method is failing most likly because (1) method arguments are not correct or "
                #                f"(2) Error happend somewhere in the audio data extraction tree structure")

    @staticmethod
    def extract_relevant_dicts_and_lists_from_function_arguments_tuples(function_arguments_tuples):
        ''''''
        '''dict_from_normal_varnames_to_parameter_valuesについて:
             　もしnormal_varnamesのvalueがNoneの場合、それに対応するintelligible_id_varnameのValueがNoneでなければ、その値にする。
           dict_from_intelligible_id_varnames_to_parameter_valuesについて:
            　 もしintelligible_id_varnamesのvalueがNoneの場合、それに対応するnormal_varnameのvalueがNoneでなければ、その値にする。
        '''
        '''
        WARNING:
            The implementation of this function (or method) depends on the fact that 
            when I list a dictionary by 'list(a_dict), the order in a_dict is preserved....
    
        '''
        dict_from_normal_varnames_to_parameter_values = {
        }  # >---- pass to next node (to be used when I save to SQl database, so that I can read and understand)
        dict_from_intelligible_id_varnames_to_parameter_values = {}  # > --- pass to next node (to be used to read values from cache)

        list_of_normal_varnames = []
        list_of_intelligible_id_varnames = []
        list_of_list_of_possible_values = []

        i = 0
        while i + 2 < len(function_arguments_tuples):
            normal_varname = function_arguments_tuples[i][0]
            value_assigned_to_normal_varname_value = function_arguments_tuples[i][1]
            intelligible_id_varname = function_arguments_tuples[i + 1][0]
            value_assigned_to_intelligible_id_varname = function_arguments_tuples[i + 1][1]
            possible_values_varname = function_arguments_tuples[i + 2][0]
            possible_values_list = function_arguments_tuples[i + 2][1]

            assert not (value_assigned_to_normal_varname_value is None and
                        value_assigned_to_intelligible_id_varname is None and
                        possible_values_list is None)

            list_of_normal_varnames.append(normal_varname)
            list_of_intelligible_id_varnames.append(intelligible_id_varname)
            list_of_list_of_possible_values.append(possible_values_list)

            if value_assigned_to_normal_varname_value is None and value_assigned_to_intelligible_id_varname is not None:
                dict_from_normal_varnames_to_parameter_values[normal_varname] = value_assigned_to_intelligible_id_varname
                dict_from_intelligible_id_varnames_to_parameter_values[
                    intelligible_id_varname] = value_assigned_to_intelligible_id_varname
            elif value_assigned_to_normal_varname_value is not None and value_assigned_to_intelligible_id_varname is None:
                dict_from_normal_varnames_to_parameter_values[normal_varname] = value_assigned_to_normal_varname_value
                dict_from_intelligible_id_varnames_to_parameter_values[
                    intelligible_id_varname] = value_assigned_to_normal_varname_value
            elif value_assigned_to_normal_varname_value is not None and value_assigned_to_intelligible_id_varname is not None:
                assert value_assigned_to_normal_varname_value == value_assigned_to_intelligible_id_varname
                dict_from_normal_varnames_to_parameter_values[normal_varname] = value_assigned_to_normal_varname_value
                dict_from_intelligible_id_varnames_to_parameter_values[
                    intelligible_id_varname] = value_assigned_to_intelligible_id_varname
            elif value_assigned_to_normal_varname_value is None and value_assigned_to_intelligible_id_varname is None:
                '''This Case is handled outside this function'''
                pass
            else:
                raise Exception()

            i = i + 3

        my_utils.data_structure_helpers.assert_all_values_in_a_dict_are_all_none_or_all_non_none_if_a_dict_is_not_empty(
            dict_from_normal_varnames_to_parameter_values)
        my_utils.data_structure_helpers.assert_all_values_in_a_dict_are_all_none_or_all_non_none_if_a_dict_is_not_empty(
            dict_from_intelligible_id_varnames_to_parameter_values)
        gh.assert_value_equality(len(list_of_normal_varnames), len(list_of_intelligible_id_varnames))
        gh.assert_value_equality(len(list_of_normal_varnames), len(list_of_list_of_possible_values))

        return dict_from_normal_varnames_to_parameter_values, \
            dict_from_intelligible_id_varnames_to_parameter_values, \
            list_of_normal_varnames, \
            list_of_intelligible_id_varnames, \
            list_of_list_of_possible_values

    @staticmethod
    def get_two_lists_about_upper_nodes_in_the_tree_structure___return_as_str___I_randomly_wrote_this_function_for_debugging(current_node):
        list_of_column_names_of_relational_db = []
        list_of_values_to_save_to_relational_db = []

        current = current_node
        while True:
            column_name = current.column_name_of_relational_db
            if column_name is not None:
                value = current.value_to_save_to_relational_db
                assert value is not None  # This assertation is actually not needed. The same assertaion exists in the __init__ of TreeClassesCommon as of Sep 10, 2022
                previous_column_name = current.attribute_alias19482
                list_of_column_names_of_relational_db.append(column_name)
                list_of_values_to_save_to_relational_db.append(value)

            assert hasattr(current, gh.varnameof(current.parent_node)), f"all tree nodes need to have 'TreeClassesCommon' as super class"
            if current.parent_node is None:
                break
            else:
                current = current.parent_node
        list_of_column_names_of_relational_db.reverse()
        list_of_values_to_save_to_relational_db.reverse()

        dict_to_return = {}
        for i in range(len(list_of_column_names_of_relational_db)):
            dict_to_return[list_of_column_names_of_relational_db[i]] = list_of_values_to_save_to_relational_db[i]
        return str(dict_to_return)
