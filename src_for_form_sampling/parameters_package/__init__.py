"""
This __init__.py file provides API to this parameters package.
"""
import pprint

import my_utils
from . import all_params
from my_utils import terminal_interaction_helpers

'''API Variables'''
PrmsContainer = all_params.PrmsContainer
prms_stack = all_params.StructuredPrms.get_prms_stack()


'''API Functions'''
'''Func 1'''
save_a_single_prms_combo_to_config_file_and_return_the_config_file_path = \
    all_params.StructuredPrms.save_a_single_prms_combo_to_config_file_and_return_the_config_file_path


'''Func 2'''
save_multiple_parameter_combos_to_config_files_and_return_the_path_to_config_files_dir = \
    all_params.StructuredPrms.save_multiple_parameter_combos_to_config_files_and_return_the_path_to_config_files_dir


'''Func 3'''


def set_up_params_for_a_single_run():
    path_to_yaml_config_file = all_params.StructuredPrms.save_a_single_prms_combo_to_config_file_and_return_the_config_file_path()
    PrmsContainer.set_values_from_structured_yaml_config_file(path_to_yaml_config_file)


'''Func 4'''


def generate_parameters_config_files_for_parallel_computation():
    save_multiple_parameter_combos_to_config_files_and_return_the_path_to_config_files_dir()


'''if __name__ == '__main__':'''
if __name__ == "__main__":
    set_up_params_for_a_single_run()
    generate_parameters_config_files_for_parallel_computation()
