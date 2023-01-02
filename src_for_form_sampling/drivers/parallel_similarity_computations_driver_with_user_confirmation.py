import my_utils.file_and_dir_interaction_helpers
import my_utils.terminal_interaction_helpers
from src_for_form_sampling.parameters_package import PrmsContainer
from my_utils import misc_helpers as gh
import os
import my_utils.parallel_execution_handlers
from src_for_form_sampling import parameters_package
import my_utils as mu


def _ask_if_it_is_okay_to_execute():
    my_utils.terminal_interaction_helpers.confirm_dangerous_operation_with_kboard_input_or_exit_from_sys("Are you using 'time' command?")

    if my_utils.terminal_interaction_helpers.ask_for_input_on_terminal_and_get_true_or_false(f'Deleting all json files in {PrmsContainer.path_to_similarity_matrix_json_cache_directory}?'):
        my_utils.delete_all_jsons_in_a_folder(PrmsContainer.path_to_similarity_matrix_json_cache_directory)
        my_utils.terminal_message_start_print()
        print('similarity matrix jsons are now deleted!!!')
        my_utils.terminal_message_end_print()


def _dynamically_generate_python3_shell_command_list():
    config_dir = parameters_package.save_multiple_parameter_combos_to_config_files_and_return_the_path_to_config_files_dir()

    shell_command_list = []
    for a_yaml_file_path in my_utils.get_list_of_paths_to_all_files_in_a_dir(config_dir):
        assert a_yaml_file_path.endswith('.yaml')

        if os.path.isabs(a_yaml_file_path):
            complete_path_from_this_file_to_a_yaml_file = a_yaml_file_path
        else:
            # TODO: This else clause is not really tested.....
            complete_path_from_this_file_to_a_yaml_file = config_dir + a_yaml_file_path

        a_python_file_to_run = 'one_similarity_computation_driver.py'
        if not os.path.exists("./" + a_python_file_to_run):
            raise Exception("The one similarity computation driver python code needs to be in the same directory")

        a_python_command_line_argument_str = "no " + str(complete_path_from_this_file_to_a_yaml_file)
        assert my_utils.does_a_file_exist(a_python_file_to_run)

        a_python_command = f"python3 {os.getcwd()}/{a_python_file_to_run} {a_python_command_line_argument_str}"
        # e.g. "python3 one_similarity_computation.py no ./yamlfolder/23098.yaml"
        shell_command_list.append(a_python_command)

    return shell_command_list


def does_a_file_exists_in_the_same_dir(file_name):
    return my_utils.does_a_file_exist(file_name)


if __name__ == "__main__":
    _ask_if_it_is_okay_to_execute()
    a_num = my_utils.parse_a_positional_argument_from_terminal('n_of_parallel_processes')
    commands = _dynamically_generate_python3_shell_command_list()
    my_utils.parallel_execution_handlers.execute_list_of_commands_parallely(
        list_of_shell_command_strings=commands,
        n_of_parallel_processes=a_num
    )
