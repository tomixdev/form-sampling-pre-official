import argparse
import atexit
import inspect
import sys
import traceback
import warnings
import pprint

from . import misc_helpers as mh

SUPRESS_DEBUG_PRINTING = False


def debuginfo(value_or_message, varname_str=None, indent_level=None, err_max_depth=4):
    if SUPRESS_DEBUG_PRINTING:
        return

    assert value_or_message is not None, f"{mh.varnameof(value_or_message)} is NONE!!"

    if varname_str is not None:
        mh.assert_class(varname_str, str)

    indents = ''
    an_indent = '    '
    if indent_level is not None:
        mh.assert_class(indent_level, int)
        assert indent_level >= 1
        for i in range(0, indent_level):
            indents += an_indent

    terminal_message_start_print()

    list_of_callers = []
    for i in range(1, err_max_depth+1):
        if i < len(inspect.stack()):
            caller = inspect.getframeinfo(inspect.stack()[i][0])
            list_of_callers.append(caller)

    list_of_callers.reverse()
    for a_caller in list_of_callers:
        print(f"{a_caller.filename}:{a_caller.lineno} | {indents} {a_caller.function}")

    if varname_str is None:
        if isinstance(value_or_message, (tuple, list, dict)):
            pprint.pprint(f"{indents}{value_or_message}")
        else:
            print(f"{indents}{value_or_message}")
    else:
        print(f"{indents}{varname_str} = {str(value_or_message)}")

    terminal_message_end_print()


def warninginfo(message):
    terminal_message_start_print()
    caller = inspect.getframeinfo(inspect.stack()[1][0])
    print(f"{caller.filename}:{caller.lineno} | {caller.function}")
    print("MyOwnWARNING: " + message)
    terminal_message_end_print()


def pring_progress_message(message):
    print(message)


def ask_for_input_on_terminal_and_get_true_or_false(message):
    terminal_message_start_print()
    print(message)
    confirmation_input_str = 'esg2gfhnl'
    while True:
        a_keyboard_input = input(f"Enter '{confirmation_input_str}' to confirm or type 'no': ")
        if a_keyboard_input.lower() == 'esg2gfhnl':
            terminal_message_end_print()
            return True
        elif a_keyboard_input.lower() == 'no':
            return False
        else:
            continue


def confirm_dangerous_operation_with_kboard_input_or_exit_from_sys(message, err_max_depth=4):
    terminal_message_start_print()

    list_of_callers = []
    for i in range(1, err_max_depth+1):
        if i < len(inspect.stack()):
            caller = inspect.getframeinfo(inspect.stack()[i][0])
            list_of_callers.append(caller)

    list_of_callers.reverse()
    for a_caller in list_of_callers:
        print(f"{a_caller.filename}:{a_caller.lineno} | {a_caller.function}")

    print(message)
    confirmation_str = 'esg2gfhnl'  # TODO: make this a random string in the future (for what? for security????)

    a_kboard_input = input(f"Type '{confirmation_str}' to confirm or your answer or type 'no': ")
    while True:
        if a_kboard_input == confirmation_str:
            terminal_message_end_print()
            return
        elif a_kboard_input == 'no':
            terminal_message_end_print()
            sys.exit()
        else:
            a_kboard_input = input("Keyboard input spelling is perhaps wrong....?????? Type again: ")


def terminal_message_start_print():
    print("_____________________________________________________________________________________________________________________________")


def terminal_message_end_print():
    print("-----------------------------------------------------------------------------------------------------------------------------")


def set_terminal_warnings_print_setting(warning_with_traceback: bool, completely_supress_all_warnings: bool):
    assert isinstance(warning_with_traceback, bool)
    assert isinstance(completely_supress_all_warnings, bool)

    def my_own_showwarning_with_traceback(message, category, filename, lineno, file=None, line=None):
        terminal_message_start_print()
        log = file if hasattr(file, 'write') else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))
        terminal_message_end_print()

    def bring_back_showwarning_setting_to_default():
        warnings.showwarning = _python_default_showwarning

    _python_default_showwarning = warnings.showwarning
    atexit.register(bring_back_showwarning_setting_to_default)
    if warning_with_traceback:
        warnings.showwarning = my_own_showwarning_with_traceback
    else:
        bring_back_showwarning_setting_to_default()

    if completely_supress_all_warnings:
        warnings.filterwarnings("ignore")
    else:
        warnings.showwarning = _python_default_showwarning

    # TODO: Not sure the following code is necessary
    if __name__ == '__main__':
        warnings.warn('Not sure if the following code is necessary', DeprecationWarning, stacklevel=2)
        bring_back_showwarning_setting_to_default()


def parse_a_positional_argument_from_terminal(an_argument_variable_name, convert_to_nums_if_possible=True):
    # Tested on 20221103
    parser = argparse.ArgumentParser()
    parser.add_argument(an_argument_variable_name)
    args = parser.parse_args()
    a_val = getattr(args, an_argument_variable_name)

    if convert_to_nums_if_possible:
        return mh.convert_str_to_int_or_float_if_possible(a_val)
    else:
        return a_val


def parse_a_keyword_argument_from_terminal(an_argument_variable_name):
    # Tested on 20221103
    parser = argparse.ArgumentParser()
    parser.add_argument("--" + an_argument_variable_name)
    args = parser.parse_args()

    a_val = getattr(args, an_argument_variable_name)

    return a_val


def parse_list_of_positional_arguments_from_terminal(list_of_argument_variable_names, convert_to_nums_if_possible=True):
    # Tested on 20221103
    parser = argparse.ArgumentParser()
    for an_argument_variable_name in list_of_argument_variable_names:
        parser.add_argument(an_argument_variable_name)
    args = parser.parse_args()

    list_to_return = [getattr(args, an_argument_variable_name) for an_argument_variable_name in list_of_argument_variable_names]

    if convert_to_nums_if_possible:
        list_to_return = [mh.convert_str_to_int_or_float_if_possible(a_val) for a_val in list_to_return]

    return list_to_return
