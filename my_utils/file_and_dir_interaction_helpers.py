import glob
import os
import platform
import shutil
import warnings

from . import misc_helpers
from . import terminal_interaction_helpers
from . import data_structure_helpers


def create_a_dir_if_not_exists(path_to_dir):
    assert isinstance(path_to_dir, str)
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)


def delete_a_file(path_to_file, confirmation_needed=True):
    assert isinstance(path_to_file, str)
    assert os.path.isfile(path_to_file)

    if os.path.exists(path_to_file):
        if confirmation_needed:
            terminal_interaction_helpers.confirm_dangerous_operation_with_kboard_input_or_exit_from_sys(
                f'Are you sure you want to delete the file {path_to_file}? (message from : {delete_a_file.__name__}'
            )
        else:
            pass
        os.remove(path_to_file)
    else:
        raise Exception(f"File '{path_to_file}' does not exist")


def delete_a_dir(path_to_dir, confirmation_needed=True):
    assert isinstance(path_to_dir, str)
    assert os.path.isdir(path_to_dir)

    if os.path.exists(path_to_dir):
        if confirmation_needed:
            terminal_interaction_helpers.confirm_dangerous_operation_with_kboard_input_or_exit_from_sys(
                f"Are you sure you want to delete the directory '{path_to_dir}'? (message from : {delete_a_dir.__name__})"
            )
        else:
            pass
        shutil.rmtree(path_to_dir)
    else:
        raise Exception(f"Directory '{path_to_dir}' does not exist")


def does_a_file_exist(relative_path_to_file):
    assert isinstance(relative_path_to_file, str)
    return os.path.exists(relative_path_to_file)

    '''
    my_file = pathlib.Path(relative_path_to_file)
    if my_file.is_file():
        return True
    else:
        return False
    '''


def does_a_dir_exist(path_to_dir):
    return os.path.isdir(path_to_dir)


def delete_all_jsons_in_a_folder(path_to_folder):
    assert isinstance(path_to_folder, str)
    assert os.path.isdir(path_to_folder)
    assert are_all_files_in_a_directory_json_files(path_to_folder)

    for file in glob.glob(path_to_folder + '/*.json'):
        os.remove(file)


def are_all_files_in_a_directory_json_files(path_to_dir):
    assert isinstance(path_to_dir, str)
    assert os.path.isdir(path_to_dir)

    for file in glob.glob(path_to_dir + '/*'):
        if not file.endswith('.json'):
            return False
    return True


def delete_all_pickles_in_a_folder(path_to_folder):
    assert isinstance(path_to_folder, str)
    assert os.path.isdir(path_to_folder)

    for file in glob.glob(path_to_folder + '/*.pickle'):
        os.remove(file)


def get_list_of_entire_relative_path_to_all_audio_files_in_a_folder(path_to_folder):
    warnings.warn(
        f"This function's name is grammatically wrong, use '{get_list_of_entire_relative_paths_to_all_audio_files_in_a_directory.__name__}' instead", DeprecationWarning, stacklevel=2)

    raise_exception_if_path_to_a_folder_is_in_wrong_format(path_to_folder)

    list_of_wav_files = glob.glob(path_to_folder + '*.wav')
    list_of_mp3_files = glob.glob(path_to_folder + '*.mp3')
    list_of_aiff_files = glob.glob(path_to_folder + '*.aiff')
    list_of_aif_files = glob.glob(path_to_folder + '*.aif')

    list_of_audio_files = list_of_wav_files + list_of_mp3_files + list_of_aiff_files + list_of_aif_files

    return list_of_audio_files


def get_list_of_entire_relative_paths_to_all_audio_files_in_a_directory(path_to_directory):
    return get_list_of_entire_relative_path_to_all_audio_files_in_a_folder(path_to_folder=path_to_directory)


def get_list_of_paths_to_all_files_in_a_dir(dir_path):
    """
    If dir_path is ABSOLUTE path: -> returns list of ABSOLUTE paths
    If dir_path is RELATIVE path: -> returns list of RELATIVE paths
    """

    assert isinstance(dir_path, str)
    assert os.path.isdir(dir_path)
    raise_exception_if_path_to_a_folder_is_in_wrong_format(dir_path)
    assert dir_path.endswith('/')

    return glob.glob(dir_path + '*')


def raise_exception_if_path_to_a_folder_is_in_wrong_format(path_to_a_folder):
    misc_helpers.assert_class(path_to_a_folder, str)
    if not os.path.isabs(path_to_a_folder) and path_to_a_folder[0] == '/':
        raise Exception(f"The relative path to a dir should NOT start with '/'! \n{path_to_a_folder}")
    if path_to_a_folder[-1] != '/':
        raise Exception(f"The relative path to a dir has to end in '/'! \n{path_to_a_folder}")


def get_creation_or_modification_date_of_a_file(path_to_file):
    path_to_file = r'%s' % path_to_file

    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime


def write_a_list_text_file(a_list, relative_path_and_name):

    file = open(relative_path_and_name, 'w+')
    content = str(a_list)
    file.write(content)
    file.close()


def get_a_list_from_text_file(relative_path_and_name):
    numbers = None

    with open(relative_path_and_name) as f:
        numbers = f.read().splitlines()

    for i in range(0, len(numbers)):
        numbers[i] = float(numbers[i])

    return numbers
