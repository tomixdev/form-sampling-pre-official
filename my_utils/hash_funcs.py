import copy
import hashlib
import my_utils.file_and_dir_interaction_helpers
from . import misc_helpers as mh
import json
import numpy as np
import warnings

read_first_time_str = 'read_first_time'


def compute_hash_value_from_json_file(relative_path_to_json):
    assert relative_path_to_json[-5:] == '.json'
    with open(relative_path_to_json, 'rb') as f:
        bytes = f.read()
        readable_hash = hashlib.sha256(bytes).hexdigest()
        return readable_hash


def hash_a_string(a_string):
    assert isinstance(a_string, str)
    return hashlib.sha256(a_string.encode('utf-8')).hexdigest()


def hash_a_ndarray(a_ndarray):
    assert isinstance(a_ndarray, np.ndarray)
    return hashlib.sha256(a_ndarray.tobytes()).hexdigest()


def hash_a_list(a_list):
    mh.assert_class(a_list, list)
    dhash = hashlib.sha256()  # hashlib.md5()
    encoded = json.dumps(tuple(a_list)).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def hash_a_list_as_multiset(a_list):
    mh.assert_class(a_list, list)

    a_new_json_dumped_str_list = []
    for an_elment in a_list:
        json_dumped_str = json.dumps(an_elment)
        a_new_json_dumped_str_list.append(json_dumped_str)

    return hash_a_list(sorted(a_new_json_dumped_str_list))


def compute_hash_value_of_a_set_of_json_files_from_hash_value_str_list(list_as_multiset_of_json_file_hash_values):
    assert isinstance(list_as_multiset_of_json_file_hash_values, list)
    assert all(isinstance(item, str) for item in list_as_multiset_of_json_file_hash_values)

    list_as_multiset_of_json_file_hash_values = copy.deepcopy(list_as_multiset_of_json_file_hash_values)  # このSentenceぜったい必要!!これがないと、もとのListのOrderがめちゃくちゃになる!!!
    list_as_multiset_of_json_file_hash_values.sort()
    string_to_be_hashed = ''
    for a_hash_value in list_as_multiset_of_json_file_hash_values:
        string_to_be_hashed += a_hash_value

    return hash_a_string(string_to_be_hashed)


def hash_a_tuple(a_tuple):
    mh.assert_class(a_tuple, tuple)
    return hash_a_list(list(a_tuple))


def hash_an_audio_extraction_parameter_combo_list(a_parameter_combo_list):
    mh.assert_class(a_parameter_combo_list, list)

    a_modified_parameter_combo_list = []
    for a_parameter in a_parameter_combo_list:
        if isinstance(a_parameter, dict):
            a_hash_value = hash_a_dict(a_parameter)
        elif isinstance(a_parameter, list):
            a_hash_value = hash_a_list(a_parameter)
        elif isinstance(a_parameter, np.ndarray):
            a_hash_value = hash_a_ndarray(a_parameter)
        elif isinstance(a_parameter, str):
            a_hash_value = hash_a_string(a_parameter)
        elif isinstance(a_parameter, dict):
            a_hash_value = hash_a_dict(a_parameter)
        elif isinstance(a_parameter, tuple):
            a_hash_value = hash_a_tuple(a_parameter)
        elif mh.is_number(a_parameter):
            from my_utils.my_db_utils import serialize_a_value_to_store_in_sql_db
            a_hash_value = serialize_a_value_to_store_in_sql_db(a_parameter)
        else:
            raise Exception(f"{a_parameter.__class__} cannot be hashed")

        a_modified_parameter_combo_list.append(a_hash_value)

    return hash_a_list(a_modified_parameter_combo_list)


def hash_a_dict(a_dict):
    mh.assert_class(a_dict, dict)
    '''
    https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html.
    '''
    ''' 
    TODO: If a value of dict is a mutable object (for example, a list or nd array), then, I need to hash that dict.
    For example, what I can do is:
    for a_key, a_value in a_dict.items():
        if a_value is an instance of list:
            hash_a_list (a_value)
        elif a_value is an instance of dict:
            hash_a_dict (a_value)
        elif a_value is an instance of ndarray:
            hash_a_ndarray(a_value)
    '''

    dhash = hashlib.sha256()  # hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(tuple(sorted(a_dict.items()))).encode()
    dhash.update(encoded)
    return dhash.hexdigest()
    # return hash(tuple(sorted(a_dict.items())))


def compute_list_of_audio_hash_values_in_audio_folder(path_to_audio_folder):
    my_utils.file_and_dir_interaction_helpers.raise_exception_if_path_to_a_folder_is_in_wrong_format(path_to_audio_folder)
    list_of_relative_paths_to_audio = my_utils.file_and_dir_interaction_helpers.get_list_of_entire_relative_path_to_all_audio_files_in_a_folder(
        path_to_audio_folder)
    list_of_audio_hash_values = []
    for a_relative_path in list_of_relative_paths_to_audio:
        a_hash_value = compute_hash_value_of_a_file(a_relative_path)
        list_of_audio_hash_values.append(a_hash_value)

    return list_of_audio_hash_values


def compute_hash_value_of_a_file(path_to_file):
    ''''''
    '''
    TODO: I can change the hashing algorithm freely if I need to later.
    '''
    return compute_the_sha256_hash_of_a_file(path_to_file)


def compute_hash_of_a_file_from_filepath_string(path_to_file):
    warnings.warn('Deprecated!!', DeprecationWarning, stacklevel=2)
    string_for_hashing = str(path_to_file) + str(
        my_utils.file_and_dir_interaction_helpers.get_creation_or_modification_date_of_a_file(path_to_file))
    encoded_string = string_for_hashing.encode('utf-8')
    return hashlib.sha256(encoded_string).hexdigest()


def compute_the_sha256_hash_of_a_file(path_to_file):
    warnings.warn('Deprecated!!', DeprecationWarning, stacklevel=2)

    path_to_file = r'%s' % path_to_file

    hash_value_to_return = None

    with open(path_to_file, 'rb') as opened_file:
        content = opened_file.read()
        sha256 = hashlib.sha256()
        sha256.update(content)
        hash_value_to_return = sha256.hexdigest()

    return hash_value_to_return
