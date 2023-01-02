import json

from . import file_and_dir_interaction_helpers
import numpy as np
import os


def convert_a_variable_to_json_serializable_datatypes(item):
    if isinstance(item, list):
        return item
    elif isinstance(item, dict):
        return item
    elif isinstance(item, tuple):
        return list(item)
    elif isinstance(item, str):
        return item
    elif isinstance(item, int):
        return item
    elif isinstance(item, float):
        return item
    elif isinstance(item, bool):
        return item
    elif isinstance(item, np.ndarray):
        if len(item.shape) == 1 or len(item.shape) == 2:
            return item.tolist()
        else:
            raise Exception(f"{item.__class__} is not JSON serializable with my current implementation of json writer.")
    elif isinstance(item, np.int64):
        return int(item)
    elif isinstance(item, np.float64):
        return float(item)
    elif isinstance(item, np.bool_):
        return bool(item)
    elif isinstance(item, np.float32):
        return float(item)
    elif isinstance(item, np.int32):
        return int(item)
    elif isinstance(item, np.float16):
        return float(item)
    elif isinstance(item, np.int16):
        return int(item)
    elif isinstance(item, np.uint64):
        return int(item)
    elif isinstance(item, np.uint32):
        return int(item)
    elif isinstance(item, np.uint16):
        return int(item)
    elif isinstance(item, np.uint8):
        return int(item)
    elif isinstance(item, np.int8):
        return int(item)
    else:
        raise Exception(f"{item.__class__} is not JSON serializable")


class JsonReader:
    def __init__(self, path_to_json_file):
        assert isinstance(path_to_json_file, str)
        assert file_and_dir_interaction_helpers.does_a_file_exist(path_to_json_file)
        self.path_to_json_file = path_to_json_file
        self.a_dict_or_list = None
        self.load_json_file_as_dict()

    def load_json_file_as_dict(self):
        with open(self.path_to_json_file, 'r') as f:
            self.a_dict_or_list = json.load(f)

    def get_value_from_json_dict(self, key_str):
        assert isinstance(self.a_dict_or_list, dict)
        assert isinstance(key_str, str)
        return self.a_dict_or_list[key_str]

    def read_as_list(self):
        with open(self.path_to_json_file, 'r') as f:
            self.a_dict_or_list = json.load(f)

        assert isinstance(self.a_dict_or_list, list)

        return self.a_dict_or_list

    def read_as_ndarray(self):
        return np.array(self.read_as_list())


class JsonWriter:
    def __init__(self,
                 value_to_save_to_json,
                 path_to_json_file,
                 auto_convert_json_unserializable_datatypes=True,
                 delete_json_if_fails=False):
        assert isinstance(path_to_json_file, str)
        assert path_to_json_file[-5:] == ".json"
        self.value_to_save_to_json = value_to_save_to_json
        self.path_to_json_file = path_to_json_file
        self.auto_convert_json_unserializable_datatypes = auto_convert_json_unserializable_datatypes
        self.delete_json_if_fails = delete_json_if_fails
        self.save_dict_as_json_file()

    def save_dict_as_json_file(self):
        def MyException():
            if self.delete_json_if_fails and file_and_dir_interaction_helpers.does_a_file_exist(self.path_to_json_file):
                assert self.path_to_json_file[-5:] == ".json"
                os.remove(self.path_to_json_file)
            raise Exception('Json Dump Failed')

        try:
            with open(self.path_to_json_file, 'w') as f:
                if isinstance(self.value_to_save_to_json, list):
                    to_save_to_json = self.value_to_save_to_json
                    if self.auto_convert_json_unserializable_datatypes:
                        to_save_to_json = self._convert_list_to_json_serializable_datatypes(to_save_to_json)
                    json.dump(to_save_to_json, f, indent=0)
                elif isinstance(self.value_to_save_to_json, dict):
                    to_save_to_json = self.value_to_save_to_json
                    if self.auto_convert_json_unserializable_datatypes:
                        to_save_to_json = self._convert_dict_to_json_serializable_datatyles(to_save_to_json)
                    json.dump(to_save_to_json, f, indent=4)
                elif isinstance(self.value_to_save_to_json, np.ndarray):
                    to_save_to_json = self.value_to_save_to_json.tolist()
                    if self.auto_convert_json_unserializable_datatypes:
                        to_save_to_json = self._convert_list_to_json_serializable_datatypes(to_save_to_json)
                    json.dump(to_save_to_json, f, indent=0)
                else:
                    MyException()
        except:
            MyException()

    def _convert_list_to_json_serializable_datatypes(self, a_list):
        assert isinstance(a_list, list)
        new_list = []
        for item in a_list:
            converted_item = convert_a_variable_to_json_serializable_datatypes(item)
            if isinstance(converted_item, list):
                converted_item = self._convert_list_to_json_serializable_datatypes(converted_item)
            if isinstance(converted_item, dict):
                converted_item = self._convert_dict_to_json_serializable_datatyles(converted_item)

            new_list.append(converted_item)
        return new_list

    def _convert_dict_to_json_serializable_datatyles(self, a_dict):
        assert isinstance(a_dict, dict)

        new_dict = {}
        for a_key, a_value in a_dict.items():
            converted_value = convert_a_variable_to_json_serializable_datatypes(a_value)
            if isinstance(converted_value, list):
                converted_value = self._convert_list_to_json_serializable_datatypes(converted_value)
            if isinstance(converted_value, dict):
                converted_value = self._convert_dict_to_json_serializable_datatyles(converted_value)

            new_dict[a_key] = converted_value

        return new_dict
