import copy
import inspect
import pprint
import yaml


def convert_dict_to_yamle_compatible_value_recursively(a_dict):
    assert isinstance(a_dict, dict)
    a_dict = copy.deepcopy(a_dict)

    for k, v in a_dict.items():
        a_dict[k] = convert_a_python_value_to_a_yaml_compatible_value(v)

    return a_dict


def convert_a_python_value_to_a_yaml_compatible_value(a_python_val):
    if isinstance(a_python_val, (int, float, str, bool)):
        return a_python_val
    elif inspect.isfunction(a_python_val):
        return copy.deepcopy(a_python_val)
    elif isinstance(a_python_val, dict):
        return a_python_val
    elif isinstance(a_python_val, list):
        return a_python_val
    else:
        raise NotImplementedError(f"\n type: "
                                  f"\n {type(a_python_val)} "
                                  f"\n value: "
                                  f"\n {pprint.pformat(a_python_val)}")


def save_dict_to_yaml(a_dict, filepath):
    assert isinstance(a_dict, dict)
    assert isinstance(filepath, str)
    a_dict = copy.deepcopy(a_dict)

    for k, v in a_dict.items():
        a_dict[k] = convert_a_python_value_to_a_yaml_compatible_value(v)

    with open(filepath, 'w') as file:
        yaml.dump(a_dict, file, indent=4)


def load_dict_from_a_yaml(filepath):
    assert isinstance(filepath, str)

    with open(filepath, 'r') as file:
        yaml_dict = yaml.load(file, Loader=yaml.FullLoader)
    return yaml_dict


if __name__ == "__main__":
    """Testers"""
    a_sample_nested_dict = {'a': 1, 'b': 2, 'c': {'d': 3, 'e': 4, 'f': {'g': 5, 'h': 6}}}
    path = './../../data_computed/temp_files_before_organized_into_mlruns/parameter_config_files_generated_dynamically_at_run_time/20221116_084111_2033823463483833_3065236.yaml'
    save_dict_to_yaml(a_sample_nested_dict, path)
