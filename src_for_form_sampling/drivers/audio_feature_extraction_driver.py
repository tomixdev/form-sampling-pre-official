"""Driver for Audio Data Extraction"""
import datetime
import sys
import pprint

import my_utils.terminal_interaction_helpers
from my_utils import misc_helpers as mh
import clear_all_cache_and_db
from src_for_form_sampling.preprocessing_audio_feature_extraction import audio_analysis_tree_classes
from src_for_form_sampling import parameters_package
from src_for_form_sampling.parameters_package import PrmsContainer
import mlflow
import my_utils as mu


########################################################################################################################
######################### Setting up and confirming with user input ####################################################
########################################################################################################################
parameters_package.set_up_params_for_a_single_run()

my_utils.confirm_dangerous_operation_with_kboard_input_or_exit_from_sys(
    f"The parameters stack I am using is {[a_cls.__name__ for a_cls in parameters_package.prms_stack]}'. Are you sure to run?"
)

my_utils.confirm_dangerous_operation_with_kboard_input_or_exit_from_sys(
    "Are you measuring execution time from terminal by 'time' command?"
)

my_utils.confirm_dangerous_operation_with_kboard_input_or_exit_from_sys(
    "Did you clear cache or db? (Maybe you do not need to do it. In that case, just proceed)"
)

my_utils.confirm_dangerous_operation_with_kboard_input_or_exit_from_sys(
    "If honban codes, do not just run this driver code. "
    "Also run audio similarity matrix calculation on terminal. "
    "In this way, some similarity values will be calculated even if audio data extraction fails."
)

clear_all_cache_and_db.deletions()


########################################################################################################################
######################## Program Execution #############################################################################
########################################################################################################################
execution_start_time = datetime.datetime.now()

audio_analysis_tree_classes.AudioFolderAnalysis(
    path_to_audio_folder=PrmsContainer.path_to_audio_folder,
    path_to_audio_extracted_data_cache_folder=PrmsContainer.path_to_audio_extracted_features_cache_directory
)

total_execution_time = str(datetime.datetime.now() - execution_start_time)

########################################################################################################################
######################## Log Results  ##################################################################################
########################################################################################################################
mlflow.set_tracking_uri(uri=f'file://{mh.to_absolute_path(PrmsContainer.path_to_dir_of_mlflow_run_results_for_audio_feature_extraction)}')
mlflow.set_experiment(experiment_name=mh.get_module_python_file_name(__name__))
with mlflow.start_run(run_name=''):
    mlflow.log_text(total_execution_time, artifact_file=mu.varnameof(total_execution_time) + '.txt')

    prm_dict = mh.get_dict_of_user_defined_public_class_attributes_excluding_methods(PrmsContainer)
    prm_dict = mu.convert_values_of_absolute_path_strings_in_a_dict_into_relative_path_strings(prm_dict)
    mlflow.log_params(prm_dict)

    mlflow.log_artifact(PrmsContainer.PATH_TO_A_CONFIG_FILE_THAT_WAS_USED_TO_SET_UP_SPECIFIC_PRMS, artifact_path='')
    mlflow.log_artifact(PrmsContainer.path_to_db_that_stores_audio_feature_extraction_results, artifact_path='')

mu.delete_a_file(PrmsContainer.PATH_TO_A_CONFIG_FILE_THAT_WAS_USED_TO_SET_UP_SPECIFIC_PRMS, confirmation_needed=True)
my_utils.delete_a_file(PrmsContainer.PATH_TO_A_CONFIG_FILE_THAT_WAS_USED_TO_SET_UP_SPECIFIC_PRMS, confirmation_needed=True)
# gh.delete_a_file(PrmsContainer.path_to_db_that_stores_audio_feature_extraction_results)
