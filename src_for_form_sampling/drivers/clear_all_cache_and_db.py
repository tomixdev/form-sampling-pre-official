import my_utils.file_and_dir_interaction_helpers
import my_utils.terminal_interaction_helpers
from src_for_form_sampling.parameters_package import PrmsContainer
from my_utils import misc_helpers as gh
import os


def deletions():
    if my_utils.file_and_dir_interaction_helpers.does_a_dir_exist(PrmsContainer.path_to_audio_extracted_features_cache_directory):
        my_utils.terminal_interaction_helpers.terminal_message_start_print()
        if my_utils.terminal_interaction_helpers.ask_for_input_on_terminal_and_get_true_or_false(f"Deleting all pickle and json files in {PrmsContainer.path_to_audio_extracted_features_cache_directory}?"):
            my_utils.file_and_dir_interaction_helpers.delete_all_pickles_in_a_folder(PrmsContainer.path_to_audio_extracted_features_cache_directory)
            my_utils.file_and_dir_interaction_helpers.delete_all_jsons_in_a_folder(PrmsContainer.path_to_audio_extracted_features_cache_directory)
            print(f"Deleted all pickle and json files in {PrmsContainer.path_to_audio_extracted_features_cache_directory}")
        my_utils.terminal_interaction_helpers.terminal_message_end_print()

    if my_utils.file_and_dir_interaction_helpers.does_a_file_exist(PrmsContainer.path_to_db_that_stores_audio_feature_extraction_results):
        my_utils.terminal_interaction_helpers.terminal_message_start_print()
        if my_utils.terminal_interaction_helpers.ask_for_input_on_terminal_and_get_true_or_false(f"Deleting {PrmsContainer.path_to_db_that_stores_audio_feature_extraction_results}?"):
            os.remove(PrmsContainer.path_to_db_that_stores_audio_feature_extraction_results)
            print(f"Deleted {PrmsContainer.path_to_db_that_stores_audio_feature_extraction_results}")
        my_utils.terminal_interaction_helpers.terminal_message_end_print()

    if my_utils.file_and_dir_interaction_helpers.does_a_dir_exist(PrmsContainer.path_to_similarity_matrix_json_cache_directory):
        my_utils.terminal_interaction_helpers.terminal_message_start_print()
        if my_utils.terminal_interaction_helpers.ask_for_input_on_terminal_and_get_true_or_false(f"Deleting all json files in {PrmsContainer.path_to_similarity_matrix_json_cache_directory}?"):
            my_utils.file_and_dir_interaction_helpers.delete_all_jsons_in_a_folder(PrmsContainer.path_to_similarity_matrix_json_cache_directory)
            print(f"Deleted all json files in {PrmsContainer.path_to_similarity_matrix_json_cache_directory}")
        my_utils.terminal_interaction_helpers.terminal_message_end_print()

    if my_utils.file_and_dir_interaction_helpers.does_a_file_exist(PrmsContainer.path_to_similarity_matching_db):
        my_utils.terminal_interaction_helpers.terminal_message_start_print()
        if my_utils.terminal_interaction_helpers.ask_for_input_on_terminal_and_get_true_or_false(f"Deleting {PrmsContainer.path_to_similarity_matching_db}?"):
            os.remove(PrmsContainer.path_to_similarity_matching_db)
            print(f"Deleted {PrmsContainer.path_to_similarity_matching_db}")
        my_utils.terminal_interaction_helpers.terminal_message_end_print()


if __name__ == '__main__':
    deletions()
