from .imports import *

def save_nested_list(file_path, nested_list):
    """Helper function to save affinity graphs."""
    np.savez_compressed(file_path, *nested_list)

def load_nested_list(file_path):
    """Helper function to load affinity graphs."""
    loaded_data = np.load(file_path,allow_pickle=True)
    loaded_nested_list = []
    for key in loaded_data.keys():
        loaded_nested_list.append(loaded_data[key])
    return loaded_nested_list


