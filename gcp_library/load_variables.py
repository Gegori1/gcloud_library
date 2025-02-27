import yaml
import os

def get_config_file_path(include_filename=True):
    """
    Dynamically determines the path to the config.yml file.

    Args:
        include_filename (bool): Whether to include the filename in the path.
                                 Defaults to True.

    Returns:
        str: The absolute path to the config.yml file.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Traverse up the directory tree until the config.yml file is found
    current_dir = script_dir
    while True:
        config_path = os.path.join(current_dir, 'config.yml')
        if os.path.exists(config_path):
            if include_filename:
                return config_path
            else:
                return current_dir
        
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            # Reached the root directory without finding the file
            raise FileNotFoundError("config.yml not found in any parent directory.")
        current_dir = parent_dir

def get_vars():
    """
    Loads variables from the config.yml file.

    Returns:
        dict: A dictionary containing the configuration variables.
    """
    config_path = get_config_file_path()
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
