import yaml
import os

def get_config_file_path(filename="config.yml", include_filename=True):
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
        config_path = os.path.join(current_dir, filename)
        if os.path.exists(config_path):
            if include_filename:
                return config_path
            else:
                return current_dir
        
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            # Reached the root directory without finding the file
            raise FileNotFoundError(f"{filename} not found in any parent directory.")
        current_dir = parent_dir
    
def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])

def get_vars(filename='config.yml'):
    """
    Loads variables from the config.yml file.

    Returns:
        dict: A dictionary containing the configuration variables.
    """
    config_file_path = get_config_file_path(filename)
    yaml.SafeLoader.add_constructor('!join', join)

    if config_file_path:
        with open(config_file_path, 'r') as file:
            return yaml.safe_load(file)
    return None
