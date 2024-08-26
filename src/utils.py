import os
import joblib
import json
import yaml

def get_config():
    """
    Retrieves the configuration from the config.yaml file.

    Returns:
        dict: The configuration dictionary.
    """
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)