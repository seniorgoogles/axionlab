from includes_ml2 import *

def _load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
