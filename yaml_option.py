import yaml
from typing import Any, Dict

def merge_config(default_config: Dict[str, Any], user_config: Dict[str, Any]
                 ) -> Dict[str, Any]:
    for key, value in user_config.items():
        if isinstance(value, dict) and key in default_config:
            # Recursively merge dictionaries
            merge_config(default_config[key], value)
        else:
            default_config[key] = value
    return default_config

def load_config(default_config_path: str, user_config_path: str) -> Dict[str, Any]:
    print(default_config_path)
    with open(default_config_path, 'r') as stream:
        try:
            default_config = yaml.safe_load(stream) or {}
        except yaml.YAMLError as exc:
            print(exc)
            default_config = {}

    with open(user_config_path, 'r') as stream:
        try:
            user_config = yaml.safe_load(stream) or {}
        except yaml.YAMLError as exc:
            print(exc)
            user_config = {}

    # Merge user config with default config
    merged_config = merge_config(default_config, user_config)

    # Warn for unused config items
    def check_unused_keys(default: Dict[str, Any], user: Dict[str, Any], path: str = ''):
        if path.startswith('data.'):
            return
        for key in user:
            if key not in default:
                print(f"Warning: Unused key {path + key} in the user config")
            elif isinstance(user[key], dict) and isinstance(default.get(key), dict):
                check_unused_keys(default[key], user[key], path + key + '.')

    check_unused_keys(default_config, user_config)

    return merged_config

if __name__ == '__main__':
    default_config_path = 'default_config.yaml'  # replace with your default config file path
    user_config_path = 'user_config.yaml'  # replace with your user config file path
    config = load_config(default_config_path, user_config_path)
