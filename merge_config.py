#!/bin/env python3

'''
python merge_config.py user_config.yaml patch.yaml --replacements old1:new1 old2:new2
'''

import yaml
import argparse
from typing import Any, Dict

from yaml_option import merge_config, load_config

def replace_strings_in_config(config: Dict[str, Any], replacements: Dict[str, str]) -> Dict[str, Any]:
    for key, value in config.items():
        if isinstance(value, str):
            for old, new in replacements.items():
                if old in value:
                    config[key] = value.replace(old, new)
        elif isinstance(value, dict):
            replace_strings_in_config(value, replacements)
    return config

def set_nested_key(config: Dict[str, Any], key_path: str, value: Any,
                   dtype: str) -> None:
    if dtype:
        if dtype == 'int':
            value = int(value)
        elif dtype == 'float':
            value = float(value)
        elif dtype == 'bool':
            value = value.lower() in ['true', '1']
        elif dtype == 'str':
            value = str(value)
        else:
            raise ValueError(f'Unknown dtype: {dtype}')

    keys = key_path.split('.')
    for key in keys[:-1]:
        config = config.setdefault(key, {})
    config[keys[-1]] = value


def main():
    parser = argparse.ArgumentParser(description='Merge YAML configs.')
    parser.add_argument('default_config_path', type=str, help='Path to the default config file.')
    parser.add_argument('user_config_path', type=str, help='Path to the user config file.')
    parser.add_argument('-r', '--replacements', nargs='+', type=str, help='Pairs of strings to replace in the format old:new.')
    parser.add_argument('-o', '--output', type=str, help='Path to the output file.')
    parser.add_argument('-O', '--override', nargs='+', type=str, help='Options to override in the format key:value.')

    args = parser.parse_args()

    print(f"{args.default_config_path} used, with patches from {args.user_config_path}")
    config = load_config(args.default_config_path, args.user_config_path)

    if args.replacements:
        replacements = dict(replacement.split(':') for replacement in args.replacements)
        config = replace_strings_in_config(config, replacements)

    if args.override:
        for override in args.override:
            split = override.split(':')
            key_path, value = split[0], split[1]
            dtype = split[2] if len(split) > 2 else None
            set_nested_key(config, key_path, value, dtype)

    if args.output:
        with open(args.output, 'w') as f:
            yaml.dump(config, f)
    else:
        print(yaml.dump(config))

if __name__ == '__main__':
    main()
