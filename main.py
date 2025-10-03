#!/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, Any, Union, List
import warnings
import traceback
import sys

import torch
from torch import nn

import model.combined_model
from yaml_option import load_config, merge_config
import utility
from data import get_loader
from model.combined_model import CombinedModel
# import loss   # We should not use the "Loss" class in our code
from trainer import Trainer

def _get_loader_config(set_name: str, data_config: Dict[str, Union[str, dict]]):
    '''
    @param data_config: should be config['data']
    '''
    out = dict(data_config[set_name])
    for key, value in data_config.items():
        if key not in out and not isinstance(value, dict):
            out[key] = value
    return out

def _test_and_add_key(keys: List[str], data_config: Dict[str, Union[str, dict]], key: str):
    if (key in data_config and 
        ('enabled' not in data_config[key] or data_config[key]['enabled'])):
        keys.append(key)

def showwarning(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    str_ = warnings.formatwarning(message, category, filename, lineno, line)
    log.write(str_)
    traceback.print_stack(file=log)

if __name__ == '__main__':
    warnings.showwarning = showwarning  # Make sure warning messages are printed with stack trace
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='User config file path')
    args = parser.parse_args()
    default_config_path = Path(__file__).parent / 'options/default.yaml'
    user_config_path = args.config if args.config else default_config_path
    config = load_config(default_config_path, user_config_path)
    torch.manual_seed(config['seed'])

    # The 'test_only' option is structured in the 'log' section in the 
    # config file, but the logger needs it. 
    config['log']['test_only'] = config['test']['test_only']
    logger = utility.Logger(config['log'])  # This is originally named as "checkpoint"

    if logger.ok:
        loaders = { }
        if config['test']['test_only']:
            # test mode
            loader_keys = [ ]  # TODO
            _test_and_add_key(loader_keys, config['data'], 'test')
        else:
            # training mode
            loader_keys = []
            _test_and_add_key(loader_keys, config['data'], 'train')
        for key in loader_keys:
            loaders[key] = get_loader(
                    config=_get_loader_config(key, config['data']),
                    is_train=(not config['test']['test_only']))
        model = CombinedModel(config).cuda()
        if config['hardware']['n_GPUs'] > 1:
            model = nn.DataParallel(model, range(config['hardware']['n_GPUs']))
        # The format of the following function call defines the API for trainer
        trainer = Trainer(config, model, loaders, logger) 
        if not config['test']['test_only']:
            # training mode
            trainer.train()
        else:
            # test mode
            trainer.test()


    
