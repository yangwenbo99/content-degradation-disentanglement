import argparse
from pathlib import Path
from typing import Dict, Any, Union
import warnings
import traceback
import sys

import torch
from torch import nn

# Add the parent directory to the Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


from model.combined_model import CombinedModel
from yaml_option import load_config
import utility
from data import get_loader
from trainer import Trainer

def showwarning(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    str_ = warnings.formatwarning(message, category, filename, lineno, line)
    log.write(str_)
    traceback.print_stack(file=log)

def load_model_from_config(config_path: Union[str, Path], use_cuda: bool = True) -> CombinedModel:
    warnings.showwarning = showwarning  # Make sure warning messages are printed with stack trace
    default_config_path = Path(__file__).parent.parent / 'options/default.yaml'
    user_config_path = config_path if config_path else default_config_path
    config = load_config(default_config_path, user_config_path)
    torch.manual_seed(config['seed'])

    # The 'test_only' option is structured in the 'log' section in the 
    # config file, but the logger needs it. 
    config['log']['test_only'] = config['test']['test_only']
    logger = utility.Logger(config['log'])  # This is originally named as "checkpoint"

    if logger.ok:
        model = CombinedModel(config)
        if use_cuda:
            model = model.cuda()
        if config['hardware']['n_GPUs'] > 1:
            model = nn.DataParallel(model, range(config['hardware']['n_GPUs']))
        return model, config, logger
    else:
        raise RuntimeError("Logger initialization failed.")

def load_config_(config_path: Union[str, Path]) -> Dict[str, Any]:
    warnings.showwarning = showwarning  # Make sure warning messages are printed with stack trace
    default_config_path = Path(__file__).parent.parent / 'options/default.yaml'
    user_config_path = config_path if config_path else default_config_path
    config = load_config(default_config_path, user_config_path)
    return config


def load_model_(config: Dict[str, Any], use_cuda: bool = True) -> CombinedModel:
    torch.manual_seed(config['seed'])

    config['log']['test_only'] = config['test']['test_only']
    logger = utility.Logger(config['log'])  # This is originally named as "checkpoint"

    model = CombinedModel(config)
    if logger.ok:
        model = CombinedModel(config)
        if use_cuda:
            model = model.cuda()
        if config['hardware']['n_GPUs'] > 1:
            model = nn.DataParallel(model, range(config['hardware']['n_GPUs']))
        return model, logger
    else:
        raise RuntimeError("Logger initialization failed.")

def load_trainer_from_config(model, config, logger, use_cuda: bool = True) -> Trainer:
    # model, config, logger = load_model_from_config(config_path, use_cuda)
    loaders = { }
    if config['test']['test_only']:
        # test mode
        loader_keys = []
        _test_and_add_key(loader_keys, config['data'], 'test')
    '''
    else:
        # training mode
        loader_keys = []
        _test_and_add_key(loader_keys, config['data'], 'train')
                '''
    for key in loader_keys:
        loaders[key] = get_loader(
                config=_get_loader_config(key, config['data']),
                is_train=(not config['test']['test_only']))
    trainer = Trainer(config, model, loaders, logger)
    return trainer

def _get_loader_config(set_name: str, data_config: Dict[str, Union[str, dict]]):
    '''
    @param data_config: should be config['data']
    '''
    out = dict(data_config[set_name])
    for key, value in data_config.items():
        if key not in out and not isinstance(value, dict):
            out[key] = value
    return out

def _test_and_add_key(keys: list, data_config: Dict[str, Union[str, dict]], key: str):
    if (key in data_config and 
        ('enabled' not in data_config[key] or data_config[key]['enabled'])):
        keys.append(key)

# Example usage in a Jupyter notebook:
# from load_model import load_model_from_config, load_trainer_from_config
# model, config, logger = load_model_from_config('path/to/config.yaml')
# trainer = load_trainer_from_config('path/to/config.yaml')
