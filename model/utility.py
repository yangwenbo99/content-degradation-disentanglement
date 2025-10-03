import os
from typing import Union
from importlib import import_module

import torch
import torch.nn as nn

def get_model(args, set_parallel=True) -> nn.Module:
    '''Get a model from the directory.

    @param args: the command arguments
    @param set_parallel: if set, the model will automatically be set to 
                         DataParrallel if more than 1 GPU used. 
    '''

    device = device('cpu' if args.cpu else 'cuda')
    module = import_module('model.' + args.model)
    model: nn.Module = module.make_model(args).to(device)
    if set_parallel and not args.cpu and args.n_GPUs > 1:
        model = nn.DataParallel(model, range(args.n_GPUs))

def get_barebone(model: Union[nn.Module, nn.DataParallel]
                 ) -> nn.Module:
    '''Get the underlying model for modification, saving and loading
    '''
    if type(model) == nn.DataParallel:
        return model.module
    else:
        return model

