import copy
from importlib import import_module
from typing import Dict, Union

import yaml
from timm.data.loader import MultiEpochsDataLoader
from torch.utils.data.dataloader import DataLoader

def get_loader(
        config: Dict[str, Union[str, int]],
        is_train=True,
        ):
    assert 'patch_size' in config, 'Put patch_size in the dataset config'
    dataset_name: str = config['type']
    module = import_module('data.' + dataset_name.lower())  # load the right dataset loader module
    dataset = getattr(module, dataset_name.replace("_", ""))(config, train=is_train)  # load the dataset, args.data_train is the  dataset name
    if is_train:
        return MultiEpochsDataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            pin_memory=True,
            # pin_memory=False,  # Let's see whether this stops the explosion of memory usage... no
            num_workers=config['n_threads'],
            drop_last=True,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=config['n_threads'],
            drop_last=False,
        )
