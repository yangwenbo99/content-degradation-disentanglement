import io
import os
from typing import Dict, Union

import lmdb
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as transforms
import torchvision.transforms.functional as F

from .base_dataset import BaseDataset

class DummyDataset(BaseDataset):
    """This dataset class mimics the SingleDataset but returns random data for debugging purposes."""

    def __init__(self, opt: Dict[str, Union[str, int]], train=True, *args, **kwargs):
        """Initialize the DummyDataset class with the same parameters as SingleDataset."""
        BaseDataset.__init__(self, opt)
        self.crop_size = opt['crop_size']
        self.patch_size = opt['patch_size']
        self.batch_size = opt['batch_size']
        self.single_image = 'single_image' in opt and opt['single_image']
        self.num_images = 10000  # Assuming a fixed number of dummy images

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return a dummy data point with random tensors mimicking real data.

        @param index: a random integer for data indexing
        @return: a dictionary containing dummy image data and paths
        """
        A_img1 = torch.rand(3, self.crop_size, self.crop_size)
        A_img2 = torch.rand(3, self.crop_size, self.crop_size) if not self.single_image else A_img1
        A_img = torch.stack((A_img1, A_img2), 0) if not self.single_image else A_img1

        res = {"A_img": A_img, "A_path": f"dummy_path_A_{index}"}

        if 'csv_B_column' in self.opt and self.opt['csv_B_column']:
            B_img = A_img1 + 0.1 * torch.rand(3, self.crop_size, self.crop_size)
            res.update({"B_img": B_img, "B_path": f"dummy_path_B_{index}"})

        return res

    def __len__(self) -> int:
        """Return the total number of dummy images."""
        return self.num_images
