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


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt: Dict[str, Union[str, int]], train=True, *args, **kwargs):
        """Initialize this dataset class.

        @param train: If set, this dataset is for training.  Otherwise, it is for testing.
        @param opt: stores all the experiment flags; should be a dict.  
            It should contains the following contents:
            - type (str): The type of the dataset. Should be 'Single_DataSet' for this class.
            - list_type (str): The type of the list. Should be 'paired_csv' for paired training or 'unpaired_csv' for unpaired training
            - n_threads (int): The number of threads for loading the dataset.
            - batch_size (int): The batch size for the data loader.
            - crop_size (int): The size of the crop patch.
            - patch_size (int): The final patch size. The images will be resized to this size.
            - use_lmdb (bool, optional, assumed to be False): Whether to use LMDB for storing the dataset.
            - db_basedir (str): The base directory for the LMDB database. This must be presented if use_lmdb is True.
            - root (str): The root directory of the dataset.  At this moment, only useful if not using LMDB. 
            - csv (str): The path to the CSV file containing the dataset.
            - csv_A_column (str): The column name in the CSV file for image A. Should be 'name' for paired training.
            - csv_B_column (str): The column name in the CSV file for image B. Should be 'ref_name' for paired training and empty for unpaired. 
            - csv_path_is_relative (bool): Whether the paths in the CSV file are relative to the root directory.
            - suppress_crop (bool): If True, the images will not be cropped.
        """
        BaseDataset.__init__(self, opt)

        self.use_db = opt['use_lmdb'] if 'use_lmdb' in opt else False
        if self.use_db:
            print(f"[*] Using lmdb: {opt['db_basedir']}")
            self.use_db = True
            self.db_basedir = opt['db_basedir']
        self.list_type = opt['list_type']
        self.use_relative_path = opt['csv_path_is_relative'] if 'csv_path_is_relative' in opt else False
        self.suppress_crop = opt['suppress_crop'] if 'suppress_crop' in opt else False  # NOTE: reserved for future use
        self.single_image = 'single_image' in opt and opt['single_image']

        self.min_size = opt['min_size'] if 'min_size' in opt else -1

        if 'csv' in self.list_type:  #type: ignore
            csv_file: str = opt['csv']
            if not train:
                csv_file = csv_file.replace("train", "test")
            self.df = pd.read_csv(csv_file)
            self.A_paths = self.df[opt['csv_A_column']].tolist()
            self.B_paths = ''
            if opt['csv_B_column']:
                assert self.list_type == 'paired_csv'
                print(f"[*] Using GT: {opt['csv_B_column']}")
                self.B_paths = self.df[opt['csv_B_column']].tolist()
            else:
                assert self.list_type == 'unpaired_csv'
        else:
            assert False, 'Invalid list type'

        ts = []
        if (not self.suppress_crop and opt['crop_size'] != opt['patch_size']):
            ts.append(transforms.Resize(opt['patch_size']))
        ts += [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
        self.transform = transforms.Compose(ts)

    def __init_db__(self):
        """Initialize the database."""
        db = lmdb.open(self.db_basedir, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = db.begin(write=False)

    def __get_img__(self, img_path: str) -> Image.Image:
        """
        Get the image from the given path.

        @param img_path: The path of the image.
        @return: The image in RGB format.
        """
        if self.use_db:
            imgbuf = self.txn.get(img_path.encode())
            img = Image.open(io.BytesIO(imgbuf)).convert('RGB')
        else:
            if self.use_relative_path:
                img_path = os.path.join(self.root, img_path)
            img = Image.open(img_path).convert('RGB')
        return img

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        @param index: a random integer for data indexing
        @return: a dictionary that contains the following key-value pairs:
                 - 'A_img': [2, 3, h, w] two images patches cropped from an image in domain A. 
                 - 'A_path': the path of the image A
                 If used for paired learning, the following keys will also 
                 be presented:
                 - 'B_img': [3, h, w] an image in domain B
                 - 'B_path': the path of the image B
        """
        if self.use_db and (not hasattr(self, "txn") or self.txn is None):
            self.__init_db__()

        A_path = self.A_paths[index]
        A_img_ori = self.__get_img__(A_path)

        if self.min_size > 0 and A_img_ori.height < self.min_size:
            # If the image is smaller than the minimum size, resize it
            # In our experiments, this is only needed for FFHQ
            assert A_img_ori.width == A_img_ori.height
            A_img_ori = F.resize(A_img_ori, self.min_size)

        # random crop
        crop_size = self.opt['crop_size']
        crop_i_1 = torch.randint(0, A_img_ori.height - crop_size + 1, size=(1,)).item()
        crop_j_1 = torch.randint(0, A_img_ori.width - crop_size + 1, size=(1,)).item()
        crop_i_2 = torch.randint(0, A_img_ori.height - crop_size + 1, size=(1,)).item()
        crop_j_2 = torch.randint(0, A_img_ori.width - crop_size + 1, size=(1,)).item()

        if not self.suppress_crop:
            A_img1 = self.transform(F.crop(A_img_ori, crop_i_1, crop_j_1, crop_size, crop_size))
            if not self.single_image:
                A_img2 = self.transform(F.crop(A_img_ori, crop_i_2, crop_j_2, crop_size, crop_size))
        else:
            A_img1 = A_img2 = self.transform(A_img_ori)
        if not self.single_image:
            A_img = torch.stack((A_img1, A_img2), 0)
        else:
            A_img = A_img1

        res = {"A_img": A_img, "A_path": A_path}

        if self.B_paths:
            B_path = self.B_paths[index]
            B_img_ori = self.__get_img__(B_path)

            if B_img_ori.size != A_img_ori.size:
                B_img_ori = B_img_ori.resize(A_img_ori.size)

            if not self.suppress_crop:
                B_img = self.transform(F.crop(B_img_ori, crop_i_1, crop_j_1, crop_size, crop_size))
            else:
                B_img = self.transform(B_img_ori)
            res.update({"B_img": B_img, "B_path": B_path})

        return res

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)

if __name__ == "__main__":
    # Testing code
    import matplotlib.pyplot as plt

    def test_single_dataset():
        # Define the options for the SingleDataset
        opt = {
            'type': 'Single_DataSet',
            'list_type': 'paired_csv',
            'n_threads': 1,
            'batch_size': 1,
            'crop_size': 128,
            'patch_size': 128,
            'use_lmdb': False,
            'root': '/mnt/hdd/datasets/ffhq',
            'csv': '/mnt/hdd/datasets/ffhq/filelist.csv',
            'csv_A_column': 'name',
            'csv_B_column': 'ref_name',
            'csv_path_is_relative': True,
            'suppress_crop': False
        }

        # Initialize the dataset
        dataset = SingleDataset(opt)

        # Get the first image and its cropped version
        data = dataset[0]
        A_img = data['A_img'][0]  # get the first cropped image
        A_img = A_img.permute(1, 2, 0)  # change from CxHxW to HxWxC for visualization

        # Display the original image
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow((A_img + 1) / 2)  # denormalize
        plt.title('Original Image')

        # Display the cropped image
        A_img_cropped = data['A_img'][1]  # get the second cropped image
        A_img_cropped = A_img_cropped.permute(1, 2, 0)  # change from CxHxW to HxWxC for visualization
        plt.subplot(1, 2, 2)
        plt.imshow((A_img_cropped + 1) / 2)  # denormalize
        plt.title('Cropped Image')

        plt.show()
    test_single_dataset()
