import abc
import os

import torch

# from utils import make_dir


class BaseNet(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, name):
        super(BaseNet, self).__init__()
        self.name = name

    def save(self, dir_path, file_name):
        assert False, "Do not use the same method in this class"
        if file_name is None:
            file_name = self.name

        make_dir(dir_path)
        checkpoint_path = os.path.join(dir_path, file_name + '.pt')
        torch.save(self.state_dict(), checkpoint_path)

    def load(self, dir_path, file_name):
        if file_name is None:
            file_name = self.name

        checkpoint_path = os.path.join(dir_path, file_name + '.pt')
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        if file_name == "disc":
            try:
                self.load_state_dict(ckpt)
            except RuntimeError:
                # ignore the module model.0.weight
                print("[!] Disc: Ignore the module model.0.weight")
                new_ckpt = {k: v for k, v in ckpt.items() if k != 'model.0.weight'}
                self.load_state_dict(new_ckpt, strict=False)
        else:
            self.load_state_dict(ckpt)
