import json
from typing import Callable, List, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from IQA_pytorch import DISTS
from kornia.filters import gaussian_blur2d as gblur
from piq import information_weighted_ssim, psnr, ssim, vif_p
from piq.functional import gaussian_filter
from piq.ssim import _reduce
from torch import nn
from kornia.color import rgb_to_lab

from model import common

def get_color_losses(color_loss_args: str, n_GPUs=-1):
    '''Get color loss function

    The check for whether color loss is enabled is shofted to the trainer. 
    '''
    color_loss = []
    color_loss_w = []
    color_loss_name = []
    for closs_raw in color_loss_args.strip(",").split(","):
        clossw, closs = closs_raw.split("*")
        clossw = float(clossw)
        closs = closs.lower()
        color_loss_w.append(clossw)
        color_loss_name.append(closs)
        if closs == "lab":
            loss_fn = ColorLoss(to_lab=True, normalize_minmax=True).cuda()
            loss_fn.eval()
            color_loss.append(loss_fn.cuda())
        elif closs == "rgb":
            loss_fn = ColorLoss(to_lab=False, normalize_minmax=False).cuda()
            loss_fn.eval()
            color_loss.append(loss_fn.cuda())
        elif closs == "rgbn":
            loss_fn = ColorLoss(to_lab=False, normalize_minmax=True).cuda()
            loss_fn.eval()
            color_loss.append(loss_fn.cuda())
        elif closs == "resnet_softmax":
            from .resnet_latent import ResNetDist
            assert n_GPUs > 0
            resnet_dist = ResNetDist(apply_softmax=True).cuda()
            if n_GPUs > 1:
                resnet_dist = nn.DataParallel(resnet_dist, range(n_GPUs))
                color_loss.append(lambda x, y: resnet_dist(x, y).mean())
            else:
                color_loss.append(resnet_dist)
        else:
            assert False, f'No color loss with the name {closs} available'
        print(f"[*] Using color loss: {closs}, weight: {clossw}")
    return color_loss, color_loss_w, color_loss_name


class ColorLoss(nn.Module):
    '''
    Note that the normalization method is different from the previous version
    '''
    def __init__(self, to_lab=False, normalize_minmax=False):
        super(ColorLoss, self).__init__()
        self.to_lab = to_lab
        self.normalize_minmax = normalize_minmax
        channel = 2 if to_lab else 3
        self._gaussian = common.GaussionSmoothLayer(channel=channel, kernel_size=45, sigma=50)
        for param in self._gaussian.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = (x + 1) / 2                 # [-1, 1] -> [0, 1]
        y = (y + 1) / 2
        if self.normalize_minmax:
            xmin = x.amin(dim=[1, 2, 3], keepdim=True)
            xmax = x.amax(dim=[1, 2, 3], keepdim=True)
            ymin = y.amin(dim=[1, 2, 3], keepdim=True)
            ymax = y.amax(dim=[1, 2, 3], keepdim=True)
            x = (x - xmin) / (xmax - xmin)
            y = (y - ymin) / (ymax - ymin)
        if self.to_lab:
            x = rgb_to_lab(x)[:, 1:3, :, :] / 128       # [-128, 127] -> [-1, 1]
            y = rgb_to_lab(y)[:, 1:3, :, :] / 128
        x = self._gaussian(x)
        y = self._gaussian(y)

        loss = (x - y).abs().mean()
        return loss

def test_color_losses():
    # Test get_color_losses function
    color_loss_args = "1*lab,2*rgb,3*rgbn"
    color_loss, color_loss_w, color_loss_name = get_color_losses(color_loss_args, n_GPUs=1)
    assert len(color_loss) == 3
    assert len(color_loss_w) == 3
    assert len(color_loss_name) == 3

    # Test ColorLoss class
    color_loss_fn = color_loss[0]
    x = torch.randn(1, 3, 224, 224).cuda()
    y = torch.randn(1, 3, 224, 224).cuda()
    loss = color_loss_fn(x, y)
    assert loss is not None

if __name__ == "__main__":
    test_color_losses()