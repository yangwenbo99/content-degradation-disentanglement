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

from model import common
from .LPIPSvgg import LPIPSvgg
from .color import ColorLoss

def get_latent_losses(loss_args: str, n_GPUs=-1):
    '''
    @param n_GPUs: the number of GPUs for constructing DataParallel.  
                   Not actually used now. 
    '''
    losses = []
    loss_weights = []
    loss_names = []
    for loss_raw in loss_args.strip(",").split(","):
        loss_weight, loss_type = loss_raw.split("*")
        if loss_weight.find("+") != -1:
            loss_weight_A = float(loss_weight.split("+")[0])
            loss_weight_B = float(loss_weight.split("+")[1])
        else:
            loss_weight_A = loss_weight_B = float(loss_weight)
        loss_type = loss_type.lower()
        loss_weights.append((loss_weight_A, loss_weight_B))
        loss_names.append(loss_type)
        if loss_type == "l1":
            losses.append(nn.L1Loss().cuda())
        elif loss_type == "mse":
            losses.append(nn.MSELoss().cuda())
        else:
            raise ValueError(f"[*] Unknown loss type: {loss_type}")

        print(f"[*] Using loss type: {loss_type}, weight: {loss_weight}")

    return losses, loss_weights, loss_names

