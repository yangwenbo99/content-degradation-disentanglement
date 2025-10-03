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


class ImageReranger:
    def __init__(self, body: nn.Module):
        self.body = body

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        x = torch.clip((x + 1) / 2, min=0, max=1)
        y = torch.clip((y + 1) / 2, min=0, max=1)
        return self.body(x, y)



def get_degrade_losses(degrade_loss_args: str, n_GPUs=-1):
    '''
    @param n_GPUs: the number of GPUs for constructing DataParallel.  u
                   Only used in some loss functions
    '''
    degrade_losses = []
    degrade_loss_weights = []
    degrade_loss_names = []
    for dloss_raw in degrade_loss_args.strip(",").split(","):
        dlossw, dloss = dloss_raw.split("*")
        if dlossw.find("+") != -1:
            dlossA_w = float(dlossw.split("+")[0])
            dlossB_w = float(dlossw.split("+")[1])
        else:
            dlossA_w = dlossB_w = float(dlossw)
        dloss = dloss.lower()
        degrade_loss_weights.append((dlossA_w, dlossB_w))
        degrade_loss_names.append(dloss)
        if dloss == "l1":
            degrade_losses.append(nn.L1Loss().cuda())
        elif dloss == "mse":
            degrade_losses.append(nn.MSELoss().cuda())
        elif dloss == "iw_ssim":
            from piq import InformationWeightedSSIMLoss
            degrade_losses.append(ImageReranger(InformationWeightedSSIMLoss(kernel_size=9,
                                                              scale_weights=torch.tensor([0.0448, 0.2856, 0.3001, 0.2363]))))
        elif dloss == "ssim":
            from piq import SSIMLoss
            degrade_losses.append(ImageReranger(SSIMLoss(downsample=False)))
        elif dloss == "msim3":
            degrade_losses.append(ImageReranger(msim3_loss))
        elif dloss == "noise":
            degrade_losses.append(noise_consistency_loss)
        elif dloss == "noise2":
            degrade_losses.append(ImageReranger(noise_consistency2_loss))
        elif dloss == "noise_energy":
            degrade_losses.append(noise_energy_distance)
        elif dloss == "noise_energy_color":
            degrade_losses.append(
                lambda x, y: noise_energy_distance(
                    x, y, underline=energy_distance_v))
        elif dloss == "noise_energy_spatial":
            degrade_losses.append(noise_energy_distance_spatial)
        elif dloss == "noise_energy_spatial_nonneg":
            degrade_losses.append(
                lambda x, y: noise_energy_distance_spatial(x, y).clamp_min(0))
        elif dloss == "gssim":
            degrade_losses.append(global_sim)
        elif dloss == "dists":
            assert n_GPUs > 0
            dists = DISTS_Loss().cuda()
            dists = nn.DataParallel(dists, range(n_GPUs))
            degrade_losses.append(lambda x, y: dists(x, y).mean())
        elif dloss == "lpips":
            lpips = LPIPS_Loss().cuda()
            lpips = nn.DataParallel(lpips, range(n_GPUs))
            degrade_losses.append(lambda x, y: lpips(x, y).mean())
        elif dloss == "rgbn":
            loss_fn = ColorLoss(to_lab=False, normalize_minmax=True).cuda()
            loss_fn.eval()
            degrade_losses.append(loss_fn.cuda())
        else:
            raise ValueError(f"[*] Unknown degrade loss: {dloss}")

        print(f"[*] Using degrade loss: {dloss}, weight: {dlossw}")

    return degrade_losses, degrade_loss_weights, degrade_loss_names



def _msim3_per_channel(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor,
                       data_range: Union[float, int] = 1., k1: float = 0.01,
                       k2: float = 0.03, non_neg=False, log=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if x.size(-1) < kernel.size(-1) or x.size(-2) < kernel.size(-2):
        raise ValueError(f'Kernel size can\'t be greater than actual input size. '
                         f'Input size: {x.size()}. Kernel size: {kernel.size()}')

    c1 = k1 ** 2
    c2 = k2 ** 2
    n_channels = x.size(1)
    kn = kernel.shape[2]
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=kn // 2, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=kn // 2, groups=n_channels)

    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding=kn // 2, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding=kn // 2, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=kn // 2, groups=n_channels) - mu_xy

    if non_neg or log:
        mu_xy = mu_xy.clamp_min(0)
        sigma_xy = sigma_xy.clamp_min(0)

    # Contrast sensitivity (CS) with alpha = beta = gamma = 1.
    if not log:
        cs = (2. * sigma_xy + c2) / (2 * torch.max(sigma_xx, sigma_yy) + c2)
    else:
        cs = torch.log(2. * sigma_xy + c2) - torch.log(2 * torch.max(sigma_xx, sigma_yy) + c2)

    # Structural similarity (SSIM)
    if not log:
        ss = (2. * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs
    else:
        ss = torch.log(2. * mu_xy + c1) - torch.log(mu_xx + mu_yy + c1) + cs

    ssim_val = ss.mean(dim=(-1, -2))
    cs = cs.mean(dim=(-1, -2))
    return ssim_val, cs


def msim3(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11, kernel_sigma: float = 1.5,
          data_range: Union[int, float] = 1., reduction: str = 'mean', full: bool = False,
          downsample: bool = True, k1: float = 0.01, k2: float = 0.03, non_neg=False, log=False) -> List[torch.Tensor]:
    '''
    modified SSIM (3)
    '''
    assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'

    x = x / float(data_range)
    y = y / float(data_range)

    # Averagepool image if the size is large enough
    f = max(1, round(min(x.size()[-2:]) / 256))
    if (f > 1) and downsample:
        x = F.avg_pool2d(x, kernel_size=f)
        y = F.avg_pool2d(y, kernel_size=f)

    kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1).to(y)
    _compute_ssim_per_channel = _msim3_per_channel
    ssim_map, cs_map = _compute_ssim_per_channel(
        x=x, y=y, kernel=kernel, data_range=data_range, k1=k1, k2=k2,
        non_neg=non_neg, log=log)
    ssim_val = ssim_map.mean(1)
    cs = cs_map.mean(1)

    ssim_val = _reduce(ssim_val, reduction)
    cs = _reduce(cs, reduction)

    if full:
        return [ssim_val, cs]

    return ssim_val


def msim3_loss(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11, kernel_sigma: float = 1.5,
               data_range: Union[int, float] = 1., reduction: str = 'mean', full: bool = False,
               downsample: bool = True, k1: float = 0.01, k2: float = 0.03, non_neg=False, log=False) -> torch.Tensor:
    '''
    Computes the MSIM3 loss between two images x and y.

    TODO:  TEST:
    k2 may be larger than the original, for numerical stability
    '''
    ssim_val = msim3(x, y, kernel_size, kernel_sigma, data_range, reduction,
                     full, downsample, k1, k2, non_neg=non_neg, log=log)
    if not log:
        loss = 1 - ssim_val
    else:
        loss = - ssim_val
    return loss.mean()


def _noise_consistency_per_channel(
        x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor,
        data_range: Union[float, int] = 1.,
        moment_weights=[1.0, 1.0, 1.0, 1.0], p=1.0) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if x.size(-1) < kernel.size(-1) or x.size(-2) < kernel.size(-2):
        raise ValueError(f'Kernel size can\'t be greater than actual input size. '
                         f'Input size: {x.size()}. Kernel size: {kernel.size()}')

    n_channels = x.size(1)
    kn = kernel.shape[2]

    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=kn // 2, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=kn // 2, groups=n_channels)

    x_c = torch.ones_like(x)
    y_c = torch.ones_like(y)
    x2 = (x - mu_x) ** 2
    y2 = (y - mu_y) ** 2
    maps = []
    for moment_weight in moment_weights:
        x_c = x_c * x2
        y_c = y_c * y2
        mu_x_c = F.conv2d(x_c, weight=kernel, stride=1, padding=kn // 2, groups=n_channels)
        mu_y_c = F.conv2d(y_c, weight=kernel, stride=1, padding=kn // 2, groups=n_channels)
        dmap = torch.abs(mu_x_c - mu_y_c) ** p
        maps.append(dmap)

    for i in range(len(maps)):
        maps[i] = maps[i].mean(dim=(-1, -2))
    return maps


def noise_consistency_loss(
        x: torch.Tensor, y: torch.Tensor,
        kernel_size: int = 11, kernel_sigma: float = 1.5,
        data_range: Union[int, float] = 1.,
        full: bool = False, downsample: bool = True,
        moment_weights=[1.0, 1, 1, 1], p=1.0, mean=True) -> List[torch.Tensor]:
    assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'

    x = x / float(data_range)
    y = y / float(data_range)

    # Averagepool image if the size is large enough
    f = max(1, round(min(x.size()[-2:]) / 256))
    if (f > 1) and downsample:
        x = F.avg_pool2d(x, kernel_size=f)
        y = F.avg_pool2d(y, kernel_size=f)

    kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1).to(y)
    _compute_per_channel = _noise_consistency_per_channel
    consistency_scores_channels = _compute_per_channel(
        x=x, y=y, kernel=kernel, data_range=data_range,
        moment_weights=moment_weights, p=p)
    consistency_scores = [x.mean(1) for x in consistency_scores_channels]
    consistency_score = torch.zeros_like(consistency_scores[0])
    for w, s in zip(moment_weights, consistency_scores):
        consistency_score += w * s

    if full:
        return consistency_scores, consistency_scores_channels

    if mean:
        return consistency_score.mean()
    else:
        return consistency_score


def _noise_consistency2_per_channel(
        x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor,
        data_range: Union[float, int] = 1.,
        moment_weights=[1.0, 1.0, 1.0, 1.0], p=1.0) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if x.size(-1) < kernel.size(-1) or x.size(-2) < kernel.size(-2):
        raise ValueError(f'Kernel size can\'t be greater than actual input size. '
                         f'Input size: {x.size()}. Kernel size: {kernel.size()}')

    n_channels = x.size(1)
    kn = kernel.shape[2]

    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=kn // 2, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=kn // 2, groups=n_channels)

    x_c = torch.ones_like(x)
    y_c = torch.ones_like(y)
    x2 = (x - mu_x) ** 2
    y2 = (y - mu_y) ** 2
    scores = []
    for moment_weight in moment_weights:
        x_c = x_c * x2
        y_c = y_c * y2
        scores.append(ssim(x_c, y_c))
        # sim = (2 * x_c * y_c + 0.01) / (x_c**2 + y_c**2 + 0.01)
        # scores.append(sim.mean((-1, -2, -3)))

    return scores


def noise_consistency2(
        x: torch.Tensor, y: torch.Tensor,
        kernel_size: int = 11, kernel_sigma: float = 1.5,
        data_range: Union[int, float] = 1.,
        downsample: bool = True,
        moment_weights=[1.0, 0.3, 0.1, 0.03], p=1.0, mean=True) -> List[torch.Tensor]:
    assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'

    x = x / float(data_range)
    y = y / float(data_range)

    # Averagepool image if the size is large enough
    f = max(1, round(min(x.size()[-2:]) / 256))
    if (f > 1) and downsample:
        x = F.avg_pool2d(x, kernel_size=f)
        y = F.avg_pool2d(y, kernel_size=f)

    kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1).to(y)
    _compute_per_channel = _noise_consistency2_per_channel
    consistency_scores = _compute_per_channel(
        x=x, y=y, kernel=kernel, data_range=data_range,
        moment_weights=moment_weights, p=p)
    consistency_score = torch.zeros_like(consistency_scores[0])
    for w, s in zip(moment_weights, consistency_scores):
        consistency_score += w * s

    if mean:
        return consistency_score.mean()
    else:
        return consistency_score


def noise_consistency2_loss(
        x: torch.Tensor, y: torch.Tensor,
        kernel_size: int = 11, kernel_sigma: float = 1.5,
        data_range: Union[int, float] = 1.,
        downsample: bool = True,
        moment_weights=[1.0, 0.3, 0.1, 0.03], p=1.0, mean=True) -> torch.Tensor:
    '''
    Computes the MSIM3 loss between two images x and y.
    '''
    consistency = noise_consistency2(x, y, kernel_size, kernel_sigma, data_range, downsample, moment_weights, mean)
    loss = sum(moment_weights) - consistency
    return loss.mean()


def energy_distance(x: torch.Tensor, y: torch.Tensor, mean=True) -> torch.Tensor:
    """
    Compute the square of energy distance between two tensors x and y.

    Each pixel is considered as three real numbers and they are calculated
    indepedently.
    """
    assert x.shape == y.shape, f'Shape mismatch: {x.shape} vs {y.shape}'
    b, c, h, w = x.shape
    x = x.reshape((b * c, h * w))
    y = y.reshape((b * c, h * w))
    l = h * w
    if l % 2 != 0:
        l = (l // 2) * 2
        x = x[:, :l]
        y = y[:, :l]
    x_indexes = torch.randperm(l, device=x.device)
    y_indexes = torch.randperm(l, device=y.device)
    x = x[:, x_indexes]
    y = y[:, y_indexes]
    # The first half and the last half are assumed to be indepedent
    xy = torch.abs(x - y).mean(dim=-1)
    xx = torch.abs(x[:, :l // 2] - x[:, l // 2:]).mean(dim=-1)
    yy = torch.abs(y[:, :l // 2] - y[:, l // 2:]).mean(dim=-1)
    d = 2 * xy - xx - yy
    d = d.reshape((b, c))
    if mean:
        d = d.mean()
    return d


def energy_distance_v(x: torch.Tensor, y: torch.Tensor, mean=True) -> torch.Tensor:
    """
    Compute the square of energy distance between two tensors x and y.

    Each pixel is considered as a c-dimentional vector (c == 3 usually holds).
    """
    assert x.shape == y.shape, f'Shape mismatch: {x.shape} vs {y.shape}'
    b, c, h, w = x.shape
    x = x.reshape((b, c, h * w))
    y = y.reshape((b, c, h * w))
    l = h * w
    if l % 2 != 0:
        l = (l // 2) * 2
        x = x[:, :l]
        y = y[:, :l]
    x_indexes = torch.randperm(l, device=x.device)
    y_indexes = torch.randperm(l, device=y.device)
    x = x[:, :, x_indexes]
    y = y[:, :, y_indexes]
    # The first half and the last half are assumed to be indepedent
    xy = torch.linalg.vector_norm(x - y, ord=2, dim=1).mean(dim=-1)
    xx = torch.linalg.vector_norm(x[:, :, :l // 2] - x[:, :, l // 2:], ord=2, dim=1).mean(dim=-1)
    yy = torch.linalg.vector_norm(y[:, :, :l // 2] - y[:, :, l // 2:], ord=2, dim=1).mean(dim=-1)
    d = 2 * xy - xx - yy
    d = d.reshape((b,))
    if mean:
        d = d.mean()
    return d


def noise_energy_distance(x: torch.Tensor, y: torch.Tensor,
                          kernel_size: int = 11, kernel_sigma: float = 1.5,
                          mean=True, underline=energy_distance
                          ) -> torch.Tensor:
    '''
    Noise energy distance.  Works well for localtion indepedent noise
    The size is like 0.0xxx

    @param underline: either energy_distance or energy_distance_v
    '''
    b, c, h, w = x.shape
    kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(c, 1, 1, 1).to(y)
    kn = kernel_size
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=kn // 2, groups=c)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=kn // 2, groups=c)
    return underline(x - mu_x, y - mu_y, mean)


def energy_distance_spatial(x: torch.Tensor, y: torch.Tensor, mean=True) -> torch.Tensor:
    b, c, h, w = x.shape

    # Change x to (b * c, 4, (h//2) * (w//2) ) by spliting adjacent four pixels
    # to 4 channels
    x = x.unfold(2, 2, 2).unfold(3, 2, 2)   # (b, c, h//2, 2, w//2, 2)
    x = x.permute((0, 1, 3, 5, 2, 4))
    x = x.reshape(b * c, 4, (h // 2) * (w // 2))
    y = y.unfold(2, 2, 2).unfold(3, 2, 2)   # (b, c, h//2, 2, w//2, 2)
    y = y.permute((0, 1, 3, 5, 2, 4))
    y = y.reshape(b * c, 4, (h // 2) * (w // 2))

    x_e = x.view(b * c, 4, 1, (h // 2) * (w // 2))
    x_f = x.view(b * c, 1, 4, (h // 2) * (w // 2))
    y_e = y.view(b * c, 4, 1, (h // 2) * (w // 2))
    y_f = y.view(b * c, 1, 4, (h // 2) * (w // 2))
    xy = torch.abs(x_e - y_f).mean(-1)  # (b * c, 4, 4)
    xx = torch.abs(x_e - x_f).mean(-1)  # (b * c, 4, 4)
    yy = torch.abs(y_e - y_f).mean(-1)  # (b * c, 4, 4)

    # Remove xy's diagonal entries, i.e. set xy[n, m, m] = 0
    # This is important as we assume independence of noise from nearby
    # entries, but the same location of x and y, this apparently does not
    # hold.
    # xx and yy's should already be 0.
    xy_d = xy.diagonal(dim1=-2, dim2=-1)
    xy_d[:] = 0
    xx_d = xy.diagonal(dim1=-2, dim2=-1)
    xx_d[:] = 0
    yy_d = xy.diagonal(dim1=-2, dim2=-1)
    yy_d[:] = 0

    d = 2 * xy - xx - yy

    d = d.mean(dim=(-1, -2))

    d = d.reshape((b, c))
    if mean:
        d = d.mean()
    return d


def noise_energy_distance_spatial(x: torch.Tensor, y: torch.Tensor,
                                  kernel_size: int = 11, kernel_sigma: float = 1.5,
                                  mean=True) -> torch.Tensor:
    b, c, h, w = x.shape
    kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(c, 1, 1, 1).to(y)
    kn = kernel_size
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=kn // 2, groups=c)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=kn // 2, groups=c)
    return energy_distance_spatial(x - mu_x, y - mu_y, mean)


class DISTS_Loss(nn.Module):
    '''
    The adapter class for using DISTS (https://github.com/dingkeyan93/DISTS)
    as a loss function in the trainer.
    '''

    def __init__(self):
        super(DISTS_Loss, self).__init__()
        self.dists = DISTS()

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        return self.dists(y, x, as_loss=True, resize=False)


class LPIPS_Loss(nn.Module):
    '''
    The adapter class for using LPIPS (https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/LPIPSvgg.py)
    as a loss function in the trainer.
    '''

    def __init__(self):
        super(LPIPS_Loss, self).__init__()
        self.lpips = LPIPSvgg()

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        return self.lpips(y, x, as_loss=True)


def global_sim(x: torch.Tensor, y: torch.Tensor, as_loss=True) -> torch.Tensor:
    c1 = 1e-6
    c2 = 1e-6

    mu_x = x.mean(dim=(-1, -2), keepdim=True)
    mu_y = y.mean(dim=(-1, -2), keepdim=True)
    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_xx = (x ** 2).mean(dim=(-1, -2), keepdim=True) - mu_xx
    sigma_yy = (y ** 2).mean(dim=(-1, -2), keepdim=True) - mu_yy
    sigma_xy = (x * y).mean(dim=(-1, -2), keepdim=True) - mu_xy
    score1 = (2 * mu_xy + c1) / (mu_xx + mu_yy + c1)
    score2 = (2 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)
    if as_loss:
        return 1 - (0.5 * score1 + 0.5 * score2).mean()
    else:
        return (0.5 * score1 + 0.5 * score2).mean()
