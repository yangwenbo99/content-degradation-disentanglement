import abc

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def get_norm1d(norm, n_ch, affine=True):
    if norm is None:
        ret = None
    elif norm == 'bn':
        ret = nn.BatchNorm1d(n_ch)
    elif norm == 'ln':
        ret = nn.LayerNorm(n_ch, elementwise_affine=affine)
    return ret


def get_norm2d(norm, n_ch, affine=True):
    if norm is None:
        ret = None
    elif norm == 'in':
        ret = nn.InstanceNorm2d(n_ch, affine=affine)
    elif norm == 'bn':
        ret = nn.BatchNorm2d(n_ch)
    return ret


def get_norm3d(norm, n_ch, affine=True):
    if norm is None:
        ret = None
    elif norm == 'in':
        ret = nn.InstanceNorm3d(n_ch, affine=affine)
    elif norm == 'bn':
        ret = nn.BatchNorm3d(n_ch)
    return ret


class BaseModule(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, name):
        super(BaseModule, self).__init__()
        self.name = name


class Linear(BaseModule):
    def __init__(self, n_in, n_out, bias=False, activation=nn.ReLU(), norm=None, sn=False, affine=True):
        super(Linear, self).__init__('Linear')

        self.norm = get_norm1d(norm, n_in, affine)
        self.activation = activation

        self.linear = nn.Linear(n_in, n_out, bias=bias)
        if sn:
            self.linear = spectral_norm(self.linear)

    def forward(self, x):
        y = x
        if self.norm is not None:
            y = self.norm(y)
        if self.activation is not None:
            y = self.activation(y)
        y = self.linear(y)
        return y


class Conv2d(BaseModule):
    def __init__(self, n_in, n_out, kernel_size, stride=1, dilation=1, bias=False, activation=nn.ReLU(), norm=None, sn=False, affine=True):
        super(Conv2d, self).__init__('Conv2d')

        self.norm = get_norm2d(norm, n_in, affine)
        self.activation = activation

        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        if stride == 2 and kernel_size % stride == 0:
            padding = kernel_size // 2 - 1
        self.conv2d = nn.Conv2d(n_in, n_out, kernel_size, stride, padding, dilation, bias=bias)
        if sn:
            self.conv2d = spectral_norm(self.conv2d)

    def forward(self, x):
        y = x
        if self.norm is not None:
            y = self.norm(y)
        if self.activation is not None:
            y = self.activation(y)
        y = self.conv2d(y)
        return y


class Conv3d(BaseModule):
    def __init__(self, n_in, n_out, kernel_size, stride=1, dilation=1, bias=False, activation=nn.ReLU(), norm=None, sn=False, affine=True):
        super(Conv3d, self).__init__('Conv3d')

        self.norm = get_norm3d(norm, n_in, affine)
        self.activation = activation

        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        if stride == 2 and kernel_size % stride == 0:
            padding = kernel_size // 2 - 1

        kernel_size_d = 3
        stride_d = 1
        padding_d = (dilation * (kernel_size_d - 1)) // 2

        kernel_size = (kernel_size_d, kernel_size, kernel_size)
        stride = (stride_d, stride, stride)
        padding = (padding_d, padding, padding)
        self.conv3d = nn.Conv3d(n_in, n_out, kernel_size, stride, padding, dilation, bias=bias)
        if sn:
            self.conv3d = spectral_norm(self.conv3d)

    def forward(self, x):
        y = x
        if self.norm is not None:
            y = self.norm(y)
        if self.activation is not None:
            y = self.activation(y)
        y = self.conv3d(y)
        return y


class ConvTranspose2d(BaseModule):
    def __init__(self, n_in, n_out, kernel_size, stride=2, dilation=1, bias=False, activation=nn.ReLU(), norm=None, sn=False, affine=True):
        super(ConvTranspose2d, self).__init__('ConvTranspose2d')

        self.norm = get_norm2d(norm, n_in, affine)
        self.activation = activation

        padding = ((kernel_size - 1) + 1 - stride) // 2
        self.tconv2d = nn.ConvTranspose2d(n_in, n_out, kernel_size, stride, padding=padding, dilation=dilation, bias=bias)
        if sn:
            self.tconv2d = spectral_norm(self.tconv2d)

    def forward(self, x):
        y = x
        if self.norm is not None:
            y = self.norm(y)
        if self.activation is not None:
            y = self.activation(y)
        y = self.tconv2d(y)
        return y


class ConvTranspose3d(BaseModule):
    def __init__(self, n_in, n_out, kernel_size, stride=2, dilation=1, bias=False, activation=nn.ReLU(), norm=None, sn=False, affine=True):
        super(ConvTranspose3d, self).__init__('ConvTranspose3d')

        self.norm = get_norm3d(norm, n_in, affine)
        self.activation = activation

        padding = ((kernel_size - 1) + 1 - stride) // 2

        kernel_size_d = 3
        stride_d = 1
        padding_d = (kernel_size_d - 1) // 2

        kernel_size = (kernel_size_d, kernel_size, kernel_size)
        stride = (stride_d, stride, stride)
        padding = (padding_d, padding, padding)
        self.tconv3d = nn.ConvTranspose3d(n_in, n_out, kernel_size, stride, padding=padding, dilation=dilation, bias=bias)
        if sn:
            self.tconv3d = spectral_norm(self.tconv3d)

    def forward(self, x):
        y = x
        if self.norm is not None:
            y = self.norm(y)
        if self.activation is not None:
            y = self.activation(y)
        y = self.tconv3d(y)
        return y
