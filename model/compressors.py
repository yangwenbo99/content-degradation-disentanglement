import torch
from torch import nn
from compressai.entropy_models.entropy_models import EntropyBottleneck

class CompressorHeadMLP(nn.Module):
    def __init__(self, channels, layers=2, mid_channels=-1):
        '''
        @params channels: number of channels in the input tensor, same as
                          "channels" in EntropyBottleneck
        @params layers: number of layers in the MLP.
        '''
        super().__init__()
        if mid_channels < 0:
            mid_channels = channels // 4

        self.layers = []
        for _ in range(layers):
            self.layers.append(
                MLPResidualBlock(channels, mid_channels)
            )
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class MLPResidualBlock(nn.Module):
    def __init__(self, channels, mid_channels=-1):
        super().__init__()
        if mid_channels < 0:
            mid_channels = channels // 4
        self.block = nn.Sequential(
            nn.Linear(channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, channels),
        )

    def forward(self, x):
        return x + self.block(x)
