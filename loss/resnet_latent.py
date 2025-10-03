import torch
from torch import nn

class ResNetDist(nn.Module):
    def __init__(self, apply_softmax=True):
        super(ResNetDist, self).__init__()
        self.resnet18 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        self.apply_softmax = apply_softmax

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        '''
        @param x: images (b, c, h, w)
        @param y: images (b, c, h, w)

        @returns the distance between softmax(resnet18(x)) and softmax(resnet18(y)) 
        or between resnet18(x) and resnet18(y), depending on whether 
        apply_softmax is set
        '''
        x = self.resnet18(x)
        y = self.resnet18(y)

        if self.apply_softmax:
            x = torch.nn.functional.softmax(x, dim=1)
            y = torch.nn.functional.softmax(y, dim=1)

        return torch.abs(x - y).mean(dim=-1)

