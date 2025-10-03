import torch
import torch.nn as nn
import torch.nn.functional as F

C_2 = 0.03 * 4

class DiversityLoss(nn.modules.loss._Loss):
    '''Diversity loss for generated examples
    '''
    def __init__(self):
        super(DiversityLoss, self).__init__()
        # Depthwise Gaussian kernel, with window size=11 and std=1.5
        self.conv = nn.Conv2d(3, 3, groups=3,
                              kernel_size=11, stride=1, padding=5,
                              padding_mode='reflect', bias=False)
        n = 11
        std = 1.5
        x = torch.linspace(-n // 2, n // 2, n)
        y = torch.linspace(-n // 2, n // 2, n)
        x, y = torch.meshgrid(x, y)
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * std ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, n, n)
        kernel = kernel.repeat(3, 1, 1, 1)
        self.conv.weight = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        '''
        @param x: distorted image generated with first set of random states
        @param y: distorted image generated with second set of random states
        '''
        mu_x = self.conv(x)
        mu_y = self.conv(y)
        mu_xx = mu_x ** 2
        mu_yy = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_xx = self.conv(x**2) - mu_xx
        sigma_yy = self.conv(y**2) - mu_yy
        sigma_xy = self.conv(x * y) - mu_xy

        cs = (2. * sigma_xy.abs() + C_2) / (sigma_xx + sigma_yy + C_2)
        return cs.mean()




