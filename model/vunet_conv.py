import functools
import warnings

import torch
import torch.nn as nn

from .baseNet import BaseNet
from .modules import BaseModule, Conv2d, ConvTranspose2d
from .blindsrlocal import IDADepthwiseConv

act_lrelu = nn.LeakyReLU()
act_relu = nn.ReLU()


class SFTLayer(BaseModule):
    def __init__(self, cond_dim, n_in_out, activation, sn, reduction_ratio=4):
        super(SFTLayer, self).__init__('SFTLayer')

        reduce_dim = (cond_dim + n_in_out) // reduction_ratio

        self.convs_gamma = nn.ModuleList([
            Conv2d(cond_dim, reduce_dim, 1, sn=sn, activation=None),
            Conv2d(reduce_dim, n_in_out, 1, sn=sn, activation=activation)
        ])

        self.convs_beta = nn.ModuleList([
            Conv2d(cond_dim + 1, reduce_dim, 1, sn=sn, activation=None),
            Conv2d(reduce_dim, n_in_out, 1, sn=sn, activation=activation)
        ])

        self.spatial_gamma = IDADepthwiseConv(
                channels_in_x=n_in_out, channels_in_map=cond_dim,
                channels_mid=(n_in_out + cond_dim) // 2,
                channels_out=n_in_out)


    def forward(self, x, noise_z):
        b, _, h, w = x.shape

        if noise_z.dim() == 2:
            noise_z = noise_z.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)  # [B, 1, H, W]
            noise_z_sm = noise_z.unsqueeze(2).unsqueeze(3).repeat(1, 1, h // 2, w // 2)
        elif noise_z.dim() == 4:
            if noise_z.shape[-2] != h // 2 or noise_z.shape[-1] != w // 2:
                noise_z_sm = nn.functional.interpolate(noise_z, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
            else:
                noise_z_sm = noise_z

            if noise_z.shape[-2] != h or noise_z.shape[-1] != w:
                noise_z = nn.functional.interpolate(noise_z, size=(h, w), mode='bilinear', align_corners=True)

        y = noise_z
        for i in range(len(self.convs_gamma)):
            y = self.convs_gamma[i](y)
        gamma = y

        random_z = torch.randn([b, 1, h, w], device=x.device)
        y = torch.cat((noise_z, random_z), dim=1)
        for i in range(len(self.convs_beta)):
            y = self.convs_beta[i](y)
        beta = y

        ret = x * (1 + gamma) + beta + self.spatial_gamma([x, noise_z_sm])
        return ret


class SFTResBlock(BaseModule):
    def __init__(self, n_in_out, kernel_size, cond_dim, activation=None, sn=False):
        super(SFTResBlock, self).__init__('SFTResBlock')

        self.sft1 = SFTLayer(cond_dim, n_in_out, activation=activation, sn=sn)
        self.conv1 = Conv2d(n_in_out, n_in_out, kernel_size, activation=activation, sn=sn)
        self.sft2 = SFTLayer(cond_dim, n_in_out, activation=activation, sn=sn)
        self.conv2 = Conv2d(n_in_out, n_in_out, kernel_size, activation=activation, sn=sn)

    def forward(self, x, cond):
        y = x
        y = self.sft1(y, cond)
        y = self.conv1(y)
        y = self.sft2(y, cond)
        y = self.conv2(y)
        return y + x


class VUNet(BaseNet):
    def __init__(self, filters, proj_dim_noise, norm=None, sn=False, name='VUNet'):
        super(VUNet, self).__init__(name)

        N_RB1 = 4
        N_RB2 = 3
        N_RB3 = 2

        K1 = filters
        K2 = K1 * 2
        K3 = K2 * 2
        ksize = 3

        cond_dim = proj_dim_noise

        self.map_head = Conv2d(cond_dim, cond_dim, 3)

        self.conv_first = Conv2d(3, K1, ksize, sn=sn, activation=None)

        SFTResBlock_ = functools.partial(SFTResBlock, cond_dim=cond_dim, activation=act_lrelu)

        self.blocks1 = nn.ModuleList([SFTResBlock_(K1, ksize, sn=sn) for _ in range(N_RB1)])

        self.conv_down1 = Conv2d(K1, K2, 4, stride=2, sn=sn, activation=None)

        self.blocks2 = nn.ModuleList([SFTResBlock_(K2, ksize, sn=sn) for _ in range(N_RB2)])

        self.conv_down2 = Conv2d(K2, K3, 4, stride=2, sn=sn, activation=None)

        self.blocks3 = nn.ModuleList([SFTResBlock_(K3, ksize, sn=sn) for _ in range(N_RB3)])

        self.conv_up2 = ConvTranspose2d(K3, K2, 4, stride=2, activation=None, sn=sn)

        self.conv_skip2 = Conv2d(K2 + K2, K2, ksize, sn=sn, activation=act_lrelu)
        self.blocks2_ = nn.ModuleList([SFTResBlock_(K2, ksize, sn=sn) for _ in range(N_RB2)])

        self.conv_up1 = ConvTranspose2d(K2, K1, 4, stride=2, activation=None, sn=sn)

        self.conv_skip1 = Conv2d(K1 + K1, K1, ksize, sn=sn, activation=act_lrelu)
        # self.blocks1_ = nn.ModuleList([SFTResBlock_(K1, ksize, sn=sn) for _ in range(N_RB1)])
        # See technical report for the reason of modifying the last block
        self.blocks1_ = nn.ModuleList([SFTResBlock_(K1, ksize, sn=sn) for _ in range(N_RB1 - 1)])
        self.blocks1_.append(SFTResBlock(K1, ksize, cond_dim, activation=act_relu, sn=sn))

        self.conv_last = Conv2d(K1, 3, 1, sn=sn, activation=act_lrelu)

    def forward(self, clean, noise_z):
        inp = clean * 2 - 1  # [-1, 1]          # [B, 3, H, W]
        y = self.conv_first(inp)                # [B, 64, H, W]
        noise_z = self.map_head(noise_z)

        for i in range(len(self.blocks1)):
            y = self.blocks1[i](y, noise_z)
        skip1 = y

        y = self.conv_down1(y)                  # [B, 128, H // 2, W // 2]
        for i in range(len(self.blocks2)):
            y = self.blocks2[i](y, noise_z)
        skip2 = y

        y = self.conv_down2(y)                  # [B, 256, H // 4, W // 4]
        for i in range(len(self.blocks3)):
            y = self.blocks3[i](y, noise_z)
        y = self.conv_up2(y)                    # [B, 128, H // 2, W // 2]

        y = self.conv_skip2(torch.cat((y, skip2), dim=1))
        for i in range(len(self.blocks2_)):
            y = self.blocks2_[i](y, noise_z)
        y = self.conv_up1(y)                    # [B, 64, H, W]

        y = self.conv_skip1(torch.cat((y, skip1), dim=1))
        for i in range(len(self.blocks1_)):
            y = self.blocks1_[i](y, noise_z)

        y = self.conv_last(y)                   # [B, 3, H, W]
        return y


class VUNetPro(BaseNet):
    def __init__(self, filters, proj_dim_noise, norm=None, sn=False, name='VUNetPro'):
        super(VUNetPro, self).__init__(name)

        N_RB1 = 4
        N_RB2 = 3
        N_RB3 = 2
        N_RB4 = 2
        N_RB5 = 2

        K1 = filters
        K2 = K1 * 2
        K3 = K2 * 2
        K4 = K3 * 2
        K5 = K4 * 2
        ksize = 3

        cond_dim = proj_dim_noise

        self.map_head = Conv2d(cond_dim, cond_dim, 3)

        self.conv_first = Conv2d(3, K1, ksize, sn=sn, activation=None)

        SFTResBlock_ = functools.partial(SFTResBlock, cond_dim=cond_dim, activation=act_lrelu)

        self.blocks1 = nn.ModuleList([SFTResBlock_(K1, ksize, sn=sn) for _ in range(N_RB1)])
        self.conv_down1 = Conv2d(K1, K2, 4, stride=2, sn=sn, activation=None)

        self.blocks2 = nn.ModuleList([SFTResBlock_(K2, ksize, sn=sn) for _ in range(N_RB2)])
        self.conv_down2 = Conv2d(K2, K3, 4, stride=2, sn=sn, activation=None)

        self.blocks3 = nn.ModuleList([SFTResBlock_(K3, ksize, sn=sn) for _ in range(N_RB3)])
        self.conv_down3 = Conv2d(K3, K4, 4, stride=2, sn=sn, activation=None)

        self.blocks4 = nn.ModuleList([SFTResBlock_(K4, ksize, sn=sn) for _ in range(N_RB4)])
        self.conv_down4 = Conv2d(K4, K5, 4, stride=2, sn=sn, activation=None)

        self.blocks5 = nn.ModuleList([SFTResBlock_(K5, ksize, sn=sn) for _ in range(N_RB5)])

        self.conv_up4 = ConvTranspose2d(K5, K4, 4, stride=2, activation=None, sn=sn)
        self.conv_skip4 = Conv2d(K4 + K4, K4, ksize, sn=sn, activation=act_lrelu)
        self.blocks4_ = nn.ModuleList([SFTResBlock_(K4, ksize, sn=sn) for _ in range(N_RB4)])

        self.conv_up3 = ConvTranspose2d(K4, K3, 4, stride=2, activation=None, sn=sn)
        self.conv_skip3 = Conv2d(K3 + K3, K3, ksize, sn=sn, activation=act_lrelu)
        self.blocks3_ = nn.ModuleList([SFTResBlock_(K3, ksize, sn=sn) for _ in range(N_RB3)])

        self.conv_up2 = ConvTranspose2d(K3, K2, 4, stride=2, activation=None, sn=sn)
        self.conv_skip2 = Conv2d(K2 + K2, K2, ksize, sn=sn, activation=act_lrelu)
        self.blocks2_ = nn.ModuleList([SFTResBlock_(K2, ksize, sn=sn) for _ in range(N_RB2)])

        self.conv_up1 = ConvTranspose2d(K2, K1, 4, stride=2, activation=None, sn=sn)
        self.conv_skip1 = Conv2d(K1 + K1, K1, ksize, sn=sn, activation=act_lrelu)
        # self.blocks1_ = nn.ModuleList([SFTResBlock_(K1, ksize, sn=sn) for _ in range(N_RB1)])
        # See technical report for the reason of modifying the last block
        self.blocks1_ = nn.ModuleList([SFTResBlock_(K1, ksize, sn=sn) for _ in range(N_RB1 - 1)])
        self.blocks1_.append(SFTResBlock(K1, ksize, cond_dim, activation=act_relu, sn=sn))

        self.conv_last = Conv2d(K1, 3, 1, sn=sn, activation=act_lrelu)

    def forward(self, clean, noise_z):
        inp = clean * 2 - 1  # [-1, 1]          # [B, 3, H, W]
        y = self.conv_first(inp)                # [B, 64, H, W]
        noise_z = self.map_head(noise_z)

        for i in range(len(self.blocks1)):
            y = self.blocks1[i](y, noise_z)
        skip1 = y
        y = self.conv_down1(y)                  # [B, 128, H // 2, W // 2]

        for i in range(len(self.blocks2)):
            y = self.blocks2[i](y, noise_z)
        skip2 = y
        y = self.conv_down2(y)                  # [B, 256, H // 4, W // 4]

        for i in range(len(self.blocks3)):
            y = self.blocks3[i](y, noise_z)
        skip3 = y
        y = self.conv_down3(y)                  # [B, 512, H // 8, W // 8]

        for i in range(len(self.blocks4)):
            y = self.blocks4[i](y, noise_z)
        skip4 = y
        y = self.conv_down4(y)                  # [B, 1024, H // 16, W // 16]

        for i in range(len(self.blocks5)):
            y = self.blocks5[i](y, noise_z)

        y = self.conv_up4(y)                    # [B, 512, H // 8, W // 8]
        y = self.conv_skip4(torch.cat((y, skip4), dim=1))
        for i in range(len(self.blocks4_)):
            y = self.blocks4_[i](y, noise_z)

        y = self.conv_up3(y)                    # [B, 256, H // 4, W // 4]
        y = self.conv_skip3(torch.cat((y, skip3), dim=1))
        for i in range(len(self.blocks3_)):
            y = self.blocks3_[i](y, noise_z)

        y = self.conv_up2(y)                    # [B, 128, H // 2, W // 2]
        y = self.conv_skip2(torch.cat((y, skip2), dim=1))
        for i in range(len(self.blocks2_)):
            y = self.blocks2_[i](y, noise_z)

        y = self.conv_up1(y)                    # [B, 64, H, W]

        y = self.conv_skip1(torch.cat((y, skip1), dim=1))
        for i in range(len(self.blocks1_)):
            y = self.blocks1_[i](y, noise_z)

        y = self.conv_last(y)                   # [B, 3, H, W]
        return y


class BasicWrapper(nn.Module):
    def __init__(self, cls=VUNet,
                 n_filter=64, proj_dim_noise=8,
                 loc_emb_dim=4, emb_dim=256,
                 *args, **kwargs):
        '''
        @param cls: the class of the network
        @param n_filter: the number of filters in the network's layers
        @param proj_dim_noise: the dimension of the condition. 
        @param loc_emb_dim: the dimension of the local distoriton embedding
        @param emb_dim: the dimension of the global distoriton embedding
        '''
        super(BasicWrapper, self).__init__()
        self.net = cls(filters=n_filter, proj_dim_noise=proj_dim_noise, *args, **kwargs)
        self.conv = nn.Conv2d(loc_emb_dim + emb_dim, proj_dim_noise, 3, padding=1)

    def forward(
            self, x: torch.Tensor, k_v: torch.Tensor, 
            dmap: torch.Tensor, pos_emb: torch.Tensor = None):
        '''
        The parameter names are defined so, for compatibility reasons. 

        @param x: the input image
        @param k_v: the global distortion embedding
        @param dmap: the local distortion embedding
        @param pos_emb: the positional embedding, IGNORE in this wrapper.
        '''
        if pos_emb is not None:
            # Warn gently to the user
            warnings.warn("pos_emb is not used in this wrapper.")

        # assert x.shape[1] == 3, "The input image should not be accompanied by random state in this implementation"
        if x.shape[1] != 3:
            assert x.shape[1] == 4, "Wrong input dimension"
            warnings.warn("Random state added but ignored.  This is allowed, because we need to facilitate the calculation of diversity loss.")
            x = x[:, :3, ...]
        # Becasue the this VUNet already has embedded mechanism to add random 
        # states, and we done need to "improve" it right now.


        k_v = k_v.unsqueeze(2).unsqueeze(3).repeat(1, 1, dmap.shape[2], dmap.shape[3])
        cond = self.conv(torch.cat((dmap, k_v), dim=1))
        return self.net(x, cond)
