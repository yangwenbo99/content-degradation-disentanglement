# Learning Enriched Features for Fast Image Restoration and Enhancement
# Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
# IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
# https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args):
    return MIRNet_v2()  # TODO: pass parameters

class DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(DA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, self.channels_in * self.kernel_size * self.kernel_size, bias=False)
        )
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=1, padding=0, bias=True)
        self.ca = CA_layer(channels_in, channels_out, reduction)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x[0].size()

        # branch 1
        kernel = self.kernel(x[1]).view(-1, 1, self.kernel_size, self.kernel_size)
        out = self.relu(F.conv2d(x[0].view(1, -1, h, w), kernel, groups=b * c, padding=(self.kernel_size - 1) // 2))
        out = self.conv(out.view(b, -1, h, w))

        # branch 2
        out = out + self.ca(x)

        return out


class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(64, channels_in // reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        att = self.conv_du(x[1][:, :, None, None])

        return x[0] * att


class DAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction):
        super(DAB, self).__init__()

        self.da_conv1 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.da_conv2 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size, stride=1, padding=(kernel_size // 2), bias=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size, stride=1, padding=(kernel_size // 2), bias=True)
        self.conv3 = nn.Conv2d(n_feat, n_feat, kernel_size, stride=1, padding=(kernel_size // 2), bias=True)

        self.relu = nn.LeakyReLU(0.1, True)

        self.additive = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 32 * 32, bias=False)
        )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''

        out = self.relu(self.da_conv1(x))
        out = self.relu(self.conv1(out))

        add = F.interpolate(self.additive(x[1]).view(-1, 1, 32, 32), size=out.shape[-1])
        out = out + add
        out = self.conv2(out)

        out = self.relu(self.da_conv2([out, x[1]]))
        out = self.conv3(out) + x[0]

        return out


##########################################################################
# ---------- Selective Kernel Feature Fusion (SKFF) ----------

class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V


class ContextBlock(nn.Module):

    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )

        self.DAB = DAB(n_feat, kernel_size=3, reduction=8)

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x, k_v):
        x = self.DAB([x, k_v])

        # [N, C, 1, 1]
        context = self.modeling(x)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term

        return x

##########################################################################
# --------- Residual Context Block (RCB) ----------


class RCB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCB, self).__init__()

        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act,
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups)
        )

        self.act = act

        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x, k_v):
        res = self.body(x)
        res = self.act(self.gcnet(res, k_v))
        res += x
        return res


##########################################################################
# ---------- Resizing Modules ----------
class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels * chan_factor), 1, stride=1, padding=0, bias=bias)
        )

    def forward(self, x):
        return self.bot(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels // chan_factor), 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        )

    def forward(self, x):
        return self.bot(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


##########################################################################
# ---------- Multi-Scale Resiudal Block (MRB) ----------
class MRB(nn.Module):
    def __init__(self, n_feat, height, width, chan_factor, bias, groups):
        super(MRB, self).__init__()

        self.n_feat, self.height, self.width = n_feat, height, width

        self.dau_top = RCB(int(n_feat * chan_factor**0), bias=bias, groups=groups)
        self.dau_mid = RCB(int(n_feat * chan_factor**1), bias=bias, groups=groups)
        self.dau_bot = RCB(int(n_feat * chan_factor**2), bias=bias, groups=groups)

        self.down2 = DownSample(int((chan_factor**0) * n_feat), 2, chan_factor)
        self.down4 = nn.Sequential(
            DownSample(int((chan_factor**0) * n_feat), 2, chan_factor),
            DownSample(int((chan_factor**1) * n_feat), 2, chan_factor)
        )

        self.up21_1 = UpSample(int((chan_factor**1) * n_feat), 2, chan_factor)
        self.up21_2 = UpSample(int((chan_factor**1) * n_feat), 2, chan_factor)
        self.up32_1 = UpSample(int((chan_factor**2) * n_feat), 2, chan_factor)
        self.up32_2 = UpSample(int((chan_factor**2) * n_feat), 2, chan_factor)

        self.conv_out = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, bias=bias)

        # only two inputs for SKFF
        self.skff_top = SKFF(int(n_feat * chan_factor**0), 2)
        self.skff_mid = SKFF(int(n_feat * chan_factor**1), 2)

    def forward(self, x_k_v):
        x, k_v = x_k_v
        x_top = x.clone()
        x_mid = self.down2(x_top)
        x_bot = self.down4(x_top)

        x_top = self.dau_top(x_top, k_v)
        x_mid = self.dau_mid(x_mid, k_v)
        x_bot = self.dau_bot(x_bot, k_v)

        x_mid = self.skff_mid([x_mid, self.up32_1(x_bot)])
        x_top = self.skff_top([x_top, self.up21_1(x_mid)])

        x_top = self.dau_top(x_top, k_v)
        x_mid = self.dau_mid(x_mid, k_v)
        x_bot = self.dau_bot(x_bot, k_v)

        x_mid = self.skff_mid([x_mid, self.up32_2(x_bot)])
        x_top = self.skff_top([x_top, self.up21_2(x_mid)])

        out = self.conv_out(x_top)
        out = out + x

        return out

##########################################################################
# ---------- Recursive Residual Group (RRG) ----------


class RRG(nn.Module):
    def __init__(self, n_feat, n_MRB, height, width, chan_factor, bias=False, groups=1):
        super(RRG, self).__init__()
        modules_body = [MRB(n_feat, height, width, chan_factor, bias, groups) for _ in range(n_MRB)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias))
        self.body = nn.Sequential(*modules_body)

        self.n_MRB = n_MRB

    def forward(self, x_k_v):
        x, k_v = x_k_v
        res = x
        for i in range(self.n_MRB):
            res = self.body[i]([res, k_v])
        res = self.body[-1](res)
        res += x
        return res


##########################################################################
# ---------- MIRNet  -----------------------
class MIRNet_v2(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 n_feat=80,
                 chan_factor=1.5,
                 n_RRG=4,
                 n_MRB=2,
                 height=3,
                 width=2,
                 scale=1,
                 bias=False,
                 task=None
                 ):
        super(MIRNet_v2, self).__init__()

        self.task = task
        self.n_RRG = n_RRG
        RRG_groups = [1, 2, 4, 4, 4, 4]
        assert n_RRG <= len(RRG_groups)

        self.conv_in = nn.Conv2d(inp_channels, n_feat, kernel_size=3, padding=1, bias=bias)

        modules_body = []

        for i in range(n_RRG):
            modules_body.append(RRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=RRG_groups[i]))

        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1, bias=bias)

        # compress
        self.compress = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, inp_img, k_v):
        k_v = self.compress(k_v)

        shallow_feats = self.conv_in(inp_img)
        deep_feats = shallow_feats
        for i in range(self.n_RRG):
            deep_feats = self.body[i]([deep_feats, k_v])

        if self.task == 'defocus_deblurring':
            deep_feats += shallow_feats
            out_img = self.conv_out(deep_feats)

        else:
            out_img = self.conv_out(deep_feats)
            out_img += inp_img

        return out_img
