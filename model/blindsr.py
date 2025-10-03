from importlib import import_module
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Optional, Union

import model.common as common
from moco.builder import MoCo
from model.mirnet_v2 import MIRNet_v2


def make_model(args):
    return BlindSR(args)

class DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(DA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(channels_in, channels_in, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(channels_in, channels_in * self.kernel_size * self.kernel_size, bias=False)
        )
        self.conv = common.default_conv(channels_in, channels_out, 1)
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
            nn.Conv2d(channels_in, channels_in // reduction, 1, 1, 0, bias=False),
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
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(DAB, self).__init__()

        self.da_conv1 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.da_conv2 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        :param x[2]: positional embedding (if presented): 1 * ? * H * W
        '''

        out = self.relu(self.da_conv1(x))
        out = self.relu(self.conv1(out))
        out = self.relu(self.da_conv2([out, x[1]]))
        out = self.conv2(out) + x[0]

        return out


class DAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_blocks):
        super(DAG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            DAB(conv, n_feat, kernel_size, reduction)
            for _ in range(n_blocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        :param x[2]: positional embedding (if presented): 1 * ? * H * W
        '''
        res = x[0]
        if len(x) > 2:   # Positional embedding
            if x[2].shape[0] > 1:   # Positional embedding
                res += x[2][0].unsqueeze(0)
            else:
                res += x[2]
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1]])
        res = self.body[-1](res)
        res = res + x[0]

        return res


class DASR(nn.Module):
    def __init__(self, args: dict, conv=common.default_conv,
                 moco_enc_dim: Optional[int] = None,
                 model_config: Optional[Dict[str, Union[int, str]]] = None,
                 ):
        super(DASR, self).__init__()
        '''
        @param args: the whole config dict, not only the 'network' section.  
                     The missing options will be pulled from the config. dict.
        '''
        if moco_enc_dim is None:
            moco_enc_dim = args['network']['netE']['moco_enc_dim']
        if model_config is None:
            model_config = args['network']['netG']

        self.n_groups = 5
        n_blocks = 5
        n_feats = 64
        kernel_size = model_config['kernel_size']
        reduction = 8
        scale = int(model_config['scale'])
        self.model_config = model_config

        # RGB mean for DIV2K
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(255.0, rgb_mean, rgb_std)
        # self.add_mean = common.MeanShift(255.0, rgb_mean, rgb_std, 1)

        # head module
        input_size = 4 if args['network']['add_random_state'] else 3
        modules_head = [conv(input_size, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        # compress
        self.compress = nn.Sequential(
            nn.Linear(moco_enc_dim, n_feats, bias=False),
            nn.LeakyReLU(0.1, True)
        )

        # body
        modules_body = [
            DAG(common.default_conv, n_feats, kernel_size, reduction, n_blocks)
            for _ in range(self.n_groups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # tail
        modules_tail = [common.Upsampler(conv, scale, n_feats, act=False),
                        conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

        # high freq distortion
        if model_config['hf']:
            self.hf = nn.ModuleList()
            for _ in range(self.n_groups):
                self.hf.append(nn.Sequential(nn.Linear(64, 16), nn.LeakyReLU(0.1, True), nn.Linear(16, 2)))

    def forward(self, x, k_v, pos_emb: torch.Tensor = None):
        k_v = self.compress(k_v)

        # sub mean
        # x = self.sub_mean(x)

        # head
        x = self.head(x)

        # body
        res = x
        for i in range(self.n_groups):
            if self.model_config['hf']:
                hf_param = self.hf[i](k_v).unsqueeze(2).unsqueeze(3).unsqueeze(4)
                noise = hf_param[:, 0] + torch.randn_like(res) * hf_param[:, 1]
                res = res + noise.to(res.device)
            if pos_emb is None:
                res = self.body[i]([res, k_v])
            else:
                res = self.body[i]([res, k_v, pos_emb])
        res = self.body[-1](res)
        res = res + x

        # tail
        x = self.tail(res)

        # add mean
        # x = self.add_mean(x)

        return x


class Encoder(nn.Module):
    def __init__(self, dim, out_dim, k):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=k, stride=int((k - 1) / 2), padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.1, True),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out  # temporarily returned to earlier version for a brief test...
        return out, out


class LPE(nn.Module):
    def __init__(self, dim=256, out_dim=256, k=13):
        super(LPE, self).__init__()

        # distortion encoder
        LPE_p = [
            common.Conv2dBlock(3, dim // 4, k, int((k - 1) / 2), padding=0, norm='batch', activation='lrelu', pad_type='zero'),
            common.Conv2dBlock(dim // 4, dim // 4, 3, 1, padding=1, norm='batch', activation='lrelu', pad_type='zero'),
            common.Conv2dBlock(dim // 4, dim // 2, 3, 1, padding=1, norm='batch', activation='lrelu', pad_type='zero'),
            common.Conv2dBlock(dim // 2, dim, 3, 1, padding=1, norm='batch', activation='lrelu', pad_type='zero'),
        ]

        LPE_l = [
            common.Conv2dBlock(3, dim // 4, 3, 1, padding=1, norm='batch', activation='lrelu', pad_type='zero'),
            common.Conv2dBlock(dim // 4, dim // 4, 1, 1, padding=0, norm='batch', activation='lrelu', pad_type='zero'),
            common.Conv2dBlock(dim // 4, dim // 2, 3, 2, padding=1, norm='batch', activation='lrelu', pad_type='zero'),
            common.Conv2dBlock(dim // 2, dim // 2, 1, 1, padding=0, norm='batch', activation='lrelu', pad_type='zero'),
            common.Conv2dBlock(dim // 2, dim, 3, 1, padding=1, norm='batch', activation='lrelu', pad_type='zero'),
            common.Conv2dBlock(dim, dim, 1, 1, padding=0, norm='batch', activation='lrelu', pad_type='zero'),
        ]

        self.LPE_p = nn.Sequential(*LPE_p)
        self.LPE_l = nn.Sequential(*LPE_l)
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.last_mlp = nn.Sequential(nn.Linear(dim * 2, dim), nn.LeakyReLU(0.1, True), nn.Linear(dim, out_dim))

    def forward(self, x):
        x_p = self.LPE_p(x)                                                  # [B, 256, 21, 21]
        x_l = self.LPE_l(x)                                                  # [B, 256, 128, 128]

        x_p = self.avg(x_p).squeeze(-1).squeeze(-1)                          # [B, 256]
        x_l = self.avg(x_l).squeeze(-1).squeeze(-1)                          # [B, 256]
        fea = torch.cat((x_p, x_l), dim=1)                                  # [B, 512]
        out = self.last_mlp(fea)                                           # [B, 256]

        # return fea, out  # temporarily returned to earlier version for a brief test...
        return out, out


# WARNING: THE FOLLOWING CODE IS NOT USED, AND THUS NOT TESTED
# I can guarentee that it will not work, because we changed how to pass the 
# configs.
class BlindSR(nn.Module):
    def __init__(self, args):
        super(BlindSR, self).__init__()

        # Generator
        if args.GB.lower() == 'dasr':
            self.GB = DASR(args)
        elif args.GB.lower() == 'mirv2':
            self.GB = MIRNet_v2()
        else:
            raise NotImplementedError(f"[!] Distortion synthesis network {args.GB} not implemented")

        # Encoder
        self.E = MoCo(base_encoder=eval(args.moco_enc),
                      dim=args.moco_enc_dim,
                      moco_enc_kernel=args.moco_enc_kernel,
                      K=args.num_neg,
                      distance=args.moco_dist)

    def forward(self, x, gt=None, pos_emb=None,
                random_states: List[torch.Tensor] =None,
                im_hard_neg: torch.Tensor=None):
        if self.training:
            x_query = x[:, 0, ...]                          # b, c, h, w
            x_key = x[:, 1, ...]                            # b, c, h, w

            # degradation-aware represenetion learning
            x_query_fea, logits, labels = self.E(x_query, x_key, im_hard_neg)

            # degradation-aware SR
            if gt is None:
                assert random_states is None, 'Not supported for generation without gt'
                out = self.G(x_query, x_query_fea)
                return out
            else:
                if random_states is None:
                    # The following part is written in this way, because only one of
                    # the generators accepts pos_emb
                    if pos_emb is not None:
                        out = self.GB(gt, x_query_fea, pos_emb)
                    else:
                        out = self.GB(gt, x_query_fea)
                    return out, logits, labels
                else:
                    outs = [ ]
                    for random_state in random_states:
                        gtr = torch.concat((gt, random_state), dim=1)
                        # We only support DASR as GB now.
                        out = self.GB(gtr, x_query_fea, pos_emb)
                        outs.append(out)
                    return outs, logits, labels

        else:

            # degradation-aware represenetion learning
            x_query_fea = self.E(x_query, x_query)

            # degradation-aware SR
            if gt is None:
                assert random_state is None, 'Not supported for generation without gt'
                out = self.G(x_query, x_query_fea)
            else:
                if pos_emb is not None:
                    out = self.GB(gt, x_query_fea, pos_emb)
                else:
                    out = self.GB(gt, x_query_fea)

            return out
