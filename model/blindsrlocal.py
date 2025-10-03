from importlib import import_module
from typing import List, Optional, Dict, Union

import torch
import torch.nn.functional as F
from torch import nn

import model.common as common
from moco.builder import MoCo
from model.mirnet_v2 import MIRNet_v2

from model.blindsr import DASR, DAB, DAG, DA_conv, Encoder, LPE
from model.iclr_compression import ImageCompressor

MAP_RATIO = 2 # the h' and w' of map are (1 / MAP_RATIO) * h or w

def make_model(args):
    return BlindSRLocal(args)

class IDADepthwiseConv(nn.Module):
    '''
    IDA's branch 1 (See the Appendix in the proposal for documentation)

    Not actually depthwise
    '''
    def __init__(self, channels_in_x: int, channels_in_map: int, 
                 channels_mid: int, 
                 channels_out: int, ds_factor=2):
        super(IDADepthwiseConv, self).__init__()
        self.transform_map = nn.Sequential(
                # nn.Conv2d(channels_in_map, channels_mid, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ConvTranspose2d(channels_in_map, channels_mid, kernel_size=MAP_RATIO, stride=MAP_RATIO, padding=0, bias=False),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(channels_mid, channels_mid, kernel_size=3, stride=ds_factor, padding=1, bias=False),
                )
        self.transform_x = nn.Conv2d(channels_in_x, channels_mid, kernel_size=3, stride=ds_factor, padding=1, bias=False)
        # self.deconv = nn.ConvTranspose2d(
                # channels_mid, channels_out,
                # kernel_size=3, stride=2, padding=1, output_padding=1)
        # Alternatively, 
        self.deconv = nn.ConvTranspose2d(
                channels_mid, channels_out,
                kernel_size=2, stride=ds_factor, padding=0, output_padding=0)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C' * H' * W'
        '''
        res = self.transform_x(x[0]) * self.transform_map(x[1])
        return self.deconv(res)


class IDAChannelwiseConv(nn.Module):
    '''
    IDA's branch 2 (Method B) (See the Appendix in the proposal for documentation)
    '''
    def __init__(
            self, channels_in: int, channels_mid: int, channels_out: int):
        super(IDAChannelwiseConv, self).__init__()
        self.transform_map = nn.Sequential(
                # nn.Conv2d(channels_in, channels_mid, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ConvTranspose2d(channels_in, channels_mid, kernel_size=MAP_RATIO, stride=MAP_RATIO, padding=0, bias=False),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(channels_mid, channels_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Sigmoid()
                )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C * H' * W'
        '''

        return x[0] * self.transform_map(x[1])


class IDA_conv(nn.Module):
    '''
    IDA_conv inside Inhomogeneous Degradation-aware convolution block

    Current channel_in_x and channel_out should be the same
    '''
    def __init__(self, channels_in_x, channels_in_map, 
                 channels_out, ds_factor=2):
        super(IDA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in_x = channels_in_x
        self.channels_mid = (channels_in_map + channels_out) // 2   # TODO: parameter tunning

        self.depthwise = IDADepthwiseConv(
                channels_in_x, channels_in_map, self.channels_mid, channels_out, 
                ds_factor)
        self.channelwise = IDAChannelwiseConv(channels_in_map, self.channels_mid, channels_out)


        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C * H' * W'
        '''
        return self.depthwise(x) + self.channelwise(x)


class IDAB(nn.Module):
    '''
    Inhomogeneous Degradation-aware convolution block; 
    Implementing Method B (See the Appendix in the proposal for documentation) 
    '''
    def __init__(self, n_feat, n_feat_map, kernel_size, reduction):
        super(IDAB, self).__init__()

        self.da_conv1 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.da_conv2 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.ida_conv1 = IDA_conv(channels_in_x=n_feat, channels_in_map=n_feat_map,
                                  channels_out=n_feat)
        self.ida_conv2 = IDA_conv(channels_in_x=n_feat, channels_in_map=n_feat_map,
                                  channels_out=n_feat)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        :param x[2]: local degradation map: B * C' * H' * W'
        '''

        # print('>>> (IDAB)', x[0].shape, x[2].shape)
        out = self.relu(self.da_conv1([x[0], x[1]]))
        out = self.relu(self.ida_conv1([out, x[2]]))
        out = self.relu(self.da_conv2([out, x[1]]))
        out = self.ida_conv2([out, x[2]]) + x[0]

        return out


class IDAG(nn.Module):
    def __init__(self, conv, n_feat, n_feat_map, kernel_size, reduction, n_blocks):
        super(IDAG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            IDAB(n_feat, n_feat_map, kernel_size, reduction)
            for _ in range(n_blocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        :param x[2]: positional embedding (if presented): 1 * ? * H * W
        :param x[3]: local degradation map: B * C' * H' * W'
        '''
        res = x[0]
        # print('>>> (IDAG)', x[0].shape, x[3].shape)
        if len(x) > 2:   # Positional embedding
            if x[2].shape[0] > 1:   # Positional embedding
                res += x[2][0].unsqueeze(0)
            else:
                res += x[2]
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1], x[3]])
        res = self.body[-1](res)
        res = res + x[0]

        return res


class IDASR(nn.Module):
    def __init__(self, args: dict, conv=common.default_conv,
                 model_config: Optional[Dict[str, Union[int, str]]] = None,
                 ):
        super(IDASR, self).__init__()

        if model_config is None:
            model_config = args['network']['netG']

        moco_enc_dim = args['network']['netE']['moco_enc_dim']

        self.n_groups = 5
        n_blocks = 5
        n_feats = 64
        kernel_size = model_config['kernel_size']
        reduction = 8
        scale = int(model_config['scale'])
        self.channels_map = model_config['channels_map']
        n_feats_map = model_config['n_feats_map']
        self.model_config = model_config
        # self.args = args
        #TODO: Create a separate module to deal with the function.  Find vector quantization code

        # RGB mean for DIV2K
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(255.0, rgb_mean, rgb_std)
        # self.add_mean = common.MeanShift(255.0, rgb_mean, rgb_std, 1)

        # head module
        input_size = 4 if args['network']['add_random_state'] else 3
        modules_head = [conv(input_size, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)
        modules_head_map = [conv(input_size, n_feats_map, kernel_size)]
        self.head_map = nn.Sequential(*modules_head_map)

        # compress
        self.compress = nn.Sequential(
            nn.Linear(moco_enc_dim, n_feats, bias=False),
            nn.LeakyReLU(0.1, True)
        )

        # body
        modules_body = [
            IDAG(common.default_conv, n_feats, n_feats_map, kernel_size, reduction, n_blocks)
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

    def forward(self, x, k_v, dmap, pos_emb: torch.Tensor = None):
        k_v = self.compress(k_v)

        # sub mean
        # x = self.sub_mean(x)

        # head
        x = self.head(x)
        dmap = self.head_map(dmap)

        # body
        res = x
        for i in range(self.n_groups):
            if self.model_config['hf']:
                hf_param = self.hf[i](k_v).unsqueeze(2).unsqueeze(3).unsqueeze(4)
                noise = hf_param[:, 0] + torch.randn_like(res) * hf_param[:, 1]
                res = res + noise.to(res.device)
            if pos_emb is None:
                res = self.body[i]([res, k_v, None, dmap])
            else:
                res = self.body[i]([res, k_v, pos_emb, dmap])
        res = self.body[-1](res)
        res = res + x

        # tail
        x = self.tail(res)

        # add mean
        # x = self.add_mean(x)

        return x

class LPEL(nn.Module):
    '''
    LPE, local version

    Input patch size is assumed to be a multiple of 128, 
    k is assumed to be an odd number that is less than 32
    '''
    def __init__(self, dim=256, out_dim=256, k=13):
        # distortion encoder
        super(LPEL, self).__init__()

        assert k % 2 == 1, 'k should be odd'

        ############################################
        # Long-range branch
        ############################################

        # We first need to calculate the paddings
        s = (k - 1) // 2
        # Ideally, we want h = output_size, but it is impossible, because
        # output_size = floor((h + 2 * p - 1 - 2 * s) / s) * s - 2 * p' + 2 * s + 1
        # After some careful calculation, we know that
        # output_size = floor((h + 2 * (p  - p')) / s) * s
        # We set p' = 0, p = s // 2.  Then we can guarentee that the 
        # output_size >= h, and the max difference is s
        # This allows us to discard the last rows and columns of the output
        p = s // 2
        self._s = s  # Keep this for debugging

        LPE_p = [
            # [b, 3, h, w], e.g. [b, 3, 256, 256]
            common.Conv2dBlock(3, dim // 4, k, s, padding=p, norm='batch', activation='lrelu', pad_type='zero'),
            # [b, dim // 4, h // r, w // r], where r == (k-1) // 2, e.g. [b, 64, 42, 42]
            common.Conv2dBlock(dim // 4, dim // 4, 3, 1, padding=1, norm='batch', activation='lrelu', pad_type='zero'),
            # [b, dim // 4, h // r, w // r], e.g. [b, 64, 42, 42]
            common.Conv2dBlock(dim // 4, dim // 2, 3, 1, padding=1, norm='batch', activation='lrelu', pad_type='zero'),
            common.Conv2dBlock(dim // 2, dim // 2, 3, 1, padding=1, norm='batch', activation='lrelu', pad_type='zero'),
            # [b, dim // 4, h // r, w // r], e.g. [b, 64, 42, 42]
            nn.ConvTranspose2d(dim // 2, dim, k, stride=s, padding=0),
            # [b, dim, h, w], e.g. [b, 256, 256, 256]
            nn.AvgPool2d(kernel_size=2)
        ]

        ############################################
        # Short-range branch
        ############################################
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

        self.last_mlp = nn.Sequential(
                nn.Conv2d(dim*2, dim, 1),
                nn.LeakyReLU(0.1, True), 
                nn.Conv2d(dim, out_dim, 1), 
                nn.Sigmoid()
                )

    def forward(self, x):
        '''
        @param x[b, c, h, w]
        @returns [b, out_dim, h // 2, w // 2]
        '''
        ## DEBUG only:
        x_p = self.LPE_p(x)                             # [B, 256, 128, 128]
        x_l = self.LPE_l(x)                             # [B, 256, 128, 128]
        if x_p.shape[-1] != x_l.shape[-1]:
            assert 0 < x_p.shape[-1] - x_l.shape[-1] < self._s, f'{x_p.shape[-1]} != {x_l.shape[-1]}'
            x_p = x_p[:, :, :x_l.shape[-2], :x_l.shape[-1]]

        # print('...', x_p.shape, x_l.shape)
        fea = torch.cat((x_p, x_l), dim=1)              # [B, 512]
        out = self.last_mlp(fea)                        # [B, 256]

        return out, out


class LPELOri(nn.Module):
    '''
    LPE, local version
    '''
    def __init__(self, dim=256, out_dim=256, k=13):
        super(LPELOri, self).__init__()

        # distortion encoder
        LPE_p = [
            common.Conv2dBlock(3, dim // 4, k, int((k - 1) / 2), padding=0, norm='batch', activation='lrelu', pad_type='zero'),
            common.Conv2dBlock(dim // 4, dim // 4, 3, 1, padding=1, norm='batch', activation='lrelu', pad_type='zero'),
            common.Conv2dBlock(dim // 4, dim // 2, 3, 1, padding=1, norm='batch', activation='lrelu', pad_type='zero'),
            common.Conv2dBlock(dim // 2, dim // 2, 3, 1, padding=1, norm='batch', activation='lrelu', pad_type='zero'),
            nn.ConvTranspose2d(dim // 2, dim, k, stride=(k-1)//2, padding=0),
            nn.AvgPool2d(kernel_size=2)
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

        self.last_mlp = nn.Sequential(
                nn.Conv2d(dim*2, dim, 1),
                nn.LeakyReLU(0.1, True), 
                nn.Conv2d(dim, out_dim, 1), 
                nn.Sigmoid()
                )

    def forward(self, x):
        '''
        @param x[b, c, h, w]
        @returns [b, out_dim, h // 2, w // 2]
        '''
        x_p = self.LPE_p(x)                                                  # [B, 256, 128, 128]
        x_l = self.LPE_l(x)                                                  # [B, 256, 128, 128]

        # print('...', x_p.shape, x_l.shape)
        fea = torch.cat((x_p, x_l), dim=1)                                  # [B, 512]
        out = self.last_mlp(fea)                                           # [B, 256]

        return out, out


class BlindSRLocal(nn.Module):
    def __init__(self, args):
        super(BlindSRLocal, self).__init__()

        # Generator
        if args.GB.lower() == 'idasr':
            self.GB = IDASR(args)
        else:
            raise NotImplementedError(f"[!] Distortion synthesis network {args.GB} not implemented")

        # Encoder
        self.E = MoCo(base_encoder=eval(args.moco_enc),
                      dim=args.moco_enc_dim,
                      moco_enc_kernel=args.moco_enc_kernel,
                      K=args.num_neg,
                      distance=args.moco_dist)
        self.E_loc=  LPEL(dim=args.moco_enc_dim,
                          out_dim=4,
                          k=args.moco_enc_kernel,
                          )
        self.Cprs_loc = ImageCompressor(out_channel_N=64, in_channel_N=4)
        self.args = args
        # TODO: implement contrastive learning on degradation embedding

    def forward(self, x, gt=None, pos_emb=None,
                random_states: List[torch.Tensor] =None,
                im_hard_neg: torch.Tensor=None):
        if self.training:
            x_query = x[:, 0, ...]                          # b, c, h, w
            x_key = x[:, 1, ...]                            # b, c, h, w

            # degradation-aware represenetion learning
            x_query_fea, logits, labels = self.E(x_query, x_key, im_hard_neg)
            x_loc_fea, _ = self.E_loc(x_query)   # in [0, 1]
            x_loc_fea_cprs, recon_mse, bpp = self.Cprs_loc(x_loc_fea, bitrate_map=False)

            # degradation-aware SR
            if gt is None:
                assert random_states is None, 'Not supported for generation without gt'
                out = self.G(x_query, x_query_fea)
                return out
            else:
                if random_states is None:
                    out = self.GB(gt, x_query_fea, x_loc_fea_cprs, pos_emb)
                    return out, logits, labels, bpp
                else:
                    outs = [ ]
                    for random_state in random_states:
                        gtr = torch.concat((gt, random_state), dim=1)
                        # We only support IDASR as GB now.
                        out = self.GB(gtr, x_query_fea, x_loc_fea_cprs, pos_emb)
                        outs.append(out)
                    return outs, logits, labels, bpp

        else:

            # degradation-aware represenetion learning
            x_query_fea = self.E(x_query, x_query)
            x_loc_fea, _ = self.E_loc(x_query)   # in [0, 1]
            x_loc_fea_cprs, recon_mse, bpp = self.netCprs(x_loc_fea, bitrate_map=False)

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

