from importlib import import_module
from typing import List
import math
from typing import Any, Dict, List, Union

import torch
import torch.nn.functional as F
from torch import nn
from compressai.entropy_models.entropy_models import EntropyBottleneck

from .blindsr import DASR, LPE, Encoder
from .blindsrlocal import IDASR, LPEL, LPELOri
# from .blindsrlocal_abl import LPEL as LPEL_abl
from .mirnet_v2 import MIRNet_v2
from .Discriminator import DiscriminatorLinear
from moco.builder import MoCo
from .iclr_compression import ImageCompressor
from .compressors import CompressorHeadMLP


class CombinedModel(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super(CombinedModel, self).__init__()

        # Generator
        if config['network']['netG']['type'].lower() == 'dasr':
            assert False, "DASR does not support inhomogeneous distortion"
            self.netG = DASR(config)
        #elif config['network']['netG']['type'].lower() == 'idasr':
            #self.netG = IDASR(config)
        # elif (mtype := config['network']['netG']['type'].lower()).startswith('sft-wrapper'):
            # import model.vunet as vunet
            # cls = vunet.VUNetPro if 'pro' in mtype else vunet.VUNet
            # self.netG = vunet.BasicWrapper(
                    # cls,
                    # proj_dim_noise=config['network']['netG']['proj_dim_noise'],
                    # loc_emb_dim=config['network']['netE_loc']['out_dim'],
                    # emb_dim=256,  # The default value of MoCo that we didn't bother to change
                    # )
        elif (mtype := config['network']['netG']['type'].lower()).startswith('sft-1-wrapper'):
            import model.vunet_conv as vunet
            cls = vunet.VUNetPro if 'pro' in mtype else vunet.VUNet
            self.netG = vunet.BasicWrapper(
                    cls,
                    proj_dim_noise=config['network']['netG']['proj_dim_noise'],
                    loc_emb_dim=config['network']['netE_loc']['out_dim'],
                    emb_dim=256,  # The default value of MoCo that we didn't bother to change
                    )
        # elif (mtype := config['network']['netG']['type'].lower()).startswith('sft-1-abl-wrapper'):
            # import model.vunet_conv_abl as vunet
            # cls = vunet.VUNetPro if 'pro' in mtype else vunet.VUNet
            # self.netG = vunet.BasicWrapper(
                    # cls,
                    # proj_dim_noise=config['network']['netG']['proj_dim_noise'],
                    # loc_emb_dim=config['network']['netE_loc']['out_dim'],
                    # emb_dim=256,  # The default value of MoCo that we didn't bother to change
                    # )
        elif config['network']['netG']['type'].lower() == 'mirv2':
            self.netG = MIRNet_v2()
        else:
            raise NotImplementedError(f"[!] Distortion synthesis network {config['network']['netG']['type']} not implemented")

        # Encoder
        self.netE = MoCo(base_encoder=eval(config['network']['netE']['moco_enc']),
                      dim=config['network']['netE']['moco_enc_dim'],
                      moco_enc_kernel=config['network']['netE']['moco_enc_kernel'],
                      K=config['train']['num_neg'],
                      distance=config['network']['netE']['moco_dist'])

        netE_loc_type = eval(config['network']['netE_loc']['type'])
        self.netE_loc=  netE_loc_type(dim=config['network']['netE_loc']['moco_enc_dim'],
                          out_dim=config['network']['netE_loc']['out_dim'],
                          k=config['network']['netE_loc']['moco_enc_kernel'],
                          )

        # Compressor
        # self.netCprs, self.netCprs_u = self._get_Cprs(config['network']['netCprs'])
        # self.netCprs_loc, self.netCprs_loc_u = self._get_Cprs(config['network']['netCprs_loc'])
        self.netCprs = CompressorWraper(config['network']['netCprs'])
        self.netCprs_loc = CompressorWraper(config['network']['netCprs_loc'])

        # Discriminator
        _C = config['network']['netP']['C']
        NDF = config['network']['netP']['NDF']
        if config['network']['netP']['strategy'] in ['conditioned', 'qiqa']:
            self.netP = DiscriminatorLinear(_C * 2, ndf=NDF, in_width=config['data']['patch_size']) # .cuda()
        elif config['network']['netP']['strategy'] == 'patched':
            self.netP = DiscriminatorLinear(_C * 2, ndf=NDF, in_width=config['data']['patch_size'] // 4) # .cuda()
        else:
            self.netP = DiscriminatorLinear(_C, ndf=NDF, in_width=config['data']['patch_size']) # .cuda()

        self.config = config

        self.pos_emb = None
        if config['network']['positional_embedding']:
            hidx = torch.arange(0, config['data']['patch_size']).reshape(1, 1, -1, 1)
            widx = torch.arange(0, config['data']['patch_size']).reshape(1, 1, 1, -1)
            h_emb = torch.sin(hidx * (8 / 2 * torch.pi))
            w_emb = torch.sin(widx * (8 / 2 * torch.pi))
            self.pos_emb = nn.Parameter(h_emb * w_emb, requires_grad=False)
            # self.pos_emb = self.pos_emb.repeat(config['hardware']['n_GPUs'], 1, 1, 1)


    def get_state_dict(self) -> Dict[str, dict]:
        '''Get the state dicts of the parts that require saving
        '''
        return {
                'netE': self.netE.encoder_q.state_dict(), 
                'netE_loc': self.netE_loc.state_dict(), 
                'netG': self.netG.state_dict(), 
                'netCprs': self.netCprs.state_dict(), 
                'netCprs_loc': self.netCprs_loc.state_dict(), 
                'netP': self.netP.state_dict(), 
                }
        
    def load_state_dict(self, state_dict: Dict[str, dict]) -> None:
        '''Load the state dicts of the parts that require loading
        '''
        self.netE.encoder_q.load_state_dict(state_dict['netE'])
        self.netE_loc.load_state_dict(state_dict['netE_loc'])
        self.netG.load_state_dict(state_dict['netG'])
        self.netCprs.load_state_dict(state_dict['netCprs'])
        self.netCprs_loc.load_state_dict(state_dict['netCprs_loc'])
        self.netP.load_state_dict(state_dict['netP'])

    def get_param_gen(self, add_option=True) -> List[torch.nn.Parameter]:
        '''Get the parameters to be passed to optimizer for the generation parts
        '''
        f = lambda x: x.requires_grad
        if add_option: 
            return [
                    {'params': filter(f, self.netE.parameters()), 'lr': self.config['optim']['lr_E']},
                    {'params': filter(f, self.netE_loc.parameters()), 'lr': self.config['optim']['lr_E_loc']},
                    {'params': filter(f, self.netG.parameters()), 'lr': self.config['optim']['lr_G']},
                    {'params': filter(f, self.netCprs.parameters()), 'lr': self.config['optim']['lr_Cprs']},
                    {'params': filter(f, self.netCprs_loc.parameters()), 'lr': self.config['optim']['lr_Cprs_loc']},
                    ] # type: ignore
        else:
            return filter(f, list(self.netE.parameters()) + list(self.netE_loc.parameters() + list(self.netG.parameters()) + list(self.netCprs.parameters()) + list(self.netCprs_loc.parameters())))

    def get_param_dis(self, add_option=True) -> List[torch.nn.Parameter]:
        '''Get the parameters to be passed to optimizer for the discrimination parts
        '''
        f = lambda x: x.requires_grad
        if add_option: 
            return [{'params': filter(f, self.netP.parameters()), 'lr': self.config['optim']['lr_P']}] # type: ignore

        else:
            return filter(f, self.netP.parameters()) # type: ignore

    def forward_latent(self, y: torch.Tensor, compress=True):
        e, _ = self.netE.encoder_q(y)
        if compress:
            e, _, _ = self.netCprs(e)
        return e

    def forward_latent_local(self, y: torch.Tensor, compress=True):
        e, _ = self.netE_loc(y)
        if compress:
            e, _, _ = self.netCprs_loc(e)
        return e

    def forward_partial(
            self, x: torch.Tensor, 
            eg: torch.Tensor, el: torch.Tensor, 
            random_states: torch.Tensor = None):
        '''
        Partial forward.  Only generate the portion '(x, e) -> y hat'

        @returns: The generated y_hat's in a list. 
        '''
        if random_states is None: 
            random_states = [None] # type: ignore
        y_hs = [
                self.netG(_add_random_state(x, random_state),
                          eg, el, pos_emb=self.pos_emb
                          )
                for random_state in random_states
                ]
        return y_hs

    def _forward_gen(
            self, 
            x_idx, 
            y_idx, 
            x: torch.Tensor,
            y: torch.Tensor,
            y_key: torch.Tensor,
            random_states: List[torch.Tensor] = None,
            compress_g=False,
            compress_l=True,
            gen_e_h=False,
            ) -> Dict[str, torch.Tensor]:
        '''
        The common part for implementing paired and unpaired forward
        '''
        res = { }
        eg, eg_output, eg_target = self.netE(y, y_key)
        if compress_g:
            eg, eg_likelihoods, eg_bpp = self.netCprs(eg)
            res[f'eg_{y_idx}_likeliboods'] = eg_likelihoods
            res[f'eg_{y_idx}_bpp'] = eg_bpp
        el, _ = self.netE_loc(y)
        if compress_l:
            el, el_likelihoods, el_bpp = self.netCprs_loc(el)
            res[f'el_{y_idx}_likeliboods'] = el_likelihoods
            res[f'el_{y_idx}_bpp'] = el_bpp
        # NOTE: might requires modifications to the MoCo code
        if random_states is None: 
            random_states = [None] # type: ignore
        y_hs = [
                self.netG(_add_random_state(x, random_state),
                          eg, el, pos_emb=self.pos_emb
                          )
                for random_state in random_states
                ]
        res.update({
                f'eg_{y_idx}': eg,
                f'eg_{y_idx}_output': eg_output,
                f'eg_{y_idx}_target': eg_target,
                f'el_{y_idx}': el,
                f'y_{x_idx}hs': y_hs,  # After checking Pytorch's code, this works for DataParallel
                })
        if gen_e_h:
            eg_h, _ = self.netE.encoder_q(y_hs[0])
            el_h, _ = self.netE_loc(y_hs[0])
            if compress_g:
                eg_h, eg_h_likelihoods, eg_h_bpp = self.netCprs(eg_h)
                res[f'eg_{x_idx}h_likeliboods'] = eg_h_likelihoods
                res[f'eg_{x_idx}h_bpp'] = eg_h_bpp
            if compress_l:
                el_h, el_h_likelihoods, el_h_bpp = self.netCprs_loc(el_h)
                res[f'el_{x_idx}h_likeliboods'] = el_h_likelihoods
                res[f'el_{x_idx}h_bpp'] = el_h_bpp
            res[f'eg_{x_idx}h'] = eg_h
        return res

    def forward_paired(
            self, 
            x_0: torch.Tensor,
            y_0: torch.Tensor,
            y_0_key: torch.Tensor,
            random_states: List[torch.Tensor] = None,
            compress_g=False,
            compress_l=True,
            gen_e_h=False
            ) -> Dict[str, torch.Tensor]:
        '''Forward, the paired part. 

        See the technical document for details.  Basically, x is the pristine, 
        y is the corresponding distorted. 

        @param random_states: the random states.  
        @param gen_e_h: if set, e_h will be generated as well (for the first y_0h). 
        @param y_0_key: The key tensor for y_0, usually a different patch 
                        from the same image.
        @return: A dictionary containing the following key-value pairs:
                 'e_0': The degradation embedding of y_0.
                 'e_0_output': The logits outputs for contrastive learning
                 'e_0_target': The labels for contrastive learning
                 'y_0hs': The generated tensor after adding random state to x_0 and passing through the generator.
                 'e_0_likelihoods': for compress
                 'e_0_bpp': bits per pixture for compress
                 'e_0h': The encoded representation of the first element of y_0hs. This is only generated if gen_e_h is True.
                 'e_0h_likelihoods': for compress
                 'e_0h_bpp': bits per pixture for compress
        '''
        return self._forward_gen(
                0, 0, 
                x_0, y_0, y_0_key, random_states, compress_g, compress_l, gen_e_h
                )
        
    def forward_discriminator(self, y) -> torch.Tensor:
        return self.netP(y)

    def forward(self, selection: str, *inputs, **kwargs) -> Dict[str, torch.Tensor]:
        if selection == 'paired':
            return self.forward_paired(*inputs, **kwargs)
        elif selection == 'unpaired':
            return self.forward_unpaired(*inputs, **kwargs)
        elif selection == 'discriminator':
            return self.forward_discriminator(*inputs, **kwargs)
        elif selection == 'latent':
            return self.forward_latent(*inputs, **kwargs)
        elif selection == 'latent_local':
            return self.forward_latent_local(*inputs, **kwargs)
        elif selection == 'partial':
            return self.forward_partial(*inputs, **kwargs)
        else:
            assert False, "Operation not supported"

    def get_compress_loss(self) -> torch.Tensor:
        '''
        @returns: The loss for (fixing the entropy estimation of) the 
                  compression network.  For some networks, this loss is 
                  not needed, in which case, 0. (the floating-point number)
                  will be returned.
        '''
        return self.netCprs.loss() + self.netCprs_loc.loss()


def _add_random_state(x, random_state):
    if random_state is None:
        return x
    return torch.concat((x, random_state), dim=1)


class CompressorWraper(nn.Module):
    def __init__(self, config: dict[str, Union[str, int]]):
        super(CompressorWraper, self).__init__()
        self.config = config
        if config['type'].lower() == 'entropybottleneck':
            self.model = EntropyBottleneck(channels=config['channels'])
        elif config['type'].lower() == 'imagecompressor':
            self.model = ImageCompressor(
                    out_channel_N=config['out_channel_N'], 
                    in_channel_N=config['in_channel_N'])
        else:
            raise NotImplementedError(f"[!] Compression network {config['type']} not implemented")

        if 'head' in config:
            self.head = CompressorHeadMLP(config['channels'], **config['head'])
        else:
            self.head = None
        if 'tail' in config:
            self.tail = CompressorHeadMLP(config['channels'], **config['tail'])
        else:
            self.tail = None


    def forward(self, e):
        '''
        @param e: input latent
        @returns: (e_l_compressed, e_l_likelihoods, e_l_bpp).  Note that
                  some of the models, e.g. ImageCompressor does not provide
                  e_l_likelihoods directly, in which case, it would be 
                  replaced by zeros with the same shape as e_l_bpp.
        '''
        # Note that this module is shared for both global and local 
        # degradations.  The "_l" in the variable names has no significance.
        if self.head is not None:
            e = self.head(e)

        if self.config['type'].lower() == 'entropybottleneck':
            e_l, likelihoods = self.model(e)
            bpp = torch.log(likelihoods).sum(dim=1) / (-math.log(2))
        elif self.config['type'].lower() == 'imagecompressor':
            e_l, recon_mse, bpp = self.model(e)
            likelihoods = torch.zeros_like(bpp)
        else:
            raise NotImplementedError(f"[!] Compression network {self.config['type']} not implemented")

        if self.tail is not None:
            e_l = self.tail(e_l)

        return e_l, likelihoods, bpp

    def forward_modifiable(self, e):
        '''
        This version returns a mofiiable e_l.  Use forward_finish on the 
        modified e_l to get the final output.
        '''
        if self.head is not None:
            e = self.head(e)

        if self.config['type'].lower() == 'entropybottleneck':
            e_l, likelihoods = self.model(e)
            bpp = torch.log(likelihoods).sum(dim=1) / (-math.log(2))
        elif self.config['type'].lower() == 'imagecompressor':
            e_l, recon_mse, bpp = self.model(e)
            likelihoods = torch.zeros_like(bpp)
        else:
            raise NotImplementedError(f"[!] Compression network {self.config['type']} not implemented")
        return e_l

    def forward_finish(self, e_l):
        if self.tail is not None:
            e_l = self.tail(e_l)
        return e_l


    def loss(self) -> torch.Tensor:
        '''
        @returns: The loss for (fixing the entropy estimation of) the 
                  compression network.  For some networks, this loss is 
                  not needed, in which case, 0. (the floating-point number)
                  will be returned.
        '''
        if self.config['type'].lower() == 'entropybottleneck':
            return self.model.loss()
        elif self.config['type'].lower() == 'imagecompressor':
            # This implementation of ImageCompressor does not need a 
            # correction term
            return 0.
        else:
            raise NotImplementedError(f"[!] Compression network {self.config['network']['netCprs_loc']['type']} not implemented")
