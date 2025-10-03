import os
from decimal import Decimal
from typing import Dict, Union, List, Tuple, Optional, Iterable
from pathlib import Path
import itertools

import torch
from torch import Tensor
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from tqdm import trange
from torchvision import transforms
from torchvision.transforms import functional as F
from timm.data.loader import MultiEpochsDataLoader



import utility
from visualizer import Visualizer
import model
from data import single_dataset
from model.combined_model import CombinedModel
from model.utility import get_barebone
from loss.color import get_color_losses
from loss.degrad import get_degrade_losses
from loss.latent import get_latent_losses
from loss.diversity import DiversityLoss
from model.utility import get_barebone

DEBUG_MEM = False
if DEBUG_MEM:
    DEBUG_COUNT = 0
    import tracemalloc
    import gc
    import sys
    from torch.profiler import profile, record_function, ProfilerActivity


class Trainer():
    def __init__(
            self, 
            config: Dict[str, Union[str, int, dict]], 
            model: CombinedModel, 
            loaders: Dict[str, MultiEpochsDataLoader], 
            logger: utility.Logger):
        self.config = config
        self.logger = logger
        self.model = model
        self.loaders = loaders

        # Loss
        self.contra_loss = torch.nn.CrossEntropyLoss().cuda()

        barebone = get_barebone(self.model)

        # Optimizers and schedulers
        self.optimizer_gen = optim.Adam(
                params=barebone.get_param_gen(add_option=True), 
                lr=config['optim']['lr'], eps=config['optim']['epsilon'],
                betas=(config['optim']['beta1'], config['optim']['beta2']), 
                )
        self.optimizer_dis = optim.Adam(
                params=barebone.get_param_dis(add_option=True),
                lr=config['optim']['lr'], eps=config['optim']['epsilon'],
                betas=(config['optim']['beta1'], config['optim']['beta2']), 
                )
        self.scheduler_gen = lrs.StepLR(
            self.optimizer_gen, step_size=config['optim']['lr_decay_gen'],
            gamma=config['optim']['gamma_gen'])
        self.scheduler_dis = lrs.StepLR(
            self.optimizer_gen, step_size=config['optim']['lr_decay_dis'],
            gamma=config['optim']['gamma_dis'])

        # Loading
        if ckpt_name := config['log']['resume']:
            print(f"[*] Loading model from checkpoint: {ckpt_name}")
            ckpt_dir = os.path.dirname(ckpt_name)
            name_stem = Path(ckpt_name).stem
            try: 
                self.load_model(
                        ckpt_dir=ckpt_dir,
                        name=name_stem, 
                        load_optim=not config['test']['test_only'])
            except:
                print("Unable to Load optimizers")
                assert config['log']['skip_missing']
                self.load_model(
                        ckpt_dir=ckpt_dir,
                        name=name_stem, 
                        load_optim=False)
            for _ in range(len(logger.log)):
                self.scheduler_gen.step()
                self.scheduler_dis.step()

        # Loss functions
        self.degrade_loss, self.degrade_loss_w, self.degrade_loss_name = get_degrade_losses(config['loss']['degrade_loss'], n_GPUs=config['hardware']['n_GPUs'])
        self.color_loss, self.color_loss_w, self.color_loss_name = get_color_losses(config['loss']['color_loss'])
        self.latent_loss, self.latent_loss_w, self.latent_loss_name = get_latent_losses(config['loss']['latent_loss'])
        self.contrast_loss = nn.CrossEntropyLoss().cuda()
        self.diversity_loss_fn = None
        self.add_random_state = config['network']['add_random_state']
        if self.add_random_state and config['loss']['lambda_diver']:
            self.diversity_loss_fn = DiversityLoss().cuda()


        # Logging
        self.screen_output_line: List[str] = [ ]
        self.log_file_output_line: List[str] = [ ]
        self.global_step = 0
        self.epoch = 0
        if not config['test']['test_only']:
            self.visualizer = Visualizer(checkpoints_dir=logger.dir, name=config['log']['save'])
            # TODO: Fix vidualizer

        # Settings
        self._should_compress_g = config['network']['netCprs']['enabled']
        self._should_compress_l = config['network']['netCprs_loc']['enabled']

    @torch.no_grad()
    def log_var(
            self, name: str, var: Tensor, 
            short_name: str = None, on_bar=False):
        '''
        Called to prepare logging each variable. 

        @param name: name to be shown in the log file and tensorboard
        @param var: the variable to be shown
        @param short_name: the shortened name, used for on-screen output
        @param on_bar: if set, the output will be put on screen
        '''
        if (self.global_step + 1) % self.config['log']['print_every'] != 0:
            assert len(self.screen_output_line) == 0
            assert len(self.log_file_output_line) == 0
            return
        if short_name is None: 
            short_name = name
        self.logger.log_board_scalar(name, var.item(), self.global_step)
        self.screen_output_line.append(f'{short_name}: {var.item():.3f}')
        self.log_file_output_line.append(f'{name}: {var.item():.6f}')

    def log_flush(self):
        '''
        Flush logs

        @returns: description to be printed
        '''
        if (self.global_step + 1) % self.config['log']['print_every'] != 0:
            return
        if self.global_step == 0:
            print()   # Avoiding the last printed line being overriden
        self.logger.write_log(f'Epoch {self.epoch:03d}[{self.global_step:07d}]: ' + ', '.join(self.log_file_output_line))
        res = f'Epoch {self.epoch:03d}[{self.global_step:07d}]:' + ', '.join(self.screen_output_line)
        self.log_file_output_line.clear()
        self.screen_output_line.clear()
        return res

    def save_model(self, name: str = None, comments=''):
        if name is None: 
            name = f'model_{self.epoch}'
            if comments:
                name += '_' + comments
        model: CombinedModel = get_barebone(self.model)
        to_save = {
                'model': model.get_state_dict(),
                'epoch': self.epoch,
                'global_step': self.global_step
                }
        print('... saving', self.epoch, self.global_step)
        torch.save(to_save,
                   os.path.join(self.logger.dir, 'model', f"{name}.pt"))
        torch.save(to_save,
                   os.path.join(self.logger.dir, 'model', f"latest.pt"))
        torch.save(self.optimizer_gen.state_dict(), os.path.join(
                self.logger.dir, 'model', f"optim_gen_{self.epoch}.pt"))
        torch.save(self.optimizer_gen.state_dict(), os.path.join(
                self.logger.dir, 'model', "optim_gen_latest.pt"))
        torch.save(self.optimizer_dis.state_dict(), os.path.join(
                self.logger.dir, 'model', f"optim_dis_{self.epoch}.pt"))
        torch.save(self.optimizer_dis.state_dict(), os.path.join(
                self.logger.dir, 'model', "optim_dis_latest.pt"))
        
    def load_model(self, 
                   ckpt_dir: str = None,
                   name: str = None, load_optim: bool = True, 
                   optim_gen_name: str = None, optim_dis_name: str = None):
        if name is None:
            name = 'latest'
        if ckpt_dir is None:
            ckpt_dir = os.path.join(self.logger.dir, 'model')
        loaded = torch.load(os.path.join(ckpt_dir, f"{name}.pt"))
        model: CombinedModel = get_barebone(self.model)
        model.load_state_dict(loaded['model'])
        self.epoch = loaded['epoch']
        self.global_step = loaded['global_step']
        if optim_gen_name is None:
            optim_gen_name = 'optim_gen_latest'
        if optim_dis_name is None:
            optim_dis_name = 'optim_dis_latest'
        if load_optim:
            self.optimizer_gen.load_state_dict(torch.load(os.path.join(
                    ckpt_dir, f"{optim_gen_name}.pt")))
            self.optimizer_dis.load_state_dict(torch.load(os.path.join(
                    ckpt_dir, f"{optim_dis_name}.pt")))

    def get_dloss_val(self, x: torch.Tensor, y: torch.Tensor
                      ) -> Tuple[Tensor, Dict[str, Tensor]]:
        '''
        Note that the x and y in this function is different from the tech doc.
        
        @return: final_loss_val, individual_vals
        '''
        res_val = 0
        res = { }
        for dloss, (dloss_w, _), dloss_name in zip(self.degrade_loss, self.degrade_loss_w, self.degrade_loss_name):
            try:
                res[dloss_name] = dloss(x, y)
                res_val += res[dloss_name] * dloss_w
            except:
                print(f"BAD D LOSS VAL {dloss_name}")
                res[dloss_name] = (x - y).mean()
                res_val += res[dloss_name] * dloss_w
        return res_val, res

    def get_closs_val(self, x: torch.Tensor, y: torch.Tensor
                      ) -> Tuple[Tensor, Dict[str, Tensor]]:
        '''
        Note that the x and y in this function is different from the tech doc.
        
        @return: final_loss_val, individual_vals
        '''
        res_val = 0
        res = { }
        for closs, closs_w, closs_name in zip(self.color_loss, self.color_loss_w, self.color_loss_name):
            res[closs_name] = closs(x, y)
            res_val += res[closs_name] * closs_w
        return res_val, res

    def get_latent_loss_val(self, yh: Tensor, e: Tensor, 
                            eh: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        e_sg = e # y does not actually need stop gradient operation
        eh_sg = self.model.forward('latent', yh.detach())   # stop gradient
        res_val = 0
        res = { }
        for latent_loss, (latent_loss_w, _), latent_loss_name in zip(self.latent_loss, self.latent_loss_w, self.latent_loss_name):
            res[latent_loss_name] = latent_loss(e, eh)
            res[latent_loss_name] -= latent_loss(e_sg, eh_sg)
            res_val += res[latent_loss_name] * latent_loss_w
        return res_val, res


    def get_gan_loss_G(self, x, y, yh, yh_2=None, el=None, eg=None):
        '''
        Calculates the generator loss for the GAN.
        See the tech doc for the meanings of each parameter. 

        @param x: only used when GAN mode is conditioned
        @param yh_2: only needed for discriminator type 3 (qiqa)
        @param el: only needed for discriminator type 4
        @param eg: only needed for discriminator type 4

        @return: generator loss
        '''

        if self.config['network']['netP']['strategy'] == 'conditioned':
            fake_y_loss = self.model('discriminator', torch.cat((x, yh), dim=1)).mean()
        elif self.config['network']['netP']['strategy'] == 'patched':
            b, c, h, w = y.shape
            ph, pw = h // 4, w // 4
            ktop, kleft = torch.randint(high=h - ph, size=tuple()), torch.randint(high=w - pw, size=tuple())
            ptop, pleft = torch.randint(high=h - ph, size=tuple()), torch.randint(high=w - pw, size=tuple())
            # Make sure the patches are disjoint.
            while torch.abs(ktop - ptop) < ph and torch.abs(kleft - pleft) < pw:
                ptop, pleft = torch.randint(high=h - ph, size=tuple()), torch.randint(high=w - pw, size=tuple())
            q_k = y[:, :, ktop:(ktop + ph), kleft:(kleft + ph)]
            q_pred_p = yh[:, :, ptop:(ptop + ph), pleft:(pleft + ph)]
            fake_y_loss = self.model('discriminator', torch.cat((q_k, q_pred_p), dim=1)).mean()
        elif self.config['network']['netP']['strategy'] == 'qiqa':
            raise NotImplemented()  # TODO
        else:
            fake_y_loss = self.model('discriminator', yh).mean()
        return -self.config['network']['netP']['alpha'] * fake_y_loss


    def cal_one_step_gen(
            self,
            paired_vars: Optional[Dict[str, torch.Tensor]] = None,
            ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
        '''
        Perform one step of the generator training.
        If either paired_vars or unpaired_vars is missing, the training of 
        the missing part will not be carried out.  Hence, the relevant loss
        will not be calculated, and the returned dict will be empty.

        The function shall also log relevant values.
        
        @param paired_vars: A dictionary containing the following keys:
            - 'x_0': the input image
            - 'y_0': the target image
            - 'y_0_key': the key for the target image

        @returns image outputs, paired loss values

        Paired loss values will contain a value 'sum', which is a weighted 
        sum of the following items, and should be used for back-propagation:
        - 'contra': contrastive loss
        - 'bpp': bits per pixel loss
        - 'degrade': degradation loss
        - 'color': color loss
        - 'latent': latent loss
        - 'diver': (optional) diversity loss, presented if add_random_state 
                   and lambda_diver != 0
        - 'gan_gen': generator loss for the GAN
        For more details, see the tech doc. 

        In addition, the following items are also included: 
        - 'bpp_g': bits per pixel loss (global)
        - 'bpp_l': bits per pixel loss (local)
        - 'dlosses': A dictionary containing degradation loss components
        - 'closses': A dictionary containing color loss components
        - 'latent_losses': (DISABLED) A dictionary containing latent loss components
        '''
        ##################################
        # Paired
        ##################################
        random_states = None
        if self.add_random_state:
            random_states = [torch.randn_like(paired_vars['x_0'][:, 0:1, :, :]) for _ in range(2)]
        # e_0, e_0_output, e_0_target, e_0_likelihood, e_0_bpp, y_0hs, e_0h
        paired_res = self.model(
                'paired', 
                x_0=paired_vars['x_0'],
                y_0=paired_vars['y_0'],
                y_0_key=paired_vars['y_0_key'],
                random_states=random_states,
                compress_g=self._should_compress_g,
                compress_l=self._should_compress_l,
                # gen_e_h=True
                gen_e_h=False  # Removed: gen_e_h
                )  # REMOVED: hard_negative
        losses_paired: Dict[str, torch.Tensor] = { }
        # L_contra
        losses_paired['contra'] = self.contra_loss(
                paired_res['eg_0_output'], paired_res['eg_0_target'])
        # L_bpp_g
        if 'eg_0_bpp' in paired_res:
            # Only present when _should_compress_g is True
            losses_paired['bpp_g'] = paired_res['eg_0_bpp'].mean()
        # losses_paired['bpp_p'] = paired_res['e_0h_bpp'].mean()
        # losses_paired['bpp'] = losses_paired['bpp_o'] + losses_paired['bpp_p']
        # L_bpp_l
        if 'el_0_bpp' in paired_res:
            # Only present when _should_compress_l is True
            losses_paired['bpp_l'] = paired_res['el_0_bpp'].mean()
        # L_sim: a.k.a. degradation loss
        losses_paired['degrade'], losses_paired['dlosses'] = self.get_dloss_val(
                paired_vars['y_0'], paired_res['y_0hs'][0])
        # L_color
        losses_paired['color'], losses_paired['closses'] = self.get_closs_val(
                paired_vars['y_0'], paired_res['y_0hs'][0] )
        # L_latent
        # losses_paired['latent'], losses_paired['latent_losses'] = \
                # self.get_latent_loss_val(
                        # paired_res['y_0hs'][0], paired_res['e_0'], 
                        # paired_res['e_0h'])
        # L_diver: diversity loss
        if self.diversity_loss_fn is not None:
            losses_paired['diver'] = self.diversity_loss_fn(
                    paired_res['y_0hs'][0], paired_res['y_0hs'][1])
        # L_gan_gen
        losses_paired['gan_gen'] = self.get_gan_loss_G(
                x=paired_vars['x_0'],
                y=paired_vars['y_0'],
                yh=paired_res['y_0hs'][0],
                yh_2=paired_res['y_0hs'][1] if len(paired_res['y_0hs']) > 1 else None
                )
        tol = 0
        for k in ['contra', 'bpp_g', 'bpp_l', 'degrade', 'color', 'latent', 'diver', 'gan_gen']:
            if k in losses_paired:
                tol += self.config['loss']['lambda_' + k] * losses_paired[k]
        losses_paired['sum'] = tol

        net_outputs = dict(paired_res)

        return net_outputs, losses_paired


    def _augment_discriminator_input(
            self, x: Tensor, y: Tensor, 
            yh: Tensor, yh_2: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        '''
        Augment the discriminator input to make the input size match.
        
        See the tech doc for the detailed usage of each parameter.

        @param x: pristine image, only used if the discriminator is conditioned

        @returns: augmented_y, augmented_yh
        
        Note in the patched mode, the "batch size" will be changed.
        '''
        if self.config['network']['netP']['strategy'] == 'conditioned':
            y = torch.cat((x, y), dim=1)
            yh = torch.cat((x, yh), dim=1)
        elif self.config['network']['netP']['strategy'] == 'patched':
            b, c, h, w = y.shape
            ph, pw = h // 4, w // 4
            # randomly crop im_q and im_q_pred to (3 * ph, 3 * pw). The cropped region should be the same in both images.
            i, j, h, w = transforms.RandomCrop.get_params(
                    y, output_size=(3 * ph, 3 * pw))
            y = F.crop(y, i, j, h, w)  # (b, c, 3 * ph, 3 * pw)
            yh = F.crop(yh, i, j, h, w)

            # isolate the 9 patches of y and yh, so that their shape becomes (b, 9, c, ph, pw)
            y_patches = y.unfold(-2, ph, ph).unfold(-1, pw, pw)  # (b, c, ph, 3, pw, 3)
            y_patches = y_patches.permute((0, 3, 5, 1, 2, 4)).reshape((b, c, 9, ph, pw))
            yh_patches = yh.unfold(-2, ph, ph).unfold(-1, pw, pw)  # (b, c, ph, 3, pw, 3)
            yh_patches = yh_patches.permute((0, 3, 5, 1, 2, 4)).reshape((b, c, 9, ph, pw))
            # randomly select one patch out of the 9 patches of y and yh
            patch_indices = torch.randperm(9)
            y_key_patch = y_patches[:, :, patch_indices[-1:], :, :]  # (b, c, 1, ph, pw)
            y_patches = y_patches[:, :, patch_indices[:8], :, :]     # (b, c, 8, ph, pw)
            yh_patches = yh_patches[:, :, patch_indices[:8], :, :]
            y_key_patch = y_key_patch.expand(b, c, 8, ph, pw)
            # Shape
            y_patches = torch.cat((y_key_patch, y_patches), dim=1)  # (b, 2 * c, 8, ph, pw)
            y = y_patches.permute((0, 2, 1, 3, 4)).reshape(b * 8, 2 * c, ph, pw)
            yh_patches = torch.cat((y_key_patch, yh_patches), dim=1)  # (b, 2 * c, 8, ph, pw)
            yh = yh_patches.permute((0, 2, 1, 3, 4)).reshape(b * 8, 2 * c, ph, pw)
        elif self.config['network']['netP']['strategy'] == 'qiqa':
            raise NotImplemented()
        return y, yh

    def get_gradient_penalty(self, y: Tensor, yh: Tensor):
        """
        Calculates the gradient penalty for the generator loss in the WGAN-GP algorithm.

        Args:
            y (torch.Tensor): Real data samples.
            yh (torch.Tensor): Generated data samples.

        Returns:
            torch.Tensor: Gradient penalty value.

        """

        batch_size = y.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(y)
        alpha = alpha.to(y.device)
        interpolated = alpha * y.data + (1 - alpha) * yh.data
        interpolated.requires_grad = True

        # Calculate probability of interpolated examples
        prob_interpolated = self.model('discriminator', interpolated)

        # Calculate gradients of probabilities with respect to examples
        grad_outputs = torch.ones(prob_interpolated.size(), device=y.device, dtype=torch.float32)
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                        grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.config['loss']['lambda_dis_gp'] * ((gradients_norm - 1) ** 2).mean()


    def cal_one_step_dis(
            self, x: Tensor, y: Tensor, 
            yh: Tensor, yh_2: Optional[Tensor] = None
            ) -> Dict[str, Tensor]:
        y = y.detach()
        yh = yh.detach()
        yh_2 = yh_2.detach() if yh_2 is not None else None
        y, yh = self._augment_discriminator_input(x, y, yh, yh_2)
        fake_y_loss = self.model('discriminator', yh).mean()
        real_y_loss = self.model('discriminator', y).mean()
        grad_y_loss = self.get_gradient_penalty(y, yh)
        loss_y = self.config['network']['netP']['alpha'] * (fake_y_loss - real_y_loss)
        loss_yg = self.config['network']['netP']['alpha'] * grad_y_loss
        return {
                'sum': loss_y + loss_yg,
                'loss_y': loss_y, 
                'loss_yg': loss_yg
                }
        
    @torch.no_grad()
    def log_train_step_gen(
            self, 
            net_outputs: Dict[str, torch.Tensor], 
            losses_paired: Dict[str, torch.Tensor], 
            loss_compress: torch.Tensor) -> None:
        NAMES = { 
                 # short_name, name, on_bar
                 'contra': ('contra', 'L_contra', False),
                 'bpp_g': ('bpp_g', 'L_bpp_g', True),
                 'bpp_l': ('bpp_l', 'L_bpp_l', True),
                 'degrade': ('dgrd', 'L_degrad', True),
                 'color': ('color', 'L_color', True),
                 'latent': ('ltnt', 'L_latent', True),
                 'diver': ('diver', 'L_sim,diver', True),
                 'gan_gen': ('G', 'L_gan_gen', True)
                }
        for k in ['contra', 'bpp_g', 'bpp_l', 'degrade', 'color', 'latent', 'diver']:
            if k in losses_paired:
                self.log_var(var=losses_paired[k],
                             short_name='P' + NAMES[k][0],
                             name='Paired' + NAMES[k][1],
                             on_bar=NAMES[k][2])

    @torch.no_grad()
    def log_train_step_dis(
            self, prefix: str, prefix_short: str, losses: Dict[str, torch.Tensor]) -> None:
        for k in ['sum', 'loss_y', 'loss_yg']:
            self.log_var(var=losses[k],
                         short_name=prefix + k,
                         name=prefix_short + k,
                         on_bar=k == 'sum')


    def train_one_step(
            self,
            paired_vars: Optional[Dict[str, torch.Tensor]] = None,
            ) -> Dict[str, Tensor]:
        '''
        @returns: generated images
        '''
        res_imgs, paired_outs = self.cal_one_step_gen(paired_vars)
        loss: Tensor = paired_outs['sum'] if paired_vars is not None else 0
        loss_compress = get_barebone(self.model).get_compress_loss()
        loss = loss + self.config['loss']['lambda_cprs'] * loss_compress
        self.optimizer_gen.zero_grad()
        loss.backward()
        self.optimizer_gen.step()
        self.log_train_step_gen(res_imgs, paired_outs, loss_compress)

        # Let's see whether this can save memory
        for k in res_imgs:
            if isinstance(res_imgs[k], torch.Tensor):
                res_imgs[k] = res_imgs[k].detach()
            elif isinstance(res_imgs[k], list):
                res_imgs[k] = [x.detach() for x in res_imgs[k]]
            else:
                raise ValueError(f"Unknown type for res_imgs[{k}]")
        del paired_outs, loss, loss_compress


        paired_dis_outs = self.cal_one_step_dis(
                paired_vars['x_0'], paired_vars['y_0'], 
                res_imgs['y_0hs'][0], 
                res_imgs['y_0hs'][1] if len(res_imgs['y_0hs']) > 1 else None)
        self.log_train_step_dis('PairedDis', 'PD', paired_dis_outs)
        dis_loss = paired_dis_outs['sum'] if paired_vars is not None else 0
        self.optimizer_dis.zero_grad()
        dis_loss.backward()
        self.optimizer_dis.step()
        return res_imgs
        
    def train(self):
        self.model.train()
        print('Start training.')

        loaders: Dict[str, Iterable[Dict[str, Tensor]]] = {
                k: utility.cycle_iterator(v) for k, v in self.loaders.items()
                }
        max_len = max(len(l) for l in self.loaders.values())

        start_epoch = self.epoch
        with trange(start_epoch, (start_epoch + self.config['train']['epochs']) * max_len) as pbar:
            for epoch in range(start_epoch, start_epoch + self.config['train']['epochs']):

                for batch_idx in range(max_len):
                    self.global_step += 1
                    self.epoch = epoch

                    # Load data
                    assert 'train' in loaders, "No train data loader"
                    paired_vars_l = next(loaders['train'])
                    paired_vars = {
                            'x_0': paired_vars_l['B_img'].cuda(),
                            'y_0': paired_vars_l['A_img'][:, 0, ...].cuda(),
                            'y_0_key': paired_vars_l['A_img'][:, 1, ...].cuda(),
                            }

                    # Actually train one step
                    res_imgs = self.train_one_step(
                            paired_vars=paired_vars,
                            )
                    pbar.update(1)
                    if dec := self.log_flush():
                        pbar.set_description(dec)

                    # Save model
                    if (self.config['log']['save_every_step'] > 0 and 
                            self.global_step % self.config['log']['save_every_step'] == 0):
                        self.save_model()
                        # Somehow, the original implementation of self.save_model f**ked up the the memory until it explodes

                    # Draw visuals
                    if (self.global_step % self.config['log']['visualize_every'] == 0
                            or (self.config['log']['save_every_step'] > 0 and
                                self.global_step % self.config['log']['save_every_step'] == 0 
                                )):
                        should_be_perm = (self.global_step // self.config['log']['visualize_every']) % self.config['log']['permanant_visualize_every'] == 0 or max_len - batch_idx <= self.config['log']['visualize_every'] # Every n batches, or the last visualized batch in an epoch
                        self.visualize(
                                paired_vars=paired_vars,
                                paired_vars_l=paired_vars_l,
                                res_imgs=res_imgs,
                                permanent=should_be_perm
                                )

                    if DEBUG_MEM:
                        global DEBUG_COUNT
                        if DEBUG_COUNT % 30 == 0:
                            '''
                            if DEBUG_COUNT > 0:
                                snapshot = tracemalloc.take_snapshot()
                                top_stats = snapshot.statistics('lineno')
                                print("[ Top 10 Lines ]")
                                for stat in top_stats[:10]:
                                    print(stat)
                                top_stats = snapshot.statistics('filename')
                                print("[ Top 10 Files ]")
                                for stat in top_stats[:10]:
                                    print(stat)
                                top_stats = snapshot.statistics('lineno', cumulative=True)
                                print("[ Top 10 Lines ]")
                                for stat in top_stats[:10]:
                                    print(stat)
                                top_stats = snapshot.statistics('filename', cumulative=True)
                                print("[ Top 10 Files ]")
                                for stat in top_stats[:10]:
                                    print(stat)
                                    '''

                            gc.collect()
                            objs = gc.get_objects()
                            print(f"[*] ({DEBUG_COUNT}) Number of objects: {len(objs)}")
                            size = list(map(sys.getsizeof, objs))
                            pairs = list(zip(objs, size))
                            pairs.sort(key=lambda x: x[1], reverse=True)
                            # Print the top 10 largest objects
                            for idx, (obj, size) in enumerate(pairs[:10]):
                                print(f"[*] {idx}: Type: {type(obj)}, Size: {size}", end='')
                                if type(obj) == torch.storage.UntypedStorage:
                                    print(f", is pined: {obj.is_pinned()}, is shared: {obj.is_shared()}", end='')
                                print()
                            del objs, size, pairs
                        DEBUG_COUNT += 1
                        # input()

                # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
                # prof.export_memory_timeline(f"profiler_output{self.global_step}.html")

                # End of epoch operations
                # -----------------------------------------------------
                if epoch % self.config['log']['save_every'] == 0:
                    self.save_model()

                self.scheduler_gen.step()
                self.scheduler_dis.step()
                # -----------------------------------------------------


    def extract_latents(self, latent_pool_path):
        '''Extract latents (not compressed) and save them.

        If you need compressed latent for EntropyBottleneck, use 
        layer.forward_modifiable(e), then use layer.forward_finish(e) after 
        modifying it. 
        For ImageCompressor, see the source code. 
        '''
        print('Start extracting.')
        self.model.eval()  # just to be safe

        loaders: Dict[str, Iterable[Dict[str, Tensor]]] = {
                k: utility.cycle_iterator(v) for k, v in self.loaders.items()
                }
        max_len = max(len(l) for l in self.loaders.values())

        start_epoch = self.epoch
        with trange(0, max_len) as pbar:
            self.epoch = -1
            for batch_idx in range(max_len):
                self.global_step += 1

                # Load data
                assert 'train' in loaders, "No train data loader"
                with torch.no_grad():
                    paired_vars_l = next(loaders['train'])
                    paired_vars = {
                            'x_0': paired_vars_l['B_img'].cuda(),
                            'y_0': paired_vars_l['A_img'][:, 0, ...].cuda(),
                            'y_0_key': paired_vars_l['A_img'][:, 1, ...].cuda(),
                            }

                    eg = self.model(
                            'latent', 
                            y=paired_vars['y_0'],
                            compress=False)
                    el = self.model(
                            'latent_local',
                            y=paired_vars['y_0'],
                            compress=False)
                for image_idx in range(eg.shape[0]):
                    # Save the latent representation
                    latent_representation = eg[image_idx].cpu()
                    latent_representation_local = el[image_idx].cpu()
                    torch.save(latent_representation, os.path.join(latent_pool_path, f"latent_{batch_idx:05d}_{image_idx:03d}.pt"))
                    torch.save(latent_representation_local, os.path.join(latent_pool_path, f"latent_local_{batch_idx:05d}_{image_idx:03d}.pt"))
                del eg, el, latent_representation, latent_representation_local
                pbar.update(1)


    def reapply_latent(self, latent_loader, output_dir: Path):
        '''Reapplying degradation latents to clean images.  It is assumed 
        that latents are compressed & decompressed by the latent loader if 
        needed.
        '''
        from PIL import Image
        import numpy as np
        import json
        from concurrent.futures import ProcessPoolExecutor
        def save_image(img: torch.Tensor, save_path: Union[Path,str]):
            if isinstance(save_path, Path):
                save_path = str(save_path.absolute()) 
            if len(img.shape) == 4:
                img = img.squeeze(0)
            if isinstance(img, torch.Tensor):
                img = img.cpu().detach().numpy()
            img = (img + 1) / 2
            img = img.transpose(1, 2, 0)
            img = (img * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img)
            img.save(save_path)
        print('Start reapplying.')
        self.model.eval()  # just to be safe
        loader_len = len(self.loaders['reproduce'])
        loader = utility.cycle_iterator(self.loaders['reproduce'])
        path_pairs = []

        with trange(loader_len) as pbar: #, ProcessPoolExecutor() as executor:
            for batch_idx in range(loader_len): # Need to use index to avoid strange behaviour
                with torch.no_grad():
                    paired_vars_l = next(loader)
                    paired_vars = {
                            'x_0': paired_vars_l['B_img'].cuda(),
                            'y_0': paired_vars_l['A_img'][:, 0, ...].cuda(),
                            'y_0_key': paired_vars_l['A_img'][:, 1, ...].cuda(),
                            }
                    eg, el = latent_loader.get_random()
                    eg, el = eg.cuda(), el.cuda()

                    random_states = None
                    if self.add_random_state:
                        random_states = [torch.randn_like(paired_vars['x_0'][:, 0:1, :, :])]

                    yhs = self.model(
                            'partial',
                            x=paired_vars['x_0'],
                            eg=eg, el=el,
                            random_states=random_states,
                            )
                    yh = yhs[0]
                    # save the images
                    # bottlednecked for PNG compression, needs parallelization
                    for image_idx in range(yh.shape[0]):
                        # Save the latent representation
                        image = yh[image_idx].cpu()
                        ref_path = paired_vars_l['B_path'][image_idx]
                        dist_path = output_dir / f'{batch_idx:05d}_{image_idx:03d}.png'
                        path_pairs.append((ref_path, dist_path.name))
                        print(f"Saved {dist_path} from {ref_path}")
                        # executor.submit(save_image, image, dist_path)
                        save_image(image, dist_path)
                    del eg, el, yh
                pbar.update(1)
            json.dump(path_pairs, open(output_dir / 'path_pairs.json', 'w'), indent=4)


    def visualize(self, 
                  paired_vars: Dict[str, Tensor],
                  paired_vars_l: Dict[str, Tensor],
                  res_imgs: Dict[str, List[Tensor]],
                  permanent: bool = False
                  ):
        visuals = { }
        ori_paths = { }
        if paired_vars is not None:
            visuals = {
                    'x_0': paired_vars['x_0'],
                    'y_0': paired_vars['y_0'],
                    'y_0h': res_imgs['y_0hs'][0],
                    }
            ori_paths = {
                    'x_0': paired_vars_l['B_path'][0],
                    'y_0': paired_vars_l['A_path'][0]
                    }
            if len(res_imgs['y_0hs']) > 1:
                visuals['y_0h_2'] = res_imgs['y_0hs'][1]
        self.visualizer.display_current_results(
                visuals=visuals, 
                epoch=self.epoch, global_step=self.global_step, 
                original_img_path=ori_paths, 
                permanent=permanent)


def test_trainer():
    import yaml

    # Load the configuration
    with open('options/test_trainer.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize the model, dataset, and logger
    combined_model = CombinedModel(config)  # You may need to provide parameters
    dataset = single_dataset.SingleDataset(config)  # You may need to provide parameters
    logger = Logger(config)  # You may need to provide parameters

    # Initialize the trainer
    trainer = Trainer(config, combined_model, {'train': dataset}, logger)

    # Train for one step
    trainer.train_one_step()

    # Save the model
    trainer.save_model()

    # Load the model
    trainer.load_model()

    # Log the results
    trainer.log_var('test_var', torch.tensor(1.0))
    print(trainer.log_flush())

if __name__ == '__main__':
    test_trainer()

