# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch import random
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from random import randint
from training.camera_utils import get_cam2world_matrix
from torch.nn import functional as F
from torchvision.utils import save_image
import os
#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg, nerf_resolution, random_swap_prob): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, 
    pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, rank=None, random_swap_prob=0.0
    ):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = 10 # blur_init_sigma
        self.blur_fade_kimg     = 200 # blur_fade_kimg
        self.rank = rank
        self.network_type = 'eg3d'
        self.random_swap_prob = 0.5
        
    def set_nerf_init_args(self, args):
        pass

    def trans_c_to_matrix(self, c):
        bs = c.shape[0]
        c = get_cam2world_matrix(c, device=c.device)  #  [:, :3]  # b, 3, 4
        return c  # b,4,4


    def run_G(self, z, c1, c2=None, update_emas=False, global_step=None, nerf_resolution=64):
        angles = c1
        bs = c1.shape[0]
        if self.random_swap_prob > 0.0:
            swap_idx = torch.where(torch.rand(size=(bs,)) < self.random_swap_prob)[0]
            cond = c1.clone()
            cond[swap_idx] = c2[swap_idx]
        else:
            cond = c1
        cond = self.trans_c_to_matrix(cond).reshape(bs, 16)
        ws = self.G.mapping(z, cond, update_emas=update_emas)
        # if self.style_mixing_prob > 0:
        #     with torch.autograd.profiler.record_function('style_mixing'):
        #         cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
        #         cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
        #         ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c1, update_emas=False)[:, cutoff:]
        if self.network_type == 'eg3d':
            bs = z.shape[0]
            nerf_init_args = {}
            # nerf_init_args['num_steps'] = 36
            # nerf_init_args['img_size'] = nerf_resolution
            # nerf_init_args['fov'] = 12
            # nerf_init_args['nerf_noise'] = 0.5
            # nerf_init_args['ray_start'] = 0.88
            # nerf_init_args['ray_end'] = 1.12

            imgs = self.G(angles=angles, ws=ws, update_emas=update_emas, nerf_init_args=nerf_init_args) # b,3,h,w
        return imgs, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c1, gen_c2, gain, cur_nimg, nerf_resolution=64, random_swap_prob=1.0):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = 0 # max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        assert self.network_type == 'eg3d'
        self.random_swap_prob = max(1 - 0.5 * (cur_nimg / (1000 * 1e3)), 0.5)
        # print(self.random_swap_prob)
        # TODO:改变loss中cond的使用方式
        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, c1=gen_c1, c2=gen_c2, global_step=cur_nimg, nerf_resolution=nerf_resolution)
                gen_logits = self.run_D(gen_img, gen_c1, blur_sigma=blur_sigma)
             
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
    
                # if self.rank == 0:
                #     b = gen_img.shape[0]
                #     tmp = gen_img.reshape(b, 2, 3, 256, 256).reshape(b*2, 3, 256, 256)
                #     save_image(
                #         tmp.detach().cpu(), 
                #         os.path.join(f'./{cur_nimg}.png'), 
                #         nrow=2, 
                #         padding=2, 
                #         normalize=True, 
                #         range=(-1, 1)
                #         )
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # grid sampler不支持二阶导数，因此不能使用path reg
        # Gpl: Apply path length regularization. 只用部分数据
        # if phase in ['Greg', 'Gboth']:
        #     with torch.autograd.profiler.record_function('Gpl_forward'):
        #         batch_size = gen_z.shape[0] // self.pl_batch_shrink
        #         gen_img, gen_ws = self.run_G(gen_z[:batch_size], cond[:batch_size], global_step=cur_nimg)
        #         pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
        #         with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
        #             pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
        #         pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
        #         pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
        #         self.pl_mean.copy_(pl_mean.detach())
        #         pl_penalty = (pl_lengths - pl_mean).square()
        #         training_stats.report('Loss/pl_penalty', pl_penalty)
        #         loss_Gpl = pl_penalty * self.pl_weight
        #         training_stats.report('Loss/G/reg', loss_Gpl)
        #     with torch.autograd.profiler.record_function('Gpl_backward'):
        #         loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c1, c2=gen_c2, update_emas=True, global_step=cur_nimg, nerf_resolution=nerf_resolution)
                gen_logits = self.run_D(gen_img, gen_c1, blur_sigma=blur_sigma, update_emas=True)
            
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
        
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                if self.D.d_channel == 6: # 使用两个判别器
                    real_low = F.interpolate(real_img_tmp, scale_factor=0.25, mode='nearest')
                    real_low = F.interpolate(real_low, scale_factor=4, mode='nearest')
                    real_img_tmp = torch.cat([real_img_tmp.detach(), real_low.detach()], dim=1).requires_grad_(phase in ['Dreg', 'Dboth'])
                # if self.network_type == 'eg3d':
                    # real_img_tmp_resize = F.interpolate(F.interpolate(real_img_tmp, scale_factor=0.5), scale_factor=2.0)
                    # real_img_tmp = torch.cat([real_img_tmp, real_img_tmp_resize], dim=1)  # b, 6, h, w
                    # real_img_tmp = torch.clamp(real_img_tmp, -1, 1)
                    # if self.rank ==0:
                    #     b = real_img_tmp_resize.shape[0]
                    #     tmp = real_img_tmp.reshape(b, 2, 3, 256, 256).reshape(b*2, 3, 256, 256)
                    #     save_image(
                    #         tmp.detach().cpu(), 
                    #         os.path.join(f'./{cur_nimg}.png'), 
                    #         nrow=2, 
                    #         padding=2, 
                    #         normalize=True, 
                    #         range=(-1, 1)
                    #         )
                loss_Dreal = 0
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
