import random
import numpy as np
import math
import logging
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def sample_pdf(bins,
               weights,
               N_importance,
               det=False,
               eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: (N_rays, N_importance), the sampled samples
    Source: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    # (N_rays, N_samples_)
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cumsum(pdf, -1)
    # (N_rays, N_samples_+1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)
    # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack(
        [below, above], -1).view(N_rays, 2 * N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    denom[denom < eps] = 1
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / \
        denom * (bins_g[..., 1] - bins_g[..., 0])
    return samples

def fancy_integration(rgb_sigma,
                      z_vals,
                      device,
                      noise_std=0.5,
                      last_back=False,
                      white_back=False,
                      fill_mode=None):
    """
    Performs NeRF volumetric rendering.

    :param rgb_sigma: (b, h x w, num_samples, dim_rgb + dim_sigma)
    :param z_vals: (b, h x w, num_samples, 1)
    :param device:
    :param dim_rgb: rgb feature dim
    :param noise_std:
    :param last_back:
    :param white_back:
    :param clamp_mode:
    :param fill_mode:
    :return:
    - rgb_final: (b, h x w, dim_rgb)
    - depth_final: (b, h x w, 1)
    - weights: (b, h x w, num_samples, 1)
    """

    rgbs = rgb_sigma[..., :-1]  # (b, h x w, num_samples, 32)
    sigmas = rgb_sigma[..., -1:]  # (b, h x w, num_samples, 1)

    # (b, h x w, num_samples - 1, 1)
    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1]
    delta_inf = 1e10 * torch.ones_like(deltas[:, :, :1])  # (b, h x w, 1, 1)
    deltas = torch.cat([deltas, delta_inf], -2)  # (b, h x w, num_samples, 1)

    noise = torch.randn(sigmas.shape, device=device) * \
        noise_std  # (b, h x w, num_samples, 1)

    # if clamp_mode == 'softplus':
    #     alphas = 1 - torch.exp(-deltas * (F.softplus(sigmas + noise)))
    # elif clamp_mode == 'relu':
    #     # (b, h x w, num_samples, 1)
    # print(deltas.shape, sigmas.shape, noise.shape)
    alphas = 1 - torch.exp(-deltas * (F.relu(sigmas + noise)))
    # else:
    #     assert 0, "Need to choose clamp mode"

    alphas_shifted = torch.cat([torch.ones_like(
        alphas[:, :, :1]), 1 - alphas + 1e-10], -2)  # (b, h x w, num_samples + 1, 1)
    # (b, h x w, num_samples, 1)
    weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1]
    weights_sum = weights.sum(2)

    if last_back:
        weights[:, :, -1] += (1 - weights_sum)

    rgb_final = torch.sum(weights * rgbs, -2)  # (b, h x w, num_samples, 3)
    depth_final = torch.sum(weights * z_vals, -2)  # (b, h x w, num_samples, 1)

    if white_back:
        rgb_final = rgb_final + 1 - weights_sum

    if fill_mode == 'debug':
        rgb_final[weights_sum.squeeze(-1) < 0.9] = torch.tensor(
            [1., 0, 0], device=rgb_final.device)
    elif fill_mode == 'weight':
        rgb_final = weights_sum.expand_as(rgb_final)

    return rgb_final, depth_final, weights


@torch.no_grad()
def get_fine_points_and_direction(
    coarse_output,
    z_vals,
    nerf_noise,
    num_steps,
    transformed_ray_origins,
    transformed_ray_directions,
    device,
):
    """

    :param coarse_output: (b, h x w, num_samples, rgb_sigma)
    :param z_vals: (b, h x w, num_samples, 1)
    :param clamp_mode:
    :param nerf_noise:
    :param num_steps:
    :param transformed_ray_origins: (b, h x w, 3)
    :param transformed_ray_directions: (b, h x w, 3)
    :return:
    - fine_points: (b, h x w x num_steps, 3)
    - fine_z_vals: (b, h x w, num_steps, 1)
    """

    batch_size = coarse_output.shape[0]

    _, _, weights = fancy_integration(
        rgb_sigma=coarse_output,
        z_vals=z_vals,
        device=device,
        # clamp_mode=clamp_mode,
        noise_std=nerf_noise)

    # weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
    weights = rearrange(weights, "b hw s 1 -> (b hw) s") + 1e-5

    # Start new importance sampling
    # z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
    z_vals = rearrange(z_vals, "b hw s 1 -> (b hw) s")
    z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
    # z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
    # z_vals = rearrange(z_vals, "(b hw) s -> b hw s 1", b=batch_size)
    fine_z_vals = sample_pdf(bins=z_vals_mid,
                             weights=weights[:, 1:-1],
                             N_importance=num_steps,
                             det=False).detach()
    # fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
    fine_z_vals = rearrange(fine_z_vals, "(b hw) s -> b hw s 1", b=batch_size)

    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + \
        transformed_ray_directions.unsqueeze(2).contiguous() * \
        fine_z_vals.expand(-1, -1, -1, 3).contiguous()
    # fine_points = fine_points.reshape(batch_size, img_size * img_size * num_steps, 3)
    # fine_points = rearrange(fine_points, "b hw s c -> b (hw s) c")

    # if lock_view_dependence:
    #   transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
    #   transformed_ray_directions_expanded[..., -1] = -1
    # end new importance sampling
    return fine_points, fine_z_vals
