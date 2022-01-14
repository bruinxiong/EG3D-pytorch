from matplotlib import projections
from numpy.core.defchararray import title
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.nn import functional as F
from einops import rearrange, repeat


def get_init_points(nerf_resolution, fov, d_range, num_steps, device=torch.device('cpu')):
    h = w = nerf_resolution
    n_points = h * w
    pixel_locations = torch.meshgrid(
        torch.linspace(-1, 1, w, device=device), torch.linspace(-1, 1, h, device=device))
    pixel_locations = torch.stack(
        [pixel_locations[0], pixel_locations[1]],
        dim=-1).reshape(-1, 2)  # h*w,2
    # depth = torch.ones(size=(n_points, 1), device=device) * -1.0  # z轴
    depth = -torch.ones(size=(n_points, 1), device=device) / \
        np.tan((math.pi * fov / 180.0) / 2)  # (HxW, )
    points = torch.cat([pixel_locations, depth], dim=1)  # h*w, 3

    rays_d_cam = normalize_vecs(points)
    # 和CIPS不一样，我让zvals代表z轴的scale，而不是整个向量的scale
    z_vals = torch.linspace(d_range[0],
                            d_range[1],
                            num_steps,
                            device=device) \
        .reshape(1, num_steps, 1) \
        .repeat(n_points, 1, 1)  # (HxW, n, 1)
    # # 计算cos_b和cos_a
    # title_line = torch.sqrt(points[:, 2]**2 + points[:, 0]**2)  # V(x^2+z^2)
    # cos_b = points[:, 2] / title_line
    # title_line_2 = torch.sqrt(
    #     title_line**2 + points[:, 1])  # V(V(x^2+z^2) + y^2)
    # cos_a = title_line / title_line_2  # hw,
    # cos_a = cos_a.reshape(n_points, 1, 1).repeat(1, num_steps, 1)
    # cos_b = cos_b.reshape(n_points, 1, 1).repeat(1, num_steps, 1)
    # z_vals = z_vals / cos_b / cos_a  # z的间隔保持一定下，每个射线的模值
    # print(z_vals.shape)
    points = rays_d_cam.unsqueeze(1).repeat(
        1, num_steps, 1) * z_vals  # hw, n, 3
    return points, rays_d_cam, z_vals


def perturb_points(points,
                   z_vals,
                   ray_directions,
                   device):
    """
    Perturb z_vals and then points

    :param points: (n, num_rays, n_samples, 3)
    :param z_vals: (n, num_rays, n_samples, 1)
    :param ray_directions: (n, num_rays, 3)
    :param device:
    :return:
    points: (n, num_rays, n_samples, 3)
    z_vals: (n, num_rays, n_samples, 1)
    """
    # 这是假定同一射线上的点都是等间隔的
    distance_between_points = z_vals[:, :, 1:2, :] - \
        z_vals[:, :, 0:1, :]  # (n, num_rays, 1, 1)
    offset = (torch.rand(z_vals.shape, device=device) - 0.5) \
        * distance_between_points  # [-0.5, 0.5] * d, (n, num_rays, n_samples, 1)
    z_vals = z_vals + offset

    points = points + \
        offset * ray_directions.unsqueeze(2)  # (n, num_rays, n_samples, 3)
    return points, z_vals


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    out = vectors / (torch.norm(vectors, dim=-1, keepdim=True))
    return out

# https://blog.csdn.net/huangkangying/article/details/108393392
# senor = w = 2


def get_camera_mat(fov=49.13, invert=True):
    # fov = 2 * arctan( sensor / (2 * focal))
    # focal = (sensor / 2)  * 1 / (tan(0.5 * fov)) # z轴深度
    # in our case, sensor = 2 as pixels are in [-1, 1]
    focal = 1. / np.tan(0.5 * fov * np.pi/180.)
    focal = focal.astype(np.float32)
    mat = torch.tensor([
        [focal, 0., 0., 0.],
        [0., focal, 0., 0.],
        [0., 0., 1, 0.],
        [0., 0., 0., 1.]
    ]).reshape(1, 4, 4)

    if invert:
        mat = torch.inverse(mat)
    return mat


def get_hierarchical_point(sigmas, z_vals, noise_std, device, clamp_mode='relu'):
    # (b, h x w, num_samples - 1, 1)
    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1]
    delta_inf = 1e10 * torch.ones_like(deltas[:, :, :1])  # (b, h x w, 1, 1)
    deltas = torch.cat([deltas, delta_inf], -2)  # (b, h x w, num_samples, 1)

    noise = torch.randn(sigmas.shape, device=device) * \
        noise_std  # (b, h x w, num_samples, 1)

    if clamp_mode == 'softplus':
        alphas = 1 - torch.exp(-deltas * (F.softplus(sigmas + noise)))
    elif clamp_mode == 'relu':
        # (b, h x w, num_samples, 1)
        alphas = 1 - torch.exp(-deltas * (F.relu(sigmas + noise)))
    else:
        assert 0, "Need to choose clamp mode"
    # 1放在最前面，也是符合论文公式(3)，T的终点是(i-1)
    alphas_shifted = torch.cat([torch.ones_like(
        alphas[:, :, :1]), 1 - alphas + 1e-10], -2)  # (b, h x w, num_samples + 1, 1)
    # e^(x+y) = e^x + e^y 所以这里使用cumprod。 nerf原文公式（3）
    # (b, h x w, num_samples, 1)
    weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1]
    return weights


def fancy_integration(rgb_sigma,
                      z_vals,
                      noise_std=0.5,
                      last_back=False,
                      white_back=False,
                      clamp_mode='relu',
                      fill_mode=None):
    """
    # modified from CIPS-3d by yangjie
    Performs NeRF volumetric rendering.

    :param rgb_sigma: (b, h x w, num_samples, dim_rgb + dim_sigma)
    :param z_vals: (b, h x w, num_samples, 1)
    :param device:
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
    device = rgb_sigma.device
    rgbs = rgb_sigma[..., :-1]  # (b, h x w, num_samples, c)
    sigmas = rgb_sigma[..., -1:]  # (b, h x w, num_samples, 1)

    # (b, h x w, num_samples - 1, 1)
    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1]
    delta_inf = 1e10 * torch.ones_like(deltas[:, :, :1])  # (b, h x w, 1, 1)
    deltas = torch.cat([deltas, delta_inf], -2)  # (b, h x w, num_samples, 1)

    noise = torch.randn(sigmas.shape, device=device) * \
        noise_std  # (b, h x w, num_samples, 1)

    if clamp_mode == 'softplus':
        alphas = 1 - torch.exp(-deltas * (F.softplus(sigmas + noise)))
    elif clamp_mode == 'relu':
        # (b, h x w, num_samples, 1)
        alphas = 1 - torch.exp(-deltas * (F.relu(sigmas + noise)))
    else:
        assert 0, "Need to choose clamp mode"

    alphas_shifted = torch.cat([torch.ones_like(
        alphas[:, :, :1]), 1 - alphas + 1e-10], -2)  # (b, h x w, num_samples + 1, 1)
    # e^(x+y) = e^x + e^y 所以这里使用cumprod。 nerf原文公式（3）
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

 # torch.Size([65536, 1, 48, 1]) torch.Size([4, 16384, 3]) torch.Size([4, 16384, 3])
def get_fine_points(weight, z_vals, trans_cam_pos, trans_d_ray, num_steps):
    bs = z_vals.shape[0]
    z_vals = rearrange(z_vals, "b hw s 1 -> (b hw) s")
    z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
    fine_z_vals = sample_pdf(bins=z_vals_mid,
                             weights=weight[:, 1:-1],
                             N_importance=num_steps,
                             det=False).detach()
    
    fine_z_vals = rearrange(fine_z_vals, "(b hw) s -> b hw s 1", b=bs)
    fine_points = trans_cam_pos.unsqueeze(2).contiguous() + \
                  trans_d_ray.unsqueeze(2).contiguous() * \
                  fine_z_vals.expand(-1, -1, -1, 3).contiguous()
    # fine_points = rearrange(fine_points, "b hw s c -> b (hw s) c")
    return fine_points, fine_z_vals


def sample_pdf(bins,
               weights,
               N_importance,
               det=False,
               eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples-1) where N_samples is "the number of coarse samples per ray"
        weights: (N_rays, N_samples-2)
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

def angel2radius(v):
    return v / 180.0 * math.pi

def radius2angle(v):
    return v * 180.0 / math.pi


def euler2mat(angle):
    # copy from video auto encoder
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (a, b, y) in radians -- size = [B, 6], 后三个是平移向量
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    device = angle.device
    # import pdb; pdb.set_trace()

    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z) # b,
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).reshape(B, 3, 3)
    rotMat = xmat @ ymat @ zmat
    v_trans = angle[:,3:]  # b,3
    rotMat = torch.cat([rotMat, v_trans.view([B, 3, 1])], 2) # b,3,4  # F.affine_grid takes 3x4
    total_mat = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1) # b,4,4
    total_mat[:, :3,:] = rotMat
    return total_mat


def get_cam2world_matrix(angles, bs=None, device=torch.device('cpu'), random_range=(10, 30, 0)):
    # angles: bs,3;  if angles is None, 随机采样
    if angles is None:
        cam_x_angle = (torch.rand(size=(bs, 1), device=device) - 0.5) * 2 * random_range[0]
        cam_y_angle = (torch.rand(size=(bs, 1), device=device) - 0.5) * 2 * random_range[1]
        cam_z_angle = (torch.rand(size=(bs, 1), device=device) - 0.5) * 2 * random_range[2]
        # print(cam_x_angle, cam_y_angle, cam_z_angle)
        # exit()
        cam_x_angle = angel2radius(cam_x_angle)  # bs, 1
        cam_y_angle = angel2radius(cam_y_angle)
        cam_z_angle = angel2radius(cam_z_angle)
    else: # 测试阶段
        bs = angles.shape[0]
        angles = angel2radius(angles)  # bs, 3
        cam_x_angle = angles[:, 0:1]  # bs, 1
        cam_y_angle = angles[:, 1:2]
        cam_z_angle = angles[:, 2:]

    origin_pos = torch.zeros(size=(bs, 3), device=device)  # (bs, 3)
    face_center_pos = torch.tensor([0, 0, -1], device=device).unsqueeze(0).repeat(bs, 1) # bs, 3
    zeros_translation = torch.zeros(size=(bs, 3), device=device)  # (bs, 3)
    
    r_input = torch.cat([cam_x_angle, cam_y_angle, cam_z_angle], dim=1) # bs, 3
    r_input = torch.cat([r_input, zeros_translation], dim=-1) # bs, 6
    Rotation_mat = euler2mat(r_input)  # b, 4, 4
    # print(Rotation_mat.shape, face_center_pos.shape)
    homogeneous_face_center_pos = torch.ones(size=(bs, 4), device=device) # (bs, 4)
    homogeneous_face_center_pos[:, :3] = face_center_pos
    # print(Rotation_mat.shape, homogeneous_face_center_pos.shape) # bs, 4, 1
    trans_face_pos =  torch.bmm(Rotation_mat, homogeneous_face_center_pos.unsqueeze(2)).squeeze(2) # b,4
    trans_face_pos = trans_face_pos[:, :3] # b,3


    forward_vector = -trans_face_pos  # bs, 3
    # print((forward_vector ** 2).sum())
    forward_vector = normalize_vecs(forward_vector)
    # cam_pos = forward_vector + origin_pos

    r_input = torch.zeros(size=(bs, 3), device=device)  # 不旋转
    r_input = torch.cat([r_input, forward_vector], dim=1)  # bs, 6
    Trans_mat = euler2mat(r_input)
    total_mat = Trans_mat @ Rotation_mat  # b, 4, 4
    return total_mat


if __name__ == '__main__':
    points, d_ray, z_vals = get_init_points(32, 12, (0.88, 1.12), 12)  # hw, 12, 3
    RT = get_cam2world_matrix(torch.tensor([[0, 20, 0]]))[0]  # 4,4
    n_points = points.shape[0]
    # print(points.shape)
    # exit()
    homo_points = torch.ones(size=(points.shape[0], 12, 4))
    homo_points[:, :, :3] = points  
    trans_points = RT @ homo_points.reshape(n_points * 12, 4).permute(1, 0)
    trans_points = trans_points[:3, ...].permute(1, 0) # hw*12, 3
    trans_points = trans_points.reshape(n_points, 12, 3)

    # print((d_ray ** 2).sum(-1))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(points[:, 0, 0], points[:, 0, 1], points[:, 0, 2],)
    ax.plot(points[:, -1, 0], points[:, -1, 1], points[:, -1, 2],)
    ax.plot(trans_points[:, 0, 0], trans_points[:, 0, 1], trans_points[:, 0, 2],)
    ax.plot(trans_points[:, -1, 0], trans_points[:, -1, 1], trans_points[:, -1, 2],)
    # ax.plot(d_ray[:, 0], d_ray[:, 1], d_ray[:, 2])
    ax.scatter([0], [0], [0])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
