import torch
from training.training_loop_eg3d import save_image_grid
# from training.EG3d_v2 import Generator
from training.EG3d_v7 import Generator
# from training.EG3d_v3 import Generator
import dnnlib
import legacy
from torch_utils import misc
import os
from torch import nn
from training.training_loop_eg3d import setup_snapshot_image_grid, save_image_grid
import numpy as np
import imageio
from tqdm import tqdm
import math

def make_camera_trajectory(fps):
    yaws = torch.linspace(-30, 30, fps)
    # pitchs = torch.linspace(-5, 5, fps // 2)
    pitchs = torch.zeros_like(yaws)
    rolls = torch.zeros_like(pitchs)
    angles = torch.stack([pitchs, yaws, rolls], dim=-1) # fps/2, 3
    # angles = torch.cat(angles, angles[::-1, ...])
    return angles

# def make_camera_circle_trajectory(fps):
#     yaws = torch.linspace(-30, 30, fps // 2)
#     pitchs = torch.linspace(-10, 10, fps // 2)
#     yaws = torch.cat([yaws, torch.linspace(30, -30, fps // 2)])
#     pitchs = torch.cat([pitchs, torch.linspace(10, -10, fps // 2)])
#     rolls = torch.zeros_like(pitchs)
#     angles = torch.stack([pitchs, yaws, rolls], dim=-1)  # fps, 3
#     return angles

# def get_yaw_pitch_by_xyz(x, y, z):
#     yaw = torch.atan2(z, x)
#     pitch = torch.atan2(torch.sqrt(x ** 2 + z ** 2), y)
#     return yaw, pitch

# def make_camera_circle_trajectory(fps, z=0.8, fov=15):
#     # pitchs = torch.linspace(-fov, fov, fps//2) / 180 * math.pi
#     # yaws = torch.acos(z / torch.cos(fov))
    
#     # pitchs_2 = torch.flip(pitchs)
#     # yaws_2 = torch.flip(yaws)
#     beta = torch.linspace(0, 360, fps) / 180 * math.pi
#     # r = torch.sin(pitch / 180 * math.pi) ** 2
#     x = torch.cos(beta) / np.sin(fov / 180 * math.pi)
#     y = torch.sin(beta) /  np.sin(fov / 180 * math.pi)
#     z = torch.ones_like(x) * z
#     yaw, pitch = get_yaw_pitch_by_xyz(x, y, z)
#     angles = torch.stack([
#         pitch, yaw, torch.zeros_like(yaw)
#     ], dim=-1) * 180 / math.pi
#     angles[:, 0] -= 90
#     return angles
    
# def get_yaw_pitch_by_xyz(x, y, z):
#     yaw = math.atan2(z, x)
#     pitch = math.atan2(math.sqrt(x ** 2 + z ** 2), y)
#     return yaw, pitch

# def make_camera_circle_trajectory(r=1,
#                                      alpha=3.141592 / 6,  # 30度
#                                      num_samples=36,
#                                      periods=1):
#     num_samples = num_samples * periods
#     xyz = np.zeros((num_samples, 3), dtype=np.float32)


#     xyz[:, 2] = r * math.cos(alpha)  # Z轴截面位置
#     z_sin = r * math.sin(alpha) # 画圆的半径

#     for idx, t in enumerate(np.linspace(1, 0, num_samples)):
#         beta = t * 2 * math.pi * periods  # 360 -> 0
#         xyz[idx, 0] = z_sin * math.cos(beta)
#         xyz[idx, 1] = z_sin * math.sin(beta)
#     lookup = - xyz

#     yaws = np.zeros(num_samples)
#     pitchs = np.zeros(num_samples)
#     for idx, (x, y, z) in enumerate(xyz):
#         yaw, pitch = get_yaw_pitch_by_xyz(x, y, z)
#         yaws[idx] = yaw
#         pitchs[idx] = pitch

#     pitchs = torch.from_numpy(pitchs)
#     yaws = torch.from_numpy(yaws)
#     angles = torch.stack([
#         pitchs,
#         yaws,
#         torch.zeros_like(pitchs)
#     ], dim=-1)
#     angles = angles / math.pi * 180
#     angles[:, 0] -= 90
#     angles[:, 1] -= 90
#     return angles

def make_camera_circle_trajectory(num_samples):
    fps = num_samples
    max_pitch = 15
    max_yaw = 30
    pitch1 = torch.linspace(max_pitch, 0, fps // 4)
    yaws1 = torch.linspace(0, max_yaw, fps // 4)

    pitch2 = torch.linspace(0, -max_pitch, fps // 4)
    yaws2 = torch.linspace(max_yaw, 0, fps//4)

    pitch3 = torch.linspace(-max_pitch, 0, fps//4)
    yaws3 = torch.linspace(0, -max_yaw, fps//4)

    pitch4 = torch.linspace(0, max_pitch, fps // 4)
    yaws4 = torch.linspace(-max_yaw, 0, fps//4)

    pitch = torch.cat([pitch1, pitch2, pitch3, pitch4])
    yaws = torch.cat([yaws1, yaws2, yaws3, yaws4])
    angles = torch.stack(
        [
            pitch, yaws, torch.zeros_like(yaws)
        ], dim=-1
    )
    return angles

def trans_to_img(img, drange):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    img = img.transpose(1, 2, 0)
    return img


root = '/home/yangjie08/stylegan3-main/training-runs/EG3d_v7/00002-stylegan2-images1024x1024-gpus8-batch64-gamma1/network-snapshot-019353.pkl'
save_root = './gen_video_examples/cicle'
os.makedirs(save_root, exist_ok=True)
fps = 30 * 4


G_kwargs = dnnlib.EasyDict(
    z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), 
    init_point_kwargs=dnnlib.EasyDict(
        nerf_resolution=128,
        fov=12,
        d_range=(0.88, 1.12),
    ),
    use_noise=False,  # 关闭noise
    nerf_decoder_kwargs=dnnlib.EasyDict(
        in_c=32,
        mid_c=64,
        out_c=32, 
    )
)
nerf_init_args = {}
nerf_init_args['num_steps'] = 36
nerf_init_args['img_size'] = 64
nerf_init_args['fov'] = 12
nerf_init_args['nerf_noise'] = 0.5
nerf_init_args['ray_start'] = 0.88
nerf_init_args['ray_end'] = 1.12

G_kwargs.mapping_kwargs.num_layers = 8 
G_kwargs.fused_modconv_default = 'inference_only'
G_kwargs.conv_clamp = None
common_kwargs = dict(c_dim= 0, #12, 
        img_resolution=256,
        img_channels= 96,
        backbone_resolution=128,
        rank=0,
    )

# G = Generator(**G_kwargs, **common_kwargs)
# G.cuda()
# G.eval()
# G.requires_grad_(False)
# if root:
#     print(f'Resuming from "{root}"')
#     cwd = os.getcwd()
#     os.chdir('./training')
#     with dnnlib.util.open_url(root) as f:
#         resume_data = legacy.load_network_pkl(f)
#     for name, module in [('G_ema', G)]:
#        misc.copy_params_and_buffers(resume_data[name], module, require_all=True)
#     os.chdir(cwd)

# device = torch.device('cuda')
# with dnnlib.util.open_url(root) as f:
#     G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
#     G.eval()

# meta_data = {'noise_mode': 'const'}


# video_num = 40
# grid_z = torch.randn([video_num, G.z_dim], device=device)
# # angles = make_camera_trajectory(fps)
# angles = make_camera_circle_trajectory(num_samples=fps)
# angles = angles.cuda()
# print(angles)
# for idx in range(video_num):
#     z = grid_z[idx:idx+1]
#     z = z.expand(fps, -1)
#     images = []
#     for f_idx in tqdm(range(fps)):
#         zz = z[f_idx:f_idx+1]
#         image = G(z=zz, angles=angles[f_idx:f_idx+1], nerf_init_args=nerf_init_args, **meta_data)[:, :3].squeeze()  # 3,h,w
#         image = image.cpu().numpy()
#         image = trans_to_img(image, (-1, 1))
#         images.append(image)

#     save_dir = os.path.join(save_root, f'{idx}')
#     os.makedirs(save_dir, exist_ok=True)
#     for f_idx in tqdm(range(fps)):
#         image = images[f_idx]
#         imageio.imwrite(os.path.join(save_dir, f'{f_idx}.png'), image)
    


# exit()
# make_video
import cv2
select_idx = [1,3,4,6,8,9,10,15,17,18,19,20,22,23,25,27]
# select_idx = list(range(video_num))
n_row = 4
n_col = 4

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter(
    './video_example.mp4', fourcc,
    30, (n_col*256, n_row*256))
for f_idx in range(fps):
    imgs = np.zeros(shape=(n_row*256, n_col*256, 3), dtype=np.uint8)
    for v_idx in range(len(select_idx)):
        img_path = os.path.join(save_root, f'{select_idx[v_idx]}', f'{f_idx}.png')
        img = cv2.imread(img_path)
        row = v_idx // n_col
        col = v_idx % n_col
        imgs[row*256:(row+1)*256, col*256:(col+1)*256] = img
    writer.write(imgs)
writer.release()
    