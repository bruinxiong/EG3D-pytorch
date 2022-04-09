import torch
from training.training_loop_eg3d import save_image_grid
# from training.EG3d_v14 import Generator
from training.EG3d_v16 import Generator
# from training.EG3d_v12 import Generator
import dnnlib
import legacy
from torch_utils import misc
import os
from torch import nn
from training.training_loop_eg3d import setup_snapshot_image_grid, save_image_grid
import numpy as np
torch.set_grad_enabled(False)
from tqdm import tqdm
import math
import imageio

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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


# root = '/home/yangjie08/stylegan3-main/training-runs/EG3d_v12/00003-stylegan2-images1024x1024-gpus8-batch64-gamma1/network-snapshot-004032.pkl'
# root = '/home/yangjie08/stylegan3-main/training-runs/EG3d_v14/00006-stylegan2-images1024x1024-gpus8-batch32-gamma1/network-snapshot-004000.pkl'
# root = '/home/yangjie08/stylegan3-main/training-runs/EG3d_v14/00009-stylegan2-images1024x1024-gpus8-batch32-gamma1/network-snapshot-003600.pkl'
# root = '/home/yangjie08/stylegan3-main/training-runs/EG3d_v16/00004-stylegan2-images1024x1024-gpus8-batch32-gamma1/network-snapshot-002400.pkl'
# root = '/home/yangjie08/stylegan3-main/training-runs/EG3d_v12/00010-stylegan2-images1024x1024-gpus8-batch64-gamma1/network-snapshot-017539.pkl'
# root = '/home/yangjie08/stylegan3-main/training-runs/EG3d_v14/00009-stylegan2-images1024x1024-gpus8-batch32-gamma1/network-snapshot-025000.pkl'
root = '/home/yangjie08/stylegan3-main/training-runs/EG3d_v16/00004-stylegan2-images1024x1024-gpus8-batch32-gamma1/network-snapshot-012800.pkl'

mode = 'video' # 'video' 'img'
gen_num = 4  # img mode
video_num = 4 # video mode
fps = 30 * 4
# hyper params
img_size =512 #256
nerf_init_args = {'img_size': 64}  # 64

# 不用设置
G_kwargs = dnnlib.EasyDict(
    z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), 
    use_noise=False,  # 关闭noise
    nerf_decoder_kwargs=dnnlib.EasyDict(
        in_c=32,
        mid_c=64,
        out_c=32, 
    )
)
device = torch.device('cuda')
G_kwargs.mapping_kwargs.num_layers = 8 
G_kwargs.fused_modconv_default = 'inference_only'
G_kwargs.conv_clamp = None
common_kwargs = dict(c_dim=16, #12, 
        img_resolution=img_size,
        img_channels= 96,
        backbone_resolution=None,
        rank=0,
    )
G = Generator(**G_kwargs, **common_kwargs)
G.cuda()
G.eval()
if root:
    print(f'Resuming from "{root}"')
    cwd = os.getcwd()
    os.chdir('./training')
    with dnnlib.util.open_url(root) as f:
        resume_data = legacy.load_network_pkl(f)
    for name, module in [('G_ema', G)]:
       misc.copy_params_and_buffers(resume_data[name], module, require_all=True)
    os.chdir(cwd)

# with dnnlib.util.open_url(root) as f:
#     G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
#     G.eval()

G.requires_grad_(False)
meta_data = {'noise_mode': 'const',} #  'trans_x':torch.tensor([0.5, 0, 0]).to(device) }
if mode == 'img':
    grid_size, grid_c = setup_snapshot_image_grid(gen_num, device=torch.device('cuda'))
    grid_z = torch.randn([gen_num, G.z_dim], device=device)
    h = w = img_size
    total_imgs = []
    for c in grid_c:
        cond = torch.zeros_like(c)
        images = G(z=grid_z, angles=c, cond=cond, nerf_init_args=nerf_init_args, **meta_data).reshape(-1, 3, h, w)  # b,3,h,w
        # images = G(z=grid_z, angles=c,  nerf_init_args=nerf_init_args, **meta_data)[:, :3]  # b,3,h,w
        res = images.shape[-1]
        # save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)
        total_imgs.append(images)
        if images.shape[0] == 2 * gen_num:
            grid_size[0] = 2 * gen_num    
    total_imgs = torch.stack(total_imgs, dim=1).reshape(grid_size[0]*grid_size[1], 3, res, res).cpu().numpy()  # b*5, 3,h,w

    thre_imgs = np.where((total_imgs > 1) + (total_imgs < -1), 1, 0)  # 溢出位置为1，否则为0

    save_image_grid(total_imgs, 'test_gen_images.png', drange=[-1,1], grid_size=grid_size)
    save_image_grid(thre_imgs, 'test_thre_images.png', drange=[-1,1], grid_size=grid_size)
elif mode == 'video':
    save_root = './gen_video_examples/cicle'
    grid_z = torch.randn([video_num, G.z_dim], device=device)
    # angles = make_camera_trajectory(fps)
    angles = make_camera_circle_trajectory(num_samples=fps)
    angles = angles.cuda()
    for idx in range(video_num):
        z = grid_z[idx:idx+1]
        z = z.expand(fps, -1)
        images = []
        for f_idx in tqdm(range(fps)):
            zz = z[f_idx:f_idx+1]
            cond = torch.zeros_like(angles[f_idx:f_idx+1])
            # image = G(z=zz, angles=angles[f_idx:f_idx+1], nerf_init_args=nerf_init_args, **meta_data)[:, :3].squeeze()  # 3,h,w
            image = G(z=zz, angles=angles[f_idx:f_idx+1], cond=cond, nerf_init_args=nerf_init_args, **meta_data)[:, :3].squeeze()  # b,3,h,w
            image = image.cpu().numpy()
            image = trans_to_img(image, (-1, 1))
            images.append(image)

        save_dir = os.path.join(save_root, f'{idx}')
        os.makedirs(save_dir, exist_ok=True)
        for f_idx in tqdm(range(fps)):  # TODO: 多进程执行
            image = images[f_idx]
            imageio.imwrite(os.path.join(save_dir, f'{f_idx}.png'), image)

    import cv2
    select_idx = list(range(video_num))
    n_row = int(math.sqrt(video_num))
    n_col = int(math.sqrt(video_num))
    # print(n_row, n_col)
    # exit()
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(
        './video_example.mp4', fourcc,
        30, (n_col*img_size, n_row*img_size))
    for f_idx in range(fps):
        imgs = np.zeros(shape=(n_row*img_size, n_col*img_size, 3), dtype=np.uint8)
        for v_idx in range(len(select_idx)):
            img_path = os.path.join(save_root, f'{select_idx[v_idx]}', f'{f_idx}.png')
            img = cv2.imread(img_path)
            row = v_idx // n_col
            col = v_idx % n_col
            imgs[row*img_size:(row+1)*img_size, col*img_size:(col+1)*img_size] = img
        writer.write(imgs)
    writer.release()
