import torch
from training.training_loop_eg3d import save_image_grid
# from training.EG3d_v2 import Generator
# from training.EG3d_v7 import Generator
# from training.EG3d_v8 import Generator
# from training.EG3d_v9 import Generator
from training.EG3d_v12 import Generator
import dnnlib
import legacy
from torch_utils import misc
import os
from torch import nn
from training.training_loop_eg3d import setup_snapshot_image_grid, save_image_grid
import numpy as np


# root = '/home/yangjie08/stylegan3-main/training-runs/EG3d_v12/00003-stylegan2-images1024x1024-gpus8-batch64-gamma1/network-snapshot-004032.pkl'
# root = '/home/yangjie08/stylegan3-main/training-runs/EG3d_v13/00003-stylegan2-images1024x1024-gpus8-batch32-gamma1/network-snapshot-000400.pkl'
root = '/home/yangjie08/stylegan3-main/training-runs/EG3d_v12/00006-stylegan2-images1024x1024-gpus8-batch64-gamma1/network-snapshot-016329.pkl'
gen_num = 16


G_kwargs = dnnlib.EasyDict(
    z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), 
    # init_point_kwargs=dnnlib.EasyDict(
    #     nerf_resolution=128,
    #     fov=12,
    #     d_range=(0.88, 1.12),
    # ),
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
nerf_init_args['nerf_noise'] = 0
nerf_init_args['ray_start'] = 0.88
nerf_init_args['ray_end'] = 1.12
G_kwargs.mapping_kwargs.num_layers = 8 
G_kwargs.fused_modconv_default = 'inference_only'
G_kwargs.conv_clamp = None
common_kwargs = dict(c_dim= 16, #12, 
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

device = torch.device('cuda')
with dnnlib.util.open_url(root) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    G.eval()


grid_size, grid_c = setup_snapshot_image_grid(gen_num, device=torch.device('cuda'))
grid_z = torch.randn([gen_num, G.z_dim], device=device)

meta_data = {'noise_mode': 'const',} #  'trans_x':torch.tensor([0.5, 0, 0]).to(device) }
total_imgs = []
for c in grid_c:
    images = G(z=grid_z, angles=c, nerf_init_args=nerf_init_args, **meta_data)[:, :3]  # b,3,h,w
    # images = G(z=grid_z, c=c, **meta_data)[:, :3]  # b,3,h,w
    res = images.shape[-1]
    # save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)
    total_imgs.append(images)
total_imgs = torch.stack(total_imgs, dim=1).reshape(grid_size[0]*grid_size[1], 3, res, res).cpu().numpy()  # b*5, 3,h,w
thre_imgs = np.where((total_imgs > 1) + (total_imgs < -1), 1, 0)  # 溢出位置为1，否则为0


save_image_grid(total_imgs, 'test_gen_images.png', drange=[-1,1], grid_size=grid_size)
save_image_grid(thre_imgs, 'test_thre_images.png', drange=[-1,1], grid_size=grid_size)


