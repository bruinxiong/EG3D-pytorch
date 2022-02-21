import torch
from training.training_loop_eg3d import save_image_grid
# from training.EG3d_v2 import Generator
# from training.EG3d_v7 import Generator
from training.EG3d_v8 import Generator
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


root = '/home/yangjie08/stylegan3-main/training-runs/EG3d_v8/00001-stylegan2-images1024x1024-gpus8-batch64-gamma1/network-snapshot-012096.pkl'


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
common_kwargs = dict(c_dim=0, #12, 
        img_resolution=256,
        img_channels= 96,
        backbone_resolution=128,
        rank=0,
    )

device = torch.device('cuda')
G = Generator(**G_kwargs, **common_kwargs)
G.cuda()
G.eval()
G.requires_grad_(False)
if root:
    print(f'Resuming from "{root}"')
    with dnnlib.util.open_url(root) as f:
        resume_data = legacy.load_network_pkl(f)
    for name, module in [('G_ema', G)]:
        misc.copy_params_and_buffers(resume_data[name], module, require_all=True)


# device = torch.device('cuda')
# with dnnlib.util.open_url(root) as f:
#     G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
#     G.eval()

# grid_size, grid_c = setup_snapshot_image_grid(8, device=torch.device('cuda'))
meta_data = {'noise_mode': 'const'}

grid_z = torch.randn(16, G.z_dim, device=device)
sigmas = G.get_sigma(grid_z, nerf_init_args=nerf_init_args, **meta_data).cpu().numpy()
threshold = 50.




print(sigmas.shape)
print('fraction occupied', np.mean(sigmas > threshold))
np.save('march_file.npy', sigmas)


