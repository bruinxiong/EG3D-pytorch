import torch
from training.training_loop_eg3d import save_image_grid
# from training.EG3d_v2 import Generator
# from training.EG3d_v7 import Generator
# from training.EG3d_v16 import Generator
from training.EG3d_v12 import Generator
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


# root = '/home/yangjie08/stylegan3-main/training-runs/EG3d_v8/00001-stylegan2-images1024x1024-gpus8-batch64-gamma1/network-snapshot-012096.pkl'
# root = '/home/yangjie08/stylegan3-main/training-runs/EG3d_v16/00004-stylegan2-images1024x1024-gpus8-batch32-gamma1/network-snapshot-002400.pkl'
root = '/home/yangjie08/stylegan3-main/training-runs/EG3d_v12/00006-stylegan2-images1024x1024-gpus8-batch64-gamma1/network-snapshot-016329.pkl'

G_kwargs = dnnlib.EasyDict(
    z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), 
    use_noise=False,  # 关闭noise
    nerf_decoder_kwargs=dnnlib.EasyDict(
        in_c=32,
        mid_c=64,
        out_c=32, 
    )
)

G_kwargs.mapping_kwargs.num_layers = 8 
G_kwargs.fused_modconv_default = 'inference_only'
G_kwargs.conv_clamp = None
common_kwargs = dict(c_dim= 16, #12, 
        img_resolution=512,
        img_channels= 96,
        backbone_resolution=256,
        rank=0,
    )
G = Generator(**G_kwargs, **common_kwargs)
G.cuda()
G.eval()
G.requires_grad_(False)
if root:
    print(f'Resuming from "{root}"')
    cwd = os.getcwd()
    os.chdir('./training')
    with dnnlib.util.open_url(root) as f:
        resume_data = legacy.load_network_pkl(f)
    for name, module in [('G_ema', G)]:
       misc.copy_params_and_buffers(resume_data[name], module, require_all=True)
    os.chdir(cwd)


device = torch.device('cuda')
# with dnnlib.util.open_url(root) as f:
#     G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
#     G.eval()


meta_data = {'noise_mode': 'const'}

grid_z = torch.randn(16, G.z_dim, device=device)
sigmas = G.get_sigma(grid_z, **meta_data).cpu().numpy()
threshold = 50.




print(sigmas.shape)
print('fraction occupied', np.mean(sigmas > threshold))
np.save('march_file.npy', sigmas)


