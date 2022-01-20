
import torch
from training.training_loop_eg3d import save_image_grid
from training.EG3d_v2 import Generator
# from training.EG3d_v3 import Generator
import dnnlib
import legacy
from torch_utils import misc
import os
from torch import nn



root = '/home/yangjie08/stylegan3-main/training-runs/00059-stylegan2-images1024x1024-gpus8-batch32-gamma1/network-snapshot-000204.pkl'
G_kwargs = dnnlib.EasyDict(
      z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), 
        init_point_kwargs=dnnlib.EasyDict(
            nerf_resolution=128,
            fov=12,
            d_range=(0.88, 1.12),
            num_steps=48,  # 实际会有2倍
        ),
        use_noise=False,  # 关闭noise
        nerf_decoder_kwargs=dnnlib.EasyDict(
           in_c=32,
           mid_c=64,
           out_c=32, 
        )
    )
# G_kwargs.channel_base = opts.cbase
# G_kwargs.channel_max = opts.cmax
G_kwargs.mapping_kwargs.num_layers = 8 
G_kwargs.fused_modconv_default = 'inference_only'
G_kwargs.conv_clamp = None
common_kwargs = dict(c_dim= 0, #12, 
        img_resolution=256,
        img_channels= 96,
        backbone_resolution=128,
        rank=0,
     )

G = Generator(**G_kwargs, **common_kwargs)
G.cuda()
G.eval()
G.requires_grad_(False)
# if root:
    # print(f'Resuming from "{root}"')
    # cwd = os.getcwd()
    # os.chdir('./training')
    # with dnnlib.util.open_url(root) as f:
    #     resume_data = legacy.load_network_pkl(f)
    # for name, module in [('G_ema', G)]:
    #    misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
    # os.chdir(cwd)

with open(root, 'rb') as f:
    cwd = os.getcwd()
    tar_path = os.path.abspath('./training')
    os.chdir(tar_path)
    import pickle
    G2 = pickle.load(f)['G_ema'].cuda().eval()  # torch.nn.Module
    misc.copy_params_and_buffers(G2, G, require_all=False)
    os.chdir(cwd)
gh = 4
gw = 5
grid_z = torch.randn(gh, 512).cuda()
grid_c = torch.randn(gh, gw, 3).cuda()

meta_data = {'second_sample_noise_std': 0, 'noise_mode': 'const'}
gen_list = []
for idx in range(grid_c.shape[1]):
    c = grid_c[:, idx] # gh, 3
    with torch.no_grad():
        gen_imgs = G(grid_z, c, **meta_data)[:, :3] # gh, 3, h, w
    gen_list.append(gen_imgs)
gen_list = torch.stack(gen_list, dim=1).reshape(gh*gw, 3, 256, 256).cpu().numpy()
save_image_grid(gen_list, 'example.png', (-1, 1), (gh, gw))

