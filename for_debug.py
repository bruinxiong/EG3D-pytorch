import pickle
import torch
from training.training_loop_eg3d import save_image_grid
from training.EG3d_v2 import Generator
import dnnlib


root = '/home/yangjie08/stylegan3-main/training-runs/00054-stylegan2-images1024x1024-gpus8-batch32-gamma1/network-snapshot-000992.pkl'
G_kwargs = dnnlib.EasyDict(
        class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), 
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
        ),
        )
# G_kwargs.channel_base = opts.cbase
# G_kwargs.channel_max = opts.cmax
G_kwargs.mapping_kwargs.num_layers = 8 
G_kwargs.class_name = 'training.EG3d_v2.Generator'
G_kwargs.fused_modconv_default = 'inference_only'
G_kwargs.num_fp16_res = 0
G_kwargs.conv_clamp = None
common_kwargs = dict(c_dim= 12, #12, 
        img_resolution=256,
        img_channels= 96,
        backbone_resolution=128,
        rank=0,
     )

with open(root, 'rb') as f:
    G_tmp = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
    G = Generator(**G_kwargs, **common_kwargs)
    for p1, p2 in zip(G.parameters(), G_tmp.parameters()):
        p1.copy()

# with open('/home/yangjie08/stylegan3-main/training-runs/00002-stylegan2-images1024x1024-gpus8-batch32-gamma1/network-snapshot-000000.pkl', 'rb') as f:
#     G2 = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module

# for p1, p2 in zip(G1.parameters(), G2.parameters()):
#     print((p1 - p2).abs().sum())

grid_z = torch.randn(4, 512).cuda()
grid_c = torch.randn(4, 5, 3).cuda()

meta_data = {'second_sample_noise_std': 0, 'noise_mode': 'const'}
gen_list = []
for idx in range(grid_c.shape[1]):
    c = grid_c[idx] # 4, 3
    gen_imgs = G(grid_z, c, **meta_data)[:, :3] # b, 3, h, w
    gen_list.append(gen_imgs)
gen_list = torch.stack(gen_list, dim=1).reshape(4*5, 3, 256, 256)
save_image_grid(gen_imgs, 'example.png', (-1, 1), (4, 5))
