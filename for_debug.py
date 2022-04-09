
import torch
from training.training_loop_eg3d import save_image_grid
# from training.EG3d_v2 import Generator
from training.EG3d_v16 import Generator
torch.set_grad_enabled(False)
import dnnlib
import legacy
from torch_utils import misc
import os
from torch import nn



root = '/home/yangjie08/stylegan3-main/training-runs/EG3d_v16/00000-stylegan2-images1024x1024-gpus8-batch32-gamma1/network-snapshot-001800.pkl'

nerf_init_args = {}
nerf_init_args['img_size'] = 64


# hyper params
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
    for name, module in [('G', G)]:  # G_ema
       misc.copy_params_and_buffers(resume_data[name], module, require_all=True)
    os.chdir(cwd)
for n, p in G.named_parameters():
    if torch.any(torch.isnan(p)):
        print(n)
for n, b in G.named_buffers():
    if torch.any(torch.isnan(b)):
        print(n)
exit()

# device = torch.device('cuda')
# with dnnlib.util.open_url(root) as f:
#     G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
#     G.eval()
# G.requires_grad_(False)


gh = 4
grid_z = torch.randn(gh, 512).cuda()
grid_c = torch.tensor([0,0,0]).reshape(1, -1).expand(gh, -1).cuda() # 4, 3

meta_data = {'noise_mode': 'const'}
with torch.no_grad():
    gen_imgs = G(grid_z, grid_c, nerf_init_args=nerf_init_args, **meta_data)[:, :3] # gh, 3, h, w
    # print(gen_imgs.shape, gen_imgs.max(), bgen_imgs.min())
save_image_grid(gen_imgs.cpu().numpy(), 'example.png', (-1, 1), (1, gh))

