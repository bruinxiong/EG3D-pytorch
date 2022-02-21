from numpy.core.numeric import outer
from torch._C import device
from training.networks_stylegan2 import *
from torch import nn
from training.camera_utils import *
from torch.nn import functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from glob import glob 
from training.cips_camera_utils_v2 import  my_get_world_points_and_direction
from training.pigan_utils import fancy_integration, sample_pdf

# 和V12的区别

# nerf noise直接归0
# backbone 256； nerf size 128
# decoder只有2层，使用softplus
# 只使用低分辨率
# 归一化系数从0.15 -> 0.18 
# blur sigma [1没有使用， 2使用,2还使用了并行采样代码]
# rgb feature的通道数为3 不是32
# TODO： 水平翻转； random swap;
# 基于stylegna2-ada写一版

@persistence.persistent_class
class Decoder(torch.nn.Module):
    def __init__(self, 
        in_c: int, 
        mid_c: int, 
        out_c: int, 
        activation='relu', 
        num_layers=3,
    ):
        super().__init__()
        num_layers = num_layers
        self.num_layers = num_layers
        self.fc0 = self.create_block(in_c, mid_c, activation=activation)
        for idx in range(1, num_layers - 1):
            layer = self.create_block(mid_c, mid_c, activation=activation)
            setattr(self, f'fc{idx}', layer)
        setattr(self, f'fc{num_layers - 1}', self.create_block(mid_c, out_c+1, activation='none'))

    def create_block(self, in_features: int, out_features: int, activation: str):
        if activation == 'relu':
            return torch.nn.Sequential(
                torch.nn.Linear(in_features, out_features), 
                torch.nn.ReLU()
            )
        elif activation == 'softmax':
            return torch.nn.Sequential(
                torch.nn.Linear(in_features, out_features), 
                torch.nn.Softmax(dim=-1)
            )
        elif activation == 'softplus':
            return torch.nn.Sequential(
                torch.nn.Linear(in_features, out_features), 
                torch.nn.Softplus()
            )
        elif activation == 'none':
            return torch.nn.Linear(in_features, out_features)
        else:
            raise NotImplementedError()
    def forward(self, feature: torch.Tensor):
        x = feature

        bs_n, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1) # bs_n, h, w, c

        # Main layers
        for idx in range(self.num_layers - 1):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)
        x = getattr(self, f'fc{self.num_layers - 1}')(x)
        o = x
        # o = o.reshape(bs_n, h, w, 33).permute(0, 3, 1, 2) # bs_n, c ,h, w
        return o

@persistence.persistent_class
class SuperResolutionNet(nn.Module): # 简化计算量，2倍计算量
    def __init__(self, 
            in_channels, w_dim, resolution, 
            img_channels=3,
            use_fp16 = False,
            channel_base=None,
            num_fp16_res=None,
            channel_max=None,
            # img_resolution=None,
            **block_kwargs
            ):
        super().__init__()
        self.w_dim = w_dim
        self.num_ws = 0
        # channels_dict = {0: 256, 1:128}
        channels_dict = {128: 256, 256:128, 512:64}
        self.n_layer = 1
        for idx in range(self.n_layer):
            is_last = False if idx < (self.n_layer - 1) else True
            tar_res = resolution // (self.n_layer - idx)
            block = SynthesisBlock(in_channels, channels_dict[tar_res], w_dim=w_dim, resolution=tar_res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            setattr(self, f'b{idx}', block)

            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            in_channels = channels_dict[tar_res]
        # for debug
        # self.const = torch.nn.Parameter(torch.randn([32, 128, 128])) 

    def forward(self, x, ws, **block_kwargs):
        img = None
        # for debugger
        # ****************************
        # x = self.const
        # x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        # ************************************************
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for idx in range(self.n_layer):
                block = getattr(self, f'b{idx}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv
        for idx in range(self.n_layer):
            cur_ws = block_ws[idx]
            block = getattr(self, f'b{idx}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

@persistence.persistent_class       
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels                = 96,  # Number of output color channels.
        backbone_resolution         = 128,  # stylegan2的输出，组成volume,  # 先设置为128，加快训练速度
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        nerf_decoder_kwargs      = {},
        rank = None,
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):        
        super().__init__()
        backbone_resolution = 256
        img_resolution = 256
        self.rank = rank
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.backbone_resolution = backbone_resolution
        synthesis_kwargs['use_noise'] = False
        self.super_res = SuperResolutionNet(in_channels=nerf_decoder_kwargs['out_c'], w_dim=w_dim, resolution=img_resolution, 
                use_fp16=True, **synthesis_kwargs)
        synthesis_kwargs['use_noise'] = True
        self.synthesis = SynthesisNetwork(
                    w_dim=w_dim, img_resolution=backbone_resolution, 
                    img_channels=img_channels,
                    num_fp16_res=0,
                     **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws + self.super_res.num_ws  
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

        # self.nerf_decoder = LightDecoder(**nerf_decoder_kwargs)
        nerf_decoder_kwargs['num_layers'] = 2
        nerf_decoder_kwargs['activation'] = 'softplus'
        nerf_decoder_kwargs['out_c'] = 3
        self.nerf_decoder = Decoder(**nerf_decoder_kwargs)
       
        

    def trans_c_to_matrix(self, c):
        bs = c.shape[0]
        c = get_cam2world_matrix(c, device=c.device)  #  [:, :3]  # b, 3, 4
        return c  # b,4,4


    def forward(self, z=None, angles=None, ws=None, truncation_psi=1, truncation_cutoff=None, 
                update_emas=True, nerf_init_args={}, cond=None, **synthesis_kwargs
                ):
        nerf_init_args = {}
        nerf_init_args['num_steps'] = 48
        nerf_init_args['img_size'] = 128  # nerf光线数目
        nerf_init_args['fov'] = 13.6
        nerf_init_args['nerf_noise'] = 0 # 去除noise
        nerf_init_args['ray_start'] = 0.88
        nerf_init_args['ray_end'] = 1.12
        img_size = nerf_init_args['img_size']
        self.nerf_resolution = img_size
        num_steps = nerf_init_args['num_steps']
    
        
        if ws is None: 
            # assert not self.training # 只有测试阶段走这条分支

            # cond = torch.zeros_like(angles) 
            if cond is None: 
                cond = angles # for debug 

            bs = cond.shape[0]
            cond = self.trans_c_to_matrix(cond).reshape(bs, 16)
            ws = self.mapping(z, c=cond, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        # print(ws.keys())
        backbone_feats = self.synthesis(ws[:, :self.synthesis.num_ws], update_emas=update_emas, **synthesis_kwargs)  # b,32*3,128,128
        
        assert backbone_feats.shape[1] == 96
        feat_xy, feat_yz, feat_xz = backbone_feats.chunk(3, dim=1)  # b, 32, 128, 128
        # assert feat_xy.shape[-1] == 256
        nerf_channel = feat_xy.shape[1]  # 32

        bs = feat_xy.shape[0]
        if angles is not None:
            assert angles.shape[1] == 3
            c2w_matrix = self.trans_c_to_matrix(angles)
            c2w_matrix = c2w_matrix
            transformed_points, transformed_ray_directions_expanded, \
            transformed_ray_origins, transformed_ray_directions, z_vals = \
                    my_get_world_points_and_direction(batch_size=bs, device=ws.device, c2w_matrix=c2w_matrix, **nerf_init_args)

        transformed_points = rearrange(transformed_points, "b (h w s) c -> b (h w) s c", h=img_size, s=num_steps)
        # transformed_ray_directions_expanded = rearrange(transformed_ray_directions_expanded,
        #                                                 "b (h w s) c -> b (h w) s c", h=img_size, s=num_steps)
        transformed_points = transformed_points / 0.18  # 0.12

         # 插值
        nerf_feat = self.bilinear_sample_tri_plane(
            transformed_points,
            feat_xy, feat_yz, feat_xz, 
            )  # b*n,c,h,w
        nerf_feat = self.nerf_decoder(nerf_feat)  

        volume_channel = nerf_feat.shape[-1]
        h = w = img_size
        nerf_feat = nerf_feat.reshape(bs, num_steps, h*w, volume_channel).permute(0,2,1,3) # b, hw, n, c+1
        # print(nerf_feat[..., -1].mean())
        # print(F.relu(nerf_feat[..., -1]).mean())    
        if True:
            fine_points, fine_z_vals = self.get_fine_points_and_direction(
            coarse_output=nerf_feat,
            z_vals=z_vals,
            dim_rgb=volume_channel-1,
            nerf_noise=nerf_init_args['nerf_noise'],
            num_steps=num_steps,
            transformed_ray_origins=transformed_ray_origins,
            transformed_ray_directions=transformed_ray_directions
            )
            fine_points = fine_points / 0.18  # # 0.12
            # print(fine_points.shape)
            # print(fine_points.max(), fine_points.min())
            # Model prediction on re-sampled find points
            fine_output = self.bilinear_sample_tri_plane(fine_points, 
                            feat_xy, feat_yz, feat_xz, 
                            )  # b*n,c,h,w)
            fine_output  = self.nerf_decoder(fine_output)  
            fine_output = fine_output.reshape(bs, num_steps, h*w, volume_channel).permute(0,2,1,3) # b, hw, s, c+1
            # Combine course and fine points
            all_outputs = torch.cat([fine_output, nerf_feat], dim=-2)  # (b, n, s, dim_rgb_sigma)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)  # (b, n, s, 1)
            _, indices = torch.sort(all_z_vals, dim=-2)  # (b, n, s, 1)
            all_z_vals = torch.gather(all_z_vals, -2, indices)  # (b, n, s, 1)
            # (b, n, s, dim_rgb_sigma)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, all_outputs.shape[-1]))
        else:
            all_outputs = nerf_feat
            all_z_vals = z_vals
      
        pixels_fea, depth, weights = fancy_integration(
        rgb_sigma=all_outputs,
        # dim_rgb=volume_channel-1,
        z_vals=all_z_vals,
        device=ws.device,
        noise_std=nerf_init_args['nerf_noise'])
        pixels_fea = pixels_fea.reshape(bs, h, w, -1).permute(0, 3, 1, 2)
        pixels_fea = pixels_fea.contiguous()

        return pixels_fea  # 3通道




    def bilinear_sample_tri_plane(self, points, feat_xy, feat_yz, feat_xz):
        b, hw, n = points.shape[:3]
        h = w = self.nerf_resolution
        c = feat_xy.shape[1]
        x = points[..., 0]  # b, hw, n
        y = points[..., 1]
        z = points[..., 2]
        xy = torch.stack([x, y], dim=-1).permute(0, 2, 1, 3)  # b, hw, n, 2 -> b, n, hw, 2
        xz = torch.stack([x, z], dim=-1).permute(0, 2, 1, 3)  # b, hw, n, 2 -> b, n, hw, 2
        yz = torch.stack([y, z], dim=-1).permute(0, 2, 1, 3)  # b, hw, n, 2 -> b, n, hw, 2

        xy = xy.reshape(b*n, h, w, 2)
        xz = xz.reshape(b*n, h, w, 2)
        yz = yz.reshape(b*n, h, w, 2)

        fh, fw = feat_xy.shape[2:] 
        feat_xy = feat_xy.unsqueeze(1).expand(-1, n, -1, -1, -1).reshape(b*n, c, fh, fw)
        feat_xz = feat_xz.unsqueeze(1).expand(-1, n, -1, -1, -1).reshape(b*n, c, fh, fw)
        feat_yz = feat_yz.unsqueeze(1).expand(-1, n, -1, -1, -1).reshape(b*n, c, fh, fw)
        
        xy_f = F.grid_sample(feat_xy, grid=xy, )#padding_mode='border')  # b*n, c, h, w
        xz_f = F.grid_sample(feat_xz, grid=xz)
        yz_f = F.grid_sample(feat_yz, grid=yz)

        result = xy_f + xz_f + yz_f # b*n, c, h, w
        # result = result.reshape(b, n, c, h, w).permute(0, 2, 3, 4, 1) # b, c, h, w, n
        return result

    # def bilinear_sample_tri_plane(self, points, feat_xy, feat_yz, feat_xz):
    #     b, hw, n = points.shape[:3]
    #     h = w = self.nerf_resolution
        
    #     x = points[..., 0]  # b, hw, n
    #     y = points[..., 1]
    #     z = points[..., 2]
    #     xy = torch.stack([x, y], dim=-1).permute(2, 0, 1, 3)  # b, hw, n, 2 -> n, b,hw,2
    #     xz = torch.stack([x, z], dim=-1).permute(2, 0, 1, 3)
    #     yz = torch.stack([y, z], dim=-1).permute(2, 0, 1, 3)
    #     xy = xy.reshape(n, b, h, w, 2)
    #     xz = xz.reshape(n, b, h, w, 2)
    #     yz = yz.reshape(n, b, h, w, 2)
    #     xy_list = []
    #     xz_list = []
    #     yz_list = []
    #     for idx in range(n):
    #         xy_idx = xy[idx] # b, h, w, 2
    #         xz_idx = xz[idx]
    #         yz_idx = yz[idx]
    #         xy_f = F.grid_sample(feat_xy, grid=xy_idx)  # b, c, h, w   # padding_mode='border') 
    #         xz_f = F.grid_sample(feat_xz, grid=xz_idx)
    #         yz_f = F.grid_sample(feat_yz, grid=yz_idx)
    #         xy_list.append(xy_f)
    #         xz_list.append(xz_f)
    #         yz_list.append(yz_f) 
    #     xy_list = torch.stack(xy_list, dim=1)  # b,n, c,h,w
    #     xz_list = torch.stack(xz_list, dim=1)
    #     yz_list = torch.stack(yz_list, dim=1)
    #     o = xy_list + xz_list + yz_list
    #     o = o.reshape(b*n, 32, h, w)
    #     return o
        
    @torch.no_grad()
    def get_fine_points_and_direction(self,
                                    coarse_output,
                                    z_vals,
                                    dim_rgb,
                                    nerf_noise,
                                    num_steps,
                                    transformed_ray_origins,
                                    transformed_ray_directions,
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
        device=coarse_output.device,
        # dim_rgb=dim_rgb,
        noise_std=nerf_noise)

        # weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
        weights = rearrange(weights, "b hw s 1 -> (b hw) s") + 1e-5

        #### Start new importance sampling
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
        #### end new importance sampling
        return fine_points, fine_z_vals




@persistence.persistent_class
class EG3dDiscriminator(nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0, # 4, # 4        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.):
    ):
        super().__init__()
        num_fp16_res = 0
        self.d_channel = img_channels
        self.dis = Discriminator(
            c_dim=c_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            architecture = architecture,
            channel_base=channel_base,
            channel_max=channel_max,
            num_fp16_res=num_fp16_res,
            conv_clamp=conv_clamp,
            cmap_dim=cmap_dim,
            block_kwargs=block_kwargs,
            mapping_kwargs=mapping_kwargs,
            epilogue_kwargs=epilogue_kwargs,
        )

    def trans_c_to_matrix(self, c):
        bs = c.shape[0]
        c = get_cam2world_matrix(c, device=c.device)  #  [:, :3]  # b, 3, 4
        return c  # b,4,4

    def forward(self, img, angles, update_emas=False, **block_kwargs):
        assert angles.shape[1] == 3
        c2w_matrix = self.trans_c_to_matrix(angles)
        bs = img.shape[0]
        c2w_matrix = c2w_matrix.reshape(bs, -1)
        assert c2w_matrix.shape[-1] == 16
        return self.dis(img, c2w_matrix, update_emas=update_emas, **block_kwargs)
                                                           

        










    
