from numpy.core.numeric import outer
from torch._C import device
from .networks_stylegan2 import *
from torch import nn
from .camera_utils import *
from torch.nn import functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from glob import glob 


@persistence.persistent_class
class NerfMLP(nn.Module):
    def __init__(self,):
        super().__init__()
        self.stem = nn.Linear(3, 32)

        self.layers = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
             nn.Linear(64, 64),
            nn.ReLU(),
             nn.Linear(64, 64),
            nn.ReLU(),
             nn.Linear(64, 64),
            nn.ReLU(),
             nn.Linear(64, 64),
            nn.ReLU(),
             nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.sigma = nn.Linear(64, 1)
        self.rgb = nn.Linear(64, 32)
    def forward(self, x):
        x = self.stem(x) # 32
        noise = torch.randn((x.shape[0], 32), device=x.device)
        x = torch.cat([x, noise], dim=1) # b, 64
        b = self.layers(x)
        sigma = self.sigma(b)
        rgb = self.rgb(b)
        return torch.cat([rgb, sigma], dim=1)  # b, 33
        

@persistence.persistent_class
class LightDecoder(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 1, 1, 0),
            nn.Softmax()
        )
        self.sigma = nn.Conv2d(mid_c, 1, 1, 1, 0)
        self.feat = nn.Conv2d(mid_c, out_c, 1, 1, 0)
    def forward(self, x):
        x = self.layers(x)
        sigma = self.sigma(x)
        feat = self.feat(x)
        return torch.cat([feat, sigma], dim=1) # xxx, c+1, h, w

@persistence.persistent_class
class SuperResolutionNet(nn.Module): # 简化计算量，只上采样一倍
    def __init__(self, 
            in_channels, w_dim, resolution, 
            img_channels=3,
            use_fp16 = True, # True # 根据论文描述，超分模块都使用FP16,在GAN init中设置，不是在这
            channel_base=None,
            num_fp16_res=None,
            channel_max=None,
            img_resolution=None,
            **block_kwargs
            ):
        super().__init__()
        self.w_dim = w_dim
        self.num_ws = 0
        # for res in self.block_resolutions:
        #     in_channels = channels_dict[res // 2] if res > 4 else 0
        #     out_channels = channels_dict[res]
        #     use_fp16 = (res >= fp16_resolution)
        #     is_last = (res == self.img_resolution)
        #     block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
        #         img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
        #     self.num_ws += block.num_conv
        #     if is_last:
        #         self.num_ws += block.num_torgb
        #     setattr(self, f'b{res}', block)
        channels_dict = {0: 128, 1:64}
        self.n_layer = resolution // 128 - 1
        for idx in range(self.n_layer):
            is_last = False if idx < (self.n_layer - 1) else True
            block = SynthesisBlock(in_channels, channels_dict[idx], w_dim=w_dim, resolution=resolution,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            setattr(self, f'b{idx}', block)

            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            in_channels = channels_dict[idx]
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
class ModifiedSynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels, 0 = first block.
        out_channels,                           # Number of output channels.
        w_dim,                                  # Intermediate latent (W) dimensionality.
        resolution,                             # Resolution of this block.
        img_channels,                           # Number of output color channels.
        is_last,                                # Is this the last block?
        last_channels           = 3*32,
        architecture            = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter         = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp              = 256,          # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16                = False,        # Use FP16 for this block?
        fp16_channels_last      = False,        # Use channels-last memory format with FP16?
        fused_modconv_default   = True,         # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
        **layer_kwargs,                         # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        # if is_last or architecture == 'skip':
        #     self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
        #         conv_clamp=conv_clamp, channels_last=self.channels_last)
        #     self.num_torgb += 1
        if is_last:
            self.torgb = ToRGBLayer(out_channels, last_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
        

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, update_emas=False, **layer_kwargs):
        _ = update_emas # unused
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)  # conv0执行2倍上采样
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        # if img is not None:
        #     misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
        #     img = upfirdn2d.upsample2d(img, self.resample_filter)
        # if self.is_last: # or self.architecture == 'skip':
            # y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            # y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            # img = img.add_(y) if img is not None else y
        if self.is_last:
            x = self.torgb(x)


        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

@persistence.persistent_class
class Backbone(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            # use_fp16 = (res >= fp16_resolution)
            use_fp16=False  # backbone 不需要使用FP16，超分辨率模块需要使用FP16
            is_last = (res == self.img_resolution)
            block = ModifiedSynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv))  # + block.num_torgb))
                w_idx += block.num_conv


        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, _ = block(x, img, cur_ws, **block_kwargs)
        return x

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])

class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels                = 96,  # Number of output color channels.
        backbone_resolution         = 128,  # stylegan2的输出，组成volume,  # 先设置为128，加快训练速度
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        init_point_kwargs   = {},
        nerf_decoder_kwargs      = {},
        rank = None,
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):        
        super().__init__()
        self.rank = rank
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.backbone_resolution = backbone_resolution
        # self.synthesis = Backbone(w_dim=w_dim, img_resolution=backbone_resolution, img_channels=96, **synthesis_kwargs)
        self.super_res = SuperResolutionNet(in_channels=nerf_decoder_kwargs['out_c'], w_dim=w_dim, resolution=img_resolution, 
                use_fp16=False, **synthesis_kwargs)
        self.synthesis = SynthesisNetwork(
                    w_dim=w_dim, img_resolution=backbone_resolution, 
                    img_channels=img_channels,
                    num_fp16_res=0,
                     **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws + self.super_res.num_ws  
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
        self.nerf_decoder = LightDecoder(**nerf_decoder_kwargs)
        # TODO，射线的数目好像是随着训练次数的增大而增大的
        init_points, rays_d_cam, z_vals = get_init_points(**init_point_kwargs) 
        self.d_range = init_point_kwargs['d_range']
        self.nerf_resolution = init_point_kwargs['nerf_resolution']
        self.register_buffer('init_points', init_points)  # hw,n,3
        self.register_buffer('init_rays_d_cam', rays_d_cam) # hw,3
        self.register_buffer('init_z_vals', z_vals) # (HxW, n, 1)
        # for debug
        self.nerf_mlp = NerfMLP()

    def trans_c_to_matrix(self, c):
        bs = c.shape[0]
        c = get_cam2world_matrix(c, device=c.device)[:, :3]  # b, 3, 4
        c = c.reshape(bs, 12)  # b, 12
        return c

    def forward(self, z=None, c=None, ws=None, truncation_psi=1, truncation_cutoff=None, 
                update_emas=True, second_sample_noise_std=0, **synthesis_kwargs
                ):
        # c是相机参数，作为条件, length为12 我应该只需要世界坐标系的相机位置
        if c.shape[-1] == 3: # 输入的是角度，需要转换一下
            c = self.trans_c_to_matrix(c) # bs, 12
        if ws is None:
            ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        # print(ws.keys())
        backbone_feats = self.synthesis(ws[:, :self.synthesis.num_ws], update_emas=update_emas, **synthesis_kwargs)  # b,32*3,256,256
        
        # for debug
        # ****************************************************************
        # backbone_feats = torch.cat([backbone_feats, backbone_feats], dim=1)  # b,6,h,w
        # return backbone_feats
        # ****************************************************************
        
        assert backbone_feats.shape[1] == 96
        feat_xy, feat_yz, feat_xz = backbone_feats.chunk(3, dim=1)  # b, 32, 256, 256
        nerf_channel = feat_xy.shape[1]  # 32
        bs = feat_xy.shape[0]
        init_points = self.init_points.unsqueeze(0).repeat(bs, 1, 1, 1)  # b,hw, n, 3
        init_rays_d_cam = self.init_rays_d_cam.unsqueeze(0).repeat(bs, 1, 1) # b,hw,3
        init_z_vals = self.init_z_vals.unsqueeze(0).repeat(bs, 1, 1, 1) # b,hw,n,1
        # depth扰动
        # 可能让坐标值超过-1到1, 需要grid sample的位置处理一下
        # TODO，射线的数目好像是随着训练次数的增大而增大的
        points, z_vals = perturb_points(init_points, init_z_vals, init_rays_d_cam, ws.device)
        n_points = points.shape[1]
        n_step = points.shape[2]
        # 世界坐标系相机位置旋转矩阵
        world_matrix = c[:, :12].reshape(bs, 3, 4)
        homo_world_matrix = torch.eye(4, device=ws.device) \
                .unsqueeze(0) \
                .repeat(bs, 1, 1)  # 
        homo_world_matrix[:, :3, :] = world_matrix  # b, 4, 4
        # 计算旋转点
        homo_points = torch.ones(
            (points.shape[0], points.shape[1], points.shape[2], points.shape[3] + 1),
            device=ws.device) # b,hw,12,4  : (x y z 1)
        homo_points[:, :, :, :3] = points
        # 以世界坐标系的原点为中心，xyz的范围都在-1到1之间的三维区域内
        trans_points = torch.bmm(homo_world_matrix, homo_points.reshape(bs, -1, 4).permute(0, 2, 1))\
            .permute(0, 2, 1).reshape(bs, n_points, n_step, 4)[:, :, :, :3]
        trans_points = trans_points / 0.18 # (self.d_range[1] - self.d_range[0]) * 2  # 3个维度坐标都在-1到1之间,有可能越界
        # print(trans_points.max(), trans_points.min())
        # print(trans_points[0])
        # if self.rank == 0:
        #     fig = plt.figure()
        #     p = points[0].detach().cpu().numpy()  # hw, n, 3
        #     tp = trans_points[0].detach().cpu().numpy()
        #     ax = fig.gca(projection='3d')
        #     ax.plot(p[:, 0, 0], p[:, 0, 1], p[:, 0, 2],)
        #     ax.plot(p[:, -1, 0], p[:, -1, 1], p[:, -1, 2],)
        #     ax.plot(tp[:, 0, 0], tp[:, 0, 1], tp[:, 0, 2],)
        #     ax.plot(tp[:, -1, 0], tp[:, -1, 1], tp[:, -1, 2],)
        #     ax.plot(d_ray[:, 0], d_ray[:, 1], d_ray[:, 2])
        #     ax.scatter([0], [0], [0])
        #     ax.set_xlabel('x')
        #     ax.set_ylabel('y')
        #     ax.set_zlabel('z')
        #     num_idx = len(glob('./*.png'))
        #     plt.savefig(f'{num_idx}.png')
        trans_rays_d_cam = torch.bmm(
            homo_world_matrix[..., :3, :3],
            init_rays_d_cam.permute(0, 2, 1)) \
            .permute(0, 2, 1) \
            .reshape(bs, n_points, 3)
        # 没有scale上的变化，所以z_vals不需要处理
        homo_origins = torch.zeros((bs, 4, n_points), device=ws.device)
        homo_origins[:, 3, :] = 1
        # 相机原点在世界坐标系下的位置
        trans_ray_origins = torch.bmm(
            homo_world_matrix,
            homo_origins) \
            .permute(0, 2, 1) \
            .reshape(bs, n_points, 4)
        trans_ray_origins = trans_ray_origins[..., :3] # (bs, num_rays, 3)
        # nerf_feat = self.bilinear_sample_tri_plane(
        #     trans_points,
        #     feat_xy, feat_yz, feat_xz, 
        #     )  # b,c,h,w,n
        
        # nerf_feat = nerf_feat.permute(0, 4, 1, 2, 3).reshape(bs*n_step, nerf_channel,\
        #      self.nerf_resolution, self.nerf_resolution)
        # # 使用light-weight decoder得到不透明度和RGB feature
        # nerf_feat = self.nerf_decoder(nerf_feat)  # bs*n_step, c+1, h, w

        # ***************************
        # for debug
        h = w = self.nerf_resolution
        trans_points_tmp = trans_points.reshape(-1, 3)  # bs*hw*n, 3
        nerf_feat = self.nerf_mlp(trans_points_tmp) # ..., 33
        nerf_feat = nerf_feat.reshape(bs, h, w, n_step, 33).permute(0, 3,4, 1,2).reshape(bs*n_step, 33, h, w) # bs*n, 33, h,w
        # ***************************

        # 第二次根据sigma密度采样
        h = w = self.nerf_resolution
        sigma = nerf_feat[:, -1, ...].reshape(bs, n_step, h, w).permute(0, 2, 3, 1).\
                    reshape(bs, h*w, n_step, 1) # bs, hw, n, 1
        weight = get_hierarchical_point(sigma, z_vals, device=ws.device,
                noise_std=second_sample_noise_std)
        weight = weight.reshape(bs*n_points, n_step) + 1e-5 # rearrange(weights, "b hw s 1 -> (b hw) s") + 1e-5
        # # (bs, hw, n, 3), (# (b, hw, n, 1))
        fine_points, fine_z_vals = get_fine_points(weight, z_vals, trans_ray_origins, trans_rays_d_cam, n_step)
        fine_points = fine_points / 0.18 # 忘记对fine point归一化了
        # *****************
        # for debug
        fine_points_tmp = fine_points.reshape(-1, 3)  # b*hw*n, 3
        fine_nerf_feat = self.nerf_mlp(fine_points_tmp) # ..., 33
        fine_nerf_feat = fine_nerf_feat.reshape(bs, h, w, n_step, 33).permute(0, 3,4, 1,2).reshape(bs*n_step, 33, h, w) # bs*n, 33, h,w
        # *****************

        # fine_nerf_feat = self.bilinear_sample_tri_plane(
        #     fine_points,
        #     feat_xy, feat_yz, feat_xz, 
        #     )  # b,c,h,w,n
        # fine_nerf_feat = fine_nerf_feat.permute(0, 4, 1, 2, 3).reshape(bs*n_step, nerf_channel,
                                        # h, w)
        # fine_nerf_feat = self.nerf_decoder(fine_nerf_feat)  # bs*n_step, c+1, h, w

        volume_channel = nerf_feat.shape[1]
        nerf_feat = nerf_feat.reshape(bs, n_step, volume_channel, h, w).permute(0, 3,4,1,2).\
            reshape(bs, n_points, n_step, volume_channel)
        fine_nerf_feat = fine_nerf_feat.reshape(bs, n_step, volume_channel, h, w).permute(0, 3, 4,1,2).\
            reshape(bs, n_points, n_step, volume_channel)
        nerf_feat = torch.cat([nerf_feat, fine_nerf_feat], dim=2) # bs, hw, 2n, c+1
        z_vals = torch.cat([z_vals, fine_z_vals], dim=2)  # bs, hw, 2n, 1
        _, indices = torch.sort(z_vals, dim=-2)  # (b, hw, 2n, 1)
        z_vals = torch.gather(z_vals, -2, indices)
        nerf_feat = torch.gather(nerf_feat, -2, indices.expand(-1, -1, -1, volume_channel))

        # 开始volume rendering，这里直接使用CIPS3d的代码 # 需要好好阅读一下giffare，CIPS，GRAF的代码
        # 沿用CIPS的代码
        # print(second_sample_noise_std)
        pixels_fea, _, _ = fancy_integration(nerf_feat, z_vals, noise_std=second_sample_noise_std)
        # 使用超分辨率模型，
        pixels_fea = pixels_fea.reshape(bs, h, w, volume_channel-1).permute(0, 3, 1, 2)# bs, 32, h, w
        # gen_low_img = pixels_fea[:, :3, ...] # bs,3,h,w
        ws = ws[:, self.synthesis.num_ws:]  # 超分辨率的ws
        assert ws.shape[1] == self.super_res.num_ws
        gen_high_img = self.super_res(pixels_fea, ws,  **synthesis_kwargs)  # bs, 3, h ,w 
        assert gen_high_img.shape[-1] == 256
        # 拼接两张图像然后输出
        # scale_factor = gen_high_img.shape[-1] // gen_low_img.shape[-1]
        # gen_low_img = F.interpolate(gen_low_img, scale_factor=scale_factor, mode='bilinear')
        # for debug, 两个图像都是高分辨率
        gen_imgs = torch.cat([gen_high_img, gen_high_img], dim=1) # bs, 6, h, w
        return gen_imgs


    def bilinear_sample_tri_plane(self, points, feat_xy, feat_yz, feat_xz):
        # TODO: 消除越界值的影响
        b, hw, n = points.shape[:3]
        h = w = self.nerf_resolution
        
        x = points[..., 0]  # b, hw, n
        y = points[..., 1]
        z = points[..., 2]
        xy = torch.stack([x, y], dim=-1).permute(2, 0, 1, 3)  # b, hw, n, 2 -> n, b,hw,2
        xz = torch.stack([x, z], dim=-1).permute(2, 0, 1, 3)
        yz = torch.stack([y, z], dim=-1).permute(2, 0, 1, 3)
        xy = xy.reshape(n, b, h, w, 2)
        xz = xz.reshape(n, b, h, w, 2)
        yz = yz.reshape(n, b, h, w, 2)
        xy_list = []
        xz_list = []
        yz_list = []
        for idx in range(n):
            xy_idx = xy[idx] # b, h, w, 2
            xz_idx = xz[idx]
            yz_idx = yz[idx]
            xy_f = F.grid_sample(feat_xy, grid=xy_idx, padding_mode='border')  # b, c, h, w
            xz_f = F.grid_sample(feat_xz, grid=xz_idx, padding_mode='border')
            yz_f = F.grid_sample(feat_yz, grid=yz_idx, padding_mode='border')
            xy_list.append(xy_f)
            xz_list.append(xz_f)
            yz_list.append(yz_f) 
        xy_list = torch.stack(xy_list, dim=-1)  # b, c,h,w,n
        xz_list = torch.stack(xz_list, dim=-1)
        yz_list = torch.stack(yz_list, dim=-1)
        return xy_list + xz_list + yz_list




    def adjust_trans_points(self, points, ray_d, cam_pos):
        # b,hw,n,3
        b, hw, n = points.shape[:3]
        points = points.reshape(b*hw, n, -1)  # N,12,3
        ray_d = ray_d.reshape(b*hw, 3)
        cam_pos = cam_pos.reshape(b*hw, 3)
        first_points = points[:, 0, :]  # N, 3
        last_points = points[:, -1, :]
        mod_first = torch.sqrt((first_points ** 2).sum(dim=1)) # N
        first_index = torch.where(mod_first > 1)[0]
        mod_sec = torch.sqrt((last_points ** 2).sum(dim=1)) # N
        second_index = torch.where(mod_sec > 1)[0]
        tmp_index = torch.cat([first_index, second_index]) # 2N
        # 先处理只有一侧超出坐标范围的情况
        # uniset, count = tmp_index.unique(return_counts=True)
        # diff_mask = (count == 1)
        # diff_index = uniset.masked_select(diff_mask)
        # mod_value = torch.maximum(mod_first[diff_index], mod_sec[diff_index])
        # points[diff_index] /= mod_value
        # # 在处理两侧都超过坐标范围的情况
        # diff_mask = (count != 1)
        # diff_index = uniset.masked_select(diff_mask)
        diff_index = tmp_index.unique(return_counts=False)
        mod_value = torch.maximum(mod_first[diff_index], mod_sec[diff_index]) # 取最大的模值
        diff_ray_d = ray_d[diff_index]
        points[diff_index] /= mod_value
        points = points.reshape(b, hw, n, 3)
        return points
        
class EG3dDiscriminator(nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0, # 4        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.):
    ):
        super().__init__()
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
    def forward(self, img, c, update_emas=False, **block_kwargs):
        return self.dis(img, c, update_emas=update_emas, **block_kwargs)







                                                           

        










    
