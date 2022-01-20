from numpy.core.numeric import outer
from torch._C import device
from training.networks_stylegan2 import *
from torch import nn
from training.camera_utils import *
from torch.nn import functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from glob import glob 
from training.cips_camera_utils import get_fine_points_and_direction, fancy_integration

# 和v3的区别是
# 完全按照论文描述的最终版本

# @persistence.persistent_class
# class LightDecoder(nn.Module):
#     def __init__(self, in_c, mid_c, out_c):
#         super().__init__()
#         self.fc1 = FullyConnectedLayer(in_c, mid_c, activation='lrelu')
#         self.rgb_feat = FullyConnectedLayer(mid_c, out_c, activation='linear')
#         self.sigma = FullyConnectedLayer(mid_c, 1, activation='linear')  # sigmoid relu
 

#     def forward(self, x):
#         bs_n, c, h, w = x.shape
#         x = x.permute(0, 2, 3, 1).reshape(-1, c)
#         x = self.fc1(x) 
#         sigma = self.sigma(x)  
#         sigma = F.sigmoid(sigma)
#         rgb_feat = self.rgb_feat(x)
#         rgb_feat = F.softmax(rgb_feat, dim=-1)
#         o = torch.cat([rgb_feat, sigma], dim=-1)  # x, 33
#         o = o.reshape(bs_n, h, w, 33).permute(0, 3, 1, 2)
#         return o

        # x = self.fc2(x)
        # sigma = self.sigma(x)
        # sigma = F.sigmoid(sigma)
        # x = self.fc3(x) 
        # feat = self.feat(x)
        # o = torch.cat([feat, sigma], dim=-1) # bs_n *h*w,  33
        # o = o.reshape(bs_n, h, w, 33).permute(0, 3, 1, 2) # bs_n, c ,h, w
        # return o

# light decoder的作用非常关键，使用论文提供的信息搭建的decoder，不稳定
@persistence.persistent_class
class LightDecoder(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super().__init__()
        self.fc1 = FullyConnectedLayer(in_c, mid_c, activation='lrelu')
        self.fc2 = FullyConnectedLayer(mid_c, mid_c, activation='lrelu')

        self.sigma = FullyConnectedLayer(mid_c, 1, activation='linear')
        self.fc3 = FullyConnectedLayer(mid_c, out_c, activation='lrelu')
        self.feat = FullyConnectedLayer(out_c, out_c, activation='linear')

    def forward(self, x):
        bs_n, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, c)
        x = self.fc1(x)
        x = self.fc2(x)
        sigma = self.sigma(x)
        sigma = F.sigmoid(sigma)  # TODO: 梯度裁剪
        x = self.fc3(x) 
        feat = self.feat(x)
        o = torch.cat([feat, sigma], dim=-1) # bs_n *h*w,  33
        o = o.reshape(bs_n, h, w, 33).permute(0, 3, 1, 2) # bs_n, c ,h, w
        return o

@persistence.persistent_class
class ToRGB(nn.Module):
    def __init__(self, in_c, out_c=3):
        super().__init__()
        # self.conv1 = Conv2dLayer(in_c, out_c, 1)  # 核为3崩溃，核用1的正常
        self.fc1 = FullyConnectedLayer(in_c, in_c, activation='relu')
        self.fc2 = FullyConnectedLayer(in_c, out_c)


    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(bs * h * w, c)
        # x = self.conv1(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.reshape(bs, h, w, 3).permute(0, 3, 1, 2)  # b, 3, h, w
        # x = F.tanh(x)  # -1, 1  # 加入tanh，观察是否能解决训练后期又崩溃的问题
        return x



@persistence.persistent_class
class SuperResolutionNet(nn.Module): # 简化计算量，只上采样一倍
    def __init__(self, 
            in_channels, w_dim, resolution, 
            img_channels=3,
            use_fp16 = True, # True # 根据论文描述，超分模块都使用FP16,在GAN init中设置，不是在这
            channel_base=None,
            num_fp16_res=None,
            channel_max=None,
            # img_resolution=None,
            **block_kwargs
            ):
        super().__init__()
        self.w_dim = w_dim
        self.num_ws = 0
        channels_dict = {0: 128, 1:64}
        # self.n_layer = resolution // 128 - 1
        self.n_layer = int(np.log2(resolution)) - int(np.log2(128))
        for idx in range(self.n_layer):
            is_last = False if idx < (self.n_layer - 1) else True
            block = SynthesisBlock(in_channels, channels_dict[idx], w_dim=w_dim, resolution=resolution // (self.n_layer - idx),
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
                use_fp16=True, **synthesis_kwargs)
        self.synthesis = SynthesisNetwork(
                    w_dim=w_dim, img_resolution=backbone_resolution, 
                    img_channels=img_channels,
                    num_fp16_res=0,
                    architecture='orig',
                     **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws + self.super_res.num_ws  
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
        self.nerf_decoder = LightDecoder(**nerf_decoder_kwargs)
        self.init_point_kwargs = init_point_kwargs
        self.d_range = init_point_kwargs['d_range']
        # self.nerf_resolution = init_point_kwargs['nerf_resolution']

        self.aux_to_rgb = ToRGB(32, 3)
        # self.aux_to_rgb = ToRGBLayer(32, 3, 0)

    def trans_c_to_matrix(self, c):
        bs = c.shape[0]
        c = get_cam2world_matrix(c, device=c.device)[:, :3]  # b, 3, 4
        c = c.reshape(bs, 12)  # b, 12
        return c

    def forward(self, z=None, c=None, c2=None, ws=None, truncation_psi=1, truncation_cutoff=None, 
                update_emas=True, second_sample_noise_std=0, **synthesis_kwargs
                ):
        # c是相机参数，作为条件, length为12 我应该只需要世界坐标系的相机位置
        # if c.shape[-1] == 3: # 输入的是角度，需要转换一下
        #     c = self.trans_c_to_matrix(c) # bs, 12
        if c2 is None:
            c2 = c
        # if c2.shape[-1] == 3:
        #     c2 = self.trans_c_to_matrix(c2)
        if ws is None:  # TODO 使用 C2
            ws = self.mapping(z, c1, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        # print(ws.keys())
        backbone_feats = self.synthesis(ws[:, :self.synthesis.num_ws], update_emas=update_emas, **synthesis_kwargs)  # b,32*3,256,256
        
        assert backbone_feats.shape[1] == 96
        feat_xy, feat_yz, feat_xz = backbone_feats.chunk(3, dim=1)  # b, 32, 256, 256
        nerf_channel = feat_xy.shape[1]  # 32
        bs = feat_xy.shape[0]

        self.nerf_resolution = synthesis_kwargs['nerf_resolution']
        init_points, init_rays_d_cam, init_z_vals = get_init_points(nerf_resolution=synthesis_kwargs['nerf_resolution'], device=ws.device, **self.init_point_kwargs) 
        init_points = init_points.unsqueeze(0).repeat(bs, 1, 1, 1)  # b,hw, n, 3
        init_rays_d_cam = init_rays_d_cam.unsqueeze(0).repeat(bs, 1, 1) # b,hw,3
        init_z_vals = init_z_vals.unsqueeze(0).repeat(bs, 1, 1, 1) # b,hw,n,1
        # depth扰动
        # 可能让坐标值超过-1到1, 需要grid sample的位置处理一下
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
        trans_points = trans_points / 0.13 # (self.d_range[1] - self.d_range[0]) * 2  # 3个维度坐标都在-1到1之间,有可能越界
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
        # if True:
        #     print(homo_world_matrix)
        #     print(trans_points.max(), trans_points.min())
        #     # print(trans_points[0])
        #     if self.rank == 0:
        #         fig = plt.figure()
        #         p = points[0].detach().cpu().numpy()  # hw, n, 3
        #         tp = trans_points[0].detach().cpu().numpy()
        #         ax = fig.gca(projection='3d')
        #         ax.scatter(p[:, 0, 0], p[:, 0, 1], p[:, 0, 2],)
        #         ax.scatter(p[:, -1, 0], p[:, -1, 1], p[:, -1, 2],)
        #         ax.plot(tp[:, 0, 0], tp[:, 0, 1], tp[:, 0, 2],)
        #         ax.plot(tp[:, -1, 0], tp[:, -1, 1], tp[:, -1, 2],)
        #         # ax.plot(d_ray[:, 0], d_ray[:, 1], d_ray[:, 2])
        #         ax.scatter([0], [0], [0])
        #         ax.set_xlabel('x')
        #         ax.set_ylabel('y')
        #         ax.set_zlabel('z')
        #         num_idx = len(glob('./fig_*.png'))
        #         plt.savefig(f'fig_{num_idx}.png')
        nerf_feat = self.bilinear_sample_tri_plane(
            trans_points,
            feat_xy, feat_yz, feat_xz, 
            )  # b*n,c,h,w
        # 这里的数值看着正常，没有全部相同
        # if self.rank == 0: 
        #     print('11111')
        #     print(nerf_feat[0,:, 0,5, 0], nerf_feat[0,:, 0,90, 2])
        nerf_feat = nerf_feat.permute(0, 4, 1, 2, 3).reshape(bs*n_step, nerf_channel,\
            self.nerf_resolution, self.nerf_resolution)
       
        # 使用light-weight decoder得到不透明度和RGB feature
        nerf_feat = self.nerf_decoder(nerf_feat)  # bs*n_step, c+1, h, w
        volume_channel = nerf_feat.shape[1]
        h = w = self.nerf_resolution 
        nerf_feat = nerf_feat.reshape(bs, n_step, volume_channel, h, w).permute(0, 3,4,1,2).\
                reshape(bs, n_points, n_step, volume_channel) # b, hw, n, c+1
        # if self.rank == 0:
        #     print(2222)
        #     print(nerf_feat[0,5, 0, :], nerf_feat[0,90,2, :])
        if False:  # TODO:
            fine_points, fine_z_vals = get_fine_points_and_direction(
                nerf_feat, z_vals, nerf_noise=second_sample_noise_std * 0.5,
                num_steps=n_step,
                transformed_ray_origins= trans_ray_origins,
                transformed_ray_directions= trans_rays_d_cam,
                device=nerf_feat.device
                    )
            fine_points = fine_points / 0.13 # 忘记对fine point归一化了
            fine_nerf_feat = self.bilinear_sample_tri_plane(
                    fine_points,
                    feat_xy, feat_yz, feat_xz, 
                    )  
            fine_nerf_feat = fine_nerf_feat.permute(0, 4, 1, 2, 3).reshape(bs*n_step, nerf_channel,
                                        h, w)
            fine_nerf_feat = self.nerf_decoder(fine_nerf_feat)  # bs*n_step, c+1, h, w

           
            fine_nerf_feat = fine_nerf_feat.reshape(bs, n_step, volume_channel, h, w).permute(0, 3, 4,1,2).\
                reshape(bs, n_points, n_step, volume_channel)
            nerf_feat = torch.cat([nerf_feat, fine_nerf_feat], dim=2) # bs, hw, 2n, c+1
            assert nerf_feat.shape[-2] == 2*n_step
            z_vals = torch.cat([z_vals, fine_z_vals], dim=2)  # bs, hw, 2n, 1
            _, indices = torch.sort(z_vals, dim=-2)  # (b, hw, 2n, 1)
            z_vals = torch.gather(z_vals, -2, indices)
            nerf_feat = torch.gather(nerf_feat, -2, indices.expand(-1, -1, -1, volume_channel))
        # if self.rank == 0:
        #     print(333)
        #     print(fine_nerf_feat[0,5, 0, :], fine_nerf_feat[0,90, 2, :])
        # 开始volume rendering，这里直接使用CIPS3d的代码 # 需要好好阅读一下giffare，CIPS，GRAF的代码
        # 沿用CIPS的代码
        pixels_fea, _, _ = fancy_integration(
            rgb_sigma=nerf_feat,
            z_vals=z_vals,
            device=nerf_feat.device,
            white_back=False,
            last_back=False,
            # clamp_mode=clamp_mode,
            noise_std=second_sample_noise_std)
        # 使用超分辨率模型，
        pixels_fea = pixels_fea.reshape(bs, h, w, volume_channel-1).permute(0, 3, 1, 2)# bs, 32, h, w
        if self.nerf_resolution != 128:
            pixels_fea = F.interpolate(pixels_fea, size=(128, 128), mode='bilinear', align_corners=True)
        # if self.rank == 0:
        #     print(44444)
        #     print(pixels_fea[0, :, 10:12, 10])
      
        gen_low_img = self.aux_to_rgb(pixels_fea) # bs, 3, h', w'
        ws = ws[:, self.synthesis.num_ws:]  # 超分辨率的ws
        assert ws.shape[1] == self.super_res.num_ws
        gen_high_img = self.super_res(pixels_fea, ws,  **synthesis_kwargs)  # bs, 3, h ,w 
        # assert gen_high_img.shape[-1] == 256  # 256的输出
        scale_factor = gen_high_img.shape[-1] // gen_low_img.shape[-1]
        gen_low_img = F.interpolate(gen_low_img, scale_factor=scale_factor, mode='bilinear')
        # gen_imgs = torch.cat([gen_high_img, gen_low_img], dim=1) # bs, 6, h, w
        gen_imgs = torch.cat([gen_low_img, gen_low_img], dim=1) # bs, 6, h, w
        return gen_imgs


    def bilinear_sample_tri_plane(self, points, feat_xy, feat_yz, feat_xz):
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
            xy_f = F.grid_sample(feat_xy, grid=xy_idx)  # b, c, h, w
            xz_f = F.grid_sample(feat_xz, grid=xz_idx)
            yz_f = F.grid_sample(feat_yz, grid=yz_idx)
            xy_list.append(xy_f)
            xz_list.append(xz_f)
            yz_list.append(yz_f) 
        xy_list = torch.stack(xy_list, dim=-1)  # b, c,h,w,n
        xz_list = torch.stack(xz_list, dim=-1)
        yz_list = torch.stack(yz_list, dim=-1)
        return xy_list + xz_list + yz_list

    # def bilinear_sample_tri_plane_v2(self, points, feat_xy, feat_yz, feat_xz):
     
    #     b, hw, n = points.shape[:3]
    #     h = w = self.nerf_resolution
    #     c = feat_xy.shape[1]
    #     x = points[..., 0]  # b, hw, n
    #     y = points[..., 1]
    #     z = points[..., 2]
    #     xy = torch.stack([x, y], dim=-1).permute(0, 2, 1, 3)  # b, hw, n, 2 -> b, n, hw, 2
    #     xz = torch.stack([x, z], dim=-1).permute(0, 2, 1, 3)  # b, hw, n, 2 -> b, n, hw, 2
    #     yz = torch.stack([y, z], dim=-1).permute(0, 2, 1, 3)  # b, hw, n, 2 -> b, n, hw, 2

    #     xy = xy.reshape(b*n, h, w, 2)
    #     xz = xz.reshape(b*n, h, w, 2)
    #     yz = yz.reshape(b*n, h, w, 2)

    #     fh, fw = feat_xy.shape[2:] 
    #     feat_xy = feat_xy.unsqueeze(1).expand(-1, n, -1, -1, -1).reshape(b*n, c, fh, fw)
    #     feat_xz = feat_xz.unsqueeze(1).expand(-1, n, -1, -1, -1).reshape(b*n, c, fh, fw)
    #     feat_yz = feat_yz.unsqueeze(1).expand(-1, n, -1, -1, -1).reshape(b*n, c, fh, fw)
        
    #     xy_f = F.grid_sample(feat_xy, grid=xy, padding_mode='border')  # b*n, c, h, w
    #     xz_f = F.grid_sample(feat_xz, grid=xz, padding_mode='border')
    #     yz_f = F.grid_sample(feat_yz, grid=yz, padding_mode='border')

    #     return xy_f + xz_f + yz_f  # b*n, c, h, w


@persistence.persistent_class
class EG3dDiscriminator(nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4, # 4, # 4        # Use FP16 for the N highest resolutions.
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




# TODO: 
# 尝试不使用第二次采样 [✔️]
# 尝试别的开源volume rendering实现


                                                           

        










    
