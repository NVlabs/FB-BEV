# Copyright (c) Phigent Robotics. All rights reserved.
import math
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models.builder import NECKS
from projects.occ_plugin.ops.occ_pooling import occ_pool
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import force_fp32
from torch.cuda.amp.autocast_mode import autocast
from mmdet.models.backbones.resnet import BasicBlock
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from scipy.special import erf
from scipy.stats import norm
import numpy as np
import copy
import pdb

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])
    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))
    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None

class ViewTransformerLiftSplatShoot(BaseModule):
    def __init__(self, grid_config=None, data_config=None,
                 numC_input=512, numC_Trans=64, downsample=16,
                 accelerate=False, use_bev_pool=True, vp_megvii=False,
                 vp_stero=False, **kwargs):
        super(ViewTransformerLiftSplatShoot, self).__init__()
        if grid_config is None:
            grid_config = {
                'xbound': [-51.2, 51.2, 0.8],
                'ybound': [-51.2, 51.2, 0.8],
                'zbound': [-10.0, 10.0, 20.0],
                'dbound': [1.0, 60.0, 1.0],}
        self.grid_config = grid_config
        dx, bx, nx = gen_dx_bx(self.grid_config['xbound'],
                               self.grid_config['ybound'],
                               self.grid_config['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        if data_config is None:
            data_config = {'input_size': (256, 704)}
        self.data_config = data_config
        self.downsample = downsample

        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.numC_input = numC_input
        self.numC_Trans = numC_Trans
        self.depth_net = nn.Conv2d(self.numC_input, self.D + self.numC_Trans, kernel_size=1, padding=0)
        self.geom_feats = None
        self.accelerate = accelerate
        self.use_bev_pool = use_bev_pool
        self.vp_megvii = vp_megvii
        self.vp_stereo = vp_stero

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans, bda):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        
        if intrins.shape[3] == 4: # for KITTI
            shift = intrins[:, :, :3, 3]
            points = points - shift.view(B, N, 1, 1, 1, 3, 1)
            intrins = intrins[:, :, :3, :3]
        
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        points = bda.view(B,1,1,1,1,3,3).matmul(points.unsqueeze(-1)).squeeze(-1)
        
        return points

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        if self.use_bev_pool:
            final = occ_pool(x, geom_feats, B, self.nx[2], self.nx[0],
                                   self.nx[1])
            final = final.transpose(dim0=-2, dim1=-1)
        else:
            # get tensors from the same voxel next to each other
            ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                    + geom_feats[:, 1] * (self.nx[2] * B) \
                    + geom_feats[:, 2] * B \
                    + geom_feats[:, 3]
            sorts = ranks.argsort()
            x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

            # cumsum trick
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

            # griddify (B x C x Z x X x Y)
            final = torch.zeros((B, C, nx[2], nx[1], nx[0]), device=x.device)
            final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x
        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def voxel_pooling_accelerated(self, rots, trans, intrins, post_rots, post_trans, bda, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)
        max = 300
        # flatten indices
        if self.geom_feats is None:
            geom_feats = self.get_geometry(rots, trans, intrins,
                                           post_rots, post_trans, bda)
            geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) /
                          self.dx).long()
            geom_feats = geom_feats.view(Nprime, 3)
            batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                             device=x.device, dtype=torch.long)
                                  for ix in range(B)])
            geom_feats = torch.cat((geom_feats, batch_ix), 1)

            # filter out points that are outside box
            kept1 = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
                    & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
                    & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
            idx = torch.range(0, x.shape[0] - 1, dtype=torch.long)
            x = x[kept1]
            idx = idx[kept1]
            geom_feats = geom_feats[kept1]

            # get tensors from the same voxel next to each other
            ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                    + geom_feats[:, 1] * (self.nx[2] * B) \
                    + geom_feats[:, 2] * B \
                    + geom_feats[:, 3]
            sorts = ranks.argsort()
            x, geom_feats, ranks, idx = x[sorts], geom_feats[sorts], ranks[sorts], idx[sorts]
            repeat_id = torch.ones(geom_feats.shape[0], device=geom_feats.device, dtype=geom_feats.dtype)
            curr = 0
            repeat_id[0] = 0
            curr_rank = ranks[0]

            for i in range(1, ranks.shape[0]):
                if curr_rank == ranks[i]:
                    curr += 1
                    repeat_id[i] = curr
                else:
                    curr_rank = ranks[i]
                    curr = 0
                    repeat_id[i] = curr
            kept2 = repeat_id < max
            repeat_id, geom_feats, x, idx = repeat_id[kept2], geom_feats[kept2], x[kept2], idx[kept2]

            geom_feats = torch.cat([geom_feats,
                                    repeat_id.unsqueeze(-1)], dim=-1)
            self.geom_feats = geom_feats
            self.idx = idx
        else:
            geom_feats = self.geom_feats
            idx = self.idx
            x = x[idx]

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, nx[2], nx[1], nx[0], max), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1],
        geom_feats[:, 0], geom_feats[:, 4]] = x
        final = final.sum(-1)
        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)
        return final

    def voxel_pooling_bevdepth(self, geom_feats, x):
        nx = self.nx.to(torch.long)
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).int()
        
        # FIXME
        # final = voxel_pooling(geom_feats, x.contiguous(), nx)
        final = self.voxel_pooling(geom_feats, x.contiguous(), nx)
        
        return final

    def forward(self, input):
        x, rots, trans, intrins, post_rots, post_trans, bda = input
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x)
        depth = self.get_depth_dist(x[:, :self.D])
        img_feat = x[:, self.D:(self.D + self.numC_Trans)]

        # Lift
        volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat
        if self.accelerate:
            bev_feat = self.voxel_pooling_accelerated(rots, trans, intrins,
                                                      post_rots, post_trans,
                                                      bda, volume)
        else:
            geom = self.get_geometry(rots, trans, intrins,
                                     post_rots, post_trans, bda)
            if self.vp_megvii:
                bev_feat = self.voxel_pooling_bevdepth(geom, volume)
            else:
                bev_feat = self.voxel_pooling(geom, volume)
        return bev_feat


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, norm_cfg=dict(type='BN2d')):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=build_norm_layer(norm_cfg, mid_channels)[1])
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=build_norm_layer(norm_cfg, mid_channels)[1])
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=build_norm_layer(norm_cfg, mid_channels)[1])
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=build_norm_layer(norm_cfg, mid_channels)[1])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = build_norm_layer(norm_cfg, mid_channels)[1]
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels,
                 depth_channels, cam_channels=27, norm_cfg=None):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            # nn.BatchNorm2d(mid_channels),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        
        self.bn = build_norm_layer(dict(type='GN', num_groups=9, requires_grad=True), cam_channels)[1]
        self.depth_mlp = Mlp(cam_channels, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(cam_channels, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            ASPP(mid_channels, mid_channels, norm_cfg=norm_cfg),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, x, mlp_input):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        return torch.cat([depth, context], dim=1)

class DepthAggregation(nn.Module):
    """
    pixel cloud feature extraction
    """
    def __init__(self, in_channels, mid_channels, out_channels, norm_cfg):
        super(DepthAggregation, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x):
        x = checkpoint(self.reduce_conv, x)
        short_cut = x
        x = checkpoint(self.conv, x)
        x = short_cut + x
        x = self.out_conv(x)
        return x


@NECKS.register_module()
class ViewTransformerLSSBEVDepth(ViewTransformerLiftSplatShoot):
    def __init__(self, loss_depth_weight, cam_channels=27, loss_depth_reg_weight=0.0, use_voxel_net=False, 
                 norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01), **kwargs):
        super(ViewTransformerLSSBEVDepth, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.loss_depth_reg_weight = loss_depth_reg_weight
        self.cam_channels = cam_channels
        
        self.depth_net = DepthNet(self.numC_input, self.numC_input,
                                  self.numC_Trans, self.D, cam_channels=self.cam_channels,
                                  norm_cfg=norm_cfg)
        self.depth_aggregation_net = DepthAggregation(self.numC_Trans,
                                                      self.numC_Trans,
                                                      self.numC_Trans,
                                                      norm_cfg=norm_cfg) if use_voxel_net else None

    def _forward_voxel_net(self, img_feat_with_depth):
        # BEVConv2D [n, c, d, h, w] -> [n, h, c, w, d]
        if self.depth_aggregation_net is None:
            return img_feat_with_depth
        img_feat_with_depth = img_feat_with_depth.permute(
            0, 3, 1, 4, 2).contiguous()  # [n, c, d, h, w] -> [n, h, c, w, d]
        n, h, c, w, d = img_feat_with_depth.shape
        img_feat_with_depth = img_feat_with_depth.view(-1, c, w, d)
        img_feat_with_depth = (
            self.depth_aggregation_net(img_feat_with_depth).view(
                n, h, c, w, d).permute(0, 2, 4, 1, 3).contiguous().float())
        return img_feat_with_depth

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda=None):
        B,N,_,_ = rot.shape
        if bda is None:
            bda = torch.eye(3).to(rot).view(1,3,3).repeat(B,1,1)
        bda = bda.view(B,1,3,3).repeat(1,N,1,1)
        
        if intrin.shape[-1] == 4:
            # for KITTI, the intrin matrix is 3x4
            mlp_input = torch.stack([
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                intrin[:, :, 0, 3],
                intrin[:, :, 1, 3],
                intrin[:, :, 2, 3],
                post_rot[:, :, 0, 0],
                post_rot[:, :, 0, 1],
                post_tran[:, :, 0],
                post_rot[:, :, 1, 0],
                post_rot[:, :, 1, 1],
                post_tran[:, :, 1],
                bda[:, :, 0, 0],
                bda[:, :, 0, 1],
                bda[:, :, 1, 0],
                bda[:, :, 1, 1],
                bda[:, :, 2, 2],
            ], dim=-1)
        else:
            mlp_input = torch.stack([
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                post_rot[:, :, 0, 0],
                post_rot[:, :, 0, 1],
                post_tran[:, :, 0],
                post_rot[:, :, 1, 0],
                post_rot[:, :, 1, 1],
                post_tran[:, :, 1],
                bda[:, :, 0, 0],
                bda[:, :, 0, 1],
                bda[:, :, 1, 0],
                bda[:, :, 1, 1],
                bda[:, :, 2, 2],
            ], dim=-1)
        
        sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)], dim=-1).reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        
        return mlp_input

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)
        
        # [min - step / 2, min + step / 2] creates min depth
        gt_depths = (gt_depths - (self.grid_config['dbound'][0] - self.grid_config['dbound'][2] / 2)) / self.grid_config['dbound'][2]
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]
        
        return gt_depths.float()

    def _prepare_depth_gt(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*H*W, d]
        """
        gt_depths = (gt_depths - (self.grid_config['dbound'][0] -
                                  self.grid_config['dbound'][2])) / \
                    self.grid_config['dbound'][2]
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.D + 1).view(-1,
                                                           self.D + 1)[:, 1:]
        return gt_depths.float()

    @force_fp32()
    def get_depth_reg_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        # depth_labels = self._prepare_depth_gt(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        # foreground predictions & labels
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        
        # cls_targets ==> reg_targets
        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(1, -1).type_as(depth_preds)
        depth_reg_labels = torch.sum(depth_labels * ds, dim=1)
        depth_reg_preds = torch.sum(depth_preds * ds, dim=1)
        
        with autocast(enabled=False):
            loss_depth = F.smooth_l1_loss(depth_reg_preds, depth_reg_labels, reduction='mean')

        return self.loss_depth_reg_weight * loss_depth

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        # depth_labels = self._prepare_depth_gt(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        
        return self.loss_depth_weight * depth_loss

    def forward(self, input):
        (x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input) = input[:8]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x, mlp_input)
        depth_digit = x[:, :self.D, ...]
        img_feat = x[:, self.D:self.D+self.numC_Trans, ...]
        depth_prob = self.get_depth_dist(depth_digit)
        # Lift
        volume = depth_prob.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = self._forward_voxel_net(volume)
        volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat
        if self.accelerate:
            bev_feat = self.voxel_pooling_accelerated(rots, trans, intrins,
                                                      post_rots, post_trans,
                                                      bda, volume)
        else:
            geom = self.get_geometry(rots, trans, intrins,
                                     post_rots, post_trans, bda)
            if self.vp_megvii:
                bev_feat = self.voxel_pooling_bevdepth(geom, volume)
            else:
                bev_feat = self.voxel_pooling(geom, volume)
        return bev_feat, depth_prob


class ConvBnReLU3D(nn.Module):
    """Implements of 3d convolution + batch normalization + ReLU."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        pad: int = 1,
        dilation: int = 1,
    ) -> None:
        """initialization method for convolution3D + batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=pad,
                              dilation=dilation,
                              bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return F.relu(self.bn(self.conv(x)), inplace=True)


class DepthNetStereo(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 d_bound,
                 num_ranges=4,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(DepthNetStereo, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_feat_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            ASPP(mid_channels, mid_channels, norm_cfg=norm_cfg),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
        )
        self.mu_sigma_range_net = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            nn.ConvTranspose2d(mid_channels,
                               mid_channels,
                               3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels,
                               mid_channels,
                               3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      num_ranges * 3,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )
        self.mono_depth_net = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )
        self.d_bound = d_bound
        self.num_ranges = num_ranges

    # @autocast(False)
    def forward(self, x, mlp_input):
        B, _, H, W = x.shape

        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth_feat = self.depth_se(x, depth_se)
        depth_feat = checkpoint(self.depth_feat_conv, depth_feat)
        mono_depth = checkpoint(self.mono_depth_net, depth_feat)
        mu_sigma_score = checkpoint(self.mu_sigma_range_net, depth_feat)
        mu = mu_sigma_score[:, 0:self.num_ranges, ...]
        sigma = mu_sigma_score[:, self.num_ranges:2 * self.num_ranges, ...]
        range_score = mu_sigma_score[:,
                                     2 * self.num_ranges:3 * self.num_ranges,
                                     ...]
        sigma = F.elu(sigma) + 1.0 + 1e-10
        return x, context, mu, sigma, range_score, mono_depth


@NECKS.register_module()
class ViewTransformerLSSBEVStereo(ViewTransformerLSSBEVDepth):
    def __init__(self, num_ranges=4, use_mask=True, em_iteration=3,
                 range_list=[[2, 8], [8, 16], [16, 28], [28, 58]],
                 sampling_range=3, num_samples=3,
                 k_list=None, min_sigma=1.0,
                 num_groups=8,
                 stereo_downsample_factor=4, 
                 norm_cfg=dict(type='BN2d'), **kwargs):
        super(ViewTransformerLSSBEVStereo, self).__init__(**kwargs)
        self.num_ranges = num_ranges
        self.depth_net = DepthNetStereo(self.numC_input, self.numC_input,
                                  self.numC_Trans, self.D,
                                  self.grid_config['dbound'],
                                  self.num_ranges,
                                  norm_cfg=norm_cfg)
        self.context_downsample_net = nn.Identity()
        self.use_mask = use_mask
        self.stereo_downsample_factor = stereo_downsample_factor
        self.num_ranges = num_ranges
        self.min_sigma = min_sigma
        self.sampling_range = sampling_range
        self.num_samples = num_samples
        self.num_groups=num_groups
        self.similarity_net = nn.Sequential(
            ConvBnReLU3D(in_channels=num_groups,
                         out_channels=16,
                         kernel_size=1,
                         stride=1,
                         pad=0),
            ConvBnReLU3D(in_channels=16,
                         out_channels=8,
                         kernel_size=1,
                         stride=1,
                         pad=0),
            nn.Conv3d(in_channels=8,
                      out_channels=1,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )
        self.depth_downsample_net = nn.Sequential(
            nn.Conv2d(self.D, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.D, 1, 1, 0),
        )
        if range_list is None:
            range_length = (self.grid_config['dbound'][1] -
                            self.grid_config['dbound'][0]) / num_ranges
            self.range_list = [[
                self.grid_config['dbound'][0] + range_length * i,
                self.grid_config['dbound'][0] + range_length * (i + 1)
            ] for i in range(num_ranges)]
        else:
            assert len(range_list) == num_ranges
            self.range_list = range_list
        self.em_iteration = em_iteration
        if k_list is None:
            self.register_buffer('k_list', torch.Tensor(self.depth_sampling()))
        else:
            self.register_buffer('k_list', torch.Tensor(k_list))
        if self.use_mask:
            self.mask_net = nn.Sequential(
                nn.Conv2d(self.D*2, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                BasicBlock(64, 64),
                BasicBlock(64, 64),
                nn.Conv2d(64, 1, 1, 1, 0),
                nn.Sigmoid(),
            )

    def depth_sampling(self):
        """Generate sampling range of candidates.

        Returns:
            list[float]: List of all candidates.
        """
        P_total = erf(self.sampling_range /
                      np.sqrt(2))  # Probability covered by the sampling range
        idx_list = np.arange(0, self.num_samples + 1)
        p_list = (1 - P_total) / 2 + ((idx_list / self.num_samples) * P_total)
        k_list = norm.ppf(p_list)
        k_list = (k_list[1:] + k_list[:-1]) / 2
        return list(k_list)

    def create_depth_sample_frustum(self, depth_sample, downsample_factor=16):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size']
        fH, fW = ogfH // downsample_factor, ogfW // downsample_factor
        batch_size, num_depth, _, _ = depth_sample.shape
        x_coords = (torch.linspace(0,
                                   ogfW - 1,
                                   fW,
                                   dtype=torch.float,
                                   device=depth_sample.device).view(
                                       1, 1, 1,
                                       fW).expand(batch_size, num_depth, fH,
                                                  fW))
        y_coords = (torch.linspace(0,
                                   ogfH - 1,
                                   fH,
                                   dtype=torch.float,
                                   device=depth_sample.device).view(
                                       1, 1, fH,
                                       1).expand(batch_size, num_depth, fH,
                                                 fW))
        paddings = torch.ones_like(depth_sample)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, depth_sample, paddings), -1)
        return frustum

    def homo_warping(
        self,
        stereo_feat,
        key_intrin_mats,
        sweep_intrin_mats,
        sensor2sensor_mats,
        key_ida_mats,
        sweep_ida_mats,
        depth_sample,
        frustum,
    ):
        """Used for mvs method to transfer sweep image feature to
            key image feature.

        Args:
            src_fea(Tensor): image features.
            key_intrin_mats(Tensor): Intrin matrix for key sensor.
            sweep_intrin_mats(Tensor): Intrin matrix for sweep sensor.
            sensor2sensor_mats(Tensor): Transformation matrix from key
                sensor to sweep sensor.
            key_ida_mats(Tensor): Ida matrix for key frame.
            sweep_ida_mats(Tensor): Ida matrix for sweep frame.
            depth_sample (Tensor): Depth map of all candidates.
            depth_sample_frustum (Tensor): Pre-generated frustum.
        """
        batch_size_with_num_cams, channels = stereo_feat.shape[
            0], stereo_feat.shape[1]
        height, width = stereo_feat.shape[2], stereo_feat.shape[3]
        with torch.no_grad():
            points = frustum
            points = points.reshape(points.shape[0], -1, points.shape[-1])
            points[..., 2] = 1
            # Undo ida for key frame.
            points = key_ida_mats.reshape(batch_size_with_num_cams,
                                          *key_ida_mats.shape[2:]).inverse(
                                          ).unsqueeze(1) @ points.unsqueeze(-1)
            # Convert points from pixel coord to key camera coord.
            points[..., :3, :] *= depth_sample.reshape(
                batch_size_with_num_cams, -1, 1, 1)
            num_depth = frustum.shape[1]
            points = (key_intrin_mats.reshape(
                batch_size_with_num_cams,
                *key_intrin_mats.shape[2:]).inverse().unsqueeze(1) @ points)
            points = (sensor2sensor_mats.reshape(
                batch_size_with_num_cams,
                *sensor2sensor_mats.shape[2:]).unsqueeze(1) @ points)
            # points in sweep sensor coord.
            points = (sweep_intrin_mats.reshape(
                batch_size_with_num_cams,
                *sweep_intrin_mats.shape[2:]).unsqueeze(1) @ points)
            # points in sweep pixel coord.
            points[..., :2, :] = points[..., :2, :] / points[
                ..., 2:3, :]  # [B, 2, Ndepth, H*W]

            points = (sweep_ida_mats.reshape(
                batch_size_with_num_cams,
                *sweep_ida_mats.shape[2:]).unsqueeze(1) @ points).squeeze(-1)
            neg_mask = points[..., 2] < 1e-3
            points[..., 0][neg_mask] = width * self.stereo_downsample_factor
            points[..., 1][neg_mask] = height * self.stereo_downsample_factor
            points[..., 2][neg_mask] = 1
            proj_x_normalized = points[..., 0] / (
                (width * self.stereo_downsample_factor - 1) / 2) - 1
            proj_y_normalized = points[..., 1] / (
                (height * self.stereo_downsample_factor - 1) / 2) - 1
            grid = torch.stack([proj_x_normalized, proj_y_normalized],
                               dim=2)  # [B, Ndepth, H*W, 2]

        warped_stereo_fea = F.grid_sample(
            stereo_feat,
            grid.view(batch_size_with_num_cams, num_depth * height, width, 2),
            mode='bilinear',
            padding_mode='zeros',
        )
        warped_stereo_fea = warped_stereo_fea.view(batch_size_with_num_cams,
                                                   channels, num_depth, height,
                                                   width)

        return warped_stereo_fea

    def _forward_mask(
        self,
        sweep_index,
        mono_depth_all_sweeps,
        mats_dict,
        depth_sample,
        depth_sample_frustum,
        sensor2sensor_mats,
    ):
        """Forward function to generate mask.

        Args:
            sweep_index (int): Index of sweep.
            mono_depth_all_sweeps (list[Tensor]): List of mono_depth for
                all sweeps.
            mats_dict (dict):
                sensor2ego_mats (Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats (Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats (Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats (Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat (Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            depth_sample (Tensor): Depth map of all candidates.
            depth_sample_frustum (Tensor): Pre-generated frustum.
            sensor2sensor_mats (Tensor): Transformation matrix from reference
                sensor to source sensor.

        Returns:
            Tensor: Generated mask.
        """
        num_sweeps = len(mono_depth_all_sweeps)
        mask_all_sweeps = list()
        for idx in range(num_sweeps):
            if idx == sweep_index:
                continue
            warped_mono_depth = self.homo_warping(
                mono_depth_all_sweeps[idx],
                mats_dict['intrin_mats'][:, sweep_index, ...],
                mats_dict['intrin_mats'][:, idx, ...],
                sensor2sensor_mats[idx],
                mats_dict['ida_mats'][:, sweep_index, ...],
                mats_dict['ida_mats'][:, idx, ...],
                depth_sample,
                depth_sample_frustum.type_as(mono_depth_all_sweeps[idx]),
            )
            mask = self.mask_net(
                torch.cat([
                    mono_depth_all_sweeps[sweep_index].detach(),
                    warped_mono_depth.mean(2).detach()
                ], 1))
            mask_all_sweeps.append(mask)
        return torch.stack(mask_all_sweeps).mean(0)

    def _generate_cost_volume(
            self,
            sweep_index,
            stereo_feats_all_sweeps,
            mats_dict,
            depth_sample,
            depth_sample_frustum,
            sensor2sensor_mats,
    ):
        """Generate cost volume based on depth sample.

        Args:
            sweep_index (int): Index of sweep.
            stereo_feats_all_sweeps (list[Tensor]): Stereo feature
                of all sweeps.
            mats_dict (dict):
                sensor2ego_mats (Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats (Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats (Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats (Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat (Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            depth_sample (Tensor): Depth map of all candidates.
            depth_sample_frustum (Tensor): Pre-generated frustum.
            sensor2sensor_mats (Tensor): Transformation matrix from reference
                sensor to source sensor.

        Returns:
            Tensor: Depth score for all sweeps.
        """
        batch_size, num_channels, height, width = stereo_feats_all_sweeps[
            0].shape
        # thres = int(self.mvs_weighting.split("CW")[1])
        num_sweeps = len(stereo_feats_all_sweeps)
        depth_score_all_sweeps = list()
        for idx in range(num_sweeps):
            if idx == sweep_index:
                continue
            warped_stereo_fea = self.homo_warping(
                stereo_feats_all_sweeps[idx],
                mats_dict['intrin_mats'][:, sweep_index, ...],
                mats_dict['intrin_mats'][:, idx, ...],
                sensor2sensor_mats[idx],
                mats_dict['ida_mats'][:, sweep_index, ...],
                mats_dict['ida_mats'][:, idx, ...],
                depth_sample,
                depth_sample_frustum.type_as(stereo_feats_all_sweeps[idx]),
            )
            warped_stereo_fea = warped_stereo_fea.reshape(
                batch_size, self.num_groups, num_channels // self.num_groups,
                self.num_samples, height, width)
            ref_stereo_feat = stereo_feats_all_sweeps[sweep_index].reshape(
                batch_size, self.num_groups, num_channels // self.num_groups,
                height, width)
            feat_cost = torch.mean(
                (ref_stereo_feat.unsqueeze(3) * warped_stereo_fea), axis=2)
            depth_score = self.similarity_net(feat_cost).squeeze(1)
            depth_score_all_sweeps.append(depth_score)
        return torch.stack(depth_score_all_sweeps).mean(0)

    def _forward_stereo(
        self,
        sweep_index,
        stereo_feats_all_sweeps,
        mono_depth_all_sweeps,
        mats_dict,
        sensor2sensor_mats,
        mu_all_sweeps,
        sigma_all_sweeps,
        range_score_all_sweeps,
        depth_feat_all_sweeps,
    ):
        """Forward function to generate stereo depth.

        Args:
            sweep_index (int): Index of sweep.
            stereo_feats_all_sweeps (list[Tensor]): Stereo feature
                of all sweeps.
            mono_depth_all_sweeps (list[Tensor]):
            mats_dict (dict):
                sensor2ego_mats (Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats (Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats (Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats (Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat (Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            sensor2sensor_mats(Tensor): Transformation matrix from key
                sensor to sweep sensor.
            mu_all_sweeps (list[Tensor]): List of mu for all sweeps.
            sigma_all_sweeps (list[Tensor]): List of sigma for all sweeps.
            range_score_all_sweeps (list[Tensor]): List of all range score
                for all sweeps.
            depth_feat_all_sweeps (list[Tensor]): List of all depth feat for
                all sweeps.

        Returns:
            Tensor: stereo_depth
        """
        batch_size_with_cams, _, feat_height, feat_width = \
            stereo_feats_all_sweeps[0].shape
        device = stereo_feats_all_sweeps[0].device
        d_coords = torch.arange(*self.grid_config['dbound'],
                                dtype=torch.float,
                                device=device).reshape(1, -1, 1, 1)
        d_coords = d_coords.repeat(batch_size_with_cams, 1, feat_height,
                                   feat_width)
        stereo_depth = stereo_feats_all_sweeps[0].new_zeros(
            batch_size_with_cams, self.D, feat_height, feat_width)
        mask_score = stereo_feats_all_sweeps[0].new_zeros(
            batch_size_with_cams,
            self.D,
            feat_height * self.stereo_downsample_factor //
            self.downsample,
            feat_width * self.stereo_downsample_factor //
            self.downsample,
        )
        score_all_ranges = list()
        range_score = range_score_all_sweeps[sweep_index].softmax(1)
        for range_idx in range(self.num_ranges):
            # Map mu to the corresponding interval.
            range_start = self.range_list[range_idx][0]
            mu_all_sweeps_single_range = [
                mu[:, range_idx:range_idx + 1, ...].sigmoid() *
                (self.range_list[range_idx][1] - self.range_list[range_idx][0])
                + range_start for mu in mu_all_sweeps
            ]
            sigma_all_sweeps_single_range = [
                sigma[:, range_idx:range_idx + 1, ...]
                for sigma in sigma_all_sweeps
            ]
            batch_size_with_cams, _, feat_height, feat_width =\
                stereo_feats_all_sweeps[0].shape
            mu = mu_all_sweeps_single_range[sweep_index]
            sigma = sigma_all_sweeps_single_range[sweep_index]
            for _ in range(self.em_iteration):
                depth_sample = torch.cat([mu + sigma * k for k in self.k_list],
                                         1)
                depth_sample_frustum = self.create_depth_sample_frustum(
                    depth_sample, self.stereo_downsample_factor)
                mu_score = self._generate_cost_volume(
                    sweep_index,
                    stereo_feats_all_sweeps,
                    mats_dict,
                    depth_sample,
                    depth_sample_frustum,
                    sensor2sensor_mats,
                )
                mu_score = mu_score.softmax(1)
                scale_factor = torch.clamp(
                    0.5 / (1e-4 + mu_score[:, self.num_samples //
                                           2:self.num_samples // 2 + 1, ...]),
                    min=0.1,
                    max=10)

                sigma = torch.clamp(sigma * scale_factor, min=0.1, max=10)
                mu = (depth_sample * mu_score).sum(1, keepdim=True)
                del depth_sample
                del depth_sample_frustum
            mu = torch.clamp(mu,
                             max=self.range_list[range_idx][1],
                             min=self.range_list[range_idx][0])
            range_length = int(
                (self.range_list[range_idx][1] - self.range_list[range_idx][0])
                // self.grid_config['dbound'][2])
            if self.use_mask:
                depth_sample = F.avg_pool2d(
                    mu,
                    self.downsample // self.stereo_downsample_factor,
                    self.downsample // self.stereo_downsample_factor,
                )
                depth_sample_frustum = self.create_depth_sample_frustum(
                    depth_sample, self.downsample)
                mask = self._forward_mask(
                    sweep_index,
                    mono_depth_all_sweeps,
                    mats_dict,
                    depth_sample,
                    depth_sample_frustum,
                    sensor2sensor_mats,
                )
                mask_score[:,
                           int((range_start - self.grid_config['dbound'][0]) //
                               self.grid_config['dbound'][2]):range_length +
                           int((range_start - self.grid_config['dbound'][0]) //
                               self.grid_config['dbound'][2]), ..., ] += mask
                del depth_sample
                del depth_sample_frustum
            sigma = torch.clamp(sigma, self.min_sigma)
            mu_repeated = mu.repeat(1, range_length, 1, 1)
            eps = 1e-6
            depth_score_single_range = (-1 / 2 * (
                (d_coords[:,
                          int((range_start - self.grid_config['dbound'][0]) //
                              self.grid_config['dbound'][2]):range_length + int(
                                  (range_start - self.grid_config['dbound'][0]) //
                                  self.grid_config['dbound'][2]), ..., ] - mu_repeated) /
                torch.sqrt(sigma))**2)
            depth_score_single_range = depth_score_single_range.exp()
            score_all_ranges.append(mu_score.sum(1).unsqueeze(1))
            depth_score_single_range = depth_score_single_range / (
                sigma * math.sqrt(2 * math.pi) + eps)
            stereo_depth[:,
                         int((range_start - self.grid_config['dbound'][0]) //
                             self.grid_config['dbound'][2]):range_length +
                         int((range_start - self.grid_config['dbound'][0]) //
                             self.grid_config['dbound'][2]), ..., ] = (
                                 depth_score_single_range *
                                 range_score[:, range_idx:range_idx + 1, ...])
            # del range_score
            del depth_score_single_range
            del mu_repeated
        if self.use_mask:
            return stereo_depth, mask_score
        else:
            return stereo_depth

    def forward(self, input):
        img_feat, depth_prob, rots, trans, intrins, post_rots, post_trans, bda = input
        B, N, C, H, W = img_feat.shape
        img_feat = img_feat.view(B*N,C,H,W)
        # Lift
        volume = depth_prob.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = self._forward_voxel_net(volume)
        volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat
        if self.accelerate:
            bev_feat = self.voxel_pooling_accelerated(rots, trans, intrins,
                                                      post_rots, post_trans,
                                                      bda, volume)
        else:
            geom = self.get_geometry(rots, trans, intrins,
                                     post_rots, post_trans, bda)
            if self.vp_megvii:
                bev_feat = self.voxel_pooling_bevdepth(geom, volume)
            else:
                bev_feat = self.voxel_pooling(geom, volume)
        return bev_feat