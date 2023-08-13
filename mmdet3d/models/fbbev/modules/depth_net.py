# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint
from mmdet.models.backbones.resnet import BasicBlock
from mmdet.models import HEADS
import torch.utils.checkpoint as cp
from mmdet3d.models import builder
from mmcv.runner import force_fp32, auto_fp16
import torch
from torchvision.utils import make_grid
import torchvision
import matplotlib.pyplot as plt
import cv2

def convert_color(img_path):
    plt.figure()
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.imsave(img_path, img, cmap=plt.get_cmap('viridis'))
    plt.close()


def save_tensor(tensor, path, pad_value=254.0,normalize=False):
    print('save_tensor', path)
    tensor = tensor.to(torch.float).detach().cpu()
    max_ = tensor.flatten(1).max(-1).values[:, None, None]
    min_ = tensor.flatten(1).min(-1).values[:, None, None]
    tensor = (tensor-min_)/(max_-min_)
    if tensor.type() == 'torch.BoolTensor':
        tensor = tensor*255
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)
    tensor = make_grid(tensor, pad_value=pad_value, normalize=normalize).permute(1, 2, 0).numpy().copy()
    torchvision.utils.save_image(torch.tensor(tensor).permute(2, 0, 1), path)
    convert_color(path)


@HEADS.register_module()
class NaiveDepthNet(BaseModule):
    r"""Naive depthnet used in Lift-Splat-Shoot 

    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_

    Args:
        in_channels (int): Channels of input feature.
        context_channels (int): Channels of transformed feature.
    """

    def __init__(
        self,
        in_channels=512,
        context_channels=64,
        depth_channels=118,
        downsample=16,
        uniform=False,
        with_cp=False
    ):
        super(NaiveDepthNet, self).__init__()
        self.uniform = uniform
        self.with_cp = with_cp     
        self.context_channels = context_channels
        self.in_channels = in_channels
        self.D =depth_channels
        self.downsample=downsample,
        self.depth_net = nn.Conv2d(
            in_channels, self.D + self.context_channels, kernel_size=1, padding=0)
    
    @force_fp32()
    def forward(self, x, mlp_input=None):
        """
        """
       
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(self.depth_net, x)
        else:
            x = self.depth_net(x)            

        depth_digit = x[:, :self.D, ...]
        context = x[:, self.D:self.D + self.context_channels, ...]
        if self.uniform:
            depth_digit = depth_digit * 0
            depth = depth_digit.softmax(dim=1)
        else:
            depth = depth_digit.softmax(dim=1)
        context = context.view(B, N,  self.context_channels, H, W)
        depth = depth.view(B, N,  self.D, H, W)
        return context, depth

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        return None



class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()
    
    @force_fp32()
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

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(
            int(mid_channels * 5), mid_channels, 1, bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()
    
    @force_fp32()
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
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
    
    @force_fp32()
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
    
    @force_fp32()
    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


@HEADS.register_module()
class CM_DepthNet(BaseModule):
    """
        Camera parameters aware depth net
    """
    def __init__(self,
                 in_channels=512,
                 context_channels=64,
                 depth_channels=118,
                 mid_channels=512,
                 use_dcn=True,
                 downsample=16,
                 grid_config=None,
                 loss_depth_weight=3.0,
                 with_cp=False,
                 se_depth_map=False,
                 sid=False,
                 bias=0.0,
                 input_size=None,
                 use_aspp=True):
        super(CM_DepthNet, self).__init__()
        self.fp16_enable=False
        self.sid=sid
        self.with_cp = with_cp
        self.downsample = downsample
        self.grid_config = grid_config
        self.loss_depth_weight = loss_depth_weight
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_channels = context_channels
        self.depth_channels = depth_channels
        self.se_depth_map = se_depth_map
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        depth_conv_input_channels = mid_channels
        downsample = None

        depth_conv_list = [
           BasicBlock(depth_conv_input_channels, mid_channels,
                                      downsample=downsample),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        ]
        if use_aspp:
            depth_conv_list.append(ASPP(mid_channels, mid_channels))
        if use_dcn:
            depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type='DCN',
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )))
        depth_conv_list.append(
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        self.depth_conv = nn.Sequential(*depth_conv_list)


  
    @force_fp32()
    def forward(self, x, mlp_input):

        # if not  x.requires_grad: 
        x = x.to(torch.float32) # FIX distill type error
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(self.reduce_conv, x)
        else:
            x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        if self.with_cp and x.requires_grad:
            context = cp.checkpoint(self.context_se, x, context_se)
        else:
            context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)


            
        if self.with_cp and depth.requires_grad:
            depth = cp.checkpoint(self.depth_conv, depth)
        else:
            depth = self.depth_conv(depth)
        depth = depth.softmax(dim=1)
        context = context.view(B, N,  self.context_channels, H, W)
        depth = depth.view(B, N, self.depth_channels, H, W)

        return context, depth


    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        B, N, _, _ = rot.shape
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
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
        ],
                                dim=-1)
        sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)],
                               dim=-1).reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input


    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        downsample = self.downsample
        # if self.downsample == 8 and self.se_depth_map:
        #    downsample = 16 
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // downsample,
                                   downsample, W // downsample,
                                   downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, downsample * downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // downsample,
                                   W // downsample)
        if not self.sid:
            gt_depths = (gt_depths - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
        else:
            gt_depths = torch.log(gt_depths) - torch.log(
                torch.tensor(self.grid_config['depth'][0]).float())
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config['depth'][1] - 1.).float() /
                self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.
        gt_depths = torch.where((gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.depth_channels + 1).view(-1, self.depth_channels + 1)[:,
                                                                           1:]
        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 1, 3, 4,
                                          2).contiguous().view(-1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return dict(loss_depth=self.loss_depth_weight * depth_loss)




@HEADS.register_module()
class CM_ContextNet(nn.Module):
    """
        Camera parameters aware depth net
    """
    def __init__(self,
                 in_channels=512,
                 context_channels=64,
                 mid_channels=512,
                 with_cp=False,
                 ):
        super(CM_ContextNet, self).__init__()
        self.with_cp = with_cp
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_channels = context_channels
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware

    
    @force_fp32()
    def forward(self, x, mlp_input):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(self.reduce_conv, x)
        else:
            x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        if self.with_cp and x.requires_grad:
            context = cp.checkpoint(self.context_se, x, context_se)
        else:
            context = self.context_se(x, context_se)
        context = self.context_conv(context)
        context = context.view(B, N,  self.context_channels, H, W)
        return context
