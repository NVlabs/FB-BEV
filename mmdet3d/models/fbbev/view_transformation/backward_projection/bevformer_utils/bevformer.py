# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .spatial_cross_attention_depth import DA_MSDeformableAttention
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import  build_neck


@TRANSFORMER.register_module()
class BEVFormer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_cams=6,
                 encoder=None,
                 embed_dims=256,
                 output_dims=256,     
                 use_cams_embeds=True,
                 **kwargs):
        super(BEVFormer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.output_dims = output_dims
    
        self.use_cams_embeds = use_cams_embeds

        self.init_layers()       

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
      

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, DA_MSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.cams_embeds)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def forward(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            # grid_length=[0.512, 0.512],
            bev_pos=None,
            cam_params=None,
            gt_bboxes_3d=None,
            pred_img_depth=None,
            prev_bev=None,
            bev_mask=None,
            **kwargs):
        """
        obtain bev features.
        """

        bs = mlvl_feats[0].size(0)

        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            else:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype) * 0
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            cam_params=cam_params,
            gt_bboxes_3d=gt_bboxes_3d,
            pred_img_depth=pred_img_depth,
            prev_bev=prev_bev,
            bev_mask=bev_mask,
            **kwargs
        )

        return bev_embed

