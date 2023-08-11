# Copyright (c) OpenMMLab. All rights reserved.
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
from mmdet3d.models.builder import build_loss


@HEADS.register_module()
class FRPN(BaseModule):
    r"""
    Args:
        in_channels (int): Channels of input feature.
        context_channels (int): Channels of transformed feature.
    """

    def __init__(
        self,
        in_channels=512,
        scale_factor=1,
        mask_thre = 0.4,
    ):
        super(FRPN, self).__init__()
        self.mask_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, 1, kernel_size=3, padding=1, stride=1),
            )
        self.upsample = nn.Upsample(scale_factor = scale_factor , mode ='bilinear',align_corners = True)
        self.dice_loss = build_loss(dict(type='CustomDiceLoss', use_sigmoid=True, loss_weight=1.))
        self.ce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.13]))  # From lss
        self.mask_thre = mask_thre

    def forward(self, input):
        """
        """
        bev_mask = self.mask_net(input)            
        bev_mask = self.upsample(bev_mask)
        return bev_mask
    
    
    def get_bev_mask_loss(self, gt_bev_mask, pred_bev_mask):
        bs, bev_h, bev_w = gt_bev_mask.shape
        b = gt_bev_mask.reshape(bs , bev_w * bev_h).permute(1, 0).to(torch.float)
        a = pred_bev_mask.reshape(bs, bev_w * bev_h).permute(1, 0)
        mask_ce_loss = self.ce_loss(a, b)
        mask_dice_loss = self.dice_loss(pred_bev_mask.reshape(bs, -1), gt_bev_mask.reshape(bs, -1))
        return dict(mask_ce_loss=mask_ce_loss, mask_dice_loss=mask_dice_loss)
