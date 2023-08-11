import math
from functools import partial
from mmdet3d.models.builder import BACKBONES
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as cp
import pdb
from mmcv.runner import BaseModule
import spconv.pytorch as spconv
from mmcv.runner import BaseModule, force_fp32
def get_inplanes():
    return [64, 128, 256, 512]

BIAS = True
def conv3x3x3(in_planes, out_planes, stride=1, use_spase_3dtensor=False):
    if not use_spase_3dtensor:
        Conv3d = nn.Conv3d
    else:
        Conv3d = spconv.SparseConv3d if stride!=1 else spconv.SubMConv3d

    return Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=BIAS)


def conv1x1x1(in_planes, out_planes, stride=1, use_spase_3dtensor=False):
    if not use_spase_3dtensor:
        Conv3d = nn.Conv3d
    else:
        Conv3d = spconv.SparseConv3d if stride!=1 else spconv.SubMConv3d

    return Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=BIAS)

from spconv.pytorch import functional as Fsp
class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, norm_cfg=None, use_spase_3dtensor=False):
        super().__init__()

        self.use_spase_3dtensor = use_spase_3dtensor
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

        if self.use_spase_3dtensor:
            Sequential = spconv.SparseSequential

            conv1 = conv3x3x3(in_planes, planes, stride, use_spase_3dtensor=self.use_spase_3dtensor)
            bn1 = build_norm_layer(norm_cfg, planes)[1]
            relu = nn.ReLU(inplace=True)
            conv2 = conv3x3x3(planes, planes, use_spase_3dtensor=self.use_spase_3dtensor)
            bn2 = build_norm_layer(norm_cfg, planes)[1]
            layer_list = [conv1, bn1, relu, conv2, bn2]
            
            self.layer_seq = Sequential(*layer_list)
        else:
            self.conv1 = conv3x3x3(in_planes, planes, stride, use_spase_3dtensor=self.use_spase_3dtensor)
            self.bn1 = build_norm_layer(norm_cfg, planes)[1]
            
            self.conv2 = conv3x3x3(planes, planes, use_spase_3dtensor=self.use_spase_3dtensor)
            self.bn2 = build_norm_layer(norm_cfg, planes)[1]

        self.stride = stride

    @force_fp32()
    def forward(self, x, debug=False):
        residual = x

        if self.use_spase_3dtensor:
            out = self.layer_seq(x)
            if self.downsample is not None:
                residual = self.downsample(x)
            out = Fsp.sparse_add(out, residual)
            out = out.replace_feature(self.relu(out.features))
            return out
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            return out




class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, norm_cfg=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = build_norm_layer(norm_cfg, planes * self.expansion)[1]
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    @force_fp32()
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

@BACKBONES.register_module()
class CustomResNet3D(BaseModule):
    def __init__(self,
                 depth,
                 block_inplanes=[64, 128, 256, 512],
                 block_strides=[1, 2, 2, 2],
                 out_indices=(0, 1, 2, 3),
                 n_input_channels=3,
                 shortcut_type='B',
                 with_cp=False,
                 norm_cfg=dict(type='BN3d', requires_grad=True),
                 use_spase_3dtensor=False,
                 plane2voxel=None,
                 widen_factor=1.0):
        super().__init__()
        
        layer_metas = {
            10: [1, 1, 1, 1],
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
        }
        
        if depth in [10, 18, 34]:
            block = BasicBlock
        else:
            assert depth in [50, 101]
            block = Bottleneck
        
        self.with_cp = with_cp 
        self.plane2voxel = plane2voxel
        
            
        layers = layer_metas[depth]
        self.use_spase_3dtensor = use_spase_3dtensor
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.out_indices = out_indices
        
        # replace the first several downsampling layers with the channel-squeeze layers
        Conv3d = nn.Conv3d if not self.use_spase_3dtensor else spconv.SubMConv3d
        Sequential = nn.Sequential if not self.use_spase_3dtensor else spconv.SparseSequential
        if self.use_spase_3dtensor:
            norm_cfg['type'] = 'BN1d'

        self.input_proj = Sequential(
            Conv3d(n_input_channels, self.in_planes, kernel_size=(1, 1, 1),
                      stride=(1, 1, 1), bias=False),
            build_norm_layer(norm_cfg, self.in_planes)[1],
            nn.ReLU(inplace=True),
        )
        
        self.layers = nn.ModuleList()
        for i in range(len(block_inplanes)):
            self.layers.append(self._make_layer(block, block_inplanes[i], layers[i], 
                                shortcut_type, block_strides[i], norm_cfg=norm_cfg))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, norm_cfg=None):
        downsample = None
        Sequential = nn.Sequential if not self.use_spase_3dtensor else spconv.SparseSequential
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                
                downsample = Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride, self.use_spase_3dtensor),
                    build_norm_layer(norm_cfg, planes * block.expansion)[1])

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample,
                  use_spase_3dtensor = self.use_spase_3dtensor,
                  norm_cfg=norm_cfg))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, norm_cfg=norm_cfg, use_spase_3dtensor = self.use_spase_3dtensor))

        return Sequential(*layers)
    
    @force_fp32()
    def forward(self, x):
        if self.plane2voxel is not None:
            x = x.unsqueeze(-1).repeat(1, 1, 1, 1, self.plane2voxel)
        x = self.input_proj(x)
        res = []
        for index, layer in enumerate(self.layers):
            if self.use_spase_3dtensor:
                for block in layer:
                    if self.with_cp:
                        x = cp(block, x)
                    else:
                        x = block(x)
            else:
                if self.with_cp:
                    x = cp(layer, x)
                else:
                    x = layer(x)
            
            if index in self.out_indices:
                if self.use_spase_3dtensor:
                    res.append(x.dense())
                else:
                    res.append(x)

        return res

def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model