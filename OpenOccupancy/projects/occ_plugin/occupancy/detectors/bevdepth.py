# Copyright (c) Phigent Robotics. All rights reserved.
import math
import torch
from mmcv.runner import force_fp32
import torch.nn.functional as F

from mmdet.models import DETECTORS
from mmdet3d.models import builder
from torch.utils.checkpoint import checkpoint
from mmdet3d.models.detectors import CenterPoint

import pdb

@DETECTORS.register_module()
class BEVDet(CenterPoint):
    def __init__(self, img_view_transformer=None,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None, **kwargs):
        super(BEVDet, self).__init__(**kwargs)
        
        if img_view_transformer is not None:
            self.img_view_transformer = builder.build_neck(img_view_transformer)
        else:
            self.img_view_transformer = None
        
        if img_bev_encoder_backbone is not None:
            self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        else:
            self.img_bev_encoder_backbone = torch.nn.Identity()
        
        if img_bev_encoder_neck is not None:
            self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)
        else:
            self.img_bev_encoder_neck = torch.nn.Identity()

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    @force_fp32()
    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        x = self.image_encoder(img[0])
        x = self.img_view_transformer([x] + img[1:7])
        x = self.bev_encoder(x)
        return [x]

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = None
        return (img_feats, pts_feats)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        assert self.with_pts_bbox
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def forward_test(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0],list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0], **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        combine_type = self.test_cfg.get('combine_type','output')
        if combine_type=='output':
            return self.aug_test_combine_output(points, img_metas, img, rescale)
        elif combine_type=='feature':
            return self.aug_test_combine_feature(points, img_metas, img, rescale)
        else:
            assert False

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, _ = self.extract_feat(points, img=img, img_metas=img_metas)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list


    def forward_dummy(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        img_feats, _ = self.extract_feat(points, img=img_inputs, img_metas=img_metas)
        from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
        img_metas=[dict(box_type_3d=LiDARInstance3DBoxes)]
        bbox_list = [dict() for _ in range(1)]
        assert self.with_pts_bbox
        bbox_pts = self.simple_test_pts(
            img_feats, img_metas, rescale=False)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list


@DETECTORS.register_module()
class BEVDet4D(BEVDet):
    def __init__(self, pre_process=None,
                 align_after_view_transfromation=False,
                 detach=True,
                 detach_pre_process=False, **kwargs):
        super(BEVDet4D, self).__init__(**kwargs)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)
        self.align_after_view_transfromation = align_after_view_transfromation
        self.detach = detach
        self.detach_pre_process = detach_pre_process

    @force_fp32()
    def shift_feature(self, input, trans, rots):
        n, c, h, w = input.shape
        _, v, _ = trans[0].shape

        # generate grid
        xs = torch.linspace(0, w - 1, w, dtype=input.dtype,
                            device=input.device).view(1, w).expand(h, w)
        ys = torch.linspace(0, h - 1, h, dtype=input.dtype,
                            device=input.device).view(h, 1).expand(h, w)
        grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)
        grid = grid.view(1, h, w, 3).expand(n,h,w,3).view(n, h, w, 3, 1)

        # get transformation from current lidar frame to adjacent lidar frame
        # transformation from current camera frame to current lidar frame
        c02l0 = torch.zeros((n, v, 4, 4), dtype=grid.dtype).to(grid)
        c02l0[:, :, :3, :3] = rots[0]
        c02l0[:, :, :3, 3] = trans[0]
        c02l0[:, :, 3, 3] = 1

        # transformation from adjacent camera frame to current lidar frame
        c12l0 = torch.zeros((n, v, 4, 4), dtype=grid.dtype).to(grid)
        c12l0[:, :, :3, :3] = rots[1]
        c12l0[:, :, :3, 3] = trans[1]
        c12l0[:, :, 3, 3] = 1

        # transformation from current lidar frame to adjacent lidar frame
        l02l1 = c02l0.matmul(torch.inverse(c12l0))[:, 0, :, :].view(n, 1, 1, 4, 4)
        '''
          c02l0 * inv(c12l0)
        = c02l0 * inv(l12l0 * c12l1)
        = c02l0 * inv(c12l1) * inv(l12l0)
        = l02l1 # c02l0==c12l1
        '''

        l02l1 = l02l1[:, :, :, [True, True, False, True], :][:, :, :, :,
                [True, True, False, True]]

        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.dx[0]
        feat2bev[1, 1] = self.img_view_transformer.dx[1]
        feat2bev[0, 2] = self.img_view_transformer.bx[0] - \
                         self.img_view_transformer.dx[0] / 2.
        feat2bev[1, 2] = self.img_view_transformer.bx[1] - \
                         self.img_view_transformer.dx[1] / 2.
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)
        tf = torch.inverse(feat2bev).matmul(l02l1).matmul(feat2bev)

        # transform and normalize
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=input.dtype,
                                        device=input.device)
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1,
                                                            2) * 2.0 - 1.0
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True)
        return output

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran, bda):
        x = self.image_encoder(img)
        bev_feat = self.img_view_transformer([x, rot, tran, intrin,
                                              post_rot, post_tran, bda])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat

    def extract_img_feat(self, img, img_metas):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        N = N//2
        imgs = inputs[0].view(B,N,2,3,H,W)
        imgs = torch.split(imgs,1,2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
        extra = [rots.view(B,2,N,3,3),
                 trans.view(B,2,N,3),
                 intrins.view(B,2,N,3,3),
                 post_rots.view(B,2,N,3,3),
                 post_trans.view(B,2,N,3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        bev_feat_list = []
        key_frame=True # back propagation for key frame only
        for img, rot, tran, intrin, post_rot, \
            post_tran in zip(imgs, rots, trans, intrins, post_rots, post_trans):
            if self.align_after_view_transfromation:
                rot, tran = rots[0], trans[0]
            inputs_curr = (img, rot, tran, intrin, post_rot, post_tran, bda)
            if not key_frame and self.detach:
                with torch.no_grad():
                    bev_feat = self.prepare_bev_feat(*inputs_curr)
            else:
                bev_feat = self.prepare_bev_feat(*inputs_curr)
            bev_feat_list.append(bev_feat)
            key_frame = False
        if self.align_after_view_transfromation:
            bev_feat_list[1] = self.shift_feature(bev_feat_list[1],
                                                  trans, rots)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x]


class BEVDepth_Base(object):
    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats, depth = self.extract_img_feat(img, img_metas)
        pts_feats = None
        return (img_feats, pts_feats, depth)

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(points, img=img, img_metas=img_metas)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        assert self.with_pts_bbox
        # assert len(img_inputs) == 8
        depth_gt = img_inputs[7]
        loss_depth = self.img_view_transformer.get_depth_loss(depth_gt, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        
        # some modifications
        if hasattr(self.img_view_transformer, 'loss_depth_reg_weight') and self.img_view_transformer.loss_depth_reg_weight > 0:
            losses['loss_depth_reg'] = self.img_view_transformer.get_depth_reg_loss(depth_gt, depth)

        return losses

@DETECTORS.register_module()
class BEVDepth(BEVDepth_Base, BEVDet):
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        x = self.image_encoder(img[0])

        # img: imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors
        rots, trans, intrins, post_rots, post_trans, bda = img[1:7]
        
        mlp_input = self.img_view_transformer.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)
        geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
        
        x, depth = self.img_view_transformer([x] + geo_inputs)
        x = self.bev_encoder(x)
        
        return [x], depth


@DETECTORS.register_module()
class BEVDepth4D(BEVDepth_Base, BEVDet4D):
    def prepare_bev_feat(self, img, rot, tran, intrin,
                         post_rot, post_tran, bda, mlp_input):
        x = self.image_encoder(img)
        bev_feat, depth = self.img_view_transformer([x, rot, tran, intrin,
                                              post_rot, post_tran, bda, mlp_input])
        if self.detach_pre_process and self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth

    def extract_img_feat(self, img, img_metas):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        N = N//2
        imgs = inputs[0].view(B,N,2,3,H,W)
        imgs = torch.split(imgs,1,2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
        extra = [rots.view(B,2,N,3,3),
                 trans.view(B,2,N,3),
                 intrins.view(B,2,N,3,3),
                 post_rots.view(B,2,N,3,3),
                 post_trans.view(B,2,N,3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        bev_feat_list = []
        depth_list = []
        key_frame=True # back propagation for key frame only
        for img, rot, tran, intrin, post_rot, \
            post_tran in zip(imgs, rots, trans, intrins, post_rots, post_trans):
            if self.align_after_view_transfromation:
                rot, tran = rots[0], trans[0]
            mlp_input = self.img_view_transformer.get_mlp_input(
                rots[0], trans[0], intrin,post_rot, post_tran, bda)
            inputs_curr = (img, rot, tran, intrin, post_rot, post_tran, bda, mlp_input)
            if not key_frame and self.detach:
                with torch.no_grad():
                    bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
            else:
                bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
            if not self.detach_pre_process and self.pre_process:
                bev_feat = self.pre_process_net(bev_feat)[0]
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            key_frame = False
        if self.align_after_view_transfromation:
            bev_feat_list[1] = self.shift_feature(bev_feat_list[1],
                                                  trans, rots)
            
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth_list[0]


@DETECTORS.register_module()
class BEVStereo(BEVDepth4D):
    def __init__(self, bevdet_model=False, **kwargs):
        super(BEVStereo, self).__init__(**kwargs)
        self.bevdet_model = bevdet_model

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        stereo_feat = x[0].detach()

        # if isinstance(self.img_backbone, CustomSwin):
        #     stereo_feat = stereo_feat.permute(0,2,3,1)
        #     stereo_feat = self.img_backbone.norm0(stereo_feat)
        #     stereo_feat = stereo_feat.permute(0,3,1,2)
        
        if self.bevdet_model:
            x = x[-2:]
        
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat

    def extract_img_feat(self, img, img_metas):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        N = N//2
        imgs = inputs[0].view(B,N,2,3,H,W)
        imgs = torch.split(imgs,1,2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, bda, _, sensor2sensors = inputs[1:9]
        extra = [rots.view(B,2,N,3,3),
                 trans.view(B,2,N,3),
                 intrins.view(B,2,N,3,3),
                 post_rots.view(B,2,N,3,3),
                 post_trans.view(B,2,N,3),
                 sensor2sensors.view(B,2,N,4,4)]

        sensor2ego_mats = torch.eye(4).view(1,1,1,4,4).repeat(B,2,N,1,1).to(rots)
        sensor2ego_mats[:,:,:,:3,:3] = extra[0]
        sensor2ego_mats[:,:,:,:3,3] = extra[1]
        intrin_mats = torch.eye(4).view(1,1,1,4,4).repeat(B,2,N,1,1).to(rots)
        intrin_mats[:,:,:,:3,:3] = extra[2]
        ida_mats = torch.eye(4).view(1,1,1,4,4).repeat(B,2,N,1,1).to(rots)
        ida_mats[:,:,:,:3,:3] = extra[3]
        ida_mats[:,:,:,:3,3] = extra[4]
        mats_dict = dict(sensor2ego_mats=sensor2ego_mats,
                         intrin_mats=intrin_mats,
                         ida_mats=ida_mats,
                         sensor2sensor_mats=extra[5],
                         bda_mat=bda)
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans, sensor2sensors = extra

        # forward stereo depth
        context_all_sweeps = list()
        depth_feat_all_sweeps = list()
        img_feats_all_sweeps = list()
        stereo_feats_all_sweeps = list()
        mu_all_sweeps = list()
        sigma_all_sweeps = list()
        mono_depth_all_sweeps = list()
        range_score_all_sweeps = list()
        key_frame=True # back propagation for key frame only
        for img, rot, tran, intrin, post_rot, post_tran in zip(imgs, rots, trans, intrins, post_rots, post_trans):
            if not key_frame:
                with torch.no_grad():
                    img_feats, stereo_feats = self.image_encoder(img)
                    img_feats = img_feats.view(B * N, *img_feats.shape[2:])
                    mlp_input = \
                        self.img_view_transformer.get_mlp_input(rots[0], trans[0], intrin, post_rot, post_tran, bda)
                    depth_feat, context, mu, sigma, range_score, mono_depth = \
                        self.img_view_transformer.depth_net(img_feats,
                                                            mlp_input)
                    context = self.img_view_transformer.context_downsample_net(
                        context)
            else:
                img_feats, stereo_feats = self.image_encoder(img)
                img_feats = img_feats.view(B * N, *img_feats.shape[2:])
                mlp_input = \
                    self.img_view_transformer.get_mlp_input(rots[0], trans[0], intrin,
                                                            post_rot,
                                                            post_tran, bda)
                depth_feat, context, mu, sigma, range_score, mono_depth = \
                    self.img_view_transformer.depth_net(img_feats,
                                                        mlp_input)
                context = self.img_view_transformer.context_downsample_net(
                    context)
            img_feats_all_sweeps.append(img_feats)
            stereo_feats_all_sweeps.append(stereo_feats)
            depth_feat_all_sweeps.append(depth_feat)
            context_all_sweeps.append(context)
            mu_all_sweeps.append(mu)
            sigma_all_sweeps.append(sigma)
            mono_depth_all_sweeps.append(mono_depth)
            range_score_all_sweeps.append(range_score)
            key_frame = False

        depth_score_all_sweeps = list()
        num_sweeps = 2
        for ref_idx in range(num_sweeps):
            sensor2sensor_mats = list()
            for src_idx in range(num_sweeps):
                ref2keysensor_mats = sensor2sensors[ref_idx].inverse()
                key2srcsensor_mats = sensor2sensors[src_idx]
                ref2srcsensor_mats = key2srcsensor_mats @ ref2keysensor_mats
                sensor2sensor_mats.append(ref2srcsensor_mats)
            if ref_idx == 0:
                # last iteration on stage 1 does not have propagation
                # (photometric consistency filtering)
                if self.img_view_transformer.use_mask:
                    stereo_depth, mask = self.img_view_transformer._forward_stereo(
                        ref_idx,
                        stereo_feats_all_sweeps,
                        mono_depth_all_sweeps,
                        mats_dict,
                        sensor2sensor_mats,
                        mu_all_sweeps,
                        sigma_all_sweeps,
                        range_score_all_sweeps,
                        depth_feat_all_sweeps,
                    )
                else:
                    stereo_depth = self.img_view_transformer._forward_stereo(
                        ref_idx,
                        stereo_feats_all_sweeps,
                        mono_depth_all_sweeps,
                        mats_dict,
                        sensor2sensor_mats,
                        mu_all_sweeps,
                        sigma_all_sweeps,
                        range_score_all_sweeps,
                        depth_feat_all_sweeps,
                    )
            else:
                with torch.no_grad():
                    # last iteration on stage 1 does not have
                    # propagation (photometric consistency filtering)
                    if self.img_view_transformer.use_mask:
                        stereo_depth, mask = self.img_view_transformer._forward_stereo(
                            ref_idx,
                            stereo_feats_all_sweeps,
                            mono_depth_all_sweeps,
                            mats_dict,
                            sensor2sensor_mats,
                            mu_all_sweeps,
                            sigma_all_sweeps,
                            range_score_all_sweeps,
                            depth_feat_all_sweeps,
                        )
                    else:
                        stereo_depth = self.img_view_transformer._forward_stereo(
                            ref_idx,
                            stereo_feats_all_sweeps,
                            mono_depth_all_sweeps,
                            mats_dict,
                            sensor2sensor_mats,
                            mu_all_sweeps,
                            sigma_all_sweeps,
                            range_score_all_sweeps,
                            depth_feat_all_sweeps,
                        )
            if self.img_view_transformer.use_mask:
                depth_score = (
                        mono_depth_all_sweeps[ref_idx] +
                        self.img_view_transformer.depth_downsample_net(
                            stereo_depth) * mask).softmax(1)
            else:
                depth_score = (
                        mono_depth_all_sweeps[ref_idx] +
                        self.img_view_transformer.depth_downsample_net(stereo_depth)).softmax(1)
            depth_score_all_sweeps.append(depth_score)

        # forward view transformation
        bev_feat_list = []
        key_frame=True # back propagation for key frame only
        for image_feat, depth_prob, rot, tran, intrin, post_rot, post_tran in \
            zip(context_all_sweeps, depth_score_all_sweeps, rots, trans,
                intrins, post_rots, post_trans):
            if not key_frame:
                with torch.no_grad():
                    input_curr = (image_feat.view(B,N,*image_feat.shape[1:]),
                                  depth_prob, rot, tran, intrin, post_rot,
                                  post_tran, bda)
                    bev_feat = self.img_view_transformer(input_curr)
            else:
                input_curr = (image_feat.view(B,N,*image_feat.shape[1:]),
                                  depth_prob, rot, tran, intrin, post_rot,
                                  post_tran, bda)
                bev_feat = self.img_view_transformer(input_curr)
            if self.pre_process:
                bev_feat = self.pre_process_net(bev_feat)[0]
            bev_feat_list.append(bev_feat)
            key_frame = False

        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth_score_all_sweeps[0]