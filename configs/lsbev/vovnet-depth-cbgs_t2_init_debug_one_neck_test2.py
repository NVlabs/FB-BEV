# Copyright (c) Phigent Robotics. All rights reserved.

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (640, 1600),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-51.2, 51.2, 0.4],
    'y': [-51.2, 51.2, 0.4],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 0.5],
}
grid_config_bevformer={
        'xbound': [-51.2, 51.2, 0.4],
        'ybound': [-51.2, 51.2, 0.4],
        'zbound': [-5., 3.0, 2.],
       }

voxel_size = [0.05, 0.05, 0.2]

bev_h_ = 256
bev_w_ = 256
_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_ * 2
_num_levels_=1
numC_Trans=80

multi_adj_frame_id_cfg = (1, 1+8, 1)

model = dict(
    type='LSBEV4D_Pro',
    use_bev_mask=True,
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    img_backbone=dict(
        type='VoVNetCP',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage4','stage5',)),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[768, 1024],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepth',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        depthnet_cfg=dict(use_dcn=True),
        downsample=16),
    bevformer_encoder=dict(
        type='SparseBEVFormerEncoder',
        bev_h=bev_h_,
        bev_w=bev_w_,
        layer_scale=1.,
        mask_thre=0.0,
        in_channels=numC_Trans,
        out_channels=numC_Trans,
        pc_range=point_cloud_range,
        transformer=dict(
            type='PerceptionTransformer_onlyencoder',
            rotate_prev_bev=True,
            use_shift=True,
            use_cams_embeds=False,
            rotate_center=(bev_h_//2, bev_w_//2),
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=1,
                pc_range=point_cloud_range,
                grid_config=grid_config_bevformer,
                data_config=data_config,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention_v2',
                            pc_range=point_cloud_range,
                            dbound=[1.0, 60.0, 0.5],
                            deformable_attention=dict(
                                type='MSDeformableAttention3D_v3',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
           ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
    ),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),

    pre_process=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_layer=[2,],
        num_channels=[numC_Trans,],
        stride=[1,],
        backbone_output_ids=[0,]),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[2048, 2048, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=83,

            # Scale-NMS
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])))

# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        normalize_cfg=dict(
            mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False),
        sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    #
    dict(type='LoadBEVMaskv2', bev_size=(256, 256)),
    # dict(type='PadMultiViewImage'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bev_mask',
                                'gt_depth'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, 
    normalize_cfg=dict(
        mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False
        ),
    sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadBEVMaskv2',bev_size=(256, 256)),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs', 'gt_bev_mask',])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    data_root=data_root,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6,
    train=dict( 
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_trainval.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['train', 'val', 'test']:
    data[key].update(share_data_config)
# data['train']['dataset'].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=1.4e-4, weight_decay=1e-2,
            betas=(0.9, 0.999),
            paramwise_cfg=dict(
            custom_keys={
                'img_backbone': dict(lr_mult=0.1),
            })
        )
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=200,
#     warmup_ratio=0.001,
#     step=[20,])
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[30, 34])

runner = dict(type='EpochBasedRunner', max_epochs=36)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='WechatLoggerHook'),
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        resume='/mount/data/lsbevv2/work_dirs/vovnet-depth-cbgs_t2_init_debug_one_neck_test/epoch_19_ema.pth',
    ),
    dict(
        type='SequentialControlHook',
        temporal_start_epoch=3,
    ),
]
load_from = 'ckpts/dd3d_det_final.pth'
evaluation = dict(interval=4, pipeline=test_pipeline)
# fp16 = dict(loss_scale='dynamic')
find_unused_parameters = False