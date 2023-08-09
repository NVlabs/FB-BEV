# Copyright (c) Phigent Robotics. All rights reserved.

# 2x/e24
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 8.78
# ===> barrier - IoU = 45.22
# ===> bicycle - IoU = 19.1
# ===> bus - IoU = 43.52
# ===> car - IoU = 50.24
# ===> construction_vehicle - IoU = 23.7
# ===> motorcycle - IoU = 19.75
# ===> pedestrian - IoU = 22.9
# ===> traffic_cone - IoU = 20.7
# ===> trailer - IoU = 31.88
# ===> truck - IoU = 37.65
# ===> driveable_surface - IoU = 80.3
# ===> other_flat - IoU = 37.02
# ===> sidewalk - IoU = 50.51
# ===> terrain - IoU = 53.41
# ===> manmade - IoU = 47.11
# ===> vegetation - IoU = 41.93
# ===> mIoU of 6019 samples: 37.28


# 3x/36e
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 9.49
# ===> barrier - IoU = 45.42
# ===> bicycle - IoU = 20.34
# ===> bus - IoU = 43.63
# ===> car - IoU = 50.52
# ===> construction_vehicle - IoU = 24.35
# ===> motorcycle - IoU = 20.97
# ===> pedestrian - IoU = 23.29
# ===> traffic_cone - IoU = 22.01
# ===> trailer - IoU = 33.4
# ===> truck - IoU = 37.67
# ===> driveable_surface - IoU = 80.45
# ===> other_flat - IoU = 38.03
# ===> sidewalk - IoU = 50.94
# ===> terrain - IoU = 53.77
# ===> manmade - IoU = 47.4
# ===> vegetation - IoU = 41.87
# ===> mIoU of 6019 samples: 37.86

# 4x/48e
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 10.07
# ===> barrier - IoU = 45.76
# ===> bicycle - IoU = 20.35
# ===> bus - IoU = 43.76
# ===> car - IoU = 50.54
# ===> construction_vehicle - IoU = 25.02
# ===> motorcycle - IoU = 21.56
# ===> pedestrian - IoU = 23.36
# ===> traffic_cone - IoU = 22.76
# ===> trailer - IoU = 34.22
# ===> truck - IoU = 38.05
# ===> driveable_surface - IoU = 80.54
# ===> other_flat - IoU = 38.16
# ===> sidewalk - IoU = 51.06
# ===> terrain - IoU = 53.37
# ===> manmade - IoU = 47.45
# ===> vegetation - IoU = 41.77
# ===> mIoU of 6019 samples: 38.11

# 5x/e60
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 10.12
# ===> barrier - IoU = 45.76
# ===> bicycle - IoU = 20.31
# ===> bus - IoU = 43.59
# ===> car - IoU = 50.45
# ===> construction_vehicle - IoU = 25.44
# ===> motorcycle - IoU = 22.02
# ===> pedestrian - IoU = 23.7
# ===> traffic_cone - IoU = 23.24
# ===> trailer - IoU = 33.77
# ===> truck - IoU = 37.78
# ===> driveable_surface - IoU = 80.54
# ===> other_flat - IoU = 38.4
# ===> sidewalk - IoU = 51.27
# ===> terrain - IoU = 53.85
# ===> manmade - IoU = 47.45
# ===> vegetation - IoU = 41.8
# ===> mIoU of 6019 samples: 38.2

# 6x/e72
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 10.35
# ===> barrier - IoU = 45.79
# ===> bicycle - IoU = 19.87
# ===> bus - IoU = 43.96
# ===> car - IoU = 50.62
# ===> construction_vehicle - IoU = 25.5
# ===> motorcycle - IoU = 22.27
# ===> pedestrian - IoU = 23.95
# ===> traffic_cone - IoU = 23.62
# ===> trailer - IoU = 34.53
# ===> truck - IoU = 37.88
# ===> driveable_surface - IoU = 80.68
# ===> other_flat - IoU = 38.16
# ===> sidewalk - IoU = 51.32
# ===> terrain - IoU = 53.92
# ===> manmade - IoU = 47.3
# ===> vegetation - IoU = 41.82
# ===> mIoU of 6019 samples: 38.33

# e100
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 10.82
# ===> barrier - IoU = 45.88
# ===> bicycle - IoU = 19.72
# ===> bus - IoU = 44.2
# ===> car - IoU = 50.61
# ===> construction_vehicle - IoU = 26.46
# ===> motorcycle - IoU = 22.44
# ===> pedestrian - IoU = 23.8
# ===> traffic_cone - IoU = 24.07
# ===> trailer - IoU = 34.93
# ===> truck - IoU = 37.48
# ===> driveable_surface - IoU = 80.75
# ===> other_flat - IoU = 38.51
# ===> sidewalk - IoU = 51.64
# ===> terrain - IoU = 54.38
# ===> manmade - IoU = 47.24
# ===> vegetation - IoU = 41.92
# ===> mIoU of 6019 samples: 38.52



_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
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
    'input_size': (384, 704),
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
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 32

multi_adj_frame_id_cfg = (1, 1+1, 1)

model = dict(
    type='BEVStereo4DOCC',
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVStereo',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=False,
        loss_depth_weight=0.05,
        depthnet_cfg=dict(use_dcn=False,
                          aspp_mid_channels=96,
                          stereo=True,
                          bias=5.),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
        num_layer=[1, 2, 4],
        with_cp=False,
        num_channels=[numC_Trans,numC_Trans*2,numC_Trans*4],
        stride=[1,2,2],
        backbone_output_ids=[0,1,2]),
    img_bev_encoder_neck=dict(type='LSSFPN3D',
                              in_channels=numC_Trans*7,
                              out_channels=numC_Trans),
    pre_process=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans,
        with_cp=False,
        num_layer=[1,],
        num_channels=[numC_Trans,],
        stride=[1,],
        backbone_output_ids=[0,]),
    loss_occ=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0),
    use_mask=True,
)

# Data
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar','mask_camera'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
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
            dict(type='Collect3D', keys=['points', 'img_inputs'])
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
    stereo=True,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[100,])
runner = dict(type='EpochBasedRunner', max_epochs=100)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

load_from="bevdet-r50-4d-stereo-cbgs.pth"
# fp16 = dict(loss_scale='dynamic')
