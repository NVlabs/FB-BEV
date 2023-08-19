# FB-BEV & FB-OCC: Forward-Backward View Transformations for Occupancy Prediction.

![](figs/demo_1.png)

## News
- `[2023/8/01]` FB-BEV was accepted to ICCV 2023.
- 🏆 `[2023/6/16]` FB-OCC wins both Outstanding Champion and Innovation Award  in [Autonomous Driving Challenge](https://opendrivelab.com/AD23Challenge.html#Track3) in conjunction with CVPR 2023  End-to-End Autonomous Driving Workshop and  Vision-Centric Autonomous Driving Workshop.


## Getting Started
- [Installation](docs/install.md)
- [Prepare Dataset](docs/prepare_datasets.md)
- [Training, Eval, Visualization](docs/start.md)

## INTRODUCTION

FB-BEV and FB-OCC are vision-centric autonomous driving perception algorithms based on forward-backward view transformation strategies.
 
## Model Zoo

| Backbone | Method | Lr Schd | IoU|  Config | Download |
| :---: | :---: | :---: | :---: | :---: | :---: |
| R50 | FB-OCC | 20ep | 39.1 |[config](occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e.py) |[model](https://github.com/zhiqi-li/storage/releases/download/v1.0/fbocc-r50-cbgs_depth_16f_16x4_20e.pth)|

* More model weights will be released later.

## Acknowledgement

Many thanks to these excellent open source projects:

- [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [BEVDet](https://github.com/HuangJunJie2017/BEVDet), [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D), [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy), [SoloFusion](https://github.com/Divadi/SOLOFusion)

## BibTeX
If this work is helpful for your research, please consider citing:

```
@article{Li2023FBBEV,
  title={FB-BEV: BEV Representation from Forward-Backward View Transformations},
  author={Zhiqi Li and Zhiding Yu and Wenhai Wang and Anima Anandkumar and Tong Lu and Jos{\'e} Manuel {\'A}lvarez},
  year={2023},
  journal={IEEE/CVF International Conference on Computer Vision (ICCV)},
}
```
