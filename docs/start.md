# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train FB-OCC with 8 GPUs 
```
./tools/dist_train.sh ./occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e.py 8
```

Eval FB-OCC with 8 GPUs
```
./tools/dist_test.sh./occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e.py ./path/to/ckpts.pth 8
```


