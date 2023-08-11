import os 
import os.path as osp
import sys
import mmcv
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
total_counter = defaultdict(lambda: 0)
info = mmcv.load('/mount/dnn_data/occupancy_2023/annotations.json')
p1 = '/mount/dnn_data/occupancy_2023/gts'
json_map = {}
scenes = os.listdir(p1)
for scene in tqdm(info['train_split']):
    for sample in os.listdir(osp.join(p1, scene)):
        data = np.load(osp.join(p1, scene, sample, 'labels.npz'))
        occupancy = data['semantics']
        visible_mask = data['mask_camera']
        index = (visible_mask>0).nonzero()
        seen = occupancy[index[0],index[1],index[2]]
        counter = Counter(seen)
        json_map[sample] = {}
        for a,b in counter.items():
            total_counter[int(a)]+=b
            json_map[sample][int(a)] = b
from IPython import embed
embed()
exit()
new_json_map = {}

for key in json_map.keys()
    new_json_map[key] = {}
    for k, v in json_map[key].items():
        new_json_map[key][int(k)] = int(v)
# for scene in scenes:
