from mmdet3d.datasets import build_dataset
import mmcv
from mmcv import Config, DictAction
from mmdet3d.datasets import build_dataset
cfg = Config.fromfile('/mount/data/lsbevv2/occupancy_configs/occupancy/debug.py')
dataset = build_dataset(cfg.data.test, dict(test_mode=True))
import numpy as np
import torch

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision.utils import make_grid
import torchvision
import matplotlib.pyplot as plt
import cv2
import json
import os
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


def generate_forward_transformation_matrix(bda, img_meta_dict=None):
    b = bda.size(0)
    hom_res = torch.eye(4)[None].repeat(b, 1, 1).to(bda.device)
    for i in range(b):
        hom_res[i, :3, :3] = bda[i]
    return hom_res


from segment_anything import sam_model_registry, SamPredictor




def show_mask(mask, ax, random_color=False, cls_=None):
    classname_to_color= {'ignore_class': (255, 255, 255),  
                'barrier': (112, 128, 144),  # Slategrey
                'bicycle': (220, 20, 60),  # Crimson
                'bus': (255, 127, 80),  # Coral
                'car': (255, 158, 0),  # Orange
                'construction_vehicle': (233, 150, 70),  # Darksalmon
                'motorcycle': (255, 61, 99),  # Red
                'pedestrian': (0, 0, 230),  # Blue
                'traffic_cone': (47, 79, 79),  # Darkslategrey
                'trailer': (255, 140, 0),  # Darkorange
                'truck': (255, 99, 71),  # Tomato
                'driveable_surface': (0, 207, 191),  # nuTonomy green
                'other_flat': (175, 0, 75),
                'sidewalk': (75, 0, 75),
                'terrain': (112, 180, 60),
                'manmade': (222, 184, 135),  # Burlywood
                'vegetation': (0, 175, 0)}

    colors = np.array(list(classname_to_color.values())).astype(np.uint8)
    alpha = np.ones((colors.shape[0], 1), dtype=np.uint8) * 0.5
    colors = np.hstack([colors/255, alpha])
    
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif cls_ is not None:
        color = colors[cls_]
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

idx_to_name = mmcv.load('/mount/data/lsbevv2/data/nuscenes/v1.0-trainval/category.json')
idx_to_name = [each['name'] for each in idx_to_name]
name_category = {'animal':0,
'human.pedestrian.personal_mobility':0,
'human.pedestrian.stroller':0,
'human.pedestrian.wheelchair':0,
'movable_object.debris':0,
'movable_object.pushable_pullable':0,
'static_object.bicycle_rack':0,
'vehicle.emergency.ambulance':0,
'vehicle.emergency.police':0,
'noise':0,
'static.other':0,
'vehicle.ego':0,
'movable_object.barrier':1,
'vehicle.bicycle':2,
'vehicle.bus.bendy':3,
'vehicle.bus.rigid':3,
'vehicle.car':4,
'vehicle.construction':5,
'vehicle.motorcycle':6,
'human.pedestrian.adult':7,
'human.pedestrian.child':7,
'human.pedestrian.construction_worker': 7,
'human.pedestrian.police_officer':7,
'movable_object.trafficcone': 8,
'vehicle.trailer': 9,
'vehicle.truck': 10,
'flat.driveable_surface': 11,
'flat.other': 12,
'flat.sidewalk':  13,
'flat.terrain':  14,
'static.manmade':  15,
'static.vegetation': 16}

idx_to_category = [name_category[each] for each in idx_to_name]

from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "/mount/data/segment-anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# front_1 = './data/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-07-11-11-54-16+0800__CAM_FRONT_LEFT__1531281439754844.jpg'
# import cv2

# image = cv2.imread(front_1)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
import json
from collections import defaultdict
file_path = '/mount/data/lsbevv2/data/nuscenes/bevdetv2-nuscenes_infos_val.coco.json'
data = json.load(open(file_path, 'r'))
category_map_from_det_to_set = {
        0:4,
        1:10,
        2:9,
        3:3,
        4:5,
        5:2,
        6:6,
        7:7,
        8:8,
        9:1
    }
sample_map = defaultdict(lambda: [])
image_map = defaultdict(lambda: [])
for each in data['images']:
    sample_map[each['token']].append(each['id'])
for i, each in enumerate(data['annotations']):
    image_map[each['image_id']].append(i)
    

import argparse
import random
from tqdm import tqdm
def f(gap=0):
    co = 0
    for i in tqdm(range(gap, len(dataset))):
        co +=1
        print(i)
        info = dataset[i]
        category_map = info['gt_depth'][0]
        for j in range(len(idx_to_category)):
            category_map[category_map==j] = idx_to_category[j]
        imgs =  info['img_inputs'][0][0]
        cams = [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
            'CAM_BACK', 'CAM_BACK_RIGHT'
        ]
        for ind, img in enumerate(imgs):
            img = img.permute(1, 2, 0).to(torch.uint8)
            image = img.cpu().numpy()
            predictor.set_image(image)
            per_category_map = category_map[ind]

            sample_data_token = info['img_metas'][0].data['curr']['cams'][cams[ind]]['sample_data_token']
            # if os.path.isfile(f'/mount/data/lsbevv2/data/nus_sem/{sample_data_token}.png'): continue

            bboxes =[data['annotations'][each_idx] for each_idx in image_map[sample_data_token]]
            input_boxes = []
            for bbox in bboxes:
                bbox['category_id'] = category_map_from_det_to_set[bbox['category_id']]
                x, y, w, h = bbox['bbox']
                input_boxes.append([x, y, x+w, y+h])
                # input_box = np.array([x, y, x+w, y+h]) # xyxy format
            input_boxes = torch.tensor(input_boxes, device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
            sem_masks = np.zeros([17, 900, 1600]) + 0.05
            thing_mask = np.zeros([900, 1600])
            if len(input_boxes)>0:
                try:
                    masks, scores, logits = predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=torch.tensor(transformed_boxes).to(device),
                        multimask_output=False,
                        return_logits=False,
                    )
                    masks, scores = masks.squeeze(1).cpu().numpy(), scores.squeeze(1).cpu().numpy()

                    for index, mask in enumerate(masks):
                        id = bboxes[index]['category_id']
                        sem_masks[id][mask] = scores[index] + 0.4 # 0.4 is the bias of bbox prompt campared  to point prompt
                        thing_mask[mask] = 1
                except:
                    print(sample_data_token, ' thing error!!!!')

            for stuff_class in [11, 12, 13, 14, 15, 16]:
                points = torch.tensor((per_category_map == stuff_class).nonzero())
                if points.size(0)==0: continue
                else:
                    xs = [each[0].item() for each in points]
                    ys = [each[1].item() for each in points]
                    points = points[thing_mask[xs, ys]==0]
                    if points.size(0)==0: continue
                    if points.size(0)<=5:
                        points = random.choices(points, k=min(3, points.size(0)))
                    else:
                        try:
                            y = points[:, 0].to(torch.float).mean()
                            x = points[:, 1].to(torch.float).mean()
                            right_up = random.choices(points[(points[:,0]>=y) & (points[:,1]>=x)], k=1)
                            left_up =  random.choices(points[(points[:,0]<y) & (points[:,1]>=x)], k=1)
                            right_bottom =  random.choices(points[(points[:,0]>=y) & (points[:,1]<x)], k=1)
                            left_bottom =  random.choices(points[(points[:,0]<y) & (points[:,1]<x)], k=1)
                            points = right_up + left_up + right_bottom + left_bottom
                        except:
                            points = random.choices(points, k=min(3, points.size(0)))

                    input_point = np.array([each.cpu().numpy() for each in points])[:,::-1]
                    input_label = np.array([stuff_class for _ in range(len(input_point))])
                    try:
                        masks, scores, logits = predictor.predict(
                            point_coords=input_point,
                            point_labels=input_label,
                            multimask_output=False,
                            return_logits=False,
                        )
                        sem_masks[stuff_class][masks[0]] = scores[0]
                    except:
                        print(sample_data_token, stuff_class, ' stuff_error')

            sem_masks = torch.from_numpy(sem_masks).permute(1, 2, 0).argmax(-1).numpy()
            # np.save(f'/mount/data/lsbevv2/data/nus_sem/{sample_data_token}.npy', mask=sem_masks.astype(np.uint8))
            mmcv.imwrite(sem_masks, f'/mount/data/lsbevv2/data/nus_sem/{sample_data_token}.png')
            # sem_masks_ = mmcv.imread( f'/mount/data/lsbevv2/data/nus_sem/{sample_data_token}.jpg', flag='grayscale')
            # save_tensor(torch.tensor(sem_masks), 'tensor_{i}_{ind}.png'.format(i=i, ind=ind))
            # # .permute(2, 0, 1).numpy()

            # plt.figure(figsize=(10,10))
            # plt.imshow(image)

            # for p in range(17):     
            #     print(p, (sem_masks==p).sum())   
            #     show_mask(sem_masks==p, plt.gca(), random_color=False, cls_=p)
            # plt.axis('off')
            # f = plt.gcf()
            # f.savefig('a_{i}_{ind}.png'.format(i=i, ind=ind))
            # f.clear()

    # if i==5:break
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gap')
    parser.add_argument('gap', default=0, type=int, help='gap')
    args = parser.parse_args()
    f(gap=args.gap)

