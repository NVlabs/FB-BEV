from prettytable import PrettyTable
import numpy as np

def cm_to_ious(cm):
    mean_ious = []
    cls_num = len(cm)
    for i in range(cls_num):
        tp = cm[i, i]
        p = cm[:, i].sum()
        g = cm[i, :].sum()
        union = p + g - tp
        mean_ious.append(tp / union)
    
    return mean_ious

def format_results(mean_ious, return_dic=False):
    class_map = {
        1: 'barrier',
        2: 'bicycle',
        3: 'bus',
        4: 'car',
        5: 'construction_vehicle',
        6: 'motorcycle',
        7: 'pedestrian',
        8: 'traffic_cone',
        9: 'trailer',
        10: 'truck',
        11: 'driveable_surface',
        12: 'other_flat',
        13: 'sidewalk',
        14: 'terrain',
        15: 'manmade',
        16: 'vegetation',
    }
    
    x = PrettyTable()
    x.field_names = ['class', 'IoU']
    class_names = list(class_map.values()) + ['mean']
    class_ious = mean_ious + [sum(mean_ious) / len(mean_ious)]
    dic = {}
    
    for cls_name, cls_iou in zip(class_names, class_ious):
        dic[cls_name] = round(cls_iou, 3)
        x.add_row([cls_name, round(cls_iou, 3)])
    
    if return_dic:
        return x, dic 
    else:
        return x



def format_SC_results(mean_ious, return_dic=False):
    class_map = {
        1: 'non-empty',
    }
    
    x = PrettyTable()
    x.field_names = ['class', 'IoU']
    class_names = list(class_map.values())
    class_ious = mean_ious
    dic = {}
    
    for cls_name, cls_iou in zip(class_names, class_ious):
        dic[cls_name] = np.round(cls_iou, 3)
        x.add_row([cls_name, np.round(cls_iou, 3)])
    
    if return_dic:
        return x, dic 
    else:
        return x


def format_SSC_results(mean_ious, return_dic=False):
    class_map = {
        0: 'free',
        1: 'barrier',
        2: 'bicycle',
        3: 'bus',
        4: 'car',
        5: 'construction_vehicle',
        6: 'motorcycle',
        7: 'pedestrian',
        8: 'traffic_cone',
        9: 'trailer',
        10: 'truck',
        11: 'driveable_surface',
        12: 'other_flat',
        13: 'sidewalk',
        14: 'terrain',
        15: 'manmade',
        16: 'vegetation',
    }
    
    x = PrettyTable()
    x.field_names = ['class', 'IoU']
    class_names = list(class_map.values())
    class_ious = mean_ious
    dic = {}
    
    for cls_name, cls_iou in zip(class_names, class_ious):
        dic[cls_name] = np.round(cls_iou, 3)
        x.add_row([cls_name, np.round(cls_iou, 3)])
    
    mean_ious = sum(mean_ious[1:]) / len(mean_ious[1:])
    dic['mean'] = np.round(mean_ious, 3)
    x.add_row(['mean', np.round(mean_ious, 3)])
    
    if return_dic:
        return x, dic 
    else:
        return x

def format_vel_results(mean_epe, return_dic=False):
    class_map = {
        0: 'barrier',
        1: 'bicycle',
        2: 'bus',
        3: 'car',
        4: 'construction_vehicle',
        5: 'motorcycle',
        6: 'pedestrian',
        7: 'traffic_cone',
        8: 'trailer',
        9: 'truck',
    }
    x = PrettyTable()
    x.field_names = ['class', 'EPE']
    class_names = list(class_map.values())
    class_epes = mean_epe
    dic = {}
    
    for cls_name, cls_iou in zip(class_names, class_epes):
        dic[cls_name] = np.round(cls_iou, 3)
        x.add_row([cls_name, np.round(cls_iou, 3)])

    mean_all_epe = mean_epe.mean()
    dic['mean'] = np.round(mean_all_epe, 3)
    x.add_row(['mean', np.round(mean_all_epe, 3)])
    if return_dic:
        return x, dic 
    else:
        return x