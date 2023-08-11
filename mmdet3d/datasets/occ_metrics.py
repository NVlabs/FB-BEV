import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import pickle as pkl
import argparse
import time
import torch
import sys, platform
from sklearn.neighbors import KDTree
from termcolor import colored
from pathlib import Path
from copy import deepcopy
from functools import reduce

np.seterr(divide='ignore', invalid='ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pcolor(string, color, on_color=None, attrs=None):
    """
    Produces a colored string for printing
    Parameters
    ----------
    string : str
        String that will be colored
    color : str
        Color to use
    on_color : str
        Background color to use
    attrs : list of str
        Different attributes for the string
    Returns
    -------
    string: str
        Colored string
    """
    return colored(string, color, on_color, attrs)


def getCellCoordinates(points, voxelSize):
    return (points / voxelSize).astype(np.int)


def getNumUniqueCells(cells):
    M = cells.max() + 1
    return np.unique(cells[:, 0] + M * cells[:, 1] + M ** 2 * cells[:, 2]).shape[0]


class Metric_mIoU():
    def __init__(self,
                 save_dir='.',
                 num_classes=18,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 min_d = -1,
                 max_d = 100,
                 ):
        self.class_names = ['others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                            'driveable_surface', 'other_flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation','free']
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes

        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0
        self.max_d = max_d
        self.min_d = min_d

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17
        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label
        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """

        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist


    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera):
        self.cnt += 1
        if len(semantics_pred.shape) == 4 or len(semantics_pred.shape) == 2:
            semantics_pred = semantics_pred.argmax(-1)
        
        if len(semantics_pred.shape) == 1:
            semantics_pred_ = deepcopy(semantics_gt)
            semantics_pred_[mask_camera] = semantics_pred
            semantics_pred = semantics_pred_


        xx, yy = np.meshgrid(np.arange(200), np.arange(200))
        mask = (np.stack([yy, xx], -1) -100) * 0.4
        distance_map = np.linalg.norm(mask, 2, -1)
        distance_map = (distance_map<=self.max_d) & (distance_map>=self.min_d)
        # print(semantics_pred.shape)
        # from IPython import embed
        # embed()
        # exit()

        # semantics_pred = semantics_pred[distance_map]
        # semantics_gt = semantics_gt[distance_map]
        # mask_camera = mask_camera[distance_map]
        mask_camera = mask_camera & distance_map[:,:, None]

        assert self.use_image_mask
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]
            if len(semantics_pred.shape) == 3:
                masked_semantics_pred = semantics_pred[mask_camera]
            elif len(semantics_pred.shape) == 1:
                masked_semantics_pred = semantics_pred
           
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar]
            masked_semantics_pred = semantics_pred[mask_lidar]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred

            # # pred = np.random.randint(low=0, high=17, size=masked_semantics.shape)
        _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
        self.hist += _hist

    def count_miou(self):
        res = {}
        mIoU = self.per_class_iu(self.hist)
        # assert cnt == num_samples, 'some samples are not included in the miou calculation'
        print(f'===> per class IoU of {self.cnt} samples:')
        for ind_class in range(self.num_classes-1):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 4)))
            res[self.class_names[ind_class]] = round(mIoU[ind_class] * 100, 2)

        print(f'===> mIoU of {self.cnt} samples: ' + str(round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2)))
        res['Overall'] =  round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2)
        # print(f'===> sample-wise averaged mIoU of {cnt} samples: ' + str(round(np.nanmean(mIoU_avg), 2)))

        return res


class Metric_FScore():
    def __init__(self,

                 leaf_size=10,
                 threshold_acc=0.6,
                 threshold_complete=0.6,
                 voxel_size=[0.4, 0.4, 0.4],
                 range=[-40, -40, -1, 40, 40, 5.4],
                 void=[17, 255],
                 use_lidar_mask=False,
                 use_image_mask=False, ) -> None:

        self.leaf_size = leaf_size
        self.threshold_acc = threshold_acc
        self.threshold_complete = threshold_complete
        self.voxel_size = voxel_size
        self.range = range
        self.void = void
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.cnt=0
        self.tot_acc = 0.
        self.tot_cmpl = 0.
        self.tot_f1_mean = 0.
        self.eps = 1e-8



    def voxel2points(self, voxel):
        # occIdx = torch.where(torch.logical_and(voxel != FREE, voxel != NOT_OBSERVED))
        # if isinstance(voxel, np.ndarray): voxel = torch.from_numpy(voxel)
        mask = np.logical_not(reduce(np.logical_or, [voxel == self.void[i] for i in range(len(self.void))]))
        occIdx = np.where(mask)

        points = np.concatenate((occIdx[0][:, None] * self.voxel_size[0] + self.voxel_size[0] / 2 + self.range[0], \
                                 occIdx[1][:, None] * self.voxel_size[1] + self.voxel_size[1] / 2 + self.range[1], \
                                 occIdx[2][:, None] * self.voxel_size[2] + self.voxel_size[2] / 2 + self.range[2]),
                                axis=1)
        return points

    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera ):

        # for scene_token in tqdm(preds_dict.keys()):
        self.cnt += 1
        
        if len(semantics_pred.shape) == 4 or len(semantics_pred.shape) == 2:
            semantics_pred = semantics_pred.argmax(-1)
        
        assert self.use_image_mask

        if self.use_image_mask:
            
            semantics_gt[mask_camera == False] = 255
            if len(semantics_pred.shape) == 1:
                semantics_pred_ = deepcopy(semantics_gt)
                semantics_pred_[mask_camera] = semantics_pred
                semantics_pred = semantics_pred_
            else:
                semantics_pred[mask_camera == False] = 255
        elif self.use_lidar_mask:
            semantics_gt[mask_lidar == False] = 255
            semantics_pred[mask_lidar == False] = 255
        else:
            pass

        ground_truth = self.voxel2points(semantics_gt)
        prediction = self.voxel2points(semantics_pred)
        if prediction.shape[0] == 0:
            accuracy=0
            completeness=0
            fmean=0

        else:
            prediction_tree = KDTree(prediction, leaf_size=self.leaf_size)
            ground_truth_tree = KDTree(ground_truth, leaf_size=self.leaf_size)
            complete_distance, _ = prediction_tree.query(ground_truth)
            complete_distance = complete_distance.flatten()

            accuracy_distance, _ = ground_truth_tree.query(prediction)
            accuracy_distance = accuracy_distance.flatten()

            # evaluate completeness
            complete_mask = complete_distance < self.threshold_complete
            completeness = complete_mask.mean()

            # evalute accuracy
            accuracy_mask = accuracy_distance < self.threshold_acc
            accuracy = accuracy_mask.mean()

            fmean = 2.0 / (1 / (accuracy+self.eps) + 1 / (completeness+self.eps))

        self.tot_acc += accuracy
        self.tot_cmpl += completeness
        self.tot_f1_mean += fmean

    def count_fscore(self,):
        res = {}
        base_color, attrs = 'red', ['bold', 'dark']
        print(pcolor('\n######## F score: {} #######'.format(self.tot_f1_mean / self.cnt), base_color, attrs=attrs))
        res['f-score'] = round(self.tot_f1_mean / self.cnt, 4)
        return res


import argparse
import os 
import sys
import numpy as np
from tqdm import tqdm
import time
def parse_args():
    parser = argparse.ArgumentParser(
        description='eval occupancy')
    parser.add_argument('pred_path', help='pred_path')
    parser.add_argument('--gt_path', default='/mount/data/occupancy_cvpr2023/gts', help='checkpoint file')
    parser.add_argument('--min_d', default=-1, type=int, help='min range')
    parser.add_argument('--max_d', default=100, type=int, help='max range')
    parser.add_argument(
        '--eval_fscore',
        action='store_true',
        help='whether to eval f-score.')
    args = parser.parse_args()
    return args

def eval(args):
    occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            min_d = args.min_d,
            max_d = args.max_d,
            use_image_mask=True)
    if args.eval_fscore:
        fscore_eval_metrics = Metric_FScore(
                leaf_size=10,
                threshold_acc=0.4,
                threshold_complete=0.4,
                voxel_size=[0.4, 0.4, 0.4],
                range=[-40, -40, -1, 40, 40, 5.4],
                void=[17, 255],
                use_lidar_mask=False,
                use_image_mask=True,)
    # print(len(os.listdir(args.pred_path)))
    pred_files = os.listdir(args.pred_path)
    val_splits = ['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018', 'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095', 'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103', 'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221', 'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275', 'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344', 'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524', 'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559', 'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0625', 'scene-0626', 'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636', 'scene-0637', 'scene-0638', 'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780', 'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784', 'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797', 'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907', 'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915', 'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924', 'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962', 'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972', 'scene-1059', 'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067', 'scene-1068', 'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073']
    mini_splits = ['scene-0103', 'scene-0916']
    for scene_name in tqdm(val_splits):
        if scene_name not in val_splits:continue
        for sample_token in os.listdir(os.path.join(args.gt_path, scene_name)):
            occ_gt = np.load(os.path.join(args.gt_path, scene_name, sample_token, 'labels.npz'))
            occ_pred = np.load(os.path.join(args.pred_path, scene_name+'_'+sample_token+'.npz'))['pred']
            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)
            occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)
            if args.eval_fscore:
                fscore_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)
    res = occ_eval_metrics.count_miou()
    if args.eval_fscore:
        fscore_eval_metrics.count_fscore()
        

if __name__ == '__main__':
    args = parse_args()
    eval(args)
