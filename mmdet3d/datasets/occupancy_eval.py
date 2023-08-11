from .occ_metrics import Metric_mIoU, Metric_FScore
import argparse
import os 
import sys
import nunmpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description='eval occupancy')
    parser.add_argument('pred_path', help='pred_path')
    parser.add_argument('--gt', default='/mount/data/occupancy_cvpr2023/gts', help='checkpoint file')
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
    for pred_path in os.listdir(args.pred_path):
        occ_pred = np.load(os.path.join(args.pred_path, pred_path))['pred']
        occ_gt = np.load(os.path.join(args.gt_path, pred_path.split('.')[0], 'labels.npz'))
        gt_semantics = occ_gt['semantics']
        mask_lidar = occ_gt['mask_lidar'].astype(bool)
        mask_camera = occ_gt['mask_camera'].astype(bool)
        occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)
        if args.eval_fscore:
            fscore_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)
        res = occ_eval_metrics.count_miou()
        if eval_fscore:
            fscore_eval_metrics.count_fscore()
        

if __main__:
    args = parse_args()
    eval(args)
