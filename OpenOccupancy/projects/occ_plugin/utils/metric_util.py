# -*- coding:utf-8 -*-
# author: Xinge
# @file: metric_util.py 

import numpy as np

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 2)
    
    # print('groundtruth noise classified as: ', hist[0])
    # print('predicted noise are actually: ', hist[:, 0])
    
    hist = hist[unique_label + 1, :]
    hist = hist[:, unique_label + 1]
    
    return hist

class SSCMetrics:
    def __init__(self, class_names, ignore_idx=255, empty_idx=None):
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.ignore_idx = ignore_idx
        self.empty_idx = empty_idx
        self.reset()

    def hist_info(self, n_cl, pred, gt):
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

    @staticmethod
    def compute_score(hist, correct, labeled):
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        mean_IU = np.nanmean(iu)
        mean_IU_no_back = np.nanmean(iu[1:])
        freq = hist.sum(1) / hist.sum()
        freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
        mean_pixel_acc = correct / labeled if labeled != 0 else 0

        return iu, mean_IU, mean_IU_no_back, mean_pixel_acc

    def add_batch(self, y_pred, y_true, nonsurface=None):
        self.count += 1
        mask = y_true != self.ignore_idx
        if self.empty_idx is not None:
            mask = mask & (y_true != self.empty_idx)
        if nonsurface is not None:
            mask = mask & nonsurface
        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)

        self.completion_tp += tp
        self.completion_fp += fp
        self.completion_fn += fn

        mask = y_true != self.ignore_idx
        if self.empty_idx is not None:
            mask = mask & (y_true != self.empty_idx)
        tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(
            y_pred, y_true, mask
        )
        self.tps += tp_sum
        self.fps += fp_sum
        self.fns += fn_sum

    def get_stats(self):
        if self.completion_tp != 0:
            precision = self.completion_tp / (self.completion_tp + self.completion_fp)
            recall = self.completion_tp / (self.completion_tp + self.completion_fn)
            iou = self.completion_tp / (
                self.completion_tp + self.completion_fp + self.completion_fn
            )
        else:
            precision, recall, iou = 0, 0, 0
        iou_ssc = self.tps / (self.tps + self.fps + self.fns + 1e-5)
        return {
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "iou_ssc": iou_ssc,
            "iou_ssc_mean": np.mean(iou_ssc[1:]),
        }

    def reset(self):

        self.completion_tp = 0
        self.completion_fp = 0
        self.completion_fn = 0
        self.tps = np.zeros(self.n_classes)
        self.fps = np.zeros(self.n_classes)
        self.fns = np.zeros(self.n_classes)

        self.hist_ssc = np.zeros((self.n_classes, self.n_classes))
        self.labeled_ssc = 0
        self.correct_ssc = 0

        self.precision = 0
        self.recall = 0
        self.iou = 0
        self.count = 1e-8
        self.iou_ssc = np.zeros(self.n_classes, dtype=np.float32)
        self.cnt_class = np.zeros(self.n_classes, dtype=np.float32)

    def get_score_completion(self, predict, target, nonempty=None):
        predict = np.copy(predict)
        target = np.copy(target)

        """for scene completion, treat the task as two-classes problem, just empty or occupancy"""
        _bs = predict.shape[0]  # batch size
        # ---- ignore
        predict[target == self.ignore_idx] = 0
        target[target == self.ignore_idx] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, 129600)
        predict = predict.reshape(_bs, -1)  # (_bs, _C, 129600), 60*36*60=129600
        # ---- treat all non-empty object class as one category, set them to label 1
        b_pred = np.zeros(predict.shape)
        b_true = np.zeros(target.shape)
        b_pred[predict != self.empty_idx] = 1
        b_true[target != self.empty_idx] = 1
        p, r, iou = 0.0, 0.0, 0.0
        tp_sum, fp_sum, fn_sum = 0, 0, 0
        for idx in range(_bs):
            y_true = b_true[idx, :]  # GT
            y_pred = b_pred[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_true = y_true[nonempty_idx == 1]
                y_pred = y_pred[nonempty_idx == 1]

            tp = np.array(np.where(np.logical_and(y_true == 1, y_pred == 1))).size
            fp = np.array(np.where(np.logical_and(y_true != 1, y_pred == 1))).size
            fn = np.array(np.where(np.logical_and(y_true == 1, y_pred != 1))).size
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        return tp_sum, fp_sum, fn_sum

    def get_score_semantic_and_completion(self, predict, target, nonempty=None):
        target = np.copy(target)
        predict = np.copy(predict)
        _bs = predict.shape[0]  # batch size
        _C = self.n_classes  # _C = 12
        # ---- ignore
        predict[target == self.ignore_idx] = 0
        target[target == self.ignore_idx] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, 129600)
        predict = predict.reshape(_bs, -1)  # (_bs, 129600), 60*36*60=129600

        cnt_class = np.zeros(_C, dtype=np.int32)  # count for each class
        iou_sum = np.zeros(_C, dtype=np.float32)  # sum of iou for each class
        tp_sum = np.zeros(_C, dtype=np.int32)  # tp
        fp_sum = np.zeros(_C, dtype=np.int32)  # fp
        fn_sum = np.zeros(_C, dtype=np.int32)  # fn

        for idx in range(_bs):
            y_true = target[idx, :]  # GT
            y_pred = predict[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_pred = y_pred[
                    np.where(np.logical_and(nonempty_idx == 1, y_true != self.ignore_idx))
                ]
                y_true = y_true[
                    np.where(np.logical_and(nonempty_idx == 1, y_true != self.ignore_idx))
                ]
            for j in range(_C):  # for each class
                tp = np.array(np.where(np.logical_and(y_true == j, y_pred == j))).size
                fp = np.array(np.where(np.logical_and(y_true != j, y_pred == j))).size
                fn = np.array(np.where(np.logical_and(y_true == j, y_pred != j))).size

                tp_sum[j] += tp
                fp_sum[j] += fp
                fn_sum[j] += fn

        return tp_sum, fp_sum, fn_sum