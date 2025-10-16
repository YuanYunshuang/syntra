import numpy as np
import torch


class Evaluator:
    def __init__(self):
        self.tp = 0
        self.union = 0
        self.pos_pred = 0
        self.pos_gt = 0
        self.overall_tp = 0 # including background
        self.n = 0

    def add_sample(self, pred, target, thresh=0.5):
        assert pred.shape == target.shape
        pred_cls = pred > thresh

        # iou
        tp = torch.logical_and(pred_cls, target)
        union = torch.logical_or(pred_cls, target)

        self.n += 1
        self.overall_tp += (pred_cls == target).sum() / target.numel()
        self.tp += torch.sum(tp)
        self.union += torch.sum(union)
        self.pos_pred += torch.sum(pred_cls)
        self.pos_gt += torch.sum(target)

    def eval(self):
        iou = self.tp / self.union
        prec = self.tp / self.pos_pred
        rec = self.tp / self.pos_gt
        acc = self.overall_tp / self.n

        return iou, prec, rec, acc


def cal_all(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection.sum() / union.sum()
    pos_pred = pred_mask.sum()
    pos_gt = gt_mask.sum()
    prec = intersection / pos_pred
    rec = intersection / pos_gt
    return iou, prec, rec


