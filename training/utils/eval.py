import os
import random
import json
import numpy as np
import torch

from torchvision.transforms import ToPILImage


class GlobalEvaluator:
    def __init__(self, data_root_dir, save_results=False, results_dir=None):
        self.eval_dict = {}
        self.cls_to_color = {}
        self.color_to_cls = {}
        # parse datasets colors
        for d in [x for x in os.listdir(data_root_dir) if os.path.isdir(os.path.join(data_root_dir, x))]:
            color_map_path = os.path.join(data_root_dir, d, 'color_map.json')
            if os.path.exists(color_map_path):
                with open(color_map_path, 'r') as f:
                    color_map = {k:v for k, v in json.load(f).items() if k != 'background' and k != 'nonlabeled'}
                self.cls_to_color[d] = {k: v for k, v in color_map.items()}
                self.color_to_cls[d] = {self._hash_color(v): k for k, v in color_map.items()}
                self.eval_dict[d] = {k: Evaluator() for k in color_map.keys()}
        self.save_results = save_results
        self.results_dir = results_dir
        print(f"Initialized GlobalEvaluator with datasets: {list(self.eval_dict.keys())}")
    
    def _hash_color(self, color):
        return ",".join([str(x) for x in color])
    
    def get_cls_name(self, dataset_name, color):
        color_key = self._hash_color(color)
        if dataset_name in self.color_to_cls and color_key in self.color_to_cls[dataset_name]:
            return self.color_to_cls[dataset_name][color_key]
        return None
    
    def set_logger(self, logger):
        self.logger = logger
    
    def _save_test_results(self, input, pred_masks, max_samples_per_batch=1):
        pred_masks = pred_masks.cpu()
        gt_masks = torch.cat([input.tgt_mask_batch.unsqueeze(1), input.src_mask_batch], dim=1).cpu()
        imgs = input.img_batch.cpu()
        names = input.sample_names
        notion_colors = input.notion_colors.cpu()
        
        B = len(names)
        T = len(names[0])
        N = len(notion_colors[0])
        h, w = imgs.shape[-2:]
        half_h, half_w = h // 2, w // 2

        # unnormalize images
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        imgs = imgs * std + mean

        # half the resolution
        imgs = imgs[..., ::2, ::2]
        gt_masks = gt_masks[..., ::2, ::2]
        pred_masks = pred_masks[..., ::2, ::2]

        batch_idxes = random.sample(range(B), min(B, max_samples_per_batch))
        for b in batch_idxes:
            cur_names = names[b]
            dataset_name, filename = cur_names[0].rsplit('.', 1)
            save_folder = os.path.join(self.results_dir, dataset_name)
            os.makedirs(save_folder, exist_ok=True)
            # create image grid 2x(T) with first row being images and second row being gt masks, tgt pred mask is in row 1 col T+1
            vis_img = torch.zeros((3, half_h * 4 + 30, (half_w + 10)*T - 10), dtype=torch.uint8) + 128

            for t in range(T):
                img = (imgs[b, t] * 255).byte()
                vis_img[:3, :half_h, t * (half_w + 10): t * (half_w + 10) + half_w] = img
                # vis_img[3, :half_h, t * (half_w + 10): t * (half_w + 10) + half_w] = 255
                # convert gt and pred masks to color images
                cur_gt_rgb_mask = torch.zeros((half_h, half_w, 3), dtype=torch.uint8)
                for n in range(N):
                    cur_gt_rgb_mask[gt_masks[b, t, n]] = notion_colors[b, n].byte()
                vis_img[:3, half_h+10:half_h*2+10, t * (half_w + 10): t * (half_w + 10) + half_w] = cur_gt_rgb_mask.permute(2, 0, 1)
                # vis_img[3, -half_h:, t * (half_w + 10): t * (half_w + 10) + half_w] = 255

            # rgb pred mask probability and rgb binary pred mask
            for n in range(N):
                color = notion_colors[b][n]
                pred_soft = torch.zeros((half_h, half_w, 4), dtype=torch.uint8)
                pred_hard = torch.zeros((half_h, half_w, 4), dtype=torch.uint8)
                pred_mask = pred_masks[b, n]
                pred_soft[..., :3] = (color.view(1, 1, 3).float() * pred_mask.unsqueeze(-1)).byte()
                # pred_soft[..., 3] = (pred_mask * 255).byte()
                pred_hard[..., :3] = color.view(1, 1, 3) * (pred_mask > 0.5).unsqueeze(-1)
                # pred_hard[..., 3] = (pred_mask > 0.5).int() * 255
                vis_img[:, half_h * 2 + 20: half_h * 3 + 20, n * (half_w + 10): n * (half_w + 10) + half_w] = pred_soft.permute(2, 0, 1)[:3]
                vis_img[:, half_h * 3 + 30:, n * (half_w + 10): n * (half_w + 10) + half_w] = pred_hard.permute(2, 0, 1)[:3]

            ToPILImage()(vis_img).save(os.path.join(save_folder, f"{filename}.png"))

    def add_samples(self, batch_input, batch_pred_mask):
        batch_gt_masks = batch_input.tgt_mask_batch
        batch_names = batch_input.sample_names
        batch_notion_colors = batch_input.notion_colors
        batch_pred_mask = batch_pred_mask.sigmoid()
        
        for gt_masks, pred_masks, names, notion_colors in zip(batch_gt_masks, batch_pred_mask, 
                                                              batch_names, batch_notion_colors):
            tgt_name = names[0]  # target image is the first one
            dataset_name = tgt_name.rsplit('.', 1)[0]
            for gt_mask, pred_mask, color in zip(gt_masks, pred_masks, notion_colors):
                cls_name = self.get_cls_name(dataset_name, color.tolist())
                if cls_name is not None:
                    self.eval_dict[dataset_name][cls_name].add_sample(pred_mask, gt_mask)
        
        if self.save_results:
            self._save_test_results(batch_input, batch_pred_mask, max_samples_per_batch=1)


    def eval(self):
        sumarized_results = {}
        overall_results = Evaluator()
        for dataset, eval_dict in self.eval_dict.items():
            main_dataname = dataset.split('.')[0]
            if main_dataname not in sumarized_results:
                sumarized_results[main_dataname] = {}
            self.logger.write(f"Results for subdataset: {dataset}\n")
            for cls_name, evaluator in eval_dict.items():
                iou, prec, rec, acc = evaluator.eval()
                self.logger.write(f"  Class: {cls_name} - IoU: {iou:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, Accuracy: {acc:.4f}\n")
                if cls_name not in sumarized_results[main_dataname]:
                    sumarized_results[main_dataname][cls_name] = Evaluator()
                sumarized_results[main_dataname][cls_name] += evaluator
                overall_results += evaluator
        # Sumarize subdatasets
        self.logger.write(f"Summarized Results for dataset: {main_dataname}\n")
        for cls_name, evaluator in sumarized_results[main_dataname].items():
            iou, prec, rec, acc = evaluator.eval()
            self.logger.write(f"  [SUM] Class: {cls_name} - IoU: {iou:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, Accuracy: {acc:.4f}\n")
        
        # Overall results
        iou, prec, rec, acc = overall_results.eval()
        self.logger.write(f"Overall Results across all datasets - IoU: {iou:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, Accuracy: {acc:.4f}\n")
    

class Evaluator:
    def __init__(self):
        self.tp = 0
        self.union = 0
        self.pos_pred = 0
        self.pos_gt = 0
        self.overall_tp = 0 # including background
        self.n = 0
    
    # implement "+=" operator of Evaluator
    def __iadd__(self, other):
        """
        Implement the += operator to accumulate metrics from another Evaluator.
        Example:
            evaluator1 += evaluator2
        """
        self.tp += other.tp
        self.union += other.union
        self.pos_pred += other.pos_pred
        self.pos_gt += other.pos_gt
        self.overall_tp += other.overall_tp
        self.n += other.n
        return self

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
        iou = self.tp / self.union if self.union > 0 else 0.0
        prec = self.tp / self.pos_pred if self.pos_pred > 0 else 0.0
        rec = self.tp / self.pos_gt if self.pos_gt > 0 else 0.0
        acc = self.overall_tp / self.n if self.n > 0 else 0.0

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


