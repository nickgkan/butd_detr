# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# ------------------------------------------------------------------------

import copy
from collections import defaultdict
from pathlib import Path
from numpy import positive
import numpy as np
import wandb
import os
import torch

import utils.vis_utils
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

import ipdb
st = ipdb.set_trace

def visualize_inputs(end_points):
    wandb.init(project="NAI_2d", name="2d_vis")
    
    # Visualize detected boxes
    img_ids = end_points['image_ids']
    imgs = end_points["imgs"]
    for i, image_id in enumerate(img_ids):
        detected_boxes = end_points["butd_boxes"][i]  # (Ndet, 4)
        sizes = end_points["orig_target_sizes"][i]
        detected_scores = end_points["butd_scores"][i]
        detected_masks = end_points['butd_masks'][i]
        detected_object_ids = end_points['butd_object_ids'][i]
        img_h, img_w = sizes
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
        detected_boxes = detected_boxes * scale_fct[None]
        detected_boxes = box_cxcywh_to_xyxy(detected_boxes)
        detected_boxes = detected_boxes[detected_masks]
        detected_object_ids = detected_object_ids[detected_masks]

        # convert to y1, x1, y2, x2
        detected_boxes = torch.stack([detected_boxes[:, 1], detected_boxes[:, 0], \
                            detected_boxes[:, 3], detected_boxes[:, 2]], axis=1).unsqueeze(0)
        img = torch.from_numpy(imgs[i]).unsqueeze(0)
        boxlist_vis = utils.vis_utils.summ_boxlist2d(img, detected_boxes)
        wandb.log({"detections": wandb.Image(boxlist_vis[0].permute(1, 2, 0).cpu().numpy())})
        
        if 'gt_boxes' in end_points:
            gt_boxes = end_points['gt_boxes'][i]
            
            caption = end_points['caption'][i]
            sizes = end_points["orig_target_sizes"][i]
            img_h, img_w = sizes
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
            gt_boxes = gt_boxes * scale_fct[None]
            gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
            gt_boxes = torch.stack([gt_boxes[:, 1], gt_boxes[:, 0], \
                            gt_boxes[:, 3], gt_boxes[:, 2]], axis=1).unsqueeze(0)
            boxlist_vis = utils.vis_utils.summ_boxlist2d(img, gt_boxes)[0]
            
            
            wandb.log({"gt": wandb.Image(boxlist_vis.permute(1, 2, 0).cpu().numpy(), caption=caption)})
            

def visualize_coco(results):
    wandb.init(project="NAI_2d", name="2d_vis")
    # Visualize detected boxes
    img_ids = results['image_ids']
    imgs = results["imgs"]
    
    for i, img_id in enumerate(img_ids):
        img = torch.from_numpy(imgs[i]).unsqueeze(0)
        if 'gt_boxes' in results:
            gt_boxes = results['gt_boxes'][i]
            
            caption = results['caption'][i]
            sizes = results["orig_target_sizes"][i]
            img_h, img_w = sizes
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
            gt_boxes = gt_boxes * scale_fct[None]
            gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
            gt_boxes = torch.stack([gt_boxes[:, 1], gt_boxes[:, 0], \
                            gt_boxes[:, 3], gt_boxes[:, 2]], axis=1).unsqueeze(0)
            boxlist_vis = utils.vis_utils.summ_boxlist2d(img, gt_boxes)[0]
            wandb.log({"gt": wandb.Image(boxlist_vis.permute(1, 2, 0).cpu().numpy(), caption=caption)})
            
        if 'boxes' in results:
            pred_boxes = results[img_id]['boxes']
            pred_scores = results[img_id]['scores']
            keep = (pred_scores > 0.7)
            pred_boxes = pred_boxes[keep]

            # pred_boxes = box_cxcywh_to_xyxy(pred_boxes)
            pred_boxes = torch.stack([pred_boxes[:, 1], pred_boxes[:, 0], \
                            pred_boxes[:, 3], pred_boxes[:, 2]], axis=1).unsqueeze(0)
            boxlist_vis = utils.vis_utils.summ_boxlist2d(boxlist_vis.unsqueeze(0), pred_boxes, color=[191.0, 142.0, 108.0])[0]
            wandb.log({"pred": wandb.Image(boxlist_vis.permute(1, 2, 0).cpu().numpy(), caption=caption)})
        
            