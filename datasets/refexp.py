# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MDETR 
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from collections import defaultdict
from pathlib import Path
from numpy import positive
import numpy as np

import torch
import torch.utils.data
from transformers import RobertaTokenizerFast

import util.misc as misc
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from .coco_bdetr import ModulatedDetection, make_coco_transforms
import utils.vis_utils
import ipdb
st = ipdb.set_trace
import wandb
import os

COLORS = [[0.375, 0.66, 0.089], [1, 0.38, 0]]

class RefExpDetection(ModulatedDetection):
    pass


class RefExpEvaluator(object):
    def __init__(self, refexp_gt, iou_types, k=(1, 5, 10), thresh_iou=0.5, limit=-1, visualize=False):
        assert isinstance(k, (list, tuple))
        refexp_gt = copy.deepcopy(refexp_gt)
        self.refexp_gt = refexp_gt
        self.iou_types = iou_types
        self.img_ids = self.refexp_gt.imgs.keys()
        self.predictions = {}
        self.positive_maps = {}
        self.projs = {}
        self.imgs = {}
        self.k = k
        self.thresh_iou = thresh_iou
        self.limit = limit
        self.visualize = visualize
        self.reset()
        if visualize:
            wandb.init(project="NAI_2d", name="2d_vis")
            
    def reset(self):
        self.prefixes = ['proposal_'] + [f'head{i}_' for i in range(5)] + ['last_']
        self.dets = {
            (prefix, k, mode): 0
            for prefix in self.prefixes
            for k in self.k
            for mode in ['bbs', 'bbf', 'opt']
        }

        self.gts = dict(self.dets)
        
    def print_stats(self):
        """Print accumulated accuracies."""
        if misc.is_main_process():
            mode_str = {
                'bbs': 'Box given span (soft-token)',
                'bbf': 'Box given span (contrastive)',
                'opt': 'Box given span (optimistic)',
            }
            for prefix in self.prefixes:
                for mode in ['bbs', 'bbf', 'opt']:
                        print(
                            prefix, mode_str[mode],
                            ', '.join([
                                'Top-%d: %.3f' % (
                                    k,
                                    self.dets[(prefix, k, mode)]
                                    / self.gts[(prefix, k, mode)]
                                )
                                for k in self.k
                            ])
                        )

    def synchronize_between_processes(self):
        all_dets = misc.all_gather(self.dets)
        all_gts = misc.all_gather(self.gts)

        if misc.is_main_process():
            merged_predictions = {}
            for key in all_dets[0].keys():
                merged_predictions[key] = 0
                for p in all_dets:
                    merged_predictions[key] += p[key]

            self.dets = merged_predictions

            merged_predictions = {}
            for key in all_gts[0].keys():
                merged_predictions[key] = 0
                for p in all_gts:
                    merged_predictions[key] += p[key]
                    
            self.gts = merged_predictions

    def evaluate(self, end_points):
        """
        Evaluate all accuracies.
        Args:
            end_points (dict): contains predictions and gt
        """
        for prefix in self.prefixes:
            self.evaluate_bbox_by_span_opt(end_points, prefix)
            self.evaluate_bbox_by_span(end_points, prefix)
            self.evaluate_bbox_by_contrast(end_points, prefix)

    def evaluate_bbox_by_span(self, end_points, prefix):
        predictions = end_points[f'{prefix}predictions']
        img_ids = end_points['image_ids']
        positive_map = end_points['positive_map']
        for i, image_id in enumerate(img_ids):
            ann_ids = self.refexp_gt.getAnnIds(imgIds=image_id)
            assert len(ann_ids) == 1

            target = self.refexp_gt.loadAnns(ann_ids[0])
            prediction = predictions[i]
            assert prediction is not None
            sem_scores = prediction['prob']
            span_scores = (sem_scores.unsqueeze(1) * positive_map.unsqueeze(0)).sum(-1)
            sorted_scores_boxes = sorted(
                        zip(span_scores.tolist(), prediction["boxes"].tolist()), reverse=True
                    )
            _, sorted_boxes = zip(*sorted_scores_boxes)
            sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
            
            target_bbox = target[0]["bbox"]
            converted_bbox = [
                target_bbox[0],
                target_bbox[1],
                target_bbox[2] + target_bbox[0],
                target_bbox[3] + target_bbox[1],
            ]
            gt_box = torch.as_tensor(converted_bbox).view(-1, 4)
            giou = generalized_box_iou(sorted_boxes, gt_box)
            for k in self.k:
                if max(giou[:k]) >= self.thresh_iou:
                    self.dets[(prefix, k, 'bbs')] += 1.0
                self.gts[(prefix, k, 'bbs')] += 1

    def evaluate_bbox_by_contrast(self, end_points, prefix):
        
        # we don't do contrastive alignment in proposal layer        
        if prefix == 'proposal_':
            for k in self.k:
                self.dets[(prefix, k, 'bbf')] += 0
                self.gts[(prefix, k, 'bbf')] += 1
            return

        predictions = end_points[f'{prefix}predictions']
        img_ids = end_points['image_ids']
        positive_map = end_points['positive_map']
        proj_queries = end_points[f'{prefix}proj_queries']
        proj_tokens = end_points['proj_tokens']
        for i, image_id in enumerate(img_ids):
            ann_ids = self.refexp_gt.getAnnIds(imgIds=image_id)
            assert len(ann_ids) == 1

            target = self.refexp_gt.loadAnns(ann_ids[0])
            prediction = predictions[i]
            assert prediction is not None

            
            proj_queries_ = proj_queries[i]
            proj_tokens_ = proj_tokens[i]
            contrast_scores = torch.matmul(proj_queries_, proj_tokens_.transpose(-1, -2))
            sem_scores_ = (contrast_scores / 0.07).softmax(-1)  # (Q, tokens)
            sem_scores = torch.zeros(sem_scores_.size(0), 256)
            sem_scores = sem_scores.to(sem_scores_.device)
            sem_scores[:sem_scores_.size(0), :sem_scores_.size(1)] = sem_scores_

            span_scores = (sem_scores.unsqueeze(1) * positive_map.unsqueeze(0)).sum(-1)
            sorted_scores_boxes = sorted(
                        zip(span_scores.tolist(), prediction["boxes"].tolist()), reverse=True
                    )
            _, sorted_boxes = zip(*sorted_scores_boxes)
            sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
            
            target_bbox = target[0]["bbox"]
            converted_bbox = [
                target_bbox[0],
                target_bbox[1],
                target_bbox[2] + target_bbox[0],
                target_bbox[3] + target_bbox[1],
            ]
            gt_box = torch.as_tensor(converted_bbox).view(-1, 4)
            
            giou = generalized_box_iou(sorted_boxes, gt_box)
            for k in self.k:
                if max(giou[:k]) >= self.thresh_iou:
                    self.dets[(prefix, k, 'bbf')] += 1.0
                self.gts[(prefix, k, 'bbf')] += 1

    def evaluate_bbox_by_span_opt(self, end_points, prefix):
        predictions = end_points[f'{prefix}predictions']
        img_ids = end_points['image_ids']
        for i, image_id in enumerate(img_ids):
            ann_ids = self.refexp_gt.getAnnIds(imgIds=image_id)
            assert len(ann_ids) == 1
            img_info = self.refexp_gt.loadImgs(image_id)[0]

            target = self.refexp_gt.loadAnns(ann_ids[0])
            prediction = predictions[i]
            assert prediction is not None
            sorted_scores_boxes = sorted(
                zip(prediction["scores"].tolist(), prediction["boxes"].tolist()), reverse=True
            )
            sorted_scores, sorted_boxes = zip(*sorted_scores_boxes)
            sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
            target_bbox = target[0]["bbox"]
            converted_bbox = [
                target_bbox[0],
                target_bbox[1],
                target_bbox[2] + target_bbox[0],
                target_bbox[3] + target_bbox[1],
            ]
            gt_box = torch.as_tensor(converted_bbox).view(-1, 4)
            giou = generalized_box_iou(sorted_boxes, gt_box)
            for k in self.k:
                if max(giou[:k]) >= self.thresh_iou:
                    self.dets[(prefix, k, 'opt')] += 1.0
                self.gts[(prefix, k, 'opt')] += 1

    def visualize_refexp(self, end_points):
        img_ids = end_points['image_ids']
        imgs = end_points["imgs"]
        for i, image_id in enumerate(img_ids):
            ann_ids = self.refexp_gt.getAnnIds(imgIds=image_id)
            assert len(ann_ids) == 1
            img_info = self.refexp_gt.loadImgs(image_id)[0]
            target = self.refexp_gt.loadAnns(ann_ids[0])
            target_bbox = target[0]["bbox"]
            converted_bbox = [
                target_bbox[0],
                target_bbox[1],
                target_bbox[2] + target_bbox[0],
                target_bbox[3] + target_bbox[1],
            ]
            xmin, ymin, xmax, ymax = converted_bbox
            gt_box = np.array([ymin, xmin, ymax, xmax]).reshape(-1, 4)
        
            prediction_decoder_boxes = []
            labels = []
            for prefix in self.prefixes:
                prediction = end_points[f'{prefix}predictions'][i]
                assert prediction is not None
                sorted_scores_boxes = sorted(
                    zip(prediction["scores"].tolist(), prediction["boxes"].tolist()), reverse=True
                )
                _, sorted_boxes = zip(*sorted_scores_boxes)
                xmin, ymin, xmax, ymax = sorted_boxes[0]
                boxes = np.concatenate([
                    gt_box,
                    np.array([ymin, xmin, ymax, xmax])[None],
                ], 0)
                prediction_decoder_boxes.append(boxes) # 2, 4
                labels.append(['gt', 'pred'])
                
            # Visualize detected boxes
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
            # area
            area = (detected_boxes.reshape(-1, 2, 2)[:, 1, :] - detected_boxes.reshape(-1, 2, 2)[:, 0, :]).prod(dim=1)
            # small boxes first
            sorted_area_boxes = sorted(
                zip(area.tolist(), detected_boxes.tolist(), detected_object_ids.tolist()), reverse=False
            )
            _, detected_boxes, detected_object_ids = zip(*sorted_area_boxes)
            detected_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in detected_boxes])
            
            # convert to y1, x1, y2, x2
            detected_boxes = torch.stack([detected_boxes[:, 1], detected_boxes[:, 0], \
                                detected_boxes[:, 3], detected_boxes[:, 2]], axis=1)
            img = torch.from_numpy(imgs[i]).unsqueeze(0)
            
            prediction_decoder_boxes = torch.from_numpy(prediction_decoder_boxes[-1][1][None]) # 2, 4
            all_boxes = []
            all_labels = []
            for box in detected_boxes:
                all_boxes.append(box.to(torch.float32))
                all_labels.append('detections')
                
            for box in prediction_decoder_boxes:
                all_boxes.append(box.to(torch.float32))
                all_labels.append('pred')
            
            all_boxes = torch.stack(all_boxes, axis=0).unsqueeze(0)
            boxlist_vis = utils.vis_utils.summ_boxlist2d(img, all_boxes, all_labels)[0].permute(1, 2, 0).cpu().numpy()
            wandb.log({"detections": wandb.Image(boxlist_vis, caption=img_info["caption"])})


def build(image_set, args):
    img_dir = Path(args.coco_path_refcoco) / "train2014"
    if args.butd:
        boxes_file = args.coco_boxes_path
    else:
        boxes_file = None

    refexp_dataset_name = args.refexp_dataset_name
    if refexp_dataset_name in ["refcoco", "refcoco+", "refcocog"]:
        if args.test:
            test_set = args.test_type
            ann_file = Path(args.refexp_ann_path) / f"finetune_{refexp_dataset_name}_{test_set}.json"
        elif image_set == "val":
            ann_file = Path(args.refexp_ann_path) / f"finetune_{refexp_dataset_name}_val.json"
        else:
            ann_file = Path(args.refexp_ann_path) / f"finetune_{refexp_dataset_name}_train.json"
    elif refexp_dataset_name in ["all"]:
        ann_file = Path(args.refexp_ann_path) / f"final_refexp_{image_set}.json"
    else:
        assert False, f"{refexp_dataset_name} not a valid datasset name for refexp"

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)
    dataset = RefExpDetection(
        img_dir,
        ann_file,
        transforms=make_coco_transforms(image_set, cautious=True),
        return_masks=False,
        return_tokens=True,
        tokenizer=tokenizer,
        image_set=image_set,
        boxes_file=boxes_file,
        butd=args.butd,
        new_contrastive=args.new_contrastive
    )
    return dataset
