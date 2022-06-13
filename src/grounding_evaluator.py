# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
"""A class to collect and evaluate language grounding results."""

import torch

from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
import utils.misc as misc

import ipdb
st = ipdb.set_trace

class GroundingEvaluator:
    """
    Evaluate language grounding.

    Args:
        only_root (bool): detect only the root noun
        thresholds (list): IoU thresholds to check
        topks (list): k to evaluate top--k accuracy
        prefixes (list): names of layers to evaluate
    """

    def __init__(self, only_root=True, thresholds=[0.25, 0.5],
                 topks=[1, 5, 10], prefixes=[]):
        """Initialize accumulators."""
        self.only_root = only_root
        self.thresholds = thresholds
        self.topks = topks
        self.prefixes = prefixes

        self.reset()

    def reset(self):
        """Reset accumulators to empty."""
        self.dets = {
            (prefix, t, k, mode): 0
            for prefix in self.prefixes
            for t in self.thresholds
            for k in self.topks
            for mode in ['bbs', 'bbf']
        }
        self.gts = dict(self.dets)

        self.dets.update({'vd': 0, 'vid': 0})
        self.dets.update({'hard': 0, 'easy': 0})
        self.dets.update({'multi': 0, 'unique': 0})
        self.gts.update({'vd': 1e-14, 'vid': 1e-14})
        self.gts.update({'hard': 1e-14, 'easy': 1e-14})
        self.gts.update({'multi': 1e-14, 'unique': 1e-14})

    def print_stats(self):
        """Print accumulated accuracies."""
        mode_str = {
            'bbs': 'Box given span (soft-token)',
            'bbf': 'Box given span (contrastive)'
        }
        for prefix in self.prefixes:
            for mode in ['bbs', 'bbf']:
                for t in self.thresholds:
                    print(
                        prefix, mode_str[mode], 'Acc%.2f:' % t,
                        ', '.join([
                            'Top-%d: %.3f' % (
                                k,
                                self.dets[(prefix, t, k, mode)]
                                / max(self.gts[(prefix, t, k, mode)], 1)
                            )
                            for k in self.topks
                        ])
                    )
        print('\nAnalysis')
        for field in ['easy', 'hard', 'vd', 'vid', 'unique', 'multi']:
            print(field, self.dets[field] / self.gts[field])

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

    def evaluate(self, end_points, prefix):
        """
        Evaluate all accuracies.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        self.evaluate_bbox_by_span(end_points, prefix)
        self.evaluate_bbox_by_contrast(end_points, prefix)

    def evaluate_bbox_by_span(self, end_points, prefix):
        """
        Evaluate bounding box IoU for top gt span detections.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # Parse gt
        positive_map, gt_bboxes = self._parse_gt(end_points)

        # Parse predictions
        sem_scores = end_points[f'{prefix}sem_cls_scores'].softmax(-1)

        if sem_scores.shape[-1] != positive_map.shape[-1]:
            sem_scores_ = torch.zeros(
                sem_scores.shape[0], sem_scores.shape[1],
                positive_map.shape[-1]).to(sem_scores.device)
            sem_scores_[:, :, :sem_scores.shape[-1]] = sem_scores
            sem_scores = sem_scores_

        # Parse predictions
        pred_center = end_points[f'{prefix}center']  # B, Q, 3
        pred_size = end_points[f'{prefix}pred_size']  # (B,Q,3) (l,w,h)
        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            # Keep scores for annotated objects only
            num_obj = int(end_points['box_label_mask'][bid].sum())
            pmap = positive_map[bid, :num_obj]
            scores = (
                sem_scores[bid].unsqueeze(0)  # (1, Q, 256)
                * pmap.unsqueeze(1)  # (obj, 1, 256)
            ).sum(-1)  # (obj, Q)

            # 10 predictions per gt box
            top = scores.argsort(1, True)[:, :10]  # (obj, 10)
            pbox = pred_bbox[bid, top.reshape(-1)]

            # IoU
            ious, _ = _iou3d_par(
                box_cxcyczwhd_to_xyzxyz(gt_bboxes[bid][:num_obj]),  # (obj, 6)
                box_cxcyczwhd_to_xyzxyz(pbox)  # (obj*10, 6)
            )  # (obj, obj*10)
            ious = ious.reshape(top.size(0), top.size(0), top.size(1))
            ious = ious[torch.arange(len(ious)), torch.arange(len(ious))]

            # Measure IoU>threshold, ious are (obj, 10)
            topks = self.topks
            for t in self.thresholds:
                thresholded = ious > t
                for k in topks:
                    found = thresholded[:, :k].any(1)
                    self.dets[(prefix, t, k, 'bbs')] += found.sum().item()
                    self.gts[(prefix, t, k, 'bbs')] += len(thresholded)

    def evaluate_bbox_by_contrast(self, end_points, prefix):
        """
        Evaluate bounding box IoU by contrasting with span features.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # Parse gt
        positive_map, gt_bboxes = self._parse_gt(end_points)

        # Parse predictions
        pred_center = end_points[f'{prefix}center']  # B, Q, 3
        pred_size = end_points[f'{prefix}pred_size']  # (B,Q,3) (l,w,h)
        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)

        proj_tokens = end_points['proj_tokens']  # (B, tokens, 64)
        proj_queries = end_points[f'{prefix}proj_queries']  # (B, Q, 64)
        sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
        sem_scores_ = (sem_scores / 0.07).softmax(-1)  # (B, Q, tokens)
        sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256)
        sem_scores = sem_scores.to(sem_scores_.device)
        sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            # Keep scores for annotated objects only
            num_obj = int(end_points['box_label_mask'][bid].sum())
            pmap = positive_map[bid, :num_obj]
            scores = (
                sem_scores[bid].unsqueeze(0)  # (1, Q, 256)
                * pmap.unsqueeze(1)  # (obj, 1, 256)
            ).sum(-1)  # (obj, Q)

            # 10 predictions per gt box
            top = scores.argsort(1, True)[:, :10]  # (obj, 10)
            pbox = pred_bbox[bid, top.reshape(-1)]

            # IoU
            ious, _ = _iou3d_par(
                box_cxcyczwhd_to_xyzxyz(gt_bboxes[bid][:num_obj]),  # (obj, 6)
                box_cxcyczwhd_to_xyzxyz(pbox)  # (obj*10, 6)
            )  # (obj, obj*10)
            ious = ious.reshape(top.size(0), top.size(0), top.size(1))
            ious = ious[torch.arange(len(ious)), torch.arange(len(ious))]

            # Measure IoU>threshold, ious are (obj, 10)
            for t in self.thresholds:
                thresholded = ious > t
                for k in self.topks:
                    found = thresholded[:, :k].any(1)
                    self.dets[(prefix, t, k, 'bbf')] += found.sum().item()
                    self.gts[(prefix, t, k, 'bbf')] += len(thresholded)
                    if prefix == 'last_':
                        found = found[0].item()
                        if k == 1 and t == self.thresholds[0]:
                            if end_points['is_view_dep'][bid]:
                                self.gts['vd'] += 1
                                self.dets['vd'] += found
                            else:
                                self.gts['vid'] += 1
                                self.dets['vid'] += found
                            if end_points['is_hard'][bid]:
                                self.gts['hard'] += 1
                                self.dets['hard'] += found
                            else:
                                self.gts['easy'] += 1
                                self.dets['easy'] += found
                            if end_points['is_unique'][bid]:
                                self.gts['unique'] += 1
                                self.dets['unique'] += found
                            else:
                                self.gts['multi'] += 1
                                self.dets['multi'] += found

    def _parse_gt(self, end_points):
        positive_map = torch.clone(end_points['positive_map'])  # (B, K, 256)
        positive_map[positive_map > 0] = 1
        gt_center = end_points['center_label'][:, :, 0:3]  # (B, K, 3)
        gt_size = end_points['size_gts']  # (B, K2,3)
        gt_bboxes = torch.cat([gt_center, gt_size], dim=-1)  # cxcyczwhd
        if self.only_root:
            positive_map = positive_map[:, :1]  # (B, 1, 256)
            gt_bboxes = gt_bboxes[:, :1]  # (B, 1, 6)
        return positive_map, gt_bboxes


class GroundingGTEvaluator:
    """
    Evaluate language grounding.

    Args:
        only_root (bool): detect only the root noun
        thresholds (list): IoU thresholds to check
        topks (list): k to evaluate top--k accuracy
        prefixes (list): names of layers to evaluate
    """

    def __init__(self, prefixes=[]):
        """Initialize accumulators."""
        self.prefixes = prefixes
        self.reset()

    def reset(self):
        """Reset accumulators to empty."""
        self.dets = {
            (prefix, mode): 0
            for prefix in self.prefixes
            for mode in ['bbs', 'bbf']
        }
        self.gts = dict(self.dets)

        self.dets.update({'vd': 0, 'vid': 0})
        self.dets.update({'hard': 0, 'easy': 0})
        self.dets.update({'multi': 0, 'unique': 0})
        self.gts.update({'vd': 1e-14, 'vid': 1e-14})
        self.gts.update({'hard': 1e-14, 'easy': 1e-14})
        self.gts.update({'multi': 1e-14, 'unique': 1e-14})

    def print_stats(self):
        """Print accumulated accuracies."""
        mode_str = {
            'bbs': 'Box given span (soft-token)',
            'bbf': 'Box given span (contrastive)'
        }
        for prefix in self.prefixes:
            for mode in ['bbs', 'bbf']:
                print(prefix, mode_str[mode], f'Acc: {self.dets[(prefix, mode)] / self.gts[(prefix, mode)]}')
        print('\nAnalysis')
        for field in ['easy', 'hard', 'vd', 'vid', 'unique', 'multi']:
            print(field, self.dets[field] / self.gts[field])

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

    def evaluate(self, end_points, prefix):
        """
        Evaluate all accuracies.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        self.evaluate_bbox_by_span(end_points, prefix)
        self.evaluate_bbox_by_contrast(end_points, prefix)

    def evaluate_bbox_by_span(self, end_points, prefix):
        """
        Evaluate bounding box IoU for top gt span detections.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # Parse gt
        positive_map, gt_bboxes = self._parse_gt(end_points)

        # Parse predictions
        sem_scores = end_points[f'{prefix}sem_cls_scores'].softmax(-1)

        if sem_scores.shape[-1] != positive_map.shape[-1]:
            sem_scores_ = torch.zeros(
                sem_scores.shape[0], sem_scores.shape[1],
                positive_map.shape[-1]).to(sem_scores.device)
            sem_scores_[:, :, :sem_scores.shape[-1]] = sem_scores
            sem_scores = sem_scores_

        # Parse predictions
        pred_center = end_points[f'{prefix}center']  # B, Q, 3
        pred_size = end_points[f'{prefix}pred_size']  # (B,Q,3) (l,w,h)
        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            all_gt_boxes = box_cxcyczwhd_to_xyzxyz(
                        end_points['all_detected_boxes'][bid][
                            end_points['all_detected_bbox_label_mask'][bid]
                        ]
                    )

            # filter out boxes with low overlap
            ious, _ = _iou3d_par(all_gt_boxes,  # (gt, 6)
                box_cxcyczwhd_to_xyzxyz(pred_bbox[bid])  # (Q, 6)
            )  # (gt, Q)
            is_correct = (ious.max(0)[0] > 0.25) * 1.0

            # Keep scores for annotated objects only
            num_obj = int(end_points['box_label_mask'][bid].sum())
            pmap = positive_map[bid, :num_obj]
            scores = (
                sem_scores[bid].unsqueeze(0)  # (1, Q, 256)
                * pmap.unsqueeze(1)  # (obj, 1, 256)
            ).sum(-1)  # (obj, Q)
            scores = scores * is_correct[None]
            top = scores.argsort(1, True)[:, 0]  # (obj)
            pbox = pred_bbox[bid, top.reshape(-1)]

            # new pbox is the gt box with highest overlap with old pbox
            ious, _ = _iou3d_par(all_gt_boxes,  # (gt, 6)
                    box_cxcyczwhd_to_xyzxyz(pbox)  # (Q, 6)
                )
            pbox = end_points['all_detected_boxes'][bid][
                            end_points['all_detected_bbox_label_mask'][bid]
                        ][ious.argmax()]
            found = int((pbox == gt_bboxes[bid]).all())
            self.dets[(prefix, 'bbs')] += found
            self.gts[(prefix, 'bbs')] += 1

    def evaluate_bbox_by_contrast(self, end_points, prefix):
        """
        Evaluate bounding box IoU by contrasting with span features.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # Parse gt
        positive_map, gt_bboxes = self._parse_gt(end_points)

        # Parse predictions
        pred_center = end_points[f'{prefix}center']  # B, Q, 3
        pred_size = end_points[f'{prefix}pred_size']  # (B,Q,3) (l,w,h)
        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)

        proj_tokens = end_points['proj_tokens']  # (B, tokens, 64)
        proj_queries = end_points[f'{prefix}proj_queries']  # (B, Q, 64)
        sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
        sem_scores_ = (sem_scores / 0.07).softmax(-1)  # (B, Q, tokens)
        sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256)
        sem_scores = sem_scores.to(sem_scores_.device)
        sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            all_gt_boxes = box_cxcyczwhd_to_xyzxyz(
                    end_points['all_detected_boxes'][bid][
                        end_points['all_detected_bbox_label_mask'][bid]
                    ]
                )
            ious, _ = _iou3d_par(all_gt_boxes,  # (gt, 6)
                box_cxcyczwhd_to_xyzxyz(pred_bbox[bid])  # (Q, 6)
            )  # (gt, Q)
            is_correct = (ious.max(0)[0] > 0.25) * 1.0
            # Keep scores for annotated objects only
            num_obj = int(end_points['box_label_mask'][bid].sum())
            pmap = positive_map[bid, :num_obj]
            scores = (
                sem_scores[bid].unsqueeze(0)  # (1, Q, 256)
                * pmap.unsqueeze(1)  # (obj, 1, 256)
            ).sum(-1)  # (obj, Q)
            scores = scores * is_correct[None]

            # 10 predictions per gt box
            top = scores.argsort(1, True)[:, 0]  # (obj, 10)
            pbox = pred_bbox[bid, top.reshape(-1)]

            # IoU
            ious, _ = _iou3d_par(
                all_gt_boxes,  # (obj, 6)
                box_cxcyczwhd_to_xyzxyz(pbox)  # (obj*10, 6)
            )  # (obj, obj*10)
            pbox = end_points['all_detected_boxes'][bid][
                            end_points['all_detected_bbox_label_mask'][bid]
                        ][ious.argmax()]
            found = int((pbox == gt_bboxes[bid]).all())

            # Measure IoU>threshold, ious are (obj, 10)
            self.dets[(prefix, 'bbf')] += found
            self.gts[(prefix, 'bbf')] += 1
            if prefix == 'last_':
                if end_points['is_view_dep'][bid]:
                    self.gts['vd'] += 1
                    self.dets['vd'] += found
                else:
                    self.gts['vid'] += 1
                    self.dets['vid'] += found
                if end_points['is_hard'][bid]:
                    self.gts['hard'] += 1
                    self.dets['hard'] += found
                else:
                    self.gts['easy'] += 1
                    self.dets['easy'] += found
                if end_points['is_unique'][bid]:
                    self.gts['unique'] += 1
                    self.dets['unique'] += found
                else:
                    self.gts['multi'] += 1
                    self.dets['multi'] += found

    def _parse_gt(self, end_points):
        positive_map = torch.clone(end_points['positive_map'])  # (B, K, 256)
        positive_map[positive_map > 0] = 1
        gt_center = end_points['center_label'][:, :, 0:3]  # (B, K, 3)
        gt_size = end_points['size_gts']  # (B, K2,3)
        gt_bboxes = torch.cat([gt_center, gt_size], dim=-1)  # cxcyczwhd
        positive_map = positive_map[:, :1]  # (B, 1, 256)
        gt_bboxes = gt_bboxes[:, :1]  # (B, 1, 6)
        return positive_map, gt_bboxes
