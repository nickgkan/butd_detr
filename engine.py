# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2021 Carnegie Mellon University. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR 
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# -------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable, Dict, Optional
from numpy import positive
from numpy.lib.shape_base import expand_dims


import torch
import util.misc as utils
from datasets.bdetr_coco_eval import CocoEvaluator
from datasets.refexp import RefExpEvaluator
from datasets.flickr_eval import FlickrEvaluator
from datasets.data_prefetcher import data_prefetcher, targets_to
from datasets.visualize import visualize_inputs, visualize_coco
from util.optim import update_ema, adjust_learning_rate
import ipdb
st = ipdb.set_trace

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    weight_dict: Dict[str, float],
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    max_norm: float = 0,
    wandb=None,
    model_ema: Optional[torch.nn.Module] = None,
):
    model.train()
    if criterion is not None:
        criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1000

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    num_training_steps = int(len(data_loader) * args.epochs)

    for i, _ in enumerate(metric_logger.log_every(
            range(len(data_loader)), print_freq, header)):
        curr_step = epoch * len(data_loader) + i
        captions = [t["caption"] for t in targets]
        positive_map = torch.cat([t["positive_map"] for t in targets])
        memory_cache = None
        butd_boxes = None
        butd_masks = None
        butd_classes = None
        if args.butd:
            butd_boxes = torch.stack([t['butd_boxes'] for t in targets], dim=0)
            butd_masks = torch.stack([t['butd_masks'] for t in targets], dim=0)
            butd_classes = torch.stack([t['butd_classes'] for t in targets], dim=0)
        memory_cache = model(
            samples,
            captions,
            encode_and_save=True,
            butd_boxes=butd_boxes,
            butd_classes=butd_classes,
            butd_masks=butd_masks
        )
        outputs = model(
            samples, captions, encode_and_save=False,
            memory_cache=memory_cache,
            butd_boxes=butd_boxes,
            butd_classes=butd_classes,
            butd_masks=butd_masks
        )

        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(outputs, targets, positive_map))

        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys()
            if k in weight_dict
            )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if args.wandb and curr_step % 10000 == 0:
            wandb.log({f'train_loss_full': loss_value}, step=curr_step)
            for k, v in loss_dict_reduced_scaled.items():
                wandb.log({f'train_loss_{k}': v}, step=curr_step)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(
                model.parameters(), max_norm)

        optimizer.step()

        if args.large_scale:
            adjust_learning_rate(
                optimizer,
                epoch,
                curr_step,
                num_training_steps=num_training_steps,
                args=args,
            )

        if model_ema is not None:
            update_ema(model, model_ema, args.ema_decay)
        
        metric_logger.update(loss=loss_value) #, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model,
    criterion,
    data_loader,
    device,
    postprocessors: Dict[str, torch.nn.Module],
    weight_dict: Dict[str, float],
    evaluator_list,
    args,
    epoch=0,
    wandb=None
):
    # Set to eval
    model.eval()
    if criterion is not None:
        criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 500, 'Test:')):
        # Move variables to device
        curr_step = epoch * len(data_loader) + i
        samples = samples.to(device)
        targets = targets_to(targets, device)
        captions = [t["caption"] for t in targets]
        positive_map = torch.cat(
            [t["positive_map"] for t in targets])

        memory_cache = None
        butd_boxes = None
        butd_masks = None
        butd_classes = None
        if args.butd:
            butd_boxes = torch.stack([t['butd_boxes'] for t in targets], dim=0)
            butd_masks = torch.stack([t['butd_masks'] for t in targets], dim=0)
            butd_classes = torch.stack([t['butd_classes'] for t in targets], dim=0)
        memory_cache = model(
            samples,
            captions,
            encode_and_save=True,
            butd_boxes=butd_boxes,
            butd_classes=butd_classes,
            butd_masks=butd_masks
        )
        outputs = model(
            samples, captions, encode_and_save=False,
            memory_cache=memory_cache,
            butd_boxes=butd_boxes,
            butd_classes=butd_classes,
            butd_masks=butd_masks
        )

        # Collect losses
        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(outputs, targets, positive_map))

        # Reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items() if k in weight_dict
        }

        if args.wandb and curr_step % 1000 == 0:
            for k, v in loss_dict_reduced_scaled.items():
                wandb.log({f'eval_loss_{k}': v}, step=curr_step)

        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()))

        # Postprocess results (to bring them in COCO format?)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        
        flickr_res = [] if "flickr_bbox" in postprocessors.keys() else None
        if "flickr_bbox" in postprocessors.keys():
            image_ids = [t["original_img_id"] for t in targets]
            sentence_ids = [t["sentence_id"] for t in targets]
            items_per_batch_element = [t["nb_eval"] for t in targets]
            positive_map_eval = torch.cat(
                [t["positive_map_eval"] for t in targets]
            ).to(device)
            flickr_results = postprocessors["flickr_bbox"](
                outputs, orig_target_sizes, positive_map_eval, items_per_batch_element,
                contrastive=False
            )
            assert len(flickr_results) == len(image_ids) == len(sentence_ids)
            for im_id, sent_id, output in zip(image_ids, sentence_ids, flickr_results):
                flickr_res.append({"image_id": im_id, "sentence_id": sent_id, "boxes": output})
            
            if args.visualize and curr_step % 10 == 0:
                if "butd_boxes" in targets[0]:
                    results['image_ids'] = [target['image_id'].item() for target in targets]
                    results['butd_boxes'] = [target['butd_boxes'] for target in targets]
                    results['butd_scores'] = [target['butd_scores'] for target in targets]
                    results['butd_masks'] = [target['butd_masks'] for target in targets]
                    results['butd_object_ids'] = [target['butd_classes'] for target in targets]
                    results['orig_target_sizes'] = orig_target_sizes
                    img, _ = samples.decompose()
                    results["imgs"] = img.detach().cpu().numpy()
                    visualize_inputs(results)
        for evaluator in evaluator_list:
            if isinstance(evaluator, FlickrEvaluator):
                    evaluator.update(flickr_res)
            elif isinstance(evaluator, RefExpEvaluator):
                results['positive_map'] = positive_map
                results['proj_tokens'] = outputs['proj_tokens']
                results['image_ids'] = [target['image_id'].item() for target in targets]
                if "butd_boxes" in targets[0]:
                    results['butd_boxes'] = [target['butd_boxes'] for target in targets]
                    results['butd_scores'] = [target['butd_scores'] for target in targets]
                    results['butd_masks'] = [target['butd_masks'] for target in targets]
                    results['butd_object_ids'] = [target['butd_classes'] for target in targets]
                    results['orig_target_sizes'] = orig_target_sizes
                evaluator.evaluate(results)
                if args.visualize and curr_step % 10 == 0:
                    img, _ = samples.decompose()
                    results["imgs"] = img.detach().cpu().numpy()
                    evaluator.visualize_refexp(results)    
            else:
                res = {
                    target['image_id'].item(): output
                    for target, output in zip(targets, results['last_predictions'])
                }
                if args.visualize and curr_step % 10 == 0:
                    if "butd_boxes" in targets[0]:
                        results['image_ids'] = [target['image_id'].item() for target in targets]
                        results['butd_boxes'] = [target['butd_boxes'] for target in targets]
                        results['butd_scores'] = [target['butd_scores'] for target in targets]
                        results['butd_masks'] = [target['butd_masks'] for target in targets]
                        results['butd_object_ids'] = [target['butd_classes'] for target in targets]
                        results['orig_target_sizes'] = orig_target_sizes
                    results['caption'] = [target['caption'] for target in targets]
                    results['gt_boxes'] = [target['boxes'] for target in targets]
                    img, _ = samples.decompose()
                    results["imgs"] = img.detach().cpu().numpy()
                evaluator.update(res)
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    for evaluator in evaluator_list:
        evaluator.synchronize_between_processes()
    refexp_res = None
    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            evaluator.accumulate()
            evaluator.summarize()
        elif isinstance(evaluator, (RefExpEvaluator)):
            evaluator.print_stats()
        elif isinstance(evaluator, FlickrEvaluator):
            flickr_res = evaluator.summarize()
        else:
            assert False, "unknown evaluator"
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            if "bbox" in postprocessors.keys():
                stats["coco_eval_bbox"] = evaluator.coco_eval["bbox"].stats.tolist()
            if "segm" in postprocessors.keys():
                stats["coco_eval_masks"] = evaluator.coco_eval["segm"].stats.tolist()

    if refexp_res is not None:
        stats.update(refexp_res)

    if flickr_res is not None:
        stats["flickr"] = flickr_res

    return stats
