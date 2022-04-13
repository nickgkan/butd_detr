# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
BEAUTY-DETR model and criterion classes.
"""
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .bdetr_transformer import build_beauty_detr_transformer
import copy
import ipdb
st = ipdb.set_trace

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class BEAUTY_DETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        num_feature_levels,
        aux_loss=True,
        contrastive_hdim=64,
        contrastive_align_loss=False,
        with_box_refine=False,
        two_stage=False,
        butd=False,
    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                        DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.butd = butd

        self.num_classes = num_classes
        hidden_dim = transformer.d_model
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.num_feature_levels = num_feature_levels
            
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if self.class_embed is not None:
            self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers # 6 + 1
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)  # in_features=256, out_features=91
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        self.contrastive_align_loss = contrastive_align_loss
        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_image = nn.Linear(
                hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_text = nn.Linear(
                hidden_dim, contrastive_hdim)

    def forward(
        self,
        samples: NestedTensor,
        captions,
        encode_and_save=True,
        memory_cache=None,
        butd_boxes=None,
        butd_classes=None,
        butd_masks=None
    ):
        """The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        if encode_and_save:
            features, pos = self.backbone(
                samples)

            srcs = []
            masks = []
            for l, feat in enumerate(features):
                src, mask = feat.decompose()
                srcs.append(self.input_proj[l](src)) 
                masks.append(mask)
                assert mask is not None
            if self.num_feature_levels > len(srcs):
                _len_srcs = len(srcs)
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:
                        src = self.input_proj[l](features[-1].tensors)
                    else:
                        src = self.input_proj[l](srcs[-1])
                    m = samples.mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src)
                    masks.append(mask)
                    pos.append(pos_l)

            query_embeds = None
            # Don't initialise query embeddings if 2 stage
            if not self.two_stage:
                query_embeds = self.query_embed.weight
            memory_cache = self.transformer(
                srcs=srcs,
                masks=masks,
                pos_embeds=pos,
                query_embed=query_embeds,
                text=captions,
                encode_and_save=True,
                butd_boxes=butd_boxes,
                butd_classes=butd_classes,
                butd_masks=butd_masks
            )

            return memory_cache

        else:
            assert memory_cache is not None

            hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
                encode_and_save=False,
                filled_memory_cache=memory_cache,
                butd_boxes=butd_boxes,
                butd_classes=butd_classes,
                butd_masks=butd_masks
            )
            out = {}
            outputs_classes = []
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self.class_embed[lvl](hs[lvl])
                tmp = self.bbox_embed[lvl](hs[lvl])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)
            outputs_class = torch.stack(outputs_classes)
            outputs_coord = torch.stack(outputs_coords)

            out.update(
                {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            )
            if self.contrastive_align_loss:
                proj_queries = F.normalize(self.contrastive_align_projection_image(hs), p=2, dim=-1)
                proj_tokens = F.normalize(
                    self.contrastive_align_projection_text(memory_cache["text_memory"]).transpose(0, 1), p=2, dim=-1
                )
                out.update(
                    {
                        "proj_queries": proj_queries[-1],
                        "proj_tokens": proj_tokens,
                        "tokenized": memory_cache["tokenized"],
                    }
                )

            if self.aux_loss:
                # TODO: Think about this
                if self.contrastive_align_loss:
                    assert proj_tokens is not None and proj_queries is not None
                    out["aux_outputs"] = [
                        {
                            "pred_logits": a,
                            "pred_boxes": b,
                            "proj_queries": c,
                            "proj_tokens": proj_tokens,
                            "tokenized": memory_cache["tokenized"],
                        }
                        for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], proj_queries[:-1])
                    ]
                else:
                    out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

            if self.two_stage:
                enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
                out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
            return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(
        self,
        num_classes,
        matcher,
        eos_coef,
        weight_dict,
        losses,
        temperature,
        focal_alpha=0.25,
        new_contrastive=False
    ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.eos_coef = eos_coef
        self.temperature = temperature
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.new_contrastive = new_contrastive

    def loss_labels(self, outputs, targets, positive_map, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        logits = outputs["pred_logits"].log_softmax(-1)
        # BS x (num_queries) x (num_tokens)

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)

        tgt_pos = positive_map[tgt_idx]
        target_sim = torch.zeros_like(logits)
        target_sim[:, :, -1] = 1
        target_sim[src_idx] = tgt_pos

        loss_ce = -(logits * target_sim).sum(-1)

        eos_coef = torch.full(
            loss_ce.shape, self.eos_coef, device=target_sim.device)
        eos_coef[src_idx] = 1

        loss_ce = loss_ce * eos_coef
        loss_ce = loss_ce.sum() / num_boxes

        losses = {"loss_ce": loss_ce}

        return losses

    def loss_labels_st(self, outputs, targets, positive_map, indices, num_boxes, log=False):
        """Soft token prediction (with objectness)."""
        logits = outputs["pred_logits"].log_softmax(-1)  # (B, Q, 256)

        # Trick to get target indices across batches
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)

        # Labels, by default lines map to the last element, no_object
        tgt_pos = positive_map[tgt_idx]
        target_sim = torch.zeros_like(logits)
        # target_sim[:, :, -1] = 1
        # start of new loss
        # handle 'not mentioned'
        tokenized = outputs["tokenized"]
        inds = tokenized['attention_mask'].sum(1) - 1
        if self.new_contrastive:
            target_sim[torch.arange(len(inds)), :, inds] = 0.5
            target_sim[torch.arange(len(inds)), :, inds - 1] = 0.5
        # end of new loss
        target_sim[src_idx] = tgt_pos

        # Compute entropy
        loss_ce = -(logits * target_sim).sum(-1)

        # Weight less 'no_object'
        eos_coef = torch.full(
            loss_ce.shape, self.eos_coef,
            device=target_sim.device
        )
        eos_coef[src_idx] = 1
        loss_ce = loss_ce * eos_coef
        loss_ce = loss_ce.sum() / num_boxes

        losses = {"loss_ce": loss_ce}

        return losses
        
    def loss_contrastive_align(self, outputs, targets, positive_map, indices, num_boxes):
        bs = outputs["proj_queries"].shape[0]
        tokenized = outputs["tokenized"]

        normalized_text_emb = outputs["proj_tokens"]  # BS x (num_tokens) x hdim
        normalized_img_emb = outputs["proj_queries"]  # BS x (num_queries) x hdim
        
        logits = (
            torch.matmul(normalized_img_emb, normalized_text_emb.transpose(-1, -2)) / self.temperature
        )  # BS x (num_queries) x (num_tokens)

        # construct a map such that positive_map[k, i,j] = True iff query i is associated to token j in batch item k
        # For efficency, the construction happens on CPU, then the whole matrix is transferred to GPU in one go.
        positive_map = torch.zeros(logits.shape, dtype=torch.bool)
        for i, ((idx_src, idx_tgt), tgt) in enumerate(zip(indices, targets)):
            if "tokens_positive" in tgt:
                cur_tokens = [tgt["tokens_positive"][j] for j in idx_tgt]
            else:
                cur_tokens = [tgt["tokens"][j] for j in idx_tgt]

            for j, tok_list in enumerate(cur_tokens):
                for (beg, end) in tok_list:
                    beg_pos = tokenized.char_to_token(i, beg)
                    end_pos = tokenized.char_to_token(i, end - 1)
                    if beg_pos is None:
                        try:
                            beg_pos = tokenized.char_to_token(beg + 1)
                            if beg_pos is None:
                                beg_pos = tokenized.char_to_token(beg + 2)
                        except:
                            beg_pos = None
                    if end_pos is None:
                        try:
                            end_pos = tokenized.char_to_token(end - 2)
                            if end_pos is None:
                                end_pos = tokenized.char_to_token(end - 3)
                        except:
                            end_pos = None
                    if beg_pos is None or end_pos is None:
                        continue

                    assert beg_pos is not None and end_pos is not None
                    positive_map[i, idx_src[j], beg_pos : end_pos + 1].fill_(True)

        positive_map = positive_map.to(logits.device)
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits  # .masked_fill(positive_map, -1000000)

        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        neg_term = negative_logits.logsumexp(2)

        nb_pos = positive_map.sum(2) + 1e-6

        box_to_token_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~boxes_with_pos, 0).sum()

        tokens_with_pos = positive_map.any(1)
        pos_term = positive_logits.sum(1)
        neg_term = negative_logits.logsumexp(1)

        nb_pos = positive_map.sum(1) + 1e-6

        tokens_to_boxes_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~tokens_with_pos, 0).sum()
        tot_loss = (box_to_token_loss + tokens_to_boxes_loss) / 2

        return {"loss_contrastive_align": tot_loss / num_boxes}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, positive_map, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, positive_map, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, positive_map, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            "contrastive_align": self.loss_contrastive_align,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, positive_map, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, positive_map):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, positive_map)
        outputs['indices'] = indices
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, positive_map, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, positive_map)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, positive_map, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets, positive_map)
            for loss in self.losses:                
                if loss == 'contrastive_align':
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                    l_dict = self.loss_labels(enc_outputs, bin_targets, positive_map, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                    continue

                l_dict = self.get_loss(loss, enc_outputs, bin_targets, positive_map, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, targets=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits_, out_bbox_, prefix_ = [], [], []
        
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            out_logits_.append(enc_outputs['pred_logits'])
            out_bbox_.append(enc_outputs['pred_boxes'])
            prefix_.append('proposal')
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                out_logits_.append(aux_outputs['pred_logits'])
                out_bbox_.append(aux_outputs['pred_boxes'])
                prefix_.append(f"head{i}")
        
        out_logits_.append(outputs['pred_logits'])
        out_bbox_.append(outputs['pred_boxes'])
        prefix_.append("last")
        
        results_dict = {}
        for out_logits, out_bbox, prefix in zip(out_logits_, out_bbox_, prefix_):
            # out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
            assert len(out_logits) == len(target_sizes)
            assert target_sizes.shape[1] == 2

            prob = out_logits.sigmoid()
            topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
            scores = topk_values
            topk_boxes = topk_indexes // out_logits.shape[2]
            labels = topk_indexes % out_logits.shape[2]
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

            # and from relative [0, 1] to absolute [0, height] coordinates
            img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            boxes = boxes * scale_fct[:, None, :]

            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
            results_dict[prefix] = results
        return results_dict

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_bdetr(args):
    num_classes = 256
    device = torch.device(args.device)

    backbone = build_backbone(args)  # Resnet

    transformer = build_beauty_detr_transformer(
        args,
        backbone.backbone.butd_class_embeddings if args.butd else None)  # encoder & decoder

    model = BEAUTY_DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        contrastive_hdim=args.contrastive_loss_hdim,
        contrastive_align_loss=args.contrastive_align_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        butd=args.butd,
    )

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef

    weight_dict['loss_giou'] = args.giou_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]
    criterion = None
    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=args.focal_alpha,
        eos_coef=args.eos_coef,
        losses=losses,
        temperature=args.temperature_NCE,
        new_contrastive=args.new_contrastive
    )
    criterion.to(device) 
                               
    return model, criterion, weight_dict
