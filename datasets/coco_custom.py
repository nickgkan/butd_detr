# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MDETR 
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import random
import json
import lmdb
import pickle
import numpy as np

import torch
import torch.utils.data
from pycocotools import mask as coco_mask
import torchvision
from transformers import RobertaTokenizerFast

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
import ipdb
st = ipdb.set_trace

class ModulatedCustomCOCODetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, return_tokens, tokenizer, is_train=False, image_set='train',
    class_name_dict=None, boxes_file=None, butd=None):
        super(ModulatedCustomCOCODetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.class_name_dict = class_name_dict
        self.prepare = ConvertCocoPolysToMask(
            return_masks,
            return_tokens,
            tokenizer=tokenizer,
            class_name_dict=self.class_name_dict,
        )
        self.is_train = is_train
        self.butd = butd
        self.boxes_file = boxes_file
        # Load LMDB File here
        if self.butd:
            self.env = lmdb.open(
                self.boxes_file,
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            with self.env.begin(write=False) as txn:
                self._image_ids = pickle.loads(txn.get("keys".encode()))
            self.butd_boxes = [None] * 1000000
            self.butd_object_ids = [None] * 1000000
            self.butd_scores = [None] * 1000000
            self.butd_masks = [None] * 1000000
        if image_set == "train100":
            self.ids = self.ids[:100]

    def __getitem__(self, idx):
        img, target = super(ModulatedCustomCOCODetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        coco_img = self.coco.loadImgs(image_id)[0]
        
        assert self.class_name_dict is not None
        class_name_list = list(self.class_name_dict.values())
        # anno = target

        # anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]
        
        # # sample class names from anno
        # anno_pos = [self.class_name_dict[str(obj["category_id"])] for obj in anno]
        # # take only 10 classes from gt
        # if len(anno_pos) > 7:
        #     anno_pos = random.sample(anno_pos, 7)
        
        # # change gt to have only the 10 sampled class ids
        # target = [obj for obj in anno if self.class_name_dict[str(obj["category_id"])] in anno_pos]

        # # sample negatives (might contain some pos, but that's ok)
        # anno_neg = random.sample(class_name_list, 3)
        
        # combine pos and neg, make caption
        # anno_all = list(set(anno_pos + anno_neg))
        # random.shuffle(anno_all)
        anno_all = class_name_list
        caption = ''.join([f"{c}. " for c in anno_all])
        dataset_name = coco_img["dataset_name"] if "dataset_name" in coco_img else None
        if self.butd:
            butd_img_id = str(coco_img["id"]).encode()

            if butd_img_id in self._image_ids:
                index = self._image_ids.index(butd_img_id)
                if self.butd_boxes[index] is not None:
                    butd_boxes = self.butd_boxes[index]
                    butd_object_ids = self.butd_object_ids[index]
                    butd_scores = self.butd_scores[index]
                    butd_masks = self.butd_masks[index]
                else:
                    with self.env.begin(write=False) as txn:
                        max_num_boxes = 99
                        conf_thresh = 0.5
                        item = pickle.loads(txn.get(butd_img_id))
                        butd_boxes = np.zeros((max_num_boxes, 4))
                        butd_object_ids = np.zeros((max_num_boxes))
                        
                        # threshold on score
                        butd_scores = item["scores"][1:][:max_num_boxes].max(1)
                        butd_masks = butd_scores > conf_thresh
                        butd_masks &= item['object_ids'][:max_num_boxes] != 0
                        if butd_masks.sum() == 0:
                            conf_thresh_ = 0.2
                            butd_masks = butd_scores > conf_thresh_
                            butd_masks &= item['object_ids'][:max_num_boxes] != 0
                            if butd_masks.sum() == 0:
                                conf_thresh_ = 0
                                butd_masks = butd_scores > conf_thresh_
                                butd_masks &= item['object_ids'][:max_num_boxes] != 0
                                if butd_masks.sum() == 0:
                                    print("Warning no bottom up boxes...")
                                    butd_masks[0] = 1
                        assert butd_masks.sum() != 0
                        butd_boxes[butd_masks] = item["boxes"].reshape(-1, 4)[1:][:max_num_boxes][butd_masks]
                        butd_object_ids[butd_masks] = item["object_ids"][:max_num_boxes][butd_masks]
                        
                        # store
                        self.butd_boxes[index] = butd_boxes
                        self.butd_object_ids[index] = butd_object_ids
                        self.butd_scores[index] = butd_scores
                        self.butd_masks[index] = butd_masks
                        
            else:
                assert False, butd_img_id
        else:
            butd_boxes = None
            butd_object_ids = None
            butd_scores = None
            butd_masks = None
        target = {
            "image_id": image_id,
            "annotations": target,
            "caption": caption,
            "butd_boxes": butd_boxes,
            "butd_object_ids": butd_object_ids,
            "butd_scores": butd_scores,
            "butd_masks": butd_masks
            }
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        target["dataset_name"] = dataset_name
        for extra_key in ["sentence_id", "original_img_id", "original_id", "task_id"]:
            if extra_key in coco_img:
                target[extra_key] = coco_img[extra_key]

        if "tokens_positive_eval" in coco_img and not self.is_train:
            tokenized = self.prepare.tokenizer(caption, return_tensors="pt")
            target["positive_map_eval"] = create_positive_map(tokenized, coco_img["tokens_positive_eval"])
            target["nb_eval"] = len(target["positive_map_eval"])

        return img, target

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

def create_positive_map(tokenized, tokens_positive):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
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
            positive_map[j, beg_pos : end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)

class ConvertCocoPolysToMask(object):
    def __init__(
        self,
        return_masks=False,
        return_tokens=False,
        tokenizer=None,
        class_name_dict=None
    ):
        self.return_masks = return_masks
        self.return_tokens = return_tokens
        self.tokenizer = tokenizer
        self.class_name_dict = class_name_dict

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        caption = target["caption"] if "caption" in target else None

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        if self.class_name_dict is not None:
            for i, obj in enumerate(anno):
                cat_name = self.class_name_dict[str(obj['category_id'])]
                start_span = caption.find(cat_name)
                end_span = start_span + len(cat_name)
                anno[i]["tokens_positive"] = [(start_span, end_span)]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        
        # load butd boxes
        if "butd_boxes" in target and target["butd_boxes"] is not None:
            butd_boxes = target["butd_boxes"]
            # guard against no butd_boxes via resizing
            butd_boxes = torch.as_tensor(butd_boxes, dtype=torch.float32).reshape(-1, 4)
            butd_boxes[:, 0::2].clamp_(min=0, max=w)
            butd_boxes[:, 1::2].clamp_(min=0, max=h)
            butd_classes = target["butd_object_ids"]
            butd_scores = target["butd_scores"]
            butd_masks = target["butd_masks"]
        else:
            butd_boxes = None

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        isfinal = None
        if anno and "isfinal" in anno[0]:
            isfinal = torch.as_tensor([obj["isfinal"] for obj in anno], dtype=torch.float)

        tokens_positive = [] if self.return_tokens else None
        if self.return_tokens and anno and "tokens" in anno[0]:
            tokens_positive = [obj["tokens"] for obj in anno]
        elif self.return_tokens and anno and "tokens_positive" in anno[0]:
            tokens_positive = [obj["tokens_positive"] for obj in anno]

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if butd_boxes is not None:
            target["butd_boxes"] = butd_boxes
            target["butd_classes"] = torch.from_numpy(butd_classes).to(dtype=torch.int64)
            target["butd_scores"] = torch.from_numpy(butd_scores)
            target["butd_masks"] = torch.from_numpy(butd_masks)
        if caption is not None:
            target["caption"] = caption
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        if tokens_positive is not None:
            target["tokens_positive"] = []

            for i, k in enumerate(keep):
                if k:
                    target["tokens_positive"].append(tokens_positive[i])

        if isfinal is not None:
            target["isfinal"] = isfinal

        # create posneg_map for **all** objects
        # mentioned in the sentence; and even for
        # no object class
        all_classes = caption.split('.')
        tokenized = self.tokenizer(caption, return_tensors="pt")
        class_name2id = {v:k for k, v in self.class_name_dict.items()}
        tokens_positive_list = []
        label_list = []
        for cls in all_classes:
            cls = cls.strip()
            if cls == '':
                continue
            start_span = caption.find(cls)
            end_span = start_span + len(cls)
            tokens_positive_ = [(start_span, end_span)]
            tokens_positive_list.append(tokens_positive_)
            label_list.append(int(class_name2id[cls]))
        posneg_map = create_positive_map(tokenized, tokens_positive_list)
        
        # positive map for no object class
        map = torch.zeros(1, posneg_map.size(-1)).to(posneg_map.device)
        map[:, -1] = 1.0
        posneg_map = torch.cat([posneg_map, map], axis=0)
        target['posneg_map'] = posneg_map
        target['label_list'] = torch.tensor(label_list)
        
        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if self.return_tokens and self.tokenizer is not None:
            assert len(target["boxes"]) == len(target["tokens_positive"])
            tokenized = self.tokenizer(caption, return_tensors="pt")
            target["positive_map"] = create_positive_map(tokenized, target["tokens_positive"])
        return image, target


def make_coco_transforms(image_set, cautious=False):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        horizontal = [] if cautious else [T.RandomHorizontalFlip()]
        return T.Compose(
            horizontal
            + [
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 1333, respect_boxes=cautious),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            # T.RandomResize([800], max_size=1333),
            normalize,
        ])
        
    if image_set == 'train100':
        return T.Compose([
            # T.RandomResize([800], max_size=1333),
            normalize,
        ])
    

    raise ValueError(f'unknown {image_set}')

def build(image_set, args):
    if image_set == 'val':
        type = "val"
        img_dir = Path(args.custom_coco_img_path_val)
    else:
        type = "train"
        img_dir = Path(args.custom_coco_img_path_train)

    ann_file = Path(args.custom_coco_ann_path) / f"instances_{type}2017.json"
    class_mapping_file = Path(args.custom_coco_id2name_path) 

    if args.butd:
        boxes_file = args.custom_coco_boxes_path
    else:
        boxes_file = None
    class_mapping_file = open(class_mapping_file, 'r')
    class_name_dict = json.load(class_mapping_file)

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)
    
    dataset = ModulatedCustomCOCODetection(
        img_dir,
        ann_file,
        transforms=make_coco_transforms(image_set, False),
        return_masks=False,
        return_tokens=True,
        image_set=image_set,
        tokenizer=tokenizer,
        class_name_dict=class_name_dict,
        boxes_file=boxes_file,
        butd=args.butd
    )
    return dataset