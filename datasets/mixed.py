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
import os
import os.path
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
import lmdb
import pickle
import numpy as np

from PIL import Image
from torchvision.datasets.vision import VisionDataset
from transformers import RobertaTokenizerFast

from .coco_bdetr import ConvertCocoPolysToMask, make_coco_transforms


class CustomCocoDetection(VisionDataset):
    """Coco-style dataset imported from TorchVision.
    It is modified to handle several image sources


    Args:
        root_coco (string): Path to the coco images
        root_vg (string): Path to the vg images
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root_coco: str,
        root_vg: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super(CustomCocoDetection, self).__init__(root_coco, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.root_coco = root_coco
        self.root_vg = root_vg

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        img_info = coco.loadImgs(img_id)[0]
        path = img_info["file_name"]

        # change 2014 to 2017
        # path = path.replace("COCO_train2014_", "", 1)
        dataset = img_info["data_source"]

        cur_root = self.root_coco if dataset == "coco" else self.root_vg
        img = Image.open(os.path.join(cur_root, path)).convert("RGB")
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.ids)


class MixedDetection(CustomCocoDetection):
    """Same as the modulated detection dataset, except with multiple img sources"""

    def __init__(self, img_folder_coco, img_folder_vg, ann_file,
                transforms, return_masks, return_tokens, tokenizer, image_set='train',
                coco_boxes_file=None, vg_boxes_file=None, butd=None, new_contrastive=False):
        super(MixedDetection, self).__init__(img_folder_coco, img_folder_vg, ann_file)
        self.new_contrastive = new_contrastive
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, return_tokens, tokenizer=tokenizer)
        self.butd = butd
        self.coco_boxes_file = coco_boxes_file
        self.vg_boxes_file = vg_boxes_file
        # Load LMDB File here
        if self.butd:
            self.coco_env = lmdb.open(
                self.coco_boxes_file,
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self.vg_env = lmdb.open(
                self.vg_boxes_file,
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            with self.coco_env.begin(write=False) as txn:
                self._image_ids_coco = pickle.loads(txn.get("keys".encode()))
            with self.vg_env.begin(write=False) as txn:
                self._image_ids_vg = pickle.loads(txn.get("keys".encode()))
                
            self.butd_boxes_coco = [None] * 1000000
            self.butd_object_ids_coco = [None] * 1000000
            self.butd_scores_coco = [None] * 1000000
            self.butd_masks_coco = [None] * 1000000
            
            self.butd_boxes_vg = [None] * 1000000
            self.butd_object_ids_vg = [None] * 1000000
            self.butd_scores_vg = [None] * 1000000
            self.butd_masks_vg = [None] * 1000000
            
        if image_set == "train100":
            self.ids = self.ids[:100]
            
    def __getitem__(self, idx):
        img, target = super(MixedDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        coco_img = self.coco.loadImgs(image_id)[0]
        caption = coco_img["caption"]
        dataset = coco_img["data_source"]
        if self.new_contrastive:
            caption += ". not mentioned"
        if self.butd:
            butd_img_id = str(coco_img["original_id"]).encode()
            if dataset == "coco":
                index = self._image_ids_coco.index(butd_img_id)
                env = self.coco_env
                butd_boxes_dataset = self.butd_boxes_coco
                butd_object_ids_dataset = self.butd_object_ids_coco
                butd_scores_dataset = self.butd_scores_coco
                butd_masks_dataset = self.butd_masks_coco
                
            else:
                index = self._image_ids_vg.index(butd_img_id)
                env = self.vg_env
                butd_boxes_dataset = self.butd_boxes_vg
                butd_object_ids_dataset = self.butd_object_ids_vg
                butd_scores_dataset = self.butd_scores_vg
                butd_masks_dataset = self.butd_masks_vg
            
            if butd_boxes_dataset[index] is not None:
                butd_boxes = butd_boxes_dataset[index]
                butd_object_ids = butd_object_ids_dataset[index]
                butd_scores = butd_scores_dataset[index]
                butd_masks = butd_masks_dataset[index]
            else:
                with env.begin(write=False) as txn:
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
                    butd_boxes_dataset[index] = butd_boxes
                    butd_object_ids_dataset[index] = butd_object_ids
                    butd_scores_dataset[index] = butd_scores
                    butd_masks_dataset[index] = butd_masks
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
        return img, target


def build(image_set, args):
    vg_img_dir = Path(args.vg_img_path)
    if args.butd:
        vg_boxes_file = args.vg_boxes_path
        coco_boxes_file = args.coco_boxes_path
    else:
        vg_boxes_file = None
        coco_boxes_file = None
    image_set_act = image_set
    if image_set == "train100" or image_set == "val":
        image_set_act = "train"
    coco_img_dir = Path(args.coco_path_refcoco) / "train2014"
    assert vg_img_dir.exists(), f"provided VG img path {vg_img_dir} does not exist"
    assert coco_img_dir.exists(), f"provided coco img path {coco_img_dir} does not exist"

    ann_file = Path(args.gqa_ann_path) / f"final_mixed_{image_set_act}.json"

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)
    dataset = MixedDetection(
        coco_img_dir,
        vg_img_dir,
        ann_file,
        transforms=make_coco_transforms(image_set, cautious=True),
        return_masks=False,
        return_tokens=True,
        tokenizer=tokenizer,
        coco_boxes_file=coco_boxes_file,
        vg_boxes_file=vg_boxes_file,
        butd=args.butd,
        image_set=image_set,
        new_contrastive=args.new_contrastive
    )

    return dataset
