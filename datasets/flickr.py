# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MDETR 
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# ------------------------------------------------------------------------

"""
Data class for the Flickr30k entities dataset. The task considered is phrase grounding.
"""
from pathlib import Path

from transformers import RobertaTokenizerFast

from .coco_bdetr import ModulatedDetection, make_coco_transforms


class FlickrDetection(ModulatedDetection):
    pass


def build(image_set, args):

    img_dir = Path(args.flickr_img_path) #/ f"{image_set}"
    if args.butd:
        boxes_file = args.flickr_boxes_path
    else:
        boxes_file = None
    if args.GT_type == "merged":
        identifier = "mergedGT"
    elif args.GT_type == "separate":
        identifier = "separateGT"
    else:
        assert False, f"{args.GT_type} is not a valid type of annotation for flickr"

    if args.test:
        ann_file = Path(args.flickr_ann_path) / f"final_flickr_{identifier}_test.json"
    elif args.debug:
        ann_file = Path(args.flickr_ann_path) / f"final_flickr_{identifier}_train.json"
    else:
        ann_file = Path(args.flickr_ann_path) / f"final_flickr_{identifier}_{image_set}.json"

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)
    dataset = FlickrDetection(
        img_dir,
        ann_file,
        transforms=make_coco_transforms(image_set, cautious=True),
        return_masks=False,
        return_tokens=True,  # args.contrastive_align_loss,
        tokenizer=tokenizer,
        is_train=image_set=="train",
        image_set=image_set,
        boxes_file=boxes_file,
        butd=args.butd,
        new_contrastive=args.new_contrastive
    )
    return dataset
