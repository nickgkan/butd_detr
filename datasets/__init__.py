# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
import torchvision
from .torchvision_datasets import CocoDetection
from .flickr import build as build_flickr
from .coco_bdetr import build as build_coco
from .refexp import build as build_refexp
from .mixed import build as build_mixed
from .vg import build as build_vg
from .coco_custom import build as build_custom_coco

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco
    if isinstance(dataset, (torchvision.datasets.CocoDetection)):
        return dataset.coco


def build_dataset(dataset_file, image_set, args):
    if dataset_file == "refexp":
        return build_refexp(image_set, args)
    if dataset_file == 'coco':
        return build_coco(image_set, args)
    if dataset_file == "flickr":
        return build_flickr(image_set, args)
    if dataset_file == "mixed":
        return build_mixed(image_set, args)
    if dataset_file == "vg":
        return build_vg(image_set, args)
    if dataset_file == "custom_coco":
        return build_custom_coco(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
