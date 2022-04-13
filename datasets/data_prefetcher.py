
# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
#------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
from typing import Any, Dict, List, Optional
import ipdb
st = ipdb.set_trace

def targets_to(targets: List[Dict[str, Any]], device):
    """Moves the target dicts to the given device."""
    excluded_keys = [
        "caption",
        "questionId",
        "tokens_positive",
        "tokens",
        "dataset_name",
        "sentence_id",
        "original_img_id",
        "nb_eval",
        "task_id",
        "original_id",
    ]
    return [{k: v.to(device, non_blocking=True)
             if k not in excluded_keys else v
             for k, v in t.items()} for t in targets]


def to_cuda(samples, targets, device):
    samples = samples.to(device, non_blocking=True)
    targets = targets_to(targets, device)
    return samples, targets


class data_prefetcher():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        self.excluded_keys = [
            "caption",
            "questionId",
            "tokens_positive",
            "tokens",
            "dataset_name",
            "sentence_id",
            "original_img_id",
            "nb_eval",
            "task_id",
            "original_id",
        ]
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_samples, self.next_targets = next(self.loader)
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            return
        
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)
            

    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            if samples is not None:
                samples.record_stream(torch.cuda.current_stream())
            if targets is not None:
                for t in targets:
                    for k, v in t.items():
                        if k not in self.excluded_keys:
                            v.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                samples, targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples = None
                targets = None
        return samples, targets
