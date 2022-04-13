# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from https://github.com/aharley/neural_3d_mapping
# ------------------------------------------------------------------------

import torch
import numpy as np
import cv2

import ipdb
st = ipdb.set_trace

def summ_boxlist2ds(rgbs, boxlist_s, labels, frame_id=None):
    B, S, N, D = list(boxlist_s.shape)
    assert(D==4)
    boxlist_vis = []
    for s in range(S):
        boxlist_vis.append(draw_boxlist2d_on_image(
            rgbs[s],
            boxlist_s[:,s:s+1].squeeze(1),
            labels=labels[s],
            frame_id=frame_id[:, s:s+1].squeeze(1)
        ))
    return boxlist_vis

def summ_boxlist2d(rgb, boxlist, labels=None, color=None):
    B, C, H, W = list(rgb.shape)
    boxlist_vis = draw_boxlist2d_on_image(rgb, boxlist, labels=labels, color=color)
    return boxlist_vis

def draw_boxlist2d_on_image(rgb, boxlist, labels=None, frame_id=None, color=None):
        B, C, H, W = list(rgb.shape)
        assert(C==3)
        B2, N, D = list(boxlist.shape)
        assert(B2==B)
        assert(D==4) # ymin, xmin, ymax, xmax
        rgb = back2color(rgb)
        # if frame_id is None:
        #     frame_id = torch.zeros(B2, N).long()
        out = draw_boxlist2d_on_image_py(
            rgb[0].cpu().numpy(),
            boxlist[0].cpu().numpy(),
            labels,
            frame_id[0].cpu().numpy() if frame_id is not None else None,
            color=color)
        out = torch.from_numpy(out).type(torch.ByteTensor).permute(2, 0, 1)
        out = torch.unsqueeze(out, dim=0)
        out = color2orig(out)
        out = torch.reshape(out, [1, C, H, W])
        return out
    
def draw_boxlist2d_on_image_py(rgb, boxlist, label=None, frame_id=None, thickness=2, color=None):
        # all inputs are numpy tensors
        # rgb is H x W x 3
        # boxlist is N x 4
        # tids is N

        rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        H, W, C = rgb.shape
        assert(C==3)
        N, D = boxlist.shape
        assert(D==4)
        
        text = {'detections': 0, 'pred': 1}
        color_list = [np.array([0.0, 255.0, 0.0]), np.array([23.0, 169.0, 96.0])]
        if frame_id is not None:
            cv2.putText(
                    rgb,
                    f'{frame_id}',
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    np.array([0.0, 0.0, 255.0]),
                    3
            )
        
        # draw
        for ind, box in enumerate(boxlist):
            # box is 4
            ymin, xmin, ymax, xmax = box

            xmin = np.clip(int(xmin), 0,  W-1)
            xmax = np.clip(int(xmax), 0,  W-1)
            ymin = np.clip(int(ymin), 0,  H-1)
            ymax = np.clip(int(ymax), 0,  H-1)
            if color is not None:
                color = color
            elif label is not None:
                color = color_list[text[label[ind]]]
            else:
                color = color_list[0]
            cv2.line(rgb, (xmin, ymin), (xmin, ymax), color, thickness, cv2.LINE_AA)
            cv2.line(rgb, (xmin, ymin), (xmax, ymin), color, thickness, cv2.LINE_AA)
            cv2.line(rgb, (xmax, ymin), (xmax, ymax), color, thickness, cv2.LINE_AA)
            cv2.line(rgb, (xmax, ymax), (xmin, ymax), color, thickness, cv2.LINE_AA)
            
        rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return rgb

def back2color(i, blacken_zeros=False):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    i = (i * torch.tensor(std).reshape(1, 3, 1, 1)) + torch.tensor(mean).reshape(1, 3, 1, 1)

    return (i*255).type(torch.ByteTensor)

def color2orig(i, blacken_zeros=False):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    i = i.float() * 1./255
    i = (i - torch.tensor(mean).reshape(1, 3, 1, 1)) / torch.tensor(std).reshape(1, 3, 1, 1)
    return i