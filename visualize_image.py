# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
import ipdb
st = ipdb.set_trace

from skimage.measure import find_contours

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import torchvision.transforms as T
from collections import defaultdict

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.375, 0.66, 0.089], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

COLORS = [[0.375, 0.66, 0.089], [1, 0.38, 0]]
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def plot_results(pil_img, scores, boxes, labels, masks=None):
    plt.figure(figsize=(16,10))
    np_image = np.array(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if masks is None:
      masks = [None for _ in range(len(scores))]
    assert len(scores) == len(boxes) == len(labels) == len(masks)
    for s, (xmin, ymin, xmax, ymax), l, mask, c in zip(scores, boxes.tolist(), labels, masks, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=6))
        text = f'{l}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

        if mask is None:
          continue
        np_image = apply_mask(np_image, mask, c)

        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
          # Subtract the padding and flip (y, x) to (x, y)
          verts = np.fliplr(verts) - 1
          p = Polygon(verts, facecolor="none", edgecolor=c)
          ax.add_patch(p)

    plt.imshow(np_image)
    plt.axis('off')
    plt.savefig('box.png')


def add_res(results, ax, color='green'):
    #for tt in results.values():
    if True:
        bboxes = results['boxes']
        labels = results['labels']
        scores = results['scores']
        keep = scores >= 0.7
        bboxes = bboxes[keep].tolist()
        labels = labels[keep].tolist()
        scores = scores[keep].tolist()
    # print(torchvision.ops.box_iou(tt['boxes'].cpu().detach(), torch.as_tensor([[xmin, ymin, xmax, ymax]])))
    
    colors = ['purple', 'yellow', 'red', 'green', 'green', 'pink']
    
    for i, (b, ll, ss) in enumerate(zip(bboxes, labels, scores)):
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color=colors[i], linewidth=3))
        cls_name = ll if isinstance(ll,str) else CLASSES[ll]
        text = f'{cls_name}'
        print(text)
        ax.text(b[0], b[1], text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))
        
def plot_inference(im, caption, model):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0).cuda()

    # propagate through the model
    memory_cache = model(img, [caption], encode_and_save=True)
    outputs = model(img, [caption], encode_and_save=False, memory_cache=memory_cache)
    # keep only predictions with 0.7+ confidence
    probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
    keep = (probas > 0.7).cpu()

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], im.size)

    # Extract the text spans predicted by each box
    positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.1).nonzero().tolist()
    predicted_spans = defaultdict(str)
    for tok in positive_tokens:
        item, pos = tok
        if pos < 255:
            span = memory_cache["tokenized"].token_to_chars(0, pos)
            predicted_spans [item] += " " + caption[span.start:span.end]

    labels = [predicted_spans [k] for k in sorted(list(predicted_spans .keys()))]
    plot_results(im, probas[keep], bboxes_scaled, labels)


def visualize_results(models, img, caption):
    plot_inference(img, caption, models)