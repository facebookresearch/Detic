# Copyright (c) Facebook, Inc. and its affiliates.
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.structures import Boxes, Instances

__all__ = ["fast_rcnn_inference"]


logger = logging.getLogger(__name__)


def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    **kw
):
    result_per_image = [
        fast_rcnn_inference_single_image(boxes_per_image, scores_per_image, image_shape, **kw)
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    topk_per_box: int,
    class_id_map: torch.Tensor=None,
    class_priority: torch.Tensor=None,
    asymmetric=False,
    filter_cls_token=False,
):
    filter_inds = torch.arange(len(boxes), device=boxes.device)

    # filter out nans
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        filter_inds = filter_inds[valid_mask]

    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor

    # filter scores below the "cls" token
    scores = scores[:, :-1]
    # if filter_cls_token:
    #     scores[scores <= scores[:, -1:]] = 0
    # get the max score
    topk_scores, topk_class_ids = torch.topk(scores, k=int(topk_per_box or 1))
    top_scores = topk_scores[:, 0]
    top_class_ids = topk_class_ids[:, 0]
    # filter by score
    filter_thresh = (top_scores > score_thresh) & (top_class_ids < scores.shape[1]-1)
    boxes = boxes[filter_thresh]
    top_scores = top_scores[filter_thresh]
    top_class_ids = top_class_ids[filter_thresh]
    topk_scores = topk_scores[filter_thresh]
    topk_class_ids = topk_class_ids[filter_thresh]
    filter_inds = filter_inds[filter_thresh]
    if class_id_map is not None:
        topk_class_ids = class_id_map[topk_class_ids]

    # from IPython import embed
    # if input(): embed()

    # 2. Apply NMS for each class independently.
    if asymmetric:
        keep, _ = asymmetric_nms(_nms_coord_trick(boxes, topk_class_ids), top_scores, class_priority, iou_threshold=nms_thresh)
    else:
        keep = batched_nms(boxes, top_scores, top_class_ids, nms_thresh)
    if topk_per_image > 0:
        keep = keep[:topk_per_image]

    # create instances
    result = Instances(
        image_shape,
        pred_boxes=Boxes(boxes[keep]),
        pred_classes=top_class_ids[keep],
        scores=top_scores[keep],
        topk_scores=topk_scores[keep],
        topk_classes=topk_class_ids[keep],
    )
    return result, filter_inds[keep]


def _nms_coord_trick(boxes, idxs):
    if boxes.numel() == 0:
        return torch.empty((0, 4), dtype=torch.int64, device=boxes.device)
    if idxs.ndim == 2:
        idxs = idxs[:, 0]
    offsets = idxs.to(boxes) * (boxes.max() + torch.tensor(1).to(boxes))
    return boxes + offsets[:, None]


def asymmetric_nms(boxes, scores, priority=None, iou_threshold=0.98):
    # Sort boxes by their confidence scores in descending order
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if priority is not None:
        indices = torch.as_tensor(np.lexsort((
            -area.cpu().numpy(),
            -priority.cpu().numpy(), 
        )), device=area.device)
    else:
        indices = torch.argsort(area, descending=True)
    boxes = boxes[indices]
    scores = scores[indices]

    selected_indices = []
    overlap_indices = []
    while len(boxes) > 0:
        # Pick the box with the highest confidence score
        b = boxes[0]
        selected_indices.append(indices[0])

        # Calculate IoU between the picked box and the remaining boxes
        zero = torch.tensor([0], device=boxes.device)
        intersection_area = (
            torch.maximum(zero, torch.minimum(b[2], boxes[1:, 2]) - torch.maximum(b[0], boxes[1:, 0])) * 
            torch.maximum(zero, torch.minimum(b[3], boxes[1:, 3]) - torch.maximum(b[1], boxes[1:, 1]))
        )
        smaller_box_area = torch.minimum(area[0], area[1:])
        # print(boxes.shape, area.shape, intersection_area.shape, smaller_box_area.shape)
        iou = intersection_area / (smaller_box_area + 1e-7)
        print(iou)

        # Filter out boxes with IoU above the threshold
        overlap_indices.append(indices[torch.where(iou > iou_threshold)[0] + 1])
        filtered_indices = torch.where(iou <= iou_threshold)[0]
        indices = indices[filtered_indices + 1]
        boxes = boxes[filtered_indices + 1]
        scores = scores[filtered_indices + 1]
        area = area[filtered_indices + 1]

    selected_indices = (
        torch.stack(selected_indices) if selected_indices else 
        torch.zeros([0], dtype=torch.int32, device=boxes.device))
    # print(nn, overlap_indices)
    # if nn>1 and input():embed()
    print(selected_indices.shape)
    input()
    return selected_indices, overlap_indices
