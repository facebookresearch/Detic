# Copyright (c) Facebook, Inc. and its affiliates.
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
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
    scores[scores <= scores[:, -1:]] = 0
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

    # from IPython import embed
    # if input(): embed()

    # 2. Apply NMS for each class independently.
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



# def fast_rcnn_inference_single_image(
#     boxes,
#     scores,
#     image_shape: Tuple[int, int],
#     score_thresh: float,
#     nms_thresh: float,
#     topk_per_image: int,
# ):
#     valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
#     if not valid_mask.all():
#         boxes = boxes[valid_mask]
#         scores = scores[valid_mask]
#     print(scores.shape, boxes.shape, valid_mask.shape)

#     scores = scores[:, :-1] # R x (C+1) => R x C
#     # Convert to Boxes to use the `clip` function ...
#     num_bbox_reg_classes = boxes.shape[1] // 4
#     boxes = Boxes(boxes.reshape(-1, 4))
#     boxes.clip(image_shape)
#     boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

#     # 1. Filter results based on detection scores. It can make NMS more efficient
#     #    by filtering out low-confidence detections.
#     filter_mask = scores > score_thresh  # R x K
#     # R' x 2. First column contains indices of the R predictions;
#     # Second column contains indices of classes.
#     filter_inds = filter_mask.nonzero()
#     print(filter_mask.shape, filter_inds.shape)
#     if num_bbox_reg_classes == 1:
#         boxes = boxes[filter_inds[:, 0], 0]
#     else:
#         boxes = boxes[filter_mask]
#     scores = scores[filter_mask]
#     print(scores.shape, boxes.shape)
#     from IPython import embed
#     if input(): embed()

#     # 2. Apply NMS for each class independently.
#     keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
#     if topk_per_image >= 0:
#         keep = keep[:topk_per_image]
#     boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

#     # TODO: group boxes back, then create scores + topk classes matrix
#     result = Instances(image_shape)
#     result.pred_boxes = Boxes(boxes)
#     result.scores = scores
#     result.pred_classes = filter_inds[:, 1]
#     # result.pred_scores = ...
#     # result.topk_pred_classes = filter_inds[:, 1]
#     return result, filter_inds[:, 0]
