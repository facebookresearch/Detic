from typing import Dict, List, Optional

import numpy as np
import torch
from detectron2.utils.logger import setup_logger

setup_logger()


import sys

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import boxes

# Add Detic and CenterNet2 paths to python path
sys.path.insert(0, "Detic/")
sys.path.insert(0, "Detic/third_party/CenterNet2/")
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.text.text_encoder import CLIPTEXT
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder

import warnings
warnings.filterwarnings("ignore")


def setup_detectron_predictor(
    config_file_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    threshold: float = 0.5,
    device: str = "cpu",
) -> DefaultPredictor:
    """Setup detectron predictor.

    Args:
        config_file: Path of the config file to be merged with the default configs.
        weights_path: Path to load model weights from. Defaults to None.
        threshold: Threshold value for predictions. Defaults to 0.5
        device: Device to be used. Defaults to `cpu`

    Returns:
        Predictor
    """
    cfg = get_cfg()
    if config_file_path is not None:
        cfg.merge_from_file(config_file_path)
        cfg.MODEL.WEIGHTS = weights_path

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DEVICE = device
    predictor = DefaultPredictor(cfg)
    return predictor


def setup_detic_predictor(
    config_file_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    threshold: float = 0.5,
    device: str = "cpu",
) -> DefaultPredictor:
    """Setup detectron predictor.

    Args:
        config_file: Path of the config file to be merged with the default configs. Defaults to None.
        weights_path: Path to load model weights from. Defaults to None.
        threshold: Threshold value for predictions. Defaults to 0.5
        device: Device to be used. Defaults to `cpu`

    Returns:
        Predictor
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    if config_file_path is not None:
        cfg.merge_from_file(config_file_path)
        cfg.MODEL.WEIGHTS = weights_path

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
    cfg.MODEL.DEVICE = device
    predictor = DefaultPredictor(cfg)
    return predictor


@torch.no_grad()
def _set_detic_vocabulary(
    predictor: DefaultPredictor, text_encoder: CLIPTEXT, vocabulary: List[str]
):
    """Set detic predictor vocabulary.

    Args:
        predictor: Detic predictor
        text_encoder: Clip text encoder
        vocabulary: Class names to detect
    """
    num_classes = len(vocabulary)
    vocabulary = [f"a {word}" for word in vocabulary]
    text_encoder.eval()
    embeddings = text_encoder(vocabulary).detach().permute(1, 0).contiguous().cpu()
    reset_cls_test(predictor.model, embeddings, num_classes)


@torch.no_grad()
def get_detic_predictions(
    image: np.ndarray,
    predictor: DefaultPredictor,
    text_encoder: Optional[CLIPTEXT] = None,
    vocabulary: Optional[List[str]] = None,
):
    """Return detic prediction bounding boxes and instance segmentation masks.

    Args:
        image: Image.
        predictor: Detic predictor.
        text_encoder: Model for obtaining word embeddings. Defaults to None.
        vocabulary: List containing words. These words will be classes to detect against in predictions. Defaults to None.

    Returns:
        Bounding boxes and instance segmentation masks
    """
    if vocabulary is not None:
        _set_detic_vocabulary(predictor, text_encoder, vocabulary)
    outputs = predictor(image)["instances"]
    return outputs.pred_boxes, outputs.pred_masks


@torch.no_grad()
def get_detectron2_predictions(image, predictor):
    """Return detectron2 prediction bounding boxes and instance segmentation masks.

    Args:
        image: Image.
        predictor: Detectron2 predictor.

    Returns:
        Bounding boxes and instance segmentation masks, respectively.
    """
    outputs = predictor(image)["instances"]
    return outputs.pred_boxes, outputs.pred_masks


def find_best_bbox(gt_box: List[int], pred_boxes: Dict[str, List[torch.Tensor]]):
    """Given a list of the bounding boxes, find the bounding boxe with the highest
    intersection over union score with respect to the given ground truth bounding box

    Args:
        gt_box: _description_
        pred_boxes: _description_

    Returns:
        Bounding boxes and instance segmentation masks, respectively.
    """
    best_iou = 0.0
    best_idx = 0
    gt_Box = boxes.Boxes([gt_box]).to(pred_boxes[0].device)
    for idx in range(len(pred_boxes)):
        pred_Box = pred_boxes[idx]
        iou = boxes.pairwise_iou(gt_Box, pred_Box).item()
        if iou > best_iou:
            best_iou = iou
            best_idx = idx
    return best_iou, best_idx


def build_vocabulary(image_id, graph, include_attributes=False):
    """Build scene vocabulary including bounding boxes and id's
    of the objects.

    Args:
        image_id: Id of the image in the scene graph
        graph: Scene graph
        include_attributes: Whether to combine object name with it's attributes
        given in the scene graph when adding vocabulary. Defaults to False.

    Returns:
        _description_
    """
    objects_data = []
    for object_id in graph[image_id]["objects"]:
        object_data = graph[image_id]["objects"][object_id]

        object_name = object_data["name"]
        if include_attributes and object_data["attributes"]:
            object_attributes = " and ".join(object_data["attributes"])
            object_name = f"{object_attributes} {object_data['name']}"

        object_x = object_data["x"]
        object_y = object_data["y"]
        object_w = object_data["w"]
        object_h = object_data["h"]
        object_bbox = [object_x, object_y, object_x + object_w, object_y + object_h]

        objects_data.append(
            {
                "id": object_id,
                "name": object_name,
                "bbox": object_bbox,
            }
        )

    return objects_data
