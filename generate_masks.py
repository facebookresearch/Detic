import argparse
import json
import os
from datetime import datetime

import cv2
import h5py
import numpy as np
from mask_utils import (
    build_text_encoder,
    build_vocabulary,
    find_best_bbox,
    get_detectron2_predictions,
    get_detic_predictions,
    setup_detectron_predictor,
    setup_detic_predictor,
)
from tqdm.auto import tqdm


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "detector", choices=["detic", "detectron2"], help="Detector for mask prediction"
    )
    parser.add_argument(
        "--img-dir",
        required=True,
        help="Directory containing the images",
    )
    parser.add_argument(
        "--scene-graph",
        required=True,
        help="Scene graph of the dataset",
    )
    parser.add_argument(
        "--config",
        help="Name of the config file of model. Check Model Zoo for available configs."
        "Defaults to Base-C2_L_R5021k_640b64_4x if detector is detic and"
        " new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ if detector is detectron2.",
    )
    parser.add_argument(
        "--weights",
        help="Pretrained model weights path."
        "detic: https://github.com/facebookresearch/Detic/blob/main/docs/MODEL_ZOO.md"
        "detectron2: https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the dataset",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold value for filterin predictions."
        "Any prediction with score below the threshold value will be discarded.",
    )
    parser.add_argument(
        "--custom-vocabulary",
        action="store_true",
        help="Whether to use custom vocabulary for detic predictions."
        "Custom vocabulary is name of the objects existing in corresponding image scene graph."
        "Won't effect the predictions if the detector is not detic.",
    )
    parser.add_argument(
        "--include-attributes",
        action="store_true",
        help="Whether to include attributes to vocabulary (if defined in the scene graph)"
        "to detic predictions. Won't effect the predictions if the detector"
        "is not detic or vocabulary is not custom.",
    )
    parser.add_argument("--use-gpu", action="store_true", help="Whether to use gpu.")
    args = parser.parse_args()
    return args


def load_scene_graph(scene_graph_path):
    with open(scene_graph_path) as f:
        return json.load(f)


def insert_to_hdf5(
    hdf5_file: h5py.File,
    image_id: str,
    image: np.ndarray,
    object_id: str,
    iou_score: float,
    **insert_kwargs,
):
    if image_id not in hdf5_file:
        hdf5_file.create_group(image_id)
        # Store image in RGB format
        hdf5_file[image_id].create_dataset(
            "image", data=image[..., ::-1], dtype=np.uint8
        )
    if "objects" not in hdf5_file[image_id]:
        hdf5_file[image_id].create_group("objects")
    group = hdf5_file[image_id]["objects"]
    if object_id not in group:
        subgroup = group.create_group(object_id)
        subgroup.create_dataset("iou", data=iou_score)
        for (name, data) in insert_kwargs.items():
            subgroup.create_dataset(name=name, data=data)
    subgroup = group[object_id]

    # If current iou score is greater than previous one, change data with the new predictions.
    if iou_score > subgroup["iou"][...]:
        subgroup["iou"][...] = iou_score
        for (name, data) in insert_kwargs.items():
            subgroup[name][...] = data


def main():
    args = setup_args()
    detector = args.detector
    images_dir = args.img_dir
    config = args.config
    weights_path = args.config
    scene_graph_path = args.scene_graph
    output_path = args.output
    threshold = args.threshold
    custom_vocabulary = args.custom_vocabulary
    include_attributes = args.include_attributes

    start_date = str(datetime.now())

    device = "cuda" if args.use_gpu else "cpu"
    if config is None:
        if detector == "detic":
            config = (
                "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
            )
            weights_path = "https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
        elif detector == "detectron2":
            config = "../configs/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
            weights_path = "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"
    scene_graph = load_scene_graph(scene_graph_path)
    output_file = h5py.File(output_path, "a")

    if detector == "detic":
        predictor = setup_detic_predictor(config, weights_path, threshold, device)
        text_encoder = None
        if custom_vocabulary:
            text_encoder = build_text_encoder()
    elif detector == "detectron2":
        predictor = setup_detectron_predictor(config, weights_path, threshold, device)

    for image_id in tqdm(scene_graph):
        image_path = os.path.join(images_dir, f"{image_id}.jpg")
        image = cv2.imread(image_path)

        objects_data = build_vocabulary(image_id, scene_graph, include_attributes)

        if detector == "detic":
            if custom_vocabulary:
                vocabulary = [object_data["name"] for object_data in objects_data]
                pred_boxes, pred_masks = get_detic_predictions(
                    image, predictor, text_encoder, vocabulary
                )
            else:
                pred_boxes, pred_masks = get_detic_predictions(image, predictor)
        elif detector == "detectron2":
            pred_boxes, pred_masks = get_detectron2_predictions(image, predictor)

        if len(pred_boxes):
            for object_data in objects_data:
                object_id = object_data["id"]
                object_name = object_data["name"]
                object_bbox = object_data["bbox"]

                best_iou, best_idx = find_best_bbox(object_bbox, pred_boxes)
                best_mask = pred_masks[best_idx]
                if best_iou >= threshold:
                    insert_to_hdf5(
                        output_file,
                        image_id,
                        image,
                        object_id,
                        best_iou,
                        mask=best_mask.cpu(),
                        name=object_name,
                    )

    end_date = str(datetime.now())

    metadata = {
        "start_date": start_date,
        "end_date": end_date,
        "params": vars(args),
    }

    if not "metadata" in output_file:
        output_file.create_dataset("metadata", data="[]")

    prev_metadata = json.loads(output_file["metadata"][...].tolist())
    output_file["metadata"][...] = json.dumps([metadata] + prev_metadata)

    output_file.close()


if __name__ == "__main__":
    main()
