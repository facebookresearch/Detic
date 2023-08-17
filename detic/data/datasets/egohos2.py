# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import glob
import json
import pickle
import numpy as np
import pandas as pd
import cv2
import tqdm

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.lvis import get_lvis_instances_meta
from detectron2.utils.visualizer import Visualizer

from detic.modeling.text.text_encoder import build_text_encoder

from shapely.geometry import Polygon, MultiPolygon


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ["custom_load_json", "custom_register_instances"]


# vocab

object_classes = [
    # 'background',
    'left hand',
    'right hand',
    'object held in left hand',
    'object held in right hand',
    'object held in both hands',
    'object acted upon by left hand',
    'object acted upon by right hand',
    'object acted upon by both hands',
]

# property_classes = [
#     'interacting',
# ]

# relation_classes = [
#     'interacting',
# ]
# ALL_CLASSES = object_classes + relation_classes
L, R, O1L, O1R, O1B, O2L, O2R, O2B = range(len(object_classes))



def custom_register_instances(name, split, metadata, image_root):
    """
    """
    meta = create_metadata(image_root, f'{name}_{split}', metadata)
    if not hasattr(meta, 'class_image_count'):
        logger.info("loading {name} {split} json on import to generate class_image_count.")
        custom_load_json(image_root, meta, name, split)
    DatasetCatalog.register(f'{name}_{split}', lambda: custom_load_json(image_root, meta, name, split))


def create_metadata(image_root, name, metadata):
    logger.info(f'creating metadata for {name}')
    meta = MetadataCatalog.get(name)
    cat_path = f"datasets/metadata/{name}_cat_count.json"
    if os.path.isfile(cat_path):
        metadata['class_image_count'] = json.load(open(cat_path))

    meta.set(
        image_root=image_root, 
        evaluator_type="lvis", 
        thing_classes=object_classes,
        **metadata
    )
    # print(meta)
    # input()
    return meta


# M5_X-Stat/YoloModel/LabeledObjects/train/M5-47_495_left_05_20_2022.jpg
# M5_X-Stat/YoloModel/LabeledObjects/train/M5-47_495_left_05_20_2022.txt


def create_dataset_dict(image_root, split, class_offset=0):
    # get file list
    img_fs = sorted(glob.glob(os.path.join(image_root, 'image/*.jpg')))
    names = [os.path.splitext(os.path.basename(f))[0] for f in img_fs]
    mask_fs = [
        os.path.join(image_root, 'label', f'{n}.png') 
        for n in names]
    
    # load annotations for each file
    dataset_dicts = []
    for i, (fname_im, fname_mask) in tqdm.tqdm(enumerate(zip(img_fs, mask_fs)), desc='loading egohos...', total=len(img_fs)):
        mask = cv2.imread(fname_mask, cv2.IMREAD_GRAYSCALE)
        dataset_dicts.append({
            "file_name": fname_im,
            "height": mask.shape[0],
            "width": mask.shape[1],
            "image_id": i,
            # "pos_category_ids": [],
            # "neg_category_ids": [],
            # "not_exhaustive_category_ids": [],
            # "captions": [],
            # "caption_features": [],
            "annotations": get_annotations(mask),
        })
    return dataset_dicts



def get_annotations(mask):
    # convert masks to contours (index 0 is background)
    masks = [mask == i+1 for i in range(len(object_classes))]
    contours = [_bool_mask_to_contour(m) for m in masks]
    (
        left, right, 
        obj1_left, obj1_right, obj1_both,
        obj2_left, obj2_right, obj2_both,
    ) = contours
    # obj1 = _bool_mask_to_contour(masks[2]|masks[3]|masks[4])
    # obj2 = _bool_mask_to_contour(masks[5]|masks[6]|masks[7])
    
    # convert to hand, object, and relation annotations
    # if the contour is empty, it will return an empty dict
    annotations = [
        _obj_ann(left, L),
        _obj_ann(right, R),
        _obj_ann(obj1_left, O1L),
        _obj_ann(obj2_left, O2L),
        _obj_ann(obj1_right, O1R),
        _obj_ann(obj2_right, O2R),
        _obj_ann(obj1_both, O1B),
        _obj_ann(obj2_both, O2B),
        # _rel_ann(left, obj1_left),
        # _rel_ann(left, obj2_left),
        # _rel_ann(right, obj1_right),
        # _rel_ann(right, obj2_right),
        # _rel_ann(left, obj1_both),
        # _rel_ann(left, obj2_both),
        # _rel_ann(right, obj1_both),
        # _rel_ann(right, obj2_both),
        # _rel_ann(obj1, obj2),
    ]

    # filter empty annotations
    return [d for d in annotations if d]


def _obj_ann(obj, i):
    return _contour_to_ann(obj, category_id=i)
    # return _contour_to_ann(obj, category_id=i, property_id=[0])

def _rel_ann(obj1, obj2, category_id):
    return _contour_to_ann(obj1 + obj2, category_id=category_id) if obj1 and obj2 else None

# def nhot(n, *idxs):
#     y = np.zeros(n, dtype=int)
#     y[tuple(i for i in idxs if i is not None)] = 1
#     return y

def _bool_mask_to_contour(mask):
    # given a boolean mask, get the contours
    if not np.any(mask): 
        return []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return [c for c in contours if cv2.contourArea(c) >= 2]

def _contour_to_ann(contours, category_id=-1, **meta):
    # given contours get the proper annotation structure
    if not contours:
        return {}
    polys = [
        Polygon(c[:, 0]).simplify(1.0, preserve_topology=False)
        for c in contours
    ]
    multi_poly = MultiPolygon([
        p for px in polys 
        for p in (px.geoms if isinstance(px, MultiPolygon) else [px])])
    return {
        "category_id": category_id,
        "segmentation": [
            np.array(poly.exterior.coords).ravel().tolist()
            for poly in multi_poly.geoms
        ],
        "bbox": multi_poly.bounds, 
        "bbox_mode": BoxMode.XYXY_ABS,
        "area": multi_poly.area,
        **meta
    }



def get_zs_weight(classes):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    return text_encoder(classes).detach().cpu().numpy()




def custom_load_json(image_root, meta, name, split, class_offset=0):
    '''
    Modifications:
      use `file_name`
      convert neg_category_ids
      add pos_category_ids
    '''
    dataset_name = f'{name}_{split}'
    assert os.path.exists(image_root), f"non-existant {dataset_name} data root {image_root}"

    # create dataset (or )
    cache_fname = os.path.join(image_root, 'detectron2_dataset.pkl')
    if not os.path.isfile(cache_fname):
        logger.info('Creating dataset dicts for {image_root} {split}')
        dataset_dicts = create_dataset_dict(image_root, split, class_offset=class_offset)
        with open(cache_fname, 'wb') as f:
            pickle.dump({'dataset': dataset_dicts}, f)
    else:
        logger.info(f"Using cached {dataset_name} dataset", cache_fname)
        with open(cache_fname, 'rb') as f:
            d = pickle.load(f)
            dataset_dicts = d['dataset']

    # get class counts
    counts = {'category_id': {}}
    for dd in dataset_dicts:
        for d in dd['annotations']:
            for c in counts:
                xs = d.get(c, -1)
                for x in (xs if isinstance(xs, (list, tuple)) else [xs]):
                    if x != -1:
                        if x >= len(object_classes):
                            print(x)
                        counts[c][x] = counts[c].get(x, 0) + 1
    class_image_count = [{'id': k, 'image_count': c} for k, c in counts['category_id'].items()]
    # if not hasattr(meta, 'class_image_count'):
    meta.class_image_count = class_image_count

    # create text-embedding weights file
    zs_path = f"datasets/metadata/{name}.npy"
    # if not os.path.isfile(zs_path):
    logger.info(f'creating weights {zs_path}')
    np.save(zs_path, get_zs_weight(meta.thing_classes))

    # create category file
    cat_path = f"datasets/metadata/{dataset_name}_cat_count.json"
    logger.info(f'creating categories {cat_path}')
    with open(cat_path, 'w') as f:
        json.dump(class_image_count, f)
    logger.info(f'{dataset_name}: {len(dataset_dicts)} dicts')
    # input()
    return dataset_dicts



# Register different splits

DATA_ROOT = "/scratch/work/ptg/EGOHOS"

_CUSTOM_SPLITS = {
    ("egohos", "train"): "train",
    ("egohos", "val"): "val",
    ("egohos", "test_indomain"): "test_indomain",
    ("egohos", "test_outdomain"): "test_outdomain",
}

metadata = {
    # "thing_classes": ALL_CLASSES,
    # "property_classes": property_classes,
    # "relation_classes": relation_classes,
}

for (key, split), dir_name in _CUSTOM_SPLITS.items():
    custom_register_instances(
        key, split, metadata,
        os.path.join(DATA_ROOT, dir_name),
    )
