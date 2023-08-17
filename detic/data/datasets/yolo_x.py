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


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ["custom_load_json", "custom_register_instances"]



def custom_register_instances(name, split, metadata, image_root):
    """
    """
    meta = create_metadata(image_root, f'{name}_{split}', metadata)
    # if not hasattr(meta, 'class_image_count'):
    #     logger.info("loading {name} {split} json on import to generate class_image_count.")
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
        **metadata
    )
    # print(meta)
    # input()
    return meta


# M5_X-Stat/YoloModel/LabeledObjects/train/M5-47_495_left_05_20_2022.jpg
# M5_X-Stat/YoloModel/LabeledObjects/train/M5-47_495_left_05_20_2022.txt


def create_dataset_dict(image_root, split, class_offset=0):
    # get file list
    img_fs = sorted(glob.glob(os.path.join(image_root, 'YoloModel/LabeledObjects', split, '*.jpg')))
    names = [os.path.splitext(os.path.basename(f))[0] for f in img_fs]
    txt_fs = [os.path.join(image_root, 'YoloModel/LabeledObjects', split, f'{n}.txt') for n in names]

    # load annotations for each file
    dataset_dicts = []
    for i, (fname_im, fname_txt) in tqdm.tqdm(
        enumerate(zip(img_fs, txt_fs)), 
        desc=f'loading {image_root}...', total=len(img_fs)
    ):
        # if not os.path.isfile(fname_txt):
        #     tqdm.tqdm.write(f'missing file: {fname_txt}')
        #     continue
        im = cv2.imread(fname_im)
        H, W, _ = im.shape
        try:
            df = pd.read_csv(fname_txt, sep=' ', header=None)
        except (pd.errors.EmptyDataError, OSError):
            tqdm.tqdm.write(f'empty file: {fname_txt} exists={os.path.isfile(fname_txt)}')
            df = pd.DataFrame()
        dataset_dicts.append({
            "file_name": fname_im,
            "height": im.shape[0],
            "width": im.shape[1],
            "image_id": i,
            # "pos_category_ids": [],
            # "neg_category_ids": [],
            # "not_exhaustive_category_ids": [],
            # "captions": [],
            # "caption_features": [],
            "annotations": [
                get_box_annotation(d, W, H, class_offset=class_offset)
                for _, d in df.iterrows()
            ],
        })
    return dataset_dicts


def get_box_annotation(d, W, H, class_offset=0, **meta):
    c, x, y, w, h = d
    return {
        "category_id": c + class_offset,
        "bbox": [x*W, y*H, w*W, h*H], 
        "bbox_mode": BoxMode.XYWH_ABS,
        "area": w*W * h*H,
        **meta
    }


def get_zs_weight(classes):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    return text_encoder(classes).detach().cpu().numpy()


def load_thing_classes(image_root):
    return [
        c.strip().replace('_', ' ')
        for c in open(os.path.join(image_root, 'YoloModel/object_names.txt')).read().splitlines()
    ]


def custom_load_json(image_roots, meta, name, split, class_offset=0):
    '''
    Modifications:
      use `file_name`
      convert neg_category_ids
      add pos_category_ids
    '''
    dataset_name = f'{name}_{split}'

    # create dataset (or )
    cache_fname = os.path.join('/scratch/work/ptg/.cache', f'{name}.pkl')
    if not os.path.isfile(cache_fname):
        dataset_dicts = []
        for image_root, offset in zip(image_roots, meta.class_offsets):
            assert os.path.exists(image_root), f"non-existant {dataset_name} data root {image_root}"
            logger.info('Creating dataset dicts for {image_root} {split}')
            dataset_dicts += create_dataset_dict(image_root, split, class_offset=offset)
        with open(cache_fname, 'wb') as f:
            pickle.dump({'dataset': dataset_dicts}, f)
    else:
        logger.info(f"Using cached {dataset_name} dataset {cache_fname}")
        with open(cache_fname, 'rb') as f:
            d = pickle.load(f)
            dataset_dicts = d['dataset']

    # get class counts
    counts = {'category_id': {i: 0 for i in range(len(meta.thing_classes))}}
    for dd in dataset_dicts:
        for d in dd['annotations']:
            for c in counts:
                xs = d.get(c, -1)
                for x in (xs if isinstance(xs, (list, tuple)) else [xs]):
                    if x != -1:
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

DATA_ROOT = "/scratch/work/ptg/BBN/skills"
SKILLS = os.listdir(DATA_ROOT)

# _CUSTOM_SPLITS = {
#     k: v
#     for d in SKILLS
#     for k, v in ({
#         (d, "train"): d,
#         (d, "val"): d,
#     }).items()
# }

metadata = {

}

# for (key, split), dir_name in _CUSTOM_SPLITS.items():
#     custom_register_instances(
#         key, split, metadata,
#         os.path.join(DATA_ROOT, dir_name),
#     )


_CUSTOM_SPLITS = {
    ('bbn', 'train'): SKILLS,
    ('bbn', 'test'): SKILLS,
}

for (key, split), skills in _CUSTOM_SPLITS.items():
    skill_roots = [os.path.join(DATA_ROOT, f) for f in skills]
    skill_classes = [load_thing_classes(f) for f in skill_roots]
    thing_classes = [c for cs in skill_classes for c in cs]
    metadata['thing_classes'] = thing_classes
    metadata['class_offsets'] = np.cumsum([len(cs) for cs in skill_classes]).tolist()
    # print(len(thing_classes))
    # print(thing_classes)
    # input()
    custom_register_instances(
        key, split, metadata,
        skill_roots,
    )