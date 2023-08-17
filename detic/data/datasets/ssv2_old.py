# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from detectron2.data.datasets.register_coco import register_coco_instances
from detic.modeling.text.text_encoder import build_text_encoder

from fvcore.common.timer import Timer

from .ssv2_categories import (
    SSV2_CATEGORIES,
    SSV2_CATEGORIES_COUNT,
)

logger = logging.getLogger('detectron2.vlpart.data.datasets.ssv2')

ATTR_TYPE_END_IDXS = [0, 30, 41, 55, 59]
ATTR_TYPE_BG_IDXS = [29, 38, 54, 58]



def load_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file in LVIS's annotation format.
    Args:
        Same as D2 LVIS dataset
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from lvis import LVIS

    json_file = PathManager.get_local_path(json_file)

    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    if dataset_name is not None:
        meta = get_instances_meta(dataset_name)
        MetadataCatalog.get(dataset_name).set(**meta)

    # sort indices for reproducible results
    img_ids = sorted(lvis_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = lvis_api.load_imgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]
    for iimg, anns_per_image in enumerate(anns):
        for iann, ann in enumerate(anns_per_image):
            anns[iimg][iann]['area'] = ann['bbox'][2] * ann['bbox'][3]

    # Sanity check that each annotation has a unique id
    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(
        ann_ids
    ), "Annotation ids in '{}' are not unique".format(json_file)

    imgs_anns = list(zip(imgs, anns))

    logger.info(
        "Loaded {} images in the LVIS format from {}".format(len(imgs_anns), json_file)
    )

    if extra_annotation_keys:
        logger.info(
            "The following extra annotation keys will be loaded: {} ".format(
                extra_annotation_keys
            )
        )
    else:
        extra_annotation_keys = []

    def get_file_name(img_root, img_dict):
        # Determine the path from the file_name field
        file_name = img_dict["file_name"]
        return os.path.join(img_root, file_name)

    dataset_dicts = []

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}

        record["file_name"] = get_file_name(image_root, img_dict)
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["not_exhaustive_category_ids"] = img_dict.get(
            "not_exhaustive_category_ids", []
        )
        record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation
            # file is buggy.
            assert anno["image_id"] == image_id
            obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}

            if dataset_name is not None and "thing_dataset_id_to_contiguous_id" in meta:
                obj["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                    anno["category_id"]
                ]
            else:
                obj["category_id"] = (
                    anno["category_id"] - 1
                )  # Convert 1-indexed to 0-indexed
            segm = anno["segmentation"]  # list[list[float]]

            #if len(segm) == 0:
            if len(segm):
                assert len(segm) > 0, segm
                obj["segmentation"] = segm
            for extra_ann_key in extra_annotation_keys:
                obj[extra_ann_key] = anno[extra_ann_key]

            if "attribute_ids" in anno:
                obj["attr_labels"] = anno["attribute_ids"]
                obj["attr_ignores"] = [
                    anno["unknown_color"],
                    anno["unknown_pattern_marking"],
                    anno["unknown_material"],
                    anno["unknown_transparency"],
                ]
            else:
                obj["attr_labels"] = ATTR_TYPE_BG_IDXS
                obj["attr_ignores"] = [1, 1, 1, 1]
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)


    zs_path = "datasets/metadata/ssv2_clip512.npy"
    if not os.path.isfile(zs_path):
        np.save(zs_path, get_zs_weight(get_labels()))
    return dataset_dicts

def get_labels():
    data = SSV2_CATEGORIES
    cat_names = [x['name'] for x in sorted(data, key=lambda x: x['id'])]
    sentences = []
    for s in cat_names:
        # find predicates
        if '(' in s and ')' in s:
            if 'is-' in s:
                sentences.append('this is '+s.replace('(is-',''))
            else:
                sentences.append('this is '+s)
            sentences[-1] = sentences[-1].replace('(','').replace(')','').replace('is fits','is').replace('-',' ').replace('this is',"it's")
        # else its a noun
        else:
            sentences.append('a '+s)

    for s in sentences:
        print(s)
    return sentences


def get_instances_meta(dataset_name):
    # we assume that ego4d is always trained in conjuction with lvis
    if "ego4d" in dataset_name or "joint" in dataset_name:
        return _get_lvis_ego4d_instances_meta()
    else:
        return _get_ssv2_instances_meta()


def _get_object_and_attribute_classes():
    assert len(SSV2_CATEGORIES) == SSV2_CATEGORIES_COUNT

    # Ensure that the category list is sorted by id
    categories = sorted(SSV2_CATEGORIES, key=lambda x: x["id"])
    #predicate_categories = sorted(SSV2_PREDICATES, key=lambda x: x["id"])

    thing_dataset_id_to_contiguous_id = {k["id"]: _i for _i, k in enumerate(categories)}
    thing_classes = [k["name"] for k in categories]
    #predicate_classes = [k["name"] for k in predicate_categories]

    return {
        "thing_classes": thing_classes,
        #"predicate_classes": predicate_classes,
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
    }


def _get_ssv2_instances_meta():
    # Use this when training on PACO-LVIS and PACO-EGO4D jointly
    # we do not currently support training only with ego4d data
    metadata = _get_object_and_attribute_classes()
    metadata["class_image_count"] = [{'id':cl['id'],'image_count':cl['instance_count']} for cl in SSV2_CATEGORIES]
    return metadata


#def _get_lvis_ego4d_instances_meta():
#    metadata = _get_object_and_attribute_classes()
#    metadata["class_image_count"] = PACO_LVIS_EGO4D_CATEGORY_IMAGE_COUNT
#    return metadata


_SSV2 = {
    "ssv2_train": ("Something_frames/images", "Something_frames/annotations/SS_TRAIN_ATTRIBUTES.json"),
    "ssv2_val": ("Something_frames/images", "Something_frames/annotations/SS_VAL_ATTRIBUTES.json"),
    "ssv2_test": ("Something_frames/images", "Something_frames/annotations/SS_TEST.json"),
    "ssv2_train_small": ("Something_frames/images", "Something_frames/annotations/SS_TRAIN_ATTRIBUTES_small.json"),
    "ssv2_val_small": ("Something_frames/images", "Something_frames/annotations/SS_VAL_ATTRIBUTES_small.json"),
    "ssv2_train_10k": ("Something_frames/images", "Something_frames/annotations/SS_TRAIN_10k.json"),
    "ssv2_val_10k": ("Something_frames/images", "Something_frames/annotations/SS_VAL_10k.json"),
    "ssv2_train_1k": ("Something_frames/images", "Something_frames/annotations/SS_TRAIN_1k.json"),
    "ssv2_val_1k": ("Something_frames/images", "Something_frames/annotations/SS_VAL_1k.json"),
    #"paco_lvis_v1_train": ("coco", "paco/annotations/paco_lvis_v1_train.json"),
    #"paco_lvis_v1_val": ("coco", "paco/annotations/paco_lvis_v1_val.json"),
    #"paco_lvis_v1_test": ("coco", "paco/annotations/paco_lvis_v1_test.json"),
    #"paco_ego4d_v1_train": ("paco/images", "paco/annotations/paco_ego4d_v1_train.json"),
    #"paco_ego4d_v1_val": ("paco/images", "paco/annotations/paco_ego4d_v1_val.json"),
    #"paco_ego4d_v1_test": ("paco/images", "paco/annotations/paco_ego4d_v1_test.json"),
    #"paco_lvis_v1_train_frequent": ("coco", "paco/annotations/paco_lvis_v1_train_frequent.json"),
    #"paco_lvis_v1_val_common_rare": ("coco", "paco/annotations/paco_lvis_v1_val_common_rare.json"),
}


def register_instances(name, metadata, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="lvis", **metadata
    )

def register_all_ssv2(root):
    for dataset_name, (image_root, json_file) in _SSV2.items():
        register_instances(
            dataset_name,
            get_instances_meta(dataset_name),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            #os.path.join(root, image_root),
            os.path.join('/vast/work/ptg', image_root),
        )

def get_zs_weight(classes):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    return text_encoder(classes).detach().cpu().numpy()

zs_path = "datasets/metadata/ssv2_clip512.npy"
if not os.path.isfile(zs_path):
    np.save(zs_path, get_zs_weight(get_labels()))


#_PACO_GOLDEN = {
#    "imagenet_golden_paco_parsed": ("imagenet/train", "imagenet/imagenet_golden_paco_parsed.json"),
#    "imagenet_golden_paco_parsed_swinbase": ("imagenet/train", "imagenet/imagenet_golden_paco_parsed_swinbase.json"),
#}
#
#def register_all_golden_paco(root):
#    for key, (image_root, json_file) in _PACO_GOLDEN.items():
#        register_coco_instances(
#            key,
#            get_instances_meta(key),
#            os.path.join(root, json_file) if "://" not in json_file else json_file,
#            os.path.join(root, image_root),
#        )




_root = os.getenv("DETECTRON2_DATASETS", "/scratch/work/ptg")
register_all_ssv2(_root)
#register_all_golden_paco(_root)
