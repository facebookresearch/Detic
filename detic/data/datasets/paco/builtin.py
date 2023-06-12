# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from .paco import get_instances_meta, register_instances

# ==== Root directories ====
_PACO_ANNOTATION_ROOT = os.environ.get(
    "PACO_ANNOTATION_ROOT", "datasets/paco/annotations"
)
_PACO_IMAGE_ROOT = os.environ.get("PACO_IMAGE_ROOT", "datasets/paco/images")
_COCO_IMAGE_ROOT = os.environ.get("COCO_IMAGE_ROOT", "datasets/coco")

# ==== Predefined datasets and splits for PACO ==========
_PREDEFINED_PACO = {
    "paco_lvis_v1_train": (
        os.path.join(_PACO_ANNOTATION_ROOT, "paco_lvis_v1_train.json"),
        _COCO_IMAGE_ROOT,
    ),
    "paco_lvis_v1_val": (
        os.path.join(_PACO_ANNOTATION_ROOT, "paco_lvis_v1_val.json"),
        _COCO_IMAGE_ROOT,
    ),
    "paco_lvis_v1_test": (
        os.path.join(_PACO_ANNOTATION_ROOT, "paco_lvis_v1_test.json"),
        _COCO_IMAGE_ROOT,
    ),
    "paco_joint_v1_train": (
        os.path.join(_PACO_ANNOTATION_ROOT, "paco_lvis_v1_train.json"),
        _COCO_IMAGE_ROOT,
    ),
    "paco_ego4d_v1_train": (
        os.path.join(_PACO_ANNOTATION_ROOT, "paco_ego4d_v1_train.json"),
        _PACO_IMAGE_ROOT,
    ),
    "paco_ego4d_v1_val": (
        os.path.join(_PACO_ANNOTATION_ROOT, "paco_ego4d_v1_val.json"),
        _PACO_IMAGE_ROOT,
    ),
    "paco_ego4d_v1_test": (
        os.path.join(_PACO_ANNOTATION_ROOT, "paco_ego4d_v1_test.json"),
        _PACO_IMAGE_ROOT,
    ),
}


def register_all_paco():
    for dataset_name, (annotation_path, image_root) in _PREDEFINED_PACO.items():
        register_instances(
            dataset_name,
            get_instances_meta(dataset_name),
            annotation_path,
            image_root,
        )


if __name__.endswith(".builtin"):
    register_all_paco()
