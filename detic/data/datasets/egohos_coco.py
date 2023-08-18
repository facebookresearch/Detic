# Copyright (c) Facebook, Inc. and its affiliates.
import os
from detectron2.data.datasets import register_coco_instances

NAME = 'egohos'
SPLITS = {
    f'{NAME}_{s}': (
        f'{NAME}/s', 
        f'{NAME}/{NAME}_{s}.json'
    )
    for s in os.listdir(f'datasets/{NAME}')
    if os.path.isdir(s)
}

for key, (image_root, json_file) in SPLITS.items():
    register_coco_instances(
        key,
        {},
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )

