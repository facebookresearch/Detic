# Copyright (c) Facebook, Inc. and its affiliates.
from .modeling.meta_arch import custom_rcnn
from .modeling.roi_heads import detic_roi_heads
from .modeling.roi_heads import res5_roi_heads
from .modeling.backbone import swintransformer
from .modeling.backbone import timm

from .inference import Detic
# try:
#     from .modeling.meta_arch import d2_deformable_detr
# except:
#     pass