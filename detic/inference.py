# from detectron2.utils.logger import setup_logger
# setup_logger()

# import some common libraries
import functools
import os
# import sys
import glob
from IPython import embed
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detic.predictor import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, random_color, ColorMode
from detectron2.modeling.roi_heads.cascade_rcnn import _ScaleGradient
from detectron2.data import MetadataCatalog
from detectron2.utils.file_io import PathManager
from detectron2.utils.events import get_event_storage

# Detic libraries
# detic_path = os.getenv('DETIC_PATH') or 'Detic'
detic_path = os.path.abspath(os.path.join(__file__, '../..'))
# sys.path.insert(0,  detic_path)
# sys.path.insert(0, os.path.join(detic_path, 'third_party/CenterNet2'))
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder
from centernet.config import add_centernet_config
from detic.data import datasets


BUILDIN_CLASSIFIER = {
    'lvis':       os.path.join(detic_path, 'datasets/metadata/lvis_v1_clip_a+cname.npy'),
    'objects365': os.path.join(detic_path, 'datasets/metadata/o365_clip_a+cnamefix.npy'),
    'openimages': os.path.join(detic_path, 'datasets/metadata/oid_clip_a+cname.npy'),
    'coco':       os.path.join(detic_path, 'datasets/metadata/coco_clip_a+cname.npy'),
    'egohos':     os.path.join(detic_path, 'datasets/metadata/egohos.npy'),
    'ssv2':     os.path.join(detic_path, 'datasets/metadata/ssv2_clip512.npy'),
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
    'egohos': 'egohos_val',
    'ssv2': 'ssv2_val',
}

device = (
    'cuda' if torch.cuda.is_available() else 
    'mps' if torch.backends.mps.is_available() else 
    'cpu')

DEFAULT_PROMPT = 'a {}'

CHECKPOINT = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
CONFIG = "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"

# from .data.datasets import egohos
# CHECKPOINT = 'output/Detic/Detic_EGOHOS_CLIP_SwinB_896b32_4x_ft4x_max-size/model_0009999.pth'
# CONFIG = "configs/Detic_EGOHOS_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"

VERSIONS = {
    None: (
        'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth',
        'configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml'
    ),
    'egohos': (
        'output/Detic/Detic_EGOHOS_CLIP_SwinB_896b32_4x_ft4x_max-size/model_0133999.pth',
        'configs/Detic_EGOHOS_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml'
    ),
    'ssv2': (
        'output/Detic/Detic_SSV2_small_CLIP_SwinB_896b32_4x_ft4x_max-size/model_0032999.pth',
        'configs/Detic_SSV2_small_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml'
    ),
}

def desc(x):
    if isinstance(x, dict):
        return {k: desc(v) for k, v in x.items()}
    if isinstance(x, (dict, list, tuple, set)):
        return type(x)(desc(xi) for xi in x)
    if hasattr(x, 'shape'):
        return f'{type(x).__name__}({x.shape}, {x.dtype})'
    return x

def path_or_url(url):
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    path = parsed_url._replace(query="").geturl()  # remove query from filename
    path = PathManager.get_local_path(path)
    return path if os.path.isfile(path) else url

@functools.lru_cache(1)
def get_text_encoder():
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    return text_encoder
text_encoder = get_text_encoder()


class Detic(nn.Module):
    def __init__(
            self, vocab=None, conf_threshold=0.5, box_conf_threshold=0.5, 
            masks=False, one_class_per_proposal=True, patch_for_embeddings=True, 
            prompt=DEFAULT_PROMPT, device=device, config=None, checkpoint=None,
    ):
        super().__init__()
        self.cfg = cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)

        if isinstance(vocab, str) and os.path.isfile(vocab):
            vocab = [x.strip() for x in open(vocab).read().splitlines()]
            vocab = [x for x in vocab if x and not x.startswith(';')]

        # get latest checkpoint for that config (if it exists)
        if config and not checkpoint:
            checkpoint = (glob.glob(os.path.join('output/Detic', os.path.splitext(os.path.basename(config))[0], 'model_*.pth')) or [None])[0]

        # get default config/checkpoint for that vocab
        _ch, _cf = (VERSIONS.get(vocab) if isinstance(vocab, str) else None) or VERSIONS[None]
        config = config or _cf
        checkpoint = checkpoint or _ch
        print(checkpoint)

        cfg.merge_from_file(os.path.join(detic_path, config))
        cfg.MODEL.WEIGHTS = path_or_url(checkpoint)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = box_conf_threshold  # set threshold for this model
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = one_class_per_proposal # For better visualization purpose. Set to False for all classes.
        cfg.MODEL.MASK_ON = masks
        cfg.MODEL.DEVICE=device # uncomment this to use cpu-only mode
        cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = os.path.join(detic_path, cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH)
        # print(cfg)
        self.predictor = DefaultPredictor(cfg)
        self.cfg = cfg

        if patch_for_embeddings:
            self.predictor.model.roi_heads.__class__ = DeticCascadeROIHeads2
            for b in self.predictor.model.roi_heads.box_predictor:
                b.__class__ = DeticFastRCNNOutputLayers2
                b.cls_score.__class__ = ZeroShotClassifier2
        
        for i, h in enumerate(self.predictor.model.roi_heads.box_predictor):
            h.test_score_thresh = conf_threshold
            # h.norm_temperature = 1

        self.text_encoder = text_encoder

        self.set_labels(vocab, prompt=prompt)

    def set_labels(self, vocab, thing_classes=None, metadata_name=None, prompt=DEFAULT_PROMPT):
        zs_weight, self.metadata, self.metadata_name = load_classifier(
            vocab, 
            thing_classes=thing_classes, 
            prompt=prompt, 
            metadata_name=metadata_name, 
            text_encoder=self.text_encoder,
            device=self.predictor.model.device, 
            norm_weight=self.predictor.model.roi_heads.box_predictor[0].cls_score.norm_weight,
            z_dim=self.cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM,
        )
        set_classifier(self.predictor.model, zs_weight)
        self.labels = np.asarray(self.metadata.thing_classes)
    # alias
    set_vocab = set_labels

    # ---------------------------------- Compute --------------------------------- #

    def forward(self, im, boxes=None, classifier=None):
        out = self.predictor(im, boxes=boxes, classifier=classifier)
        cid = out['instances'].pred_classes.detach().int().cpu().numpy()
        out['instances'].pred_labels = self.labels[cid]
        return out

    def encode_features(self, batched_inputs):
        images = self.predictor.model.preprocess_image(batched_inputs)
        features = self.predictor.model.backbone(images.tensor)
        return features, images

    def get_box_features(self, features, boxes):
        return self.predictor.model.roi_heads.get_box_features(features, boxes)

    def prepose_and_detect(self, batched_inputs, images, features, classifier=None):
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, classifier_info=(classifier, None, None))
        return self._postprocess(results, batched_inputs, images.image_sizes)

    # ------------------------------- Visualization ------------------------------ #

    def draw(self, im, outputs):
        v = Visualizer(im[:, :, ::-1], self.metadata, instance_mode=ColorMode.SEGMENTATION)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return out.get_image()[:, :, ::-1]

    # ----------------------------------- Utils ---------------------------------- #

    @staticmethod
    def group_proposals(bbox):
        bbox_unique, iv = np.unique(bbox, return_inverse=True, axis=0)
        return bbox_unique, np.arange(len(bbox_unique))[:,None] == iv[None]

    @staticmethod
    def boxnorm(xyxy, h, w):
        xyxy[:, 0] = (xyxy[:, 0]) / w
        xyxy[:, 1] = (xyxy[:, 1]) / h
        xyxy[:, 2] = (xyxy[:, 2]) / w
        xyxy[:, 3] = (xyxy[:, 3]) / h
        return xyxy

    def unpack_results(self, outputs, im):
        insts = outputs['instances'].to("cpu")
        xyxy = insts.pred_boxes.tensor.numpy()
        class_ids = insts.pred_classes.numpy().astype(int)
        confs = insts.scores.numpy()
        box_confs = insts.box_scores.numpy()
        # combine (exact) duplicate bounding boxes
        xyxy_unique, ivs = self.group_proposals(xyxy)
        xyxyn_unique = self.boxnorm(xyxy_unique, *im.shape[:2])
        labels = self.labels[class_ids]
        return xyxyn_unique, ivs, class_ids, labels, confs, box_confs

# disable jitter
def _jitter(self, c):
    return [c*255 for c in c]
Visualizer._jitter = _jitter


from torch.nn import functional as F
# from detic.modeling.meta_arch.custom_rcnn import CustomRCNN
from detic.modeling.roi_heads.detic_roi_heads import DeticCascadeROIHeads
from detic.modeling.roi_heads.detic_fast_rcnn import DeticFastRCNNOutputLayers
from detic.modeling.roi_heads.zero_shot_classifier import ZeroShotClassifier
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference

class DeticCascadeROIHeads2(DeticCascadeROIHeads):
    # xx How to get box features given a box
    # Given box features, how can we query detic with them?
    # Are the detection thresholds okay with other instances of an object?

    def get_box_features(self, features, pool_boxes, k=-1):
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, pool_boxes)
        box_features = self.box_head[k](box_features)
        x, xo = self.box_predictor[k].cls_score.encode_features(box_features)
        return x
    
    # def classify_box(self, features, boxes, classifier):
    #     features = [features[f] for f in self.box_in_features]
    #     pool_features = self.box_pooler(features, pool_boxes)
    #     scores_per_stage = []
    #     for k in range(self.num_cascade_stages):
    #         zs = self.box_head[k](pool_features)
    #         scores, cls_feats = self.box_predictor[k].class_pred(zs, (classifier, None, None))
    #         scores_per_stage.append(scores)
    #     stage_scores = [torch.stack(s, dim=1) for s in zip(*scores_per_stage)]
    #     scores = [s.mean(1).round(decimals=3) for s in stage_scores]
    #     return scores

    def _forward_box(self, features, proposals, targets=None, ann_type='box', classifier_info=(None,None,None), score_threshold=None):
        # get image and object metadata from proposals
        k = 'scores' if len(proposals) > 0 and proposals[0].has('scores') else 'objectness_logits'
        proposal_scores = [p.get(k) for p in proposals]
        objectness_logits = [p.objectness_logits for p in proposals]
        image_sizes = [x.image_size for x in proposals]

        # select certain features e.g. ['p3', 'p4', 'p5']
        features = [features[f] for f in self.box_in_features]

        # run multi-stage classification
        predictor = boxes = None
        scores_per_stage = []
        head_outputs = []
        for k in range(self.num_cascade_stages):
            if k > 0:
                # use boxes from previous iter as new proposals
                proposals = self._create_proposals_from_boxes(boxes, image_sizes, logits=objectness_logits)
                if self.training and ann_type in ['box']:
                    # match proposals to ground truth
                    proposals = self._match_and_label_boxes(
                        proposals, k, targets)

            # Run stage - get features per box
            pool_boxes = [x.proposal_boxes for x in proposals]
            # pools features using boxes
            box_features = self.box_pooler(features, pool_boxes)
            box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
            # several CNN>norm>relu layers
            box_features = self.box_head[k](box_features)
            # store 1024 features on the proposals
            for feat, p in zip(box_features.split([len(p) for p in proposals], dim=0), proposals):
                p.feat = feat

            # predict classes and regress boxes
            predictor = self.box_predictor[k]
            # does zero-shot classification against text features - linear > cosine distance
            # deltas predicted using linear+relu+linear[4]
            # cls_feats are the features after the linear layer in the zero-shot classifier
            # prop_score returned if with_softmax_prop. it's x>linear>relu>linear[n_classes+1]
            scores, cls_feats = predictor.class_pred(box_features, classifier_info)
            deltas = predictor.bbox_pred(box_features)
            # prop_score = self.prop_score(box_features)
            # scores, deltas, cls_feats, *prop_score = predictions = predictor(
            #     box_features, classifier_info=classifier_info)
            # add deltas to boxes (no prediction)
            boxes = predictor.predict_boxes((scores, deltas), proposals)
            # just applies sigmoid or softmax depending on config
            scores = predictor.predict_probs((scores,), proposals)

            scores_per_stage.append(scores)
            head_outputs.append((predictor, (scores, deltas,), proposals))

        if self.training:
            return self._training_losses(head_outputs, targets, ann_type, classifier_info)

        # aggregate scores
        stage_scores = [torch.stack(s, dim=1) for s in zip(*scores_per_stage)] # [(nprop, 3, 2)]
        scores = [s.mean(1).round(decimals=2) for s in stage_scores]
        # if self.mult_proposal_score:
        #     scores = [(s ** 0.8) * (ps[:, None] ** 0.2) for s, ps in zip(scores, proposal_scores)]
        if self.one_class_per_proposal:
            scores = [s * (s == s[:, :-1].max(dim=1)[0][:, None]).float() for s in scores]

        # down-select proposals
        # clip boxes, filter threshold, nms
        pred_instances, filt_idxs = fast_rcnn_inference(
            boxes, scores, image_sizes,
            predictor.test_score_thresh if score_threshold is None else score_threshold,
            predictor.test_nms_thresh,
            predictor.test_topk_per_image,
        )
        # ++ add clip features and box scores to instances [N boxes x 512]
        pred_instances[0].stage_scores = stage_scores[0][filt_idxs]
        pred_instances[0].raw_features = box_features[filt_idxs]
        # pred_instances[0].raw_features = feats_per_image[0][filt_idxs]
        pred_instances[0].clip_features = cls_feats[filt_idxs]
        if self.mult_proposal_score:
            pred_instances[0].box_scores = proposal_scores[0][filt_idxs]
        return pred_instances
    


    def _training_losses(self, head_outputs, targets, ann_type, classifier_info):
        losses = {}
        storage = get_event_storage()
        for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
            with storage.name_scope("stage{}".format(stage)):
                if ann_type != 'box': 
                    stage_losses = {}
                    if ann_type in ['image', 'caption', 'captiontag']:
                        weak_losses = predictor.image_label_losses(
                            predictions, proposals, 
                            image_labels=[x._pos_category_ids for x in targets],
                            classifier_info=classifier_info,
                            ann_type=ann_type)
                        stage_losses.update(weak_losses)
                else: # supervised
                    stage_losses = predictor.losses(
                        (predictions[0], predictions[1]), proposals,
                        classifier_info=classifier_info)
                    if self.with_image_labels:
                        stage_losses['image_loss'] = predictions[0].new_zeros([1])[0]
            losses.update({
                f"{k}_stage{stage}": v
                for k, v in stage_losses.items()})
        return losses



class Query:
    def __init__(self, proposals):
        self.proposals = proposals

        k = 'scores' if len(proposals) > 0 and proposals[0].has('scores') else 'objectness_logits'
        self.proposal_scores = [p.get(k) for p in proposals]
        self.objectness_logits = [p.objectness_logits for p in proposals]
        self.image_sizes = [x.image_size for x in proposals]

    def encode_box_features(self):
        # Run stage - get features per box
        pool_boxes = [x.proposal_boxes for x in self.proposals]
        # pools features using boxes
        box_features = self.box_pooler(features, pool_boxes)
        box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        # several CNN>norm>relu layers
        box_features = self.box_head[k](box_features)

    def cascade_boxes(self):
        pass




class DeticFastRCNNOutputLayers2(DeticFastRCNNOutputLayers):
    def class_pred(self, x, classifier_info=(None,None,None)):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = []
        cls_scores, x_features = self.cls_score(x, classifier=classifier_info[0])  # ++ add x_features
        scores.append(cls_scores)
   
        if classifier_info[2] is not None:
            cap_cls = classifier_info[2]
            cap_cls = cap_cls[:, :-1] if self.sync_caption_batch else cap_cls
            caption_scores, _ = self.cls_score(x, classifier=cap_cls)
            scores.append(caption_scores)
        scores = torch.cat(scores, dim=1) # B x C' or B x N or B x (C'+N)
        return scores, x_features

    def forward(self, x, classifier_info=(None,None,None)):
        scores, x_features = self.class_pred(x, classifier_info)
        proposal_deltas = self.bbox_pred(x)
        if self.with_softmax_prop:
            prop_score = self.prop_score(x)
            return scores, proposal_deltas, x_features, prop_score
        return scores, proposal_deltas, x_features  # ++ return x_features


class ZeroShotClassifier2(ZeroShotClassifier):
    cls_weight = 1
    cls_bias = 0
    def prepare_classifier(self, classifier):
        zs_weight = classifier.permute(1, 0).contiguous() # D x C'
        if self.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=0)
        return zs_weight

    def get_classifier(self, classifier=None):
        if classifier is None:
            classifier = self.zs_weight
        # else:
        #     classifier = self.prepare_classifier(classifier)
        if self.linear.weight.shape[0] != classifier.shape[0] and self.linear.weight.shape[1] == classifier.shape[0]:
            classifier = self.linear(classifier.permute(1, 0)).permute(1, 0)
        assert self.linear.weight.shape[0] == classifier.shape[0], f'{self.linear.weight.shape} != {classifier.shape}'
        return classifier  # D x C'
    
    def encode_features(self, x):
        xo = x = self.linear(x)  # ++ save linear output
        if self.norm_weight:
            x = F.normalize(x, p=2, dim=1)
        return x, xo

    def forward(self, x, classifier=None):
        '''
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        '''
        x, _ = self.encode_features(x)
        zs_weight = self.get_classifier(classifier)
        T = self.norm_temperature if self.norm_weight else 1
        y = torch.mm(T * x, zs_weight) * self.cls_weight + self.cls_bias
        return y, x


def load_classifier(
        vocab, thing_classes=None, 
        prompt='a {}', 
        metadata_name=None, 
        text_encoder=text_encoder,
        device=device, 
        norm_weight=True,
        z_dim=512,
):
    # default vocab
    if vocab is None:
        vocab = 'lvis'

    # handle ambiguity with numpy arrays
    if isinstance(vocab, np.ndarray):
        if vocab.ndim == 1:  # label np.array -> label list
            vocab = vocab.tolist()
        else:  # custom embeddings
            vocab = torch.as_tensor(vocab).float()

    # custom embeddings
    if isinstance(vocab, torch.Tensor):
        assert vocab.ndim == 2 and vocab.shape[1] == z_dim, f"Expected vocab shape (N, {z_dim})."
        if thing_classes is None:
            thing_classes = list(range(vocab.shape[0]))
        classifier = vocab.permute(1, 0)

    # text queries
    elif isinstance(vocab, (list, tuple)):
        if thing_classes is None:
            thing_classes = vocab
        
        prompt = prompt or '{}'
        classifier = text_encoder(
            [prompt.format(x) for x in vocab]
        ).detach().permute(1, 0).contiguous()#.cpu()

    # pre-defined vocabularies
    elif isinstance(vocab, str):
        if os.path.isfile(vocab):
            metadata_name = metadata_name or os.path.splitext(os.path.basename(vocab))[0]
            classifier = vocab
        else:
            metadata_name = metadata_name or BUILDIN_METADATA_PATH.get(vocab) or vocab
            classifier = BUILDIN_CLASSIFIER[vocab]

    else:
        raise ValueError("Invalid vocab. Must be a list of text classes.")   

    # build metadata
    metadata_name = metadata_name or '__vocab:' + ','.join(thing_classes)
    metadata = MetadataCatalog.get(metadata_name)
    try:
        if thing_classes is not None:
            metadata.thing_classes = list(thing_classes)
        metadata.thing_colors = [tuple(random_color(rgb=True, maximum=1)) for _ in metadata.thing_classes]
    except (AttributeError, AssertionError):
        pass

    return prepare_classifier(classifier, device, norm_weight), metadata, metadata_name

def prepare_classifier(zs_weight, device=None, norm_weight=True):
    if isinstance(zs_weight, str):
        zs_weight = torch.tensor(np.load(zs_weight), dtype=torch.float32)
        zs_weight = zs_weight.permute(1, 0).contiguous() # D x C
    bg_emb = zs_weight.new_zeros((zs_weight.shape[0], 1))
    zs_weight = torch.cat([zs_weight, bg_emb], dim=1) # D x (C + 1)
    if norm_weight:
        zs_weight = F.normalize(zs_weight, p=2, dim=0)
    if device is not None:
        zs_weight = zs_weight.to(device)
    return zs_weight

def set_classifier(model, zs_weight):
    model.roi_heads.num_classes = zs_weight.shape[1]
    for k in range(len(model.roi_heads.box_predictor)):
        del model.roi_heads.box_predictor[k].cls_score.zs_weight
        model.roi_heads.box_predictor[k].cls_score.zs_weight = zs_weight

def reset_cls_test(model, cls_path, num_classes):
    norm_weight = model.roi_heads.box_predictor[0].cls_score.norm_weight
    zs_weight = prepare_classifier(cls_path, model.device, norm_weight)
    set_classifier(model, zs_weight)
    

import supervision as sv
class Visualizer:
    def __init__(self, labels) -> None:
        self.labels = labels
        self.ba = sv.BoxAnnotator(text_scale=0.4, text_padding=1)
        self.ma = sv.MaskAnnotator()

    def as_detections(self, outputs):
        return sv.Detections(
            xyxy=outputs["instances"].pred_boxes.tensor.cpu().numpy(),
            mask=outputs["instances"].pred_masks.cpu().numpy() if hasattr(outputs["instances"], 'pred_masks') else None,
            confidence=outputs["instances"].scores.cpu().numpy(),
            class_id=outputs["instances"].pred_classes.cpu().numpy().astype(int),
        )

    def draw(self, frame, detections):
        labels = [
            f"{self.labels[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections
        ]
        # frame = self.ma.annotate(scene=frame, detections=detections)
        frame = self.ba.annotate(scene=frame, detections=detections, labels=labels)
        return frame


def run(src, vocab, out_file=True, size=480, fps_down=1, **kw):
    """Run multi-target tracker on a particular sequence.
    """
    import tqdm
    import cv2
    import supervision as sv

    # class VideoSink2(sv.VideoSink):
    #     def __enter__(self):
    #         import cv2
    #         for c in ['mp4v', 'avc1']:
    #             self._VideoSink__fourcc = cv2.VideoWriter_fourcc(*c)
    #             super().__enter__()
    #             if self._VideoSink__writer.isOpened():
    #                 return self
    #         raise RuntimeError("Could not find a video codec that works")
    # sv.VideoSink = VideoSink2

    

    kw.setdefault('masks', True)
    # kw.setdefault('conf_threshold', 0.6)
    model = Detic(['cat'], **kw).to(device)

    if isinstance(vocab, str) and os.path.isfile(vocab):
        d = np.load(vocab)
        if isinstance(d, np.ndarray):
            model.set_vocab(d)
        else:
            z, labels = d['Z'], d['labels']
            z, labels = agg_labels(z, labels)
            model.set_vocab(z, labels)
    else:
        model.set_vocab(vocab)

    classifier = model.predictor.model.roi_heads.box_predictor[0].cls_score.zs_weight

    # for h in model.predictor.model.roi_heads.box_predictor:
    #     h.cls_score.norm_temperature = 10
    # #     h.cls_score.dist_scale = 1

    if out_file is True:
        out_file='detic_'+os.path.basename(src)
    assert out_file
    print("Writing to:", os.path.abspath(out_file))

    classes = model.labels
    print("classes:", classes)

    vis = Visualizer(model.labels)
    masks_on = model.cfg.MODEL.MASK_ON
    print("using masks:", masks_on)

    single_class = model.cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL
    print("single class:", single_class)

    from pyinstrument import Profiler
    p = Profiler()
    # video_info = sv.VideoInfo.from_video_path(src)
    video_info, WH = get_video_info(src, size, fps_down)
    try:
        replaced = [False]*len(classifier)
        with sv.VideoSink(out_file, video_info=video_info) as s, p:
            pbar = tqdm.tqdm(enumerate(sv.get_video_frames_generator(src)), total=video_info.total_frames)
            for i, frame in pbar:
                # if i > 100: break
                frame = cv2.resize(frame, WH)
                outputs = model(frame, classifier)
                # bbox_unique, iv = model.group_proposals(bbox)
                detections = vis.as_detections(outputs)
                
                pbar.set_description(f'{len(detections)}' + ', '.join(set(classes[i] for i in detections.class_id)) or 'nothing')
                out_frame = vis.draw(frame.copy(), detections)
                s.write_frame(out_frame)
                # if input():embed()

                # print((outputs['instances'].stage_scores[:,:,0]*100).int())
                # # print(torch.mean(outputs['instances'].stage_scores[:,:,0], dim=1))
                # # print((outputs['instances'].scores*100).int())
                # # print((outputs['instances'].stage_boxes).int())
                # for c, cid, z in sorted(zip(detections.confidence, detections.class_id, outputs['instances'].raw_features), key=lambda x: -x[0]):
                #     if not replaced[cid]:
                #         tqdm.tqdm.write(f'Replacing {model.labels[cid]} {cid}')
                #         # classifier[:, cid] = z
                #         classifier = prepare_classifier(z[:, None])
                #         replaced[cid]=True
                #         for h in model.predictor.model.roi_heads.box_predictor:
                #             h.cls_score.norm_temperature = 1
                #             h.cls_score.dist_scale = 1/10

                #         outputs = model(frame, classifier)
                #         detections = vis.as_detections(outputs)
                #         s.write_frame(vis.draw(frame.copy(), detections))
                # #         embed()
                #     # else:
                #     #     mix = 0.05
                #     #     classifier[:, cid] = classifier[:, cid] * (1-mix) + z * mix
    finally:
        p.print()

# for h in model.predictor.model.roi_heads.box_predictor:
#     print((torch.mm(F.normalize(h.cls_score.linear(rf), p=2, dim=1), F.normalize(h.cls_score.get_classifier(classifier), p=2, dim=1))/5).sigmoid())

def get_video_info(src, size, fps_down=1, nrows=1, ncols=1):
    # get the source video info
    video_info = sv.VideoInfo.from_video_path(video_path=src)
    # make the video size a multiple of 16 (because otherwise it won't generate masks of the right size)
    aspect = video_info.width / video_info.height
    video_info.width = int(aspect*size)//16*16
    video_info.height = int(size)//16*16
    WH = video_info.width, video_info.height

    # double width because we have both detic and xmem frames
    video_info.width *= ncols
    video_info.height *= nrows
    # possibly reduce the video frame rate
    video_info.og_fps = video_info.fps
    video_info.fps /= fps_down or 1

    print(f"Input Video {src}\nsize: {WH}  fps: {video_info.fps}")
    return video_info, WH


def agg_labels(z, labels):
    zs = []
    labels = np.asarray(labels)
    ulabels = np.unique(labels)
    for l in ulabels:
        zs.append(z[labels == l].mean(0))
    return np.array(zs), ulabels


if __name__ == '__main__':
    import fire
    fire.Fire(run)
