from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
# import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
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


BUILDIN_CLASSIFIER = {
    'lvis':       os.path.join(detic_path, 'datasets/metadata/lvis_v1_clip_a+cname.npy'),
    'objects365': os.path.join(detic_path, 'datasets/metadata/o365_clip_a+cnamefix.npy'),
    'openimages': os.path.join(detic_path, 'datasets/metadata/oid_clip_a+cname.npy'),
    'coco':       os.path.join(detic_path, 'datasets/metadata/coco_clip_a+cname.npy'),
    'egohos':     os.path.join(detic_path, 'datasets/metadata/egohos.npy'),
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
    'egohos': 'egohos_val',
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DEFAULT_PROMPT = 'a {}'

CHECKPOINT = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
CONFIG = "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"

from .data.datasets import egohos
CHECKPOINT = 'output/Detic/Detic_EGOHOS_CLIP_SwinB_896b32_4x_ft4x_max-size/model_0009999.pth'
CONFIG = "configs/Detic_EGOHOS_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"

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

class Detic(nn.Module):
    def __init__(self, vocab=None, conf_threshold=0.5, box_conf_threshold=0.5, masks=False, one_class_per_proposal=True, patch_for_embeddings=True, prompt=DEFAULT_PROMPT, device=device):
        super().__init__()
        self.cfg = cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(os.path.join(detic_path, CONFIG))
        cfg.MODEL.WEIGHTS = path_or_url(CHECKPOINT)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = box_conf_threshold  # set threshold for this model
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = one_class_per_proposal # For better visualization purpose. Set to False for all classes.
        cfg.MODEL.MASK_ON = masks
        cfg.MODEL.DEVICE=device # uncomment this to use cpu-only mode
        cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = os.path.join(detic_path, cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH)
        # print(cfg)
        self.predictor = DefaultPredictor(cfg)

        if patch_for_embeddings:
            self.predictor.model.roi_heads.__class__ = DeticCascadeROIHeads2
            for b in self.predictor.model.roi_heads.box_predictor:
                b.__class__ = DeticFastRCNNOutputLayers2
                b.cls_score.__class__ = ZeroShotClassifier2
        
        for cascade_stages in range(len(self.predictor.model.roi_heads.box_predictor)):
            self.predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = conf_threshold

        self.text_encoder = build_text_encoder(pretrain=True)
        self.text_encoder.eval()

        self.set_vocab(vocab, prompt)
        
    def set_vocab(self, vocab, prompt=DEFAULT_PROMPT):
        if isinstance(vocab, (np.ndarray, torch.Tensor)):
            vocab = vocab.tolist()
        if isinstance(vocab, (list, tuple)):
            self.vocab_key = '__vocab:' + ','.join(vocab)
            self.metadata = metadata = MetadataCatalog.get(self.vocab_key)
            try:
                metadata.thing_classes = list(vocab)
                metadata.thing_colors = [tuple(random_color(rgb=True, maximum=1)) for _ in metadata.thing_classes]
            except (AttributeError, AssertionError):
                pass
            
            self.prompt = prompt = prompt or '{}'
            if isinstance(vocab, (np.ndarray, torch.Tensor)):
                classifier = torch.as_tensor(vocab).cpu()
            else:
                classifier = self.text_encoder(
                    [prompt.format(x) for x in vocab]
                ).detach().permute(1, 0).contiguous().cpu()
            self.text_features = classifier
        else:
            vocab = 'lvis' if vocab is None else vocab
            self.vocab_key = BUILDIN_METADATA_PATH[vocab]
            self.metadata = metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocab])
            classifier = BUILDIN_CLASSIFIER[vocab]    
        
        self.labels = np.asarray(metadata.thing_classes)
        reset_cls_test(self.predictor.model, classifier, len(metadata.thing_classes))

    def forward(self, im):
        return self.predictor(im)

    def draw(self, im, outputs):
        v = Visualizer(im[:, :, ::-1], self.metadata, instance_mode=ColorMode.SEGMENTATION)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return out.get_image()[:, :, ::-1]


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
from detic.modeling.roi_heads.detic_roi_heads import DeticCascadeROIHeads
from detic.modeling.roi_heads.detic_fast_rcnn import DeticFastRCNNOutputLayers
from detic.modeling.roi_heads.zero_shot_classifier import ZeroShotClassifier
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference

class DeticCascadeROIHeads2(DeticCascadeROIHeads):
    def _forward_box(self, features, proposals, targets=None, ann_type='box', classifier_info=(None,None,None)):
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
            # store 1024 features on the proposals?
            if self.add_feature_to_prop:
                n_proposals = [len(p) for p in proposals]
                feats_per_image = box_features.split(n_proposals, dim=0)
                for feat, p in zip(feats_per_image, proposals):
                    p.feat = feat

            # predict classes and regress boxes
            predictor = self.box_predictor[k]
            # does zero-shot classification against text features - linear > cosine distance
            # deltas predicted using linear+relu+linear[4]
            # cls_feats are the features after the linear layer in the zero-shot classifier
            # prop_score returned if with_softmax_prop. it's x>linear>relu>linear[n_classes+1]
            scores, deltas, cls_feats, *prop_score = predictions = predictor(
                box_features, classifier_info=classifier_info)
            # add deltas to boxes (no prediction)
            boxes = predictor.predict_boxes((scores, deltas), proposals)
            # just applies sigmoid or softmax depending on config
            scores = predictor.predict_probs((scores,), proposals)
            # scores[0][:,-1] = -torch.inf
            # scores = (F.softmax(scores[0]*1e7),)
            # print(scores[0].shape)
            # print(scores[0])
            scores_per_stage.append(scores)
            head_outputs.append((predictor, predictions, proposals))

        if self.training:
            return self._training_losses(head_outputs, targets, ann_type, classifier_info)

        # aggregate scores
        scores = [torch.mean(torch.stack(s), dim=0) for s in zip(*scores_per_stage)]
        if self.mult_proposal_score:
            scores = [(s * ps[:, None]) ** 0.5 for s, ps in zip(scores, proposal_scores)]
        if self.one_class_per_proposal:
            scores = [s * (s == s[:, :-1].max(dim=1)[0][:, None]).float() for s in scores]

        # down-select proposals
        # clip boxes, filter threshold, nms
        pred_instances, filt_idxs = fast_rcnn_inference(
            boxes, scores, image_sizes,
            predictor.test_score_thresh,
            predictor.test_nms_thresh,
            predictor.test_topk_per_image,
        )
        # ++ add clip features and box scores to instances [N boxes x 512]
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

class DeticFastRCNNOutputLayers2(DeticFastRCNNOutputLayers):
    def forward(self, x, classifier_info=(None,None,None)):
        """
        enable classifier_info
        """
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

        proposal_deltas = self.bbox_pred(x)
        if self.with_softmax_prop:
            prop_score = self.prop_score(x)
            return scores, proposal_deltas, x_features, prop_score
        return scores, proposal_deltas, x_features
        # ++ return x_features


class ZeroShotClassifier2(ZeroShotClassifier):
    def forward(self, x, classifier=None):
        '''
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        '''
        x_features = x = self.linear(x)  # ++ save linear output
        if classifier is not None:
            zs_weight = classifier.permute(1, 0).contiguous() # D x C'
            zs_weight = F.normalize(zs_weight, p=2, dim=0) \
                if self.norm_weight else zs_weight
        else:
            zs_weight = self.zs_weight
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=1)
        x = torch.mm(x, zs_weight)
        if self.use_bias:
            x = x + self.cls_bias
        return x, x_features  # ++ add x_features




def run(src, vocab, out_file=True, **kw):
    """Run multi-target tracker on a particular sequence.
    """
    import tqdm
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
    model = Detic(**kw)

    if out_file is True:
        out_file='detic_'+os.path.basename(src)
    assert out_file
    print("Writing to:", os.path.abspath(out_file))

    if vocab:
        model.set_vocab(vocab)
        model.set_vocab([c for c in model.labels if c != 'interacting'])
    classes = model.labels
    print("classes:", classes)

    box_annotator = sv.BoxAnnotator(text_scale=0.4, text_padding=1)
    mask_annotator = sv.MaskAnnotator()
    masks_on = model.cfg.MODEL.MASK_ON
    print("using masks:", masks_on)

    video_info = sv.VideoInfo.from_video_path(src)

    with sv.VideoSink(out_file, video_info=video_info) as s:
        pbar = tqdm.tqdm(enumerate(sv.get_video_frames_generator(src)), total=video_info.total_frames)
        for i, frame in pbar:
            # if i > 100: break
            outputs = model(frame)
            detections = sv.Detections.from_detectron2(outputs)
            detections = sv.Detections(
                xyxy=outputs["instances"].pred_boxes.tensor.cpu().numpy(),
                mask=outputs["instances"].pred_masks.cpu().numpy() if hasattr(outputs["instances"], 'pred_masks') else None,
                confidence=outputs["instances"].scores.cpu().numpy(),
                class_id=outputs["instances"].pred_classes.cpu().numpy().astype(int),
            )
            if masks_on:
                tqdm.tqdm.write(f'{detections.mask.shape}')
            pbar.set_description(', '.join(classes[i] for i in detections.class_id) or 'nothing')
            frame = frame.copy()
            if masks_on:
                frame = mask_annotator.annotate(
                    scene=frame,
                    detections=detections,
                )
            frame = box_annotator.annotate(
                scene=frame,
                detections=detections,
                labels=[
                    f"{classes[class_id]} {confidence:0.2f}"
                    for _, _, confidence, class_id, _
                    in detections
                ]
            )
            s.write_frame(frame)
    if s._VideoSink__writer is None:
        s._VideoSink__writer.release()
    


if __name__ == '__main__':
    import fire
    fire.Fire(run)
