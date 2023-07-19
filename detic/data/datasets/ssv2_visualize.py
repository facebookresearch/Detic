from detectron2.structures import BoxMode
from ssv2_categories import SSV2_CATEGORIES
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import numpy as np
import os
import json
import cv2
import random
import itertools
from collections import defaultdict

IMG_PATH = '/vast/work/ptg/Something_frames'
SPLIT = 'VAL_ATTRIBUTES'
NEXAMPLES = 50
S = 3

def get_category_names():
    tuples = []
    for c in SSV2_CATEGORIES:
        tuples.append((c['name'],c['id']))
    tuples = sorted(tuples, key=lambda x: x[1])
    return [c[0] for c in tuples]


CAT_NAMES = get_category_names()

def get_ss_dicts(img_dir, split=SPLIT, N=NEXAMPLES):
    json_file = os.path.join(img_dir, f'annotations/SS_{SPLIT}.json')
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for v in random.sample(imgs_anns['images'],N):
        record = {}
        
        filename = os.path.join(img_dir+'/images', v["file_name"])
        height, width = v['height'], v['width']
        
        record["file_name"] = filename
        record["image_id"] = v['id']
        record["height"] = height
        record["width"] = width
      
        annos = [x for x in imgs_anns['annotations'] if x['image_id'] == record['image_id']]
        objs = defaultdict(lambda: [])
        for anno in annos:
            objs[tuple(anno['bbox'])].append(anno['category_id'])

        record['image'] = cv2.resize(cv2.imread(record['file_name']),(0,0),fx=S,fy=S)
        record['bbox'] = S*np.array(list(objs))
        record['categories'] = [[CAT_NAMES[ci] for ci in c] for c in objs.values()]
        dataset_dicts.append([record['image'], record['bbox'],record['categories'],record['file_name']])
        
    return dataset_dicts

def draw_boxes(im, boxes, labels=None, color=(0,255,0), size=1, text_color=(0, 0, 255), spacing=3):
    boxes = np.asarray(boxes).astype(int)
    color = np.asarray(color).astype(int)
    color = color[None] if color.ndim == 1 else color
    labels = itertools.chain([] if labels is None else labels, itertools.cycle(['']))
    for xy, c in zip(boxes, itertools.cycle(color)):
        im = cv2.rectangle(im, xy[:2], xy[2:4]+xy[:2], tuple(c.tolist()), 2)
    
    for xy, label, c in zip(boxes, labels, itertools.cycle(color)):
        if label:
            if isinstance(label, list):
                im, _ = draw_text_list(
                    im, label, 0, tl=xy[:2] + spacing, scale=im.shape[1]/1400*size, 
                    space=40, color=text_color)
            else:
                im = cv2.putText(
                    im, label, xy[:2] - spacing, 
                    cv2.FONT_HERSHEY_SIMPLEX, im.shape[1]/1400*size, 
                    text_color, 1)
    return im


def draw_text_list(img, texts, i=-1, tl=(10, 50), scale=0.4, space=50, color=(255, 255, 255), thickness=1):
    for i, txt in enumerate(texts, i+1):
        cv2.putText(
            img, txt, 
            (int(tl[0]), int(tl[1]+scale*space*i)), 
            cv2.FONT_HERSHEY_COMPLEX , 
            scale, color, thickness)
    return img, i

d = f'{SPLIT}'
dataset_dicts = get_ss_dicts(IMG_PATH)
for i,d in enumerate(random.sample(dataset_dicts, NEXAMPLES)):
    print(i)
    img = draw_boxes(*d[:-1])
    fname = d[-1].split('/')[-2]
    print(fname)
    cv2.imwrite(f"test_imgs/{fname}.png", img)
