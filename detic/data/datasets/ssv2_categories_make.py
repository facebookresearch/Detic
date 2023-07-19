import json
import tqdm
import copy
import re
import os
from pddl import parse_domain, parse_problem
import pddl
import cv2

NTH = 6

FRAMES_PATH = '/vast/work/ptg/Something_frames/images'

SSV2_JSONS = {
    'TRAIN': '/vast/work/ptg/ss_metadata/train.json',
    'VAL':   '/vast/work/ptg/ss_metadata/validation.json',
    'TEST':  '/vast/work/ptg/ss_metadata/test.json',
    'LABELS':'/vast/work/ptg/ss_metadata/labels.json',
}

def load_ss_jsons(ss_jsons):
    ss_ann = {}
    for j in ss_jsons:
        dict_ = json.load(open(j))
        ss_ann.update(dict_)
    return ss_ann
SE_JSONS = [
    '/vast/work/ptg/something_else/bounding_box_smthsmth_part1.json',
    '/vast/work/ptg/something_else/bounding_box_smthsmth_part2.json',
    '/vast/work/ptg/something_else/bounding_box_smthsmth_part3.json',
    '/vast/work/ptg/something_else/bounding_box_smthsmth_part4.json',
]
SE_ann = load_ss_jsons(SE_JSONS)

GPA_PDDL = '/vast/work/ptg/gpa/20bn.pddl'
GPA_std2alpha = {
    '0000':'?a',
    '0001':'?b',
    '0002':'?c',
    '0003':'?d',
    'hand':'hand'
}
GPA_int2alpha = {
    0:'?a',
    1:'?b',
    2:'?c',
    3:'?d',
}


def parse_GAP_PDDL(GPA_PDDL):
    # Load pddl 20DB domain
    label_tuples = re.findall('\s*;\s+\d+\s(.*)\n\s+(?:\s*;.*\n\s+)?\(:action ([\w-]+)', open(GPA_PDDL).read())
    action2label = {k:v for v, k in label_tuples}
    label2action = {v.lower():k.lower() for k,v in action2label.items()}
    ssv2_domain = parse_domain(GPA_PDDL)
    ssv2_actions = {action.name:action for action in ssv2_domain.actions}
    ssv2_axioms = {str(axiom.context):axiom for axiom in ssv2_domain.axioms}
    return action2label, label2action, ssv2_actions, ssv2_axioms
action2label, label2action, ssv2_actions, ssv2_axioms = parse_GAP_PDDL(GPA_PDDL)

def get_implications(predicate,ssv2_axioms=ssv2_axioms,possible_parameters=['?a','?b','?c']):
    parameters = [w.replace(')','') for w in predicate.split() if '?' in w]
    norm_parms = [possible_parameters[i] for i in range(len(parameters))]
    norm2orig = {k:v for k,v in zip(norm_parms,parameters)}
    if 'hand' in predicate:
        norm_parms.append('hand')
        parameters.append('hand')
    pred_root = predicate.split()[:-len(norm_parms)]
    pred_root.extend(norm_parms)
    norm_pred = ' '.join(pred_root)
    for i in range(norm_pred.count('(')):
        norm_pred += ')'

    # collect implications
    implications = [norm_pred]
    if norm_pred in ssv2_axioms:
        implies_ = ssv2_axioms[norm_pred].implies
        if isinstance(implies_, (pddl.logic.base.And, pddl.logic.effects.AndEffect)):
            implications.extend([str(p) for p in implies_._operands])
        elif not isinstance(implies_, (list, tuple)):
            implications.extend([str(implies_)])
    implications = list(set(implications))

    # unnormalize parameters
    norm_imps = []
    for imp in implications:
        wlist = imp.split(' ')
        imp_params = [w.replace(')','') for w in wlist if '?' in w]
        imp_params = [norm2orig[p] for p in imp_params]
        iparam = 0
        wlist_norm = []
        for w in wlist:
            if '?' in w:
                w = imp_params[iparam]
                iparam += 1
            wlist_norm.append(w)
        norm_imp = ' '.join(wlist_norm)
        for i in range(norm_imp.count('(')-norm_imp.count(')')):
            norm_imp += ')'
        norm_imps.append(norm_imp)
    return norm_imps


def remove_fillers(x):
    fillers = ['a','an','other', 'another','more', 'the', 'this','some','different']
    xs = x.split()
    while xs and xs[0] in fillers:
        xs.pop(0)
    x = ' '.join(xs).lower()
    return x

# iterate over SE_JSONS
ALL_OBJECT_CATEGORIES = []
ALL_OBJECT_PREDICATES = []
SS_CLASSES = {}
obj_id = 0
# iterate over all SSV2 JSONS
for f in ['VAL','TRAIN']:
    print(f)
    ssv2_ = json.load(open(SSV2_JSONS[f'{f}']))
    ssv2_meta = {d['id']:d for d in ssv2_}

    # create the 'data' SS json
    SSdata = {'info':{},'licenses':[],'categories':[],'images':[],'annotations':[]}

    OBJECT_CATEGORIES = {}
    OBJECT_PREDICATES = {}
    image_id = 0
    ann_id = 0
    # iterate over videos
    for video in tqdm.tqdm(SE_ann.keys()):

        # sanity check
        # How many videos match between SS and SE?
        # many videos are missing: https://github.com/joaanna/something_else/issues/15
        if video not in ssv2_meta:
            continue

        # get video metadata
        vid_frames_ = os.listdir(os.path.join(FRAMES_PATH,video))
        vid_nframes = len(vid_frames_)
        img_0 = cv2.imread(os.path.join(FRAMES_PATH,video,vid_frames_[0]))
        img_end = cv2.imread(os.path.join(FRAMES_PATH,video,vid_frames_[-1]))
        assert img_0.shape == img_end.shape
        HEIGHT, WIDTH, _ = img_0.shape
        video_label = ssv2_meta[video]['template'].replace('[','').replace(']','')
        video_label = video_label.lower()
        if 'camera' in video_label:
            continue
        # get the video action (and flag void if objs have no attributes)
        video_action = label2action[video_label]
        video_pddl = ssv2_actions[video_action]
        void_video = True if '?void' in [str(v) for v in list(video_pddl.parameters)] else False
        # skip void videos as the pddl definition does not handle numerosity
        if void_video:
            if 'number' in video_label or 'piling' in video_label:
                continue
            else:
                print(video)
                print(video_label)
                print('objects:',video_pddl)
                print('')
                input()

        # get video preconditions
        if isinstance(video_pddl._precondition, (pddl.logic.base.And, pddl.logic.effects.AndEffect)):
            vid_precs = [str(p) for p in video_pddl._precondition._operands]
        elif not isinstance(video_pddl._precondition, (list, tuple)):
            vid_precs = [str(video_pddl._precondition)]
        # get the effects TODO: add the preconditions that did not change
        if isinstance(video_pddl._effect, (pddl.logic.base.And, pddl.logic.effects.AndEffect)):
            vid_effcs = [str(p) for p in video_pddl._effect._operands]
        elif not isinstance(video_pddl._effect, (list, tuple)):
            vid_effcs = [str(video_pddl._effect)]
        if len(vid_effcs) == 0:
            continue
        vid_precs = [p.replace('onsurface','on-surface') for p in vid_precs]
        vid_effcs = [p.replace('onsurface','on-surface') for p in vid_effcs]
            #vid_effcs = vid_precs
        # get number of pddl objects
        #print(video_action)
        #print(vid_precs)
        #print(vid_effcs)

        #effc_imps = []
        #for eff in vid_effcs:
        #    effc_imps.extend(get_implications(eff))
        #vid_effcs = effc_imps
        ##print(effc_imps)
        #for p in vid_precs:
        #    if 'not' in p:
        #        if p.replace('(not ','')[:-1] in vid_effcs:
        #            #print('[NOT] predicate changed:',p)
        #            p
        #        else:
        #            #print('[NOT] predicate same:',p)
        #            vid_effcs.append(p)
        #    else:
        #        if '(not '+p+')' in vid_effcs:
        #            p
        #            #print('predicate changed:',p)
        #        else:
        #            #print('predicate same:',p)
        #            vid_effcs.append(p)
        #    #input()
        #vid_effcs = list(set(vid_effcs))

        vid_effcs = [ff.replace('  ',' ') for ff in vid_effcs]
        vid_precs = [pp.replace('  ',' ') for pp in vid_precs]
        #print(vid_effcs)
        #input()
        npddl_objects = len(video_pddl._parameters)
        #print(video)
        #print(video_label)
        #print(vid_precs)
        #print(vid_effcs)
        #input()
        #print(eff_implc)

        # get the expected objects based on the video metadata
        ss_plhs = [remove_fillers(x) for x in ssv2_meta[video]['placeholders']]
        # count the number of ss placeholders
        nss_objects = len(ss_plhs)
        # do the number of ss placeholders and pddl parameters match?
        if npddl_objects < len(ss_plhs):
            # if not, handle accordingly
            if 'because something does not fit' in video_label:
                if len(set(ss_plhs))>3:
                    print('ss placeholders',ss_plhs)
                    print('npddl_objects',npddl_objects)
                    print('!!!something is very very wrong!!!!!')
                    input()
                else:
                    ss_plhs = ss_plhs[:2]
                    assert npddl_objects == len(ss_plhs)
            elif 'touching (without moving) part of something' in video_label:
                if len(set(ss_plhs))>3:
                    print('ss placeholders',ss_plhs)
                    print('npddl_objects',npddl_objects)
                    print('!!!something is very very wrong!!!!!')
                    input()
                else:
                    ss_plhs = ss_plhs[:1]
                    assert npddl_objects == len(ss_plhs)
            elif 'tipping something with something in it over' in video_label:
                if len(set(ss_plhs))>3:
                    print('ss placeholders',ss_plhs)
                    print('npddl_objects',npddl_objects)
                    print('!!!something is very very wrong!!!!!')
                    input()
                else:
                    if len(set(ss_plhs)) == 2:
                        ss_plhs = sorted(ss_plhs,key=ss_plhs.count,reverse=True)
                        ss_plhs = ss_plhs[-2:][::-1]
                        assert npddl_objects == len(ss_plhs)
                    else:
                        # hoping for the best
                        ss_plhs = ss_plhs[:2]
            elif 'pretending to pour something out of something' in video_label:
                if len(set(ss_plhs))>3:
                    print('ss placeholders',ss_plhs)
                    print('npddl_objects',npddl_objects)
                    print('!!!something is very very wrong!!!!!')
                    input()
                else:
                    # hoping for the best
                    ss_plhs = ss_plhs[:2]
                    assert npddl_objects == len(ss_plhs)
            else:
                print('ss placeholders',ss_plhs)
                print('npddl_objects',npddl_objects)
                print('!!!something is very very wrong!!!!!')
                input()
        elif npddl_objects > len(ss_plhs):
            print('ss placeholders',ss_plhs)
            print('npddl_objects',npddl_objects)
            print('!!!something is very very wrong!!!!!')
            input()
        
        nth_frames = vid_nframes/NTH
        # iterate over the video frames (each is considered to be an image)
        # only the frames with bounding box annotations
        for frame in SE_ann[video]:

            fname_parts = frame['name'].split('/')
            iframe = int(fname_parts[1].split('.')[0])
            if iframe >= nth_frames and iframe <= (vid_nframes-nth_frames):
                continue

            # initialize the image_dict
            image_dict = {}
            image_dict['id'] = image_id
            fname_parts = frame['name'].split('/')
            image_dict['file_name'] = fname_parts[0]+'/'+'frame_{:010d}'.format(iframe)+'.png'
            image_dict['width'] = WIDTH
            image_dict['height'] = HEIGHT

            # get frame objects and add them to the object categories metadata
            frame_objects = frame['labels']
            for obj in frame_objects:

                obj_st_cat = obj['standard_category']
                if obj_st_cat != 'hand':
                    # skip objects that overshoot the number of pddl parameters
                    if not int(obj_st_cat) < npddl_objects:
                        continue
                else:
                    continue

                # substitute with the SS object label
                if obj['category'] != 'hand' and int(obj_st_cat) < npddl_objects:
                    obj['category'] = ss_plhs[int(obj_st_cat)]
                elif obj['category'] != 'hand':
                    print('code should never go here!')
                    input()
                    assert int(obj_st_cat) < npddl_objects

                SE_bbox = [v for v in obj['box2d'].values()]
                toplx = SE_bbox[0]
                toply = SE_bbox[2]
                W = SE_bbox[1] - SE_bbox[0]
                H = SE_bbox[3] - SE_bbox[2]

                if not void_video: 
                    # checking if the object should get pre or post conditions
                    if iframe < nth_frames:
                        obj_precs = [p for p in vid_precs if GPA_std2alpha[obj_st_cat] in p]
                    elif iframe > (vid_nframes-nth_frames): 
                        obj_precs = [p for p in vid_effcs if GPA_std2alpha[obj_st_cat] in p]
                    else:
                        print('something really wrong happened!!!!')
                        input()
                    # discard predicates we cannot describe
                    obj_precs = [p for p in obj_precs if '(not (=' not in p and 'visible' not in p]

                    # filter which conditions apply to the object
                    if obj_st_cat == 'hand':
                        continue
                    else:
                        precs = []
                        for p in obj_precs:
                            if p.count('?') > 1 and f'{GPA_std2alpha[obj_st_cat]})' in p:
                                continue
                            else:
                                p = p.replace(GPA_std2alpha[obj_st_cat],'')
                            precs.append(p)
                        obj_precs = precs
                    obj_precs = [p.replace(' )',')').replace('  ',' ') for p in obj_precs if '?' not in p]
                    obj_precs = list(set(obj_precs))
                    if len(obj_precs) == 0:
                        continue

                    for p in obj_precs:
                        if p not in SS_CLASSES:
                            SS_CLASSES[p] = obj_id
                            obj_id += 1
                        if p not in OBJECT_PREDICATES:
                            OBJECT_PREDICATES[p] = {}
                            OBJECT_PREDICATES[p]['name'] = p
                            OBJECT_PREDICATES[p]['instance_count'] = 1
                            OBJECT_PREDICATES[p]['id'] = SS_CLASSES[p]
                            OBJECT_PREDICATES[p]['supercategory'] = 'ATTRIBUTE'
                            SSdata['categories'].append({'id':OBJECT_PREDICATES[p]['id'], 'name':OBJECT_PREDICATES[p]['name'], 'supercategory':'OBJECT'})
                            #print('new predicate category added to SSdat:',SSdata['categories'][-1])
                            #input()
                        else:
                            OBJECT_PREDICATES[p]['instance_count'] += 1

                        # initialize predicate annotation and add to SSdata
                        obj_dict = {}
                        obj_dict['id'] = ann_id
                        obj_dict['image_id'] = image_id
                        obj_dict['category_id'] = OBJECT_PREDICATES[p]['id']
                        obj_dict['bbox'] = [toplx,toply,W,H]
                        obj_dict['segmentation'] = []
                        SSdata['annotations'].append(obj_dict)
                        ann_id += 1
                else:
                    print('void video leak!')
                    input()
            
            # add image_dict to SSdata
            SSdata['images'].append(image_dict)
            image_id += 1

    with open(f'SS_{f}.json', 'w') as fp:
        json.dump(SSdata, fp)
    ALL_OBJECT_CATEGORIES.append(copy.deepcopy(OBJECT_CATEGORIES))
    ALL_OBJECT_PREDICATES.append(copy.deepcopy(OBJECT_PREDICATES))
    #print('done!')
    #input()

    OBJECT_CATEGORIES = {}
    for D in ALL_OBJECT_CATEGORIES:
        for k,v in D.items():
            if k not in OBJECT_CATEGORIES:
                OBJECT_CATEGORIES[k] = {}
                OBJECT_CATEGORIES[k]['name'] = v['name']
                OBJECT_CATEGORIES[k]['instance_count'] = v['instance_count']
                OBJECT_CATEGORIES[k]['id'] = v['id']
                OBJECT_CATEGORIES[k]['supercategory'] = v['supercategory']
            else:
                OBJECT_CATEGORIES[k]['instance_count'] += v['instance_count']
    OBJECT_PREDICATES = {}
    for D in ALL_OBJECT_PREDICATES:
        for k,v in D.items():
            if k not in OBJECT_PREDICATES:
                OBJECT_PREDICATES[k] = {}
                OBJECT_PREDICATES[k]['name'] = v['name']
                OBJECT_PREDICATES[k]['instance_count'] = v['instance_count']
                OBJECT_PREDICATES[k]['id'] = v['id']
                OBJECT_PREDICATES[k]['supercategory'] = v['supercategory']
            else:
                OBJECT_PREDICATES[k]['instance_count'] += v['instance_count']


    OBJECT_PREDICATES = [v for v in OBJECT_PREDICATES.values()]
    OBJECT_CATEGORIES = [v for v in OBJECT_CATEGORIES.values()]

    SS_CATEGORIES = OBJECT_CATEGORIES + OBJECT_PREDICATES
    SS_PREDICATES = OBJECT_PREDICATES


    f = open("vlpart/data/datasets/ssv2_categories.py", "w")

    f.write(f"""
    SSV2_CATEGORIES_COUNT = {len(SS_CATEGORIES)}

    SSV2_CATEGORIES = {SS_CATEGORIES}
    """)

    f.close()
