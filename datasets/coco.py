# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.utils.data
from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
import json

from arctic_tools.src.factory import fetch_dataloader
from datasets.arctic.build import fetch_dataloader as build_arctic

class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, dataset, transforms, cache_mode=False, local_rank=0, local_size=1, mode='train'):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self.etc_ann = None   
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(self.coco, dataset, self.etc_ann)
        self.mode = mode

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)

        if img is None:
            return None, target

        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        if img is None:
            return None, target

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = torch.cat((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

class ConvertCocoPolysToMask(object):
    def __init__(self, coco, dataset, etc_ann=None):
        self.coco = coco
        self.dataset = dataset

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]  #(min_x, min_y, w, h)
        # guard against no boxes via resizing
        if self.dataset == 'AssemblyHands':
            boxes = list(boxes[0].values())
            boxes = [b for b in boxes if b is not None]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        
        boxes[:, 2:] += boxes[:, :2]           # (min_x, min_y, max_x, max_y)
        boxes[:, 0::2].clamp_(min=0, max=w) 
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = None
        obj6D = None
        if self.dataset == 'AssemblyHands':
            name_to_idx = {'right':1, 'left':2}
            classes = [key for key, val in anno[0]['bbox'].items() if val is not None]
            classes = [name_to_idx[c] for c in classes]
        else:
            classes = [obj["category_id"] for obj in anno]

            obj6D = [obj["obj6Dpose"] for obj in anno]
            obj6D = torch.tensor(obj6D)
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.dataset == 'AssemblyHands':
            cam_param = torch.Tensor(sum(list(anno[0]['cam_param'].values()), []) + [0,0])
        else:
            cam_param = torch.Tensor(self.coco.loadImgs(image_id.item())[0]['cam_param'])

        keypoints = None
        if anno and "joint3d" in anno[0]:
            keypoints = [obj["joint3d"] for obj in anno]
            if self.dataset == 'AssemblyHands':
                keypoints = [keypoints[0]['right'], keypoints[0]['left']]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        if num_keypoints != len(classes):
            # print('==Not matching!==')
            assert len(classes) < num_keypoints
            try:
                keypoints = keypoints[classes.item()-1][None]
            except:
                return None, None

                 #max_y         min_y          max_x           min_x
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        if classes is not None:
            classes = classes[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        fx, fy, cx, cy, _, _ = cam_param

        # if self.dataset == 'AssemblyHands':
        #     assert len(keypoints.shape) == 3
        #     keypoints[keypoints<-1]=-1
        #     # for key in keypoints:
        #     #     for i in range(2):
        #     #         target = key[..., i]
        #     #         val = w if i == 0 else h
        #     #         # target[target > val] = val
        #     #         target[target < -1] = -1
        #     uvd = keypoints
        # else:
        uvd = torch.stack([
            cam2pixel(keypoints[i], (fx, fy), (cx, cy)) \
                if keypoints[i].sum()!=0 else \
            torch.zeros(21,3) \
                for i in range(len(classes)) 
        ])

        if self.dataset == 'FPHA':
            uvd[...,2] /= 1000
        target = {}
        target["boxes"] = boxes
        target["cam_param"] = cam_param
        target["image_id"] = image_id
        target["labels"] = classes
        target["keypoints"] = uvd

        # target['check'] = torch.tensor([num_keypoints, len(classes)])

        if obj6D is not None:
            target["obj6Dpose"] = obj6D

        # for conversion to coco api
        target["orig_size"] = torch.as_tensor([int(h), int(w)])

        return image, target


import os
def origin_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]
        

    if image_set == 'train':
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                normalize,
            ])

        if strong_aug:
            import datasets.sltransform as SLT
            
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                SLT.RandomSelectMulti([
                    SLT.RandomCrop(),
                    # SLT.Rotate(10),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),              
                # # for debug only  
                # SLT.RandomCrop(),
                # SLT.LightingNoise(),
                # SLT.AdjustBrightness(2),
                # SLT.AdjustContrast(2),
                # SLT.Rotate(10),
                normalize,
            ])
        
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set in ['val', 'test']:

        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.Compose([
                T.ResizeDebug((1280, 800)),
                normalize,
            ])   

        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])



    raise ValueError(f'unknown {image_set}')

def make_coco_transforms(image_set, img_size, make_pickle):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train' and make_pickle==False:
        return T.Compose([
            T.Resize(img_size),
            T.CollorJitter(),
            T.RandomRotation(45), #15
            normalize,
        ])
    else:
        return T.Compose([
            T.Resize(img_size),
            normalize,
        ])

def build(image_set, args):
    root = Path(args.coco_path) / args.dataset_file
    assert root.exists(), f'provided COCO path {root} does not exist'

    # ARCTIC
    if args.dataset_file == 'arctic':
        return fetch_dataloader(args, image_set, seq=args.seq)
    # AssemblyHands
    elif args.dataset_file == 'AssemblyHands':
        PATHS = {
            "train": (root , root / 'annotations/train.json'),
            "val": (root , root / 'annotations/val.json'),
        }
    # H2O
    elif args.dataset_file == 'H2O':
        PATHS = {
            "train": (root , root /  'H2O_pose_train.json'),
            "val": (root , root / 'H2O_pose_val.json'),
            "test": (root, root / 'H2O_pose_test.json')
        }
    # FPHA
    elif args.dataset_file == 'FPHA':
        PATHS = {
            "train": (root , root /  'FPHA_train.json'),
            "val": (root , root / 'FPHA_val.json'),
        }
    # COCO
    else:
        mode = 'instances'
        PATHS = {
            "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
            "train_reg": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
            "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
            "eval_debug": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
            "test": (root / "test2017", root / "annotations" / 'image_info_test-dev2017.json' ),
        }
        img_folder, ann_file = PATHS[image_set]
        from .coco_test import CocoDetection as DnCocoDectection
        dataset = DnCocoDectection(img_folder, ann_file,
                transforms=origin_coco_transforms(image_set, fix_size=False, strong_aug=False, args=args), 
                return_masks=args.masks, aux_target_hacks=None)
        return dataset

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, args.dataset_file, transforms=make_coco_transforms(image_set, args.img_size, args.make_pickle),
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(), mode=image_set)
    return dataset

