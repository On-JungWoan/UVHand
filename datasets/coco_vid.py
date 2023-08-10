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
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection_vid as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
import os
import trimesh


class CocoDetection_vid(TvCocoDetection):
    def __init__(self, img_folder, ann_file, cache_mode=False, local_rank=0, local_size=1, mode=None, args=None):
        super(CocoDetection_vid, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size, mode=mode, args=args)
        
        self.data_list = []
    def __getitem__(self, idx):

        uvd_list, cam_list, _6D_list, label_list, mano_list, action_id, img_ids, data_path = super(CocoDetection_vid, self).__getitem__(idx)
        target = {'vid_id': torch.tensor(idx), 
            'uvd':uvd_list, 
            'cam':cam_list, 
            '6D':_6D_list, 
            'label':label_list, 
            'mano':mano_list, 
            'action':torch.tensor(int(action_id)),
            'img_ids' : torch.tensor(img_ids),
            'data_path': data_path
        }
        self.data_list.append(data_path)
        return uvd_list, [target]


def build_vid(image_set, args):
    root = Path(args.coco_path) / args.dataset_file
    assert root.exists(), f'provided COCO path {root} does not exist'

    if args.dataset_file == 'H2O':
        PATHS = {
            "train": (root , root /  'H2O_action_train.json'),
            "val": (root , root / 'H2O_action_val.json'),
            "test": (root, root / 'H2O_action_test.json')
        }
    elif args.dataset_file == 'FPHA':
        PATHS = {
            "train": (root , root /  'FPHA_subset_train.json'),
            "val": (root , root / 'FPHA_subset_val.json'),
        }
        
    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection_vid(img_folder, ann_file,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(), mode=image_set, args=args)
    return dataset