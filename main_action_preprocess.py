# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import train_contact, test_contact
from models import build_model
import os
import wandb
import torch.backends.cudnn as cudnn
from models.vivit import ViViT
from manopth.manolayer import ManoLayer
import pickle
from copy import deepcopy
from collections import defaultdict
from cfg import Config
#GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr_action.sh

# wandb.init(project="contact_action", entity="hoseong", dir='/HDD/hoseong')
# wandb.run.name = 'vivit_64_random_contact'

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--lr_drop', default=60, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # dataset parameters
    parser.add_argument('--train_stage', default='action')
    parser.add_argument('--dataset_file', default='FPHA')
    parser.add_argument('--coco_path', default='/mnt/hoseong', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--num_frame', default=16, type=int)

    parser.add_argument('--output_dir', default='./weights',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # parser.add_argument('--resume', default='./weights/36.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)

    cfg = Config(args)
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    
    idx2obj = {v:k for k, v in cfg.obj2idx.items()}
    GT_obj_vertices_dict = {}
    GT_3D_bbox_dict = {}
    for i in range(1,cfg.hand_idx[0]):
        with open(os.path.join(args.coco_path, args.dataset_file, 'obj_pkl', f'{idx2obj[i]}_2000.pkl'), 'rb') as f:
            vertices = pickle.load(f)
            GT_obj_vertices_dict[i] = vertices
        with open(os.path.join(args.coco_path, args.dataset_file, 'obj_pkl', f'{idx2obj[i]}_bbox.pkl'), 'rb') as f:
            bbox = pickle.load(f)
            GT_3D_bbox_dict[i] = bbox
            
    _mano_root = 'mano/models'
    mano_left = ManoLayer(flat_hand_mean=True,
                    side="left",
                    mano_root=_mano_root,
                    use_pca=False,
                    root_rot_mode='axisang',
                    joint_rot_mode='axisang').to(device)

    mano_right = ManoLayer(flat_hand_mean=True,
                    side="right",
                    mano_root=_mano_root,
                    use_pca=False,
                    root_rot_mode='axisang',
                    joint_rot_mode='axisang').to(device)
    
    idx2obj = {v:k for k, v in cfg.obj2idx.items()}

    criterion = nn.CrossEntropyLoss()
    temporal_model = ViViT(num_classes=cfg.num_action_classes, num_frames=args.num_frame, dim=cfg.IA_dim, dataset=args.dataset_file)
    temporal_model.to(device)
    
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, num_workers=args.num_workers,
                                 pin_memory=True)

    if args.dataset_file == 'H2O':
        dataset_test = build_dataset(image_set='test', args=args)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_test = DataLoader(dataset_test, 1, sampler=sampler_test,
                                    drop_last=False, num_workers=args.num_workers,
                                    pin_memory=True)

    if args.sgd:
        optimizer = torch.optim.SGD(temporal_model.parameters(), lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(temporal_model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    output_dir = Path(args.output_dir)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = temporal_model.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
                   
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_contact(
            temporal_model, mano_left, mano_right, deepcopy(GT_3D_bbox_dict), deepcopy(GT_obj_vertices_dict), criterion, data_loader_train, optimizer, device, epoch, cfg)
        lr_scheduler.step()
        val_stats = test_contact(temporal_model, mano_left, mano_right, deepcopy(GT_3D_bbox_dict), deepcopy(GT_obj_vertices_dict), criterion, data_loader_val, optimizer, device, epoch, cfg, save_confusion = False)
        
        if args.dataset_file == 'H2O':
            test_stats = test_contact(temporal_model, mano_left, mano_right, deepcopy(GT_3D_bbox_dict), deepcopy(GT_obj_vertices_dict), criterion, data_loader_test, optimizer, device, epoch, cfg, save_json=False)
        
        utils.save_on_master({
                    'model': temporal_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, output_dir / f'FPHA_action/{epoch}.pth')


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
