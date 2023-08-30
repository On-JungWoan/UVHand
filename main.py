# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import sys
sys.path = ["./arctic_tools"] + sys.path

import time
import torch
import random
import argparse
import datetime
import numpy as np
import os.path as op
from pathlib import Path
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

import wandb
from glob import glob
from cfg import Config
import util.misc as utils
import datasets.samplers as samplers
from torch.utils.data import DataLoader

from models import build_model
from datasets import build_dataset
from engine import train_pose, test_pose
from arctic_tools.src.factory import collate_custom_fn as lstm_fn
from util.settings import get_args_parser, load_resume, extract_epoch
#GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr.sh

# main script
def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.wandb:
        if args.distributed and utils.get_local_rank() != 0:
            pass
        else:
            wandb.init(
                project='2023-ICCV-hand-New',
                entity='jeongwanon'
            )
            wandb.config.update(args)

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)
    cfg = Config(args)
    
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

    if not args.eval:
        dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    model, criterion = build_model(args, cfg)
    model.to(device)
    model_without_ddp = model

    if args.wandb:
        if args.distributed:
            if utils.get_local_rank() == 0:
                wandb.watch(model_without_ddp)
        else:
            wandb.watch(model_without_ddp)


    if args.dataset_file == 'arctic':
        if args.method == 'arctic_lstm' and args.split_window:
                collate_fn=lstm_fn
        else:
            collate_fn=utils.collate_custom_fn
    else:
        collate_fn=utils.collate_fn

    # test_train = [
    #     [dataset_train[13][0], dataset_train[13][1], dataset_train[13][2]],
    #     [dataset_train[1][0], dataset_train[1][1], dataset_train[1][2]],
    #     [dataset_train[2][0], dataset_train[2][1], dataset_train[2][2]],
    #     [dataset_train[3][0], dataset_train[3][1], dataset_train[3][2]]
    # ]
    # test_val = [
    #     [dataset_val[13][0], dataset_val[13][1], dataset_val[13][2]],
    #     [dataset_val[1][0], dataset_val[1][1], dataset_val[1][2]],
    #     [dataset_val[2][0], dataset_val[2][1], dataset_val[2][2]],
    #     [dataset_val[3][0], dataset_val[3][1], dataset_val[3][2]]
    # ]    
    # collate_fn(test_train)
    # collate_fn(test_val)

    if args.distributed:
        if args.cache_mode:
            if not args.eval:
                sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            if not args.eval:
                sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        if not args.eval:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if not args.eval:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                    collate_fn=collate_fn, num_workers=args.num_workers,
                                    pin_memory=True)
    # data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
    data_loader_val = DataLoader(dataset_val, args.val_batch_size, sampler=sampler_val,
                                drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers,
                                pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        assert not args.resume_dir
        load_resume(args, model_without_ddp, args.resume)

    print("Start training")
    start_time = time.time()

    # for evaluation
    if args.eval:
        if args.resume_dir:
            assert not args.resume
            resume_list = glob(op.join(args.resume_dir,'*'))
            resume_list.sort(key=extract_epoch)

            for resume in resume_list:
                args.resume = resume
                load_resume(model_without_ddp, resume)
                print(f"\n{'='*10} current epoch :{extract_epoch(args.resume)} {'='*10}")
                test_pose(model, criterion, data_loader_val, device, cfg, args=args, vis=args.visualization)
        else:
            test_pose(model, criterion, data_loader_val, device, cfg, args=args, vis=args.visualization)
        sys.exit(0)
        
    # for training
    else:
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                sampler_train.set_epoch(epoch)

            # collate_fn(
            #     data_loader_train.dataset[0] + data_loader_train.dataset[1] + data_loader_train.dataset[2] + data_loader_train.dataset[3]
            # )

            # train
            train_pose(
                model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, args, cfg=cfg
            )
            lr_scheduler.step()

            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, f'{args.output_dir}/{epoch}.pth')

            # evaluate
            test_pose(model, criterion, data_loader_val, device, cfg, args=args, vis=args.visualization, epoch=epoch)
            
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_known_args()[0]

    if args.dataset_file == 'arctic':
        from arctic_tools.src.parsers.parser import construct_args
        args = construct_args(parser)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)