# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import os
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

import json
import wandb
from glob import glob
from cfg import Config
import util.misc as utils
import datasets.samplers as samplers
from torch.utils.data import DataLoader
from util.slconfig import SLConfig
from models.smoothnet import ArcticSmoother, SmoothCriterion

from models import build_model
from datasets import build_dataset
from arctic_tools.src.factory import collate_custom_fn as lstm_fn
from engine import train_pose, test_pose, train_smoothnet, test_smoothnet, train_dn, eval_dn, eval_coco
from util.settings import (
    get_general_args_parser, get_deformable_detr_args_parser, get_dino_arg_parser,
    load_resume, extract_epoch, set_training_scheduler, make_arctic_environments,
    set_dino_args
)


# main script
def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.modelname == 'dino':
        set_dino_args(args)

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

    def collate_test(dataset_train=None, dataset_val=None):
        if dataset_train is not None:
            test_train = [
                [dataset_train[13][0], dataset_train[13][1], dataset_train[13][2]],
                [dataset_train[1][0], dataset_train[1][1], dataset_train[1][2]],
                [dataset_train[2][0], dataset_train[2][1], dataset_train[2][2]],
                [dataset_train[3][0], dataset_train[3][1], dataset_train[3][2]]
            ]
            tt = collate_fn(test_train)
        if dataset_val is not None:
            test_val = [
                [dataset_val[13][0], dataset_val[13][1], dataset_val[13][2]],
                [dataset_val[1][0], dataset_val[1][1], dataset_val[1][2]],
                [dataset_val[2][0], dataset_val[2][1], dataset_val[2][2]],
                [dataset_val[3][0], dataset_val[3][1], dataset_val[3][2]]
            ]    
            tv = collate_fn(test_val)
    # collate_test(dataset_val=dataset_val)

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

    optimizer, lr_scheduler = set_training_scheduler(args, model_without_ddp)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    if args.resume:
        assert not args.resume_dir
        load_resume(args, model_without_ddp, args.resume)

    print("Start training")
    start_time = time.time()

    # for evaluation
    if args.eval:
        # train smooth net
        if args.train_smoothnet:
            smoother = ArcticSmoother(args.batch_size, args.window_size).to(device)
            WEIGHT_DICT = {
                "loss_left": 1000.0,
                "loss_right": 1000.0,
                "loss_obj": 1000.0,
            }
            smoother_criterion = SmoothCriterion(args.batch_size, args.window_size, WEIGHT_DICT)
            optimizer, lr_scheduler = set_training_scheduler(args, smoother, 0.001)            
            test_smoothnet(model, smoother, criterion, data_loader_val, device, cfg, args=args, vis=args.visualization)
            sys.exit(0)

        # multiple evaluation
        if args.resume_dir:
            assert not args.resume
            assert args.modelname == 'dino' and args.dataset_file == 'arctic', 'Not implemented yet.'
            resume_list = glob(op.join(args.resume_dir,'*.pth'))
            resume_list.sort(key=extract_epoch)

            for resume in resume_list:
                args.resume = resume
                load_resume(args, model_without_ddp, resume)
                print(f"\n{'='*10} current epoch :{extract_epoch(args.resume)} {'='*10}")
                eval_dn(model, cfg, data_loader_val, device, wo_class_error=False, args=args)
            sys.exit(0)

        # evaluation script
        wo_class_error = False
        if args.dataset_file == 'COCO':
            # If you want to evaluate coco dataset, replace None to postprocessor.
            eval_coco(model, criterion, None, data_loader_val, device, args.output_dir, wo_class_error=False, args=args)
        else:
            if args.modelname == 'dino':
                eval_dn(model, cfg, data_loader_val, device, wo_class_error=wo_class_error, args=args)
            else:
                test_pose(model, data_loader_val, device, cfg, args=args, vis=args.visualization)
        sys.exit(0)

    # for training
    else:
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                sampler_train.set_epoch(epoch)

            # train smoothnet
            if args.train_smoothnet:
                smoother = ArcticSmoother(args.batch_size, args.window_size).to(device)
                WEIGHT_DICT = {
                    "loss_left": 1000.0,
                    "loss_right": 1000.0,
                    "loss_obj": 1000.0,
                }
                smoother_criterion = SmoothCriterion(args.batch_size, args.window_size, WEIGHT_DICT)
                optimizer, lr_scheduler = set_training_scheduler(args, smoother, 0.001)

                train_smoothnet(model, smoother, smoother_criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, args=args, cfg=cfg)
                lr_scheduler.step()

                utils.save_on_master({
                    'model': smoother.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, f'{args.output_dir}/{epoch}.pth')

                test_smoothnet(model, smoother, criterion, data_loader_val, device, cfg, args=args, vis=args.visualization, epoch=epoch)

            # origin training
            else:
                # for dino
                if args.modelname == 'dino':
                    train_dn(
                        model, criterion, data_loader_train, optimizer, device, epoch,
                        args.clip_max_norm, wo_class_error=False, lr_scheduler=lr_scheduler, args=args,                 
                    )

                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, f'{args.output_dir}/{epoch}.pth')

                    eval_dn(model, cfg, data_loader_val, device, wo_class_error=False, args=args, vis=args.visualization, epoch=epoch)

                # for deformable detr
                else:
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
                    test_pose(model, data_loader_val, device, cfg, args=args, vis=args.visualization, epoch=epoch)
            
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    # get general parser
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_general_args_parser()])
    args = parser.parse_known_args()[0]

    # get model parser
    # if args.modelname == 'dn_detr':
        # parser = get_dn_detr_args_parser(parser)
    if args.modelname == 'dino':
        parser = get_dino_arg_parser(parser)
    elif args.modelname == 'deformable_detr':
        parser = get_deformable_detr_args_parser(parser)
    else:
        raise Exception('Please be specific model names.')
    args = parser.parse_known_args()[0]

    # get arctic parser
    if args.dataset_file == 'arctic':
        from arctic_tools.src.parsers.parser import construct_args
        args = construct_args(parser)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # check arctic env
    make_arctic_environments(args)
    from datasets import build_dataset
    import datasets.samplers as samplers
    from engine import train_pose, test_pose    

    main(args)