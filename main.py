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
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import train_pose, test_pose
from models import build_model
import os
# import wandb
import torch.backends.cudnn as cudnn
from cfg import Config

#GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr.sh

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=True, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,            
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_keypoint', default=5, type=float,
                        help="L1 box coefficient in the matching cost") 

    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=2, type=float) 
    parser.add_argument('--keypoint_loss_coef', default=5, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--train_stage', default='pose')
    parser.add_argument('--dataset_file', default='FPHA')
    parser.add_argument('--coco_path', default='/mnt/hoseong', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./weights',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # parser.add_argument('--resume', default='./weights/paper_pose.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='./weights/FPHA_pose.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--img_size', default=(960, 540), type=tuple)
    parser.add_argument('--make_pickle', default=False, action='store_true')

    # for hand challenge
    parser.add_argument('--eval', default=False, action='store_true')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)
    cfg = Config(args)
    
    # wandb.config.update(args)
    # os.environ["CUDA_VISIBLE_DEVICES"]= '4'
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
    dataset_val[0]

    model, criterion = build_model(args, cfg)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

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
                                    collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                    pin_memory=True)
    # data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)
    
    if args.dataset_file == 'H2O':
        dataset_test = build_dataset(image_set='test', args=args)
        if args.distributed:
            if args.cache_mode:
                sampler_test = samplers.NodeDistributedSampler(dataset_test, shuffle=False)
            else:
                sampler_test = samplers.DistributedSampler(dataset_test, shuffle=False)
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
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
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        # if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        #     import copy
        #     p_groups = copy.deepcopy(optimizer.param_groups)
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     for pg, pg_old in zip(optimizer.param_groups, p_groups):
        #         pg['lr'] = pg_old['lr']
        #         pg['initial_lr'] = pg_old['initial_lr']
        #     print(optimizer.param_groups)
        #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #     # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
        #     args.override_resumed_lr_drop = True
        #     if args.override_resumed_lr_drop:
        #         print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
        #         lr_scheduler.step_size = args.lr_drop
        #         lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        #     lr_scheduler.step(lr_scheduler.last_epoch)
        #     args.start_epoch = checkpoint['epoch'] + 1

    # for i in range(40, 50):
    #     ckpt = torch.load(f'./weights/FPHA_aug45/{i}.pth')
    #     model.load_state_dict(ckpt['model'])
    #     val_stats = test_pose(model, criterion, data_loader_val, device, cfg)
    #     print(f"Val ||left : {val_stats['left']} || right : {val_stats['right']} || obj : {val_stats['obj']} || class : {val_stats['class_error']}")

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if not args.eval:
            if args.distributed:
                sampler_train.set_epoch(epoch)
            train_stats = train_pose(
                model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
            lr_scheduler.step()

            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, f'{args.output_dir}/{args.dataset_file}_aug45/{epoch}.pth')

        val_stats = test_pose(model, criterion, data_loader_val, device, cfg, vis=True)
        print(f"Val ||left : {val_stats['left']} || right : {val_stats['right']} || obj : {val_stats['obj']} || class : {val_stats['class_error']}")
        
        if args.dataset_file == 'H2O':
            test_stats = test_pose(model, criterion, data_loader_test, device, cfg)
            print(f"Test ||left : {test_stats['left']} || right : {test_stats['right']} || obj : {test_stats['obj']} || class : {test_stats['class_error']}")
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
