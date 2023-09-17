# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""

import os
import sys
import math
import wandb
import torch
import numpy as np
from tqdm import tqdm
from typing import Iterable
import matplotlib.pyplot as plt

import util.misc as utils
from util.settings import extract_epoch
from datasets.data_prefetcher import data_prefetcher
from datasets.arctic_prefetcher import data_prefetcher as arctic_prefetcher

from arctic_tools.visualizer import visualize_arctic_result
from arctic_tools.process import arctic_pre_process, prepare_data, measure_error, get_arctic_item, make_output
from util.tools import (
    extract_feature, visualize_assembly_result, eval_assembly_result, stat_round,
    create_loss_dict, create_arctic_score_dict, arctic_smoothing
)
from torch.cuda.amp import autocast
# os.environ["CUB_HOME"] = os.getcwd() + '/cub-1.10.0'

from datasets.coco_eval import CocoEvaluator

def to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, list):
        return [to_device(i, device) for i in item]
    elif isinstance(item, dict):
        return {k: to_device(v, device) for k,v in item.items()}
    else:
        raise NotImplementedError("Call Shilong if you use other containers! type: {}".format(type(item)))


def train_dn(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    # scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print(header)

    prefetcher = arctic_prefetcher(data_loader, device, prefetch=True)
    samples, targets, meta_info = prefetcher.next()
    pbar = tqdm(range(len(data_loader)))

    for _ in pbar:
        targets, meta_info = arctic_pre_process(args, targets, meta_info)

        # test_debug(targets, meta_info, B=19, h=400, w=300)
        # samples, targets, meta_info = prefetcher.next()
        # continue

        # with torch.cuda.amp.autocast(enabled=args.amp):
        if need_tgt_for_training:
            outputs = model(samples, targets=targets) 

            # if outputs['dn_meta']['output_known_lbs_bboxes']['pred_logits'].isnan().sum() != 0:
            #     outputs = model(samples, targets=targets) 

            loss_dict = criterion(outputs, targets, args, meta_info)
        else:
            raise Exception('Not implemented!')
            outputs = model(samples)
            data = prepare_data(args, outputs, targets, meta_info, cfg)
            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # for arctic
        for k, v in loss_dict.items():
            if len(v.shape) == 1:
                loss_dict[k] = v[0]

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            for k,v in (loss_dict_reduced.items()):
                print(f'{k} : {v.item()}')
            sys.exit(1)


        # amp backward function
        if args.amp:
            raise Exception('Not implemeted!')
            # optimizer.zero_grad()
            # scaler.scale(losses).backward()
            # if max_norm > 0:
            #     scaler.unscale_(optimizer)
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            # scaler.step(optimizer)
            # scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if args.debug:
            if _ == args.num_debug:
                print("BREAK!"*5)
                break

        pbar.set_postfix(
            create_loss_dict(
                loss_value, loss_dict_reduced_scaled,
                round_value=True, mode='small'
            )
        )
        samples, targets, meta_info = prefetcher.next()        


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    train_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    result = create_loss_dict(loss_value, train_stat, flag='train', mode='small')

    # for wandb
    if args is not None and args.wandb:
        if args.distributed:
            if utils.get_local_rank() != 0:
                return train_stat
        
        # save results
        save_dir = os.path.join(f'{args.output_dir}/loss.txt')
        epoch = extract_epoch(args.resume) if epoch is None else epoch
        with open(save_dir, 'a') as f:
            if args.test_viewpoint is not None:
                f.write(f"{'='*10} {args.test_viewpoint} {'='*10}\n")
            f.write(f"{'='*10} epoch : {epoch} {'='*10}\n\n")
            f.write(f"{'='*9} {args.val_batch_size}*{args.window_size}, {args.iter}iter {'='*9}\n")
            for key, val in train_stat.items():
                res = f'{key:35} : {round(val, 8)}\n'
                f.write(res)
                print(res, end='')
            f.write('\n\n')  

        # check dataset
        if args.dataset_file == 'arctic':
            wandb.log(result, step=epoch)
        elif args.dataset_file == 'AssemblyHands':
            wandb.log({
                'loss' : loss_value,
                'ce_loss' : train_stat['loss_ce'],
                'hand': train_stat['loss_hand_keypoint'], 
            }, step=epoch)    

    return train_stat


@torch.no_grad()
def eval_dn(model, cfg, data_loader, device, wo_class_error=False, args=None, vis=None, epoch=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    # set model
    model.eval()

    # set prefetcher
    prefetcher = arctic_prefetcher(data_loader, device, prefetch=True)
    samples, targets, meta_info = prefetcher.next()

    # set logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    print(header)

    # start test
    pbar = tqdm(range(len(data_loader)))
    for _ in pbar:
        targets, meta_info = arctic_pre_process(args, targets, meta_info)

        # implement & calc loss
        # with torch.cuda.amp.autocast(enabled=args.amp):
        if need_tgt_for_training:
            outputs = model(samples, targets=targets)
        else:
            outputs = model(samples)

        # vis or measure error
        data = prepare_data(args, outputs, targets, meta_info, cfg)
        if vis:
            visualize_arctic_result(args, data, 'pred')
        else:
            # smoothing
            if args.iter > 0:
                cnt = args.iter
                data.overwrite("pred.object.v.cam", arctic_smoothing(data["pred.object.v.cam"], cnt))
                data.overwrite("pred.mano.v3d.cam.r", arctic_smoothing(data["pred.mano.v3d.cam.r"], cnt))
                data.overwrite("pred.mano.v3d.cam.l", arctic_smoothing(data["pred.mano.v3d.cam.l"], cnt))

            # measure error
            try:
                stats = measure_error(data, args.eval_metrics)
            except:
                print('Fail to mesure the data of last iteration.')
                break
            
            # drop na
            for k,v in stats.items():
                not_non_idx = ~np.isnan(stats[k])
                replace_value = float(stats[k][not_non_idx].mean())
                # If all values are nan, drop that key.
                if replace_value != replace_value:
                    stats = stats.rm(k)
                else:
                    stats.overwrite(k, replace_value)
            
            # debug
            pbar.set_postfix(stat_round(**stats))
            metric_logger.update(**stats)

        if args.debug:
            if _ == args.num_debug:
                print("BREAK!"*5)
                break
        samples, targets, meta_info = prefetcher.next()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

    if args.distributed:
        if utils.get_local_rank() != 0:
            return stats

    # save results
    save_dir = os.path.join(f'{args.output_dir}/results.txt')
    epoch = extract_epoch(args.resume) if epoch is None else epoch

    with open(save_dir, 'a') as f:
        if args.test_viewpoint is not None:
            f.write(f"{'='*10} {args.test_viewpoint} {'='*10}\n")

        f.write(f"{'='*10} epoch : {epoch} {'='*10}\n\n")
        f.write(f"{'='*9} {args.val_batch_size}*{args.window_size}, {args.iter}iter {'='*9}\n")

        for key, val in stats.items():
            f.write(f'{key:30} : {val}\n')
        f.write('\n\n')

    # save wandb
    if args is not None and args.wandb:
        wandb.log(
            create_arctic_score_dict(stats), step=epoch
        )
    return stats


# just testing
def test_debug(targets, meta_info, B=0, h=224, w=224):
    from PIL import Image
    import cv2

    test = targets['keypoints'][B].view(-1, 21, 2)

    imgname = meta_info['imgname'][B]
    img = Image.open('/home/unist/Desktop/hdd/arctic/data/arctic_data/data/cropped_images/' + imgname)
    img = img.resize((w,h))
    img = np.array(img)

    color = [(255,0,0), (0,255,0), (0,0,255)]
    for i, t in enumerate(test):
        for j in range(21):
            x = int( t[j][0] * w )
            y = int( t[j][1] * h )
            cv2.line(img, (x, y), (x, y), color[i], 5)
    plt.imshow(img)

    # samples, targets, meta_info = prefetcher.next()
    # continue


def train_smoothnet(
        base_model, smoothnet, criterion, data_loader, optimizer, device, epoch, max_norm=0, args=None, cfg=None
    ):
    # set model and criterion
    base_model.eval()
    smoothnet.train()
    criterion.train()

    # set logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print(header)

    # prefetcher settings
    prefetcher = arctic_prefetcher(data_loader, device, prefetch=True)
    samples, targets, meta_info = prefetcher.next()
    pbar = tqdm(range(len(data_loader)))

    # start training
    for _ in pbar:
        targets, meta_info = arctic_pre_process(args, targets, meta_info)

        # Inference baseline model
        with torch.no_grad():
            outputs = base_model(samples)

            # select query
            base_out = get_arctic_item(outputs, cfg, args.device)
        
        #
        data = prepare_data(args, outputs, targets, meta_info, cfg).to(device)
        pred_vl = data["pred.mano.v3d.cam.l"]
        pred_vr = data["pred.mano.v3d.cam.r"]
        pred_vo = data["pred.object.v.cam"]

        # smoothing
        sm_l_v, sm_r_v, sm_o_v = smoothnet(pred_vl, pred_vr, pred_vo)
        data.overwrite("pred.mano.v3d.cam.l", sm_l_v)
        data.overwrite("pred.mano.v3d.cam.r", sm_r_v)
        data.overwrite("pred.object.v.cam", sm_o_v)
        
        # # post process
        # query_names = meta_info["query_names"]
        # K = meta_info["intrinsics"]
        # arctic_out = make_output(args, sm_root, sm_pose, sm_shape, sm_angle, query_names, K)
        # data = prepare_data(args, None, targets, meta_info, cfg, arctic_out)

        # calc losses
        loss_dict = criterion(args, data, meta_info)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # for arctic
        for k, v in loss_dict.items():
            if len(v.shape) == 1:
                loss_dict[k] = v[0]

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # loss check
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # back propagation
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(smoothnet.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(smoothnet.parameters(), max_norm)
        optimizer.step()

        # logger update
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        # for early stop
        if args.debug:
            if args.num_debug == _:
                break

        # for debug
        pbar.set_postfix(
            create_loss_dict(loss_value, loss_dict_reduced_scaled, round_value=True, mode='smoothnet')
        )
        samples, targets, meta_info = prefetcher.next()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    train_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    result = create_loss_dict(loss_value, train_stat, mode='smoothnet')
    print(result)

    # for wandb
    if args is not None and args.wandb:
        if args.distributed:
            if utils.get_local_rank() != 0:
                return train_stat
        # check dataset
        wandb.log(result, step=epoch)

    # end training process
    return train_stat


def test_smoothnet(base_model, smoothnet, criterion, data_loader, device, cfg, args=None, vis=False, epoch=None):
    # set model and criterion
    base_model.eval()
    smoothnet.eval()
    criterion.eval()

    # prefetcher settings
    prefetcher = arctic_prefetcher(data_loader, device, prefetch=True)
    samples, targets, meta_info = prefetcher.next()

    # set logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    pbar = tqdm(range(len(data_loader)))

    # start testing
    for _ in pbar:
        targets, meta_info = arctic_pre_process(args, targets, meta_info)

        with torch.no_grad():
            # Inference baseline model
            outputs = base_model(samples)

            # select query
            base_out = get_arctic_item(outputs, cfg, args.device)

            # # base output test
            # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!반드시 수정!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # t_root = [targets['mano.cam_t.wp.l'], targets['mano.cam_t.wp.r'], targets['object.cam_t.wp']]
            # t_pose = [targets['mano.pose.l'], targets['mano.pose.r']]
            # t_beta = [targets['mano.beta.l'], targets['mano.beta.r']]
            # t_obj = [targets["object.rot"], targets["object.radian"]]
            # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # query_names = meta_info["query_names"]
            # K = meta_info["intrinsics"]
            # arctic_out = make_output(args, base_out[0], base_out[1], base_out[2], t_obj, query_names, K)
            # base_data = prepare_data(args, None, targets, meta_info, cfg, arctic_out)
            # visualize_arctic_result(args, base_data, 'pred')
            # samples, targets, meta_info = prefetcher.next()
            # continue
            
            #
            data = prepare_data(args, outputs, targets, meta_info, cfg).to(device)
            pred_vl = data["pred.mano.v3d.cam.l"]
            pred_vr = data["pred.mano.v3d.cam.r"]
            pred_vo = data["pred.object.v.cam"]

            # smoothing
            try:
                sm_l_v, sm_r_v, sm_o_v = smoothnet(pred_vl, pred_vr, pred_vo)
            except:
                break
            data.overwrite("pred.mano.v3d.cam.l", sm_l_v)
            data.overwrite("pred.mano.v3d.cam.r", sm_r_v)
            data.overwrite("pred.object.v.cam", sm_o_v)

            # # post process
            # query_names = meta_info["query_names"]
            # K = meta_info["intrinsics"]
            # arctic_out = make_output(args, sm_root, sm_pose, sm_shape, sm_angle, query_names, K)
            # data = prepare_data(args, None, targets, meta_info, cfg, arctic_out)

            # vis results
            if vis:
                visualize_arctic_result(args, data, 'pred')

            # measure error
            else:
                stats = measure_error(data, args.eval_metrics)
                for k,v in stats.items():
                    not_non_idx = ~np.isnan(stats[k])
                    replace_value = float(stats[k][not_non_idx].mean())
                    # If all values are nan, drop that key.
                    if replace_value != replace_value:
                        stats = stats.rm(k)
                    else:
                        stats.overwrite(k, replace_value)

                pbar.set_postfix(stat_round(**stats))
                metric_logger.update(**stats)

            if args.debug == True:
                if args.num_debug == _:
                    break

        # next step
        samples, targets, meta_info = prefetcher.next()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # Only main process can reach out last line of this script.
    if args.distributed:
        if utils.get_local_rank() != 0:
            return stats
    
    # save results to txt
    save_dir = os.path.join(f'{args.output_dir}/results.txt')
    epoch = extract_epoch(args.resume) if epoch is None else epoch
    with open(save_dir, 'a') as f:
        if args.test_viewpoint is not None:
            f.write(f"{'='*10} {args.test_viewpoint} {'='*10}\n")
        f.write(f"{'='*10} epoch : {epoch} {'='*10}\n\n")
        for key, val in stats.items():
            f.write(f'{key:30} : {val}\n')
        f.write('\n\n')

    # save results to wandb
    if args is not None and args.wandb:
        if args.dataset_file == 'arctic':
            wandb.log(
                create_arctic_score_dict(stats), step=epoch
            )
        else:
            wandb.log(
                {
                    'mpjpe' : stats['mpjpe'],
                }, step=epoch
            )            
    return stats


def train_pose(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, args=None, cfg=None):
    # scaler = torch.cuda.amp.GradScaler(enabled=True)

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print(header)
    print_freq = 10

    # prefetcher settings
    if args.dataset_file == 'arctic':
        prefetcher = arctic_prefetcher(data_loader, device, prefetch=True)
        samples, targets, meta_info = prefetcher.next()
    else:
        prefetcher = data_prefetcher(data_loader, device, prefetch=True)
        samples, targets = prefetcher.next()
    pbar = tqdm(range(len(data_loader)))

    for _ in pbar:
        # not exist images
        if samples is None:
            samples, targets = prefetcher.next()
            continue

        # arctic pre process
        if args.dataset_file == 'arctic':
            targets, meta_info = arctic_pre_process(args, targets, meta_info)

        # for feature map extraction mode
        if args.extract:
            extract_feature(
                args, model, samples, targets, meta_info, data_loader, cfg, check_mode=True
            )

            # next samples
            if args.dataset_file == 'arctic':
                samples, targets, meta_info = prefetcher.next()
                continue
            else:
                samples, targets = prefetcher.next()
                continue

        # with torch.cuda.amp.autocast(enabled=True):
        # Training script begin from here
        outputs = model(samples)

        if args.dataset_file == 'arctic':
            # data = prepare_data(args, outputs, targets, meta_info, cfg)
            loss_dict = criterion(outputs, targets, args, meta_info)
        else:
            # check validation
            for i in range(len(targets)):
                target = targets[i]
                img_id = target['image_id'].item()
                label = [l.item()-1 for l in target['labels']]
                joint_valid = data_loader.dataset.coco.loadAnns(img_id)[0]['joint_valid']
                joint_valid = torch.stack([torch.tensor(joint_valid[:21]), torch.tensor(joint_valid[21:])]).type(torch.bool)[label]
                joint_valid = joint_valid.unsqueeze(-1).repeat(1,1,3)
                targets[i]['joint_valid'] = joint_valid
            loss_dict = criterion(outputs, targets)

        # calc losses
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # for arctic
        for k, v in loss_dict.items():
            if len(v.shape) == 1:
                loss_dict[k] = v[0]

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # loss check
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            for k,v in (loss_dict_reduced.items()):
                print(f'{k} : {v.item()}')
            sys.exit(1)

        # back propagation
        # if scaler is not None:
        #     optimizer.zero_grad()
        #     scaler.scale(losses).backward()
        #     if max_norm > 0:
        #         scaler.unscale_(optimizer)
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        #     scaler.step(optimizer)
        #     scaler.update()
        # else:        
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        # logger update
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        if args.dataset_file == 'AssemblyHands':
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        # for early stop
        if args.debug:
            if args.num_debug == _:
                break

        # for debug
        if args.dataset_file == 'arctic':
            pbar.set_postfix(
                create_loss_dict(
                    loss_value, loss_dict_reduced_scaled,
                    round_value=True, mode='small'
                )
            )
            samples, targets, meta_info = prefetcher.next()
        elif args.dataset_file == 'AssemblyHands':
            pbar.set_postfix({
                'loss' : loss_value,
                'ce_loss' : loss_dict_reduced_scaled['loss_ce'].item(),
                'hand': loss_dict_reduced_scaled['loss_hand_keypoint'].item(), 
            })
            samples, targets = prefetcher.next()

    if args.extract:
        return 0

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    train_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if args.dataset_file == 'arctic':
        result = create_loss_dict(loss_value, train_stat, flag='train', mode='small')

    # for wandb
    if args is not None and args.wandb:
        if args.distributed:
            if utils.get_local_rank() != 0:
                return train_stat
        
        # save results
        save_dir = os.path.join(f'{args.output_dir}/loss.txt')
        epoch = extract_epoch(args.resume) if epoch is None else epoch
        with open(save_dir, 'a') as f:
            if args.test_viewpoint is not None:
                f.write(f"{'='*10} {args.test_viewpoint} {'='*10}\n")
            f.write(f"{'='*10} epoch : {epoch} {'='*10}\n\n")
            f.write(f"{'='*9} {args.val_batch_size}*{args.window_size}, {args.iter}iter {'='*9}\n")
            for key, val in train_stat.items():
                res = f'{key:35} : {round(val, 8)}\n'
                f.write(res)
                print(res, end='')
            f.write('\n\n')            
        
        # check dataset
        if args.dataset_file == 'arctic':
            wandb.log(result, step=epoch)
        elif args.dataset_file == 'AssemblyHands':
            wandb.log({
                'loss' : loss_value,
                'ce_loss' : train_stat['loss_ce'],
                'hand': train_stat['loss_hand_keypoint'], 
            }, step=epoch)

    # end training process
    return train_stat


@torch.no_grad()
def test_pose(model, data_loader, device, cfg, args=None, vis=False, save_pickle=False, epoch=None):
    model.eval()

    if args.dataset_file == 'arctic':
        prefetcher = arctic_prefetcher(data_loader, device, prefetch=True)
        samples, targets, meta_info = prefetcher.next()
    else:
        prefetcher = data_prefetcher(data_loader, device, prefetch=True)
        samples, targets = prefetcher.next()

    metric_logger = utils.MetricLogger(delimiter="  ")
    pbar = tqdm(range(len(data_loader)))
    header = 'Test:'
    print(header)

    for _ in pbar:
        if args.dataset_file == 'arctic':
            targets, meta_info = arctic_pre_process(args, targets, meta_info)

        # for feature map extraction mode
        if args.extract:
            extract_feature(
                args, model, samples, targets, meta_info, data_loader, cfg
            )

            # next samples
            if args.dataset_file == 'arctic':
                samples, targets, meta_info = prefetcher.next()
                continue
            else:
                samples, targets = prefetcher.next()
                continue

        # Testing script begin from here
        # with torch.cuda.amp.autocast(enabled=True):
        outputs = model(samples)

        if args.dataset_file == 'arctic':
            data = prepare_data(args, outputs, targets, meta_info, cfg)

            # cnt = args.iter
            # data.overwrite("pred.object.v.cam", arctic_smoothing(data["pred.object.v.cam"], cnt))
            # data.overwrite("pred.mano.v3d.cam.r", arctic_smoothing(data["pred.mano.v3d.cam.r"], cnt))
            # data.overwrite("pred.mano.v3d.cam.l", arctic_smoothing(data["pred.mano.v3d.cam.l"], cnt))                    

        if args.visualization:
            # assert samples.tensors.shape[0] == 1
            if args.dataset_file == 'arctic':
                visualize_arctic_result(args, data, 'pred')
            elif args.dataset_file == 'AssemblyHands':
                visualize_assembly_result(args, cfg, outputs, targets, data_loader)
        else:
            if args.dataset_file == 'AssemblyHands':
                # measure error
                stats = eval_assembly_result(outputs, targets, cfg, data_loader)

                pbar.set_postfix({
                    'MPJPE': stats['mpjpe'],
                    })                    
            else:
                def test():
                    import arctic_tools.common.torch_utils as torch_utils
                    from arctic_tools.common.xdict import xdict
                    from copy import deepcopy
                    
                    origin = measure_error(data, args.eval_metrics)
                    origin_cdev = torch_utils.nanmean(torch.tensor(origin['cdev/ho']))

                    cnt = 20
                    test_data = deepcopy(data)
                    test_data.overwrite("pred.object.v.cam", arctic_smoothing(test_data["pred.object.v.cam"], cnt))
                    test_data.overwrite("pred.mano.v3d.cam.r", arctic_smoothing(test_data["pred.mano.v3d.cam.r"], cnt))
                    test_data.overwrite("pred.mano.v3d.cam.l", arctic_smoothing(test_data["pred.mano.v3d.cam.l"], cnt))

                    replace = measure_error(test_data, args.eval_metrics)
                    replace_cdev = torch_utils.nanmean(torch.tensor(replace['cdev/ho']))
                    print(replace_cdev)

                    visualize_arctic_result(args, test_data, 'pred')
                    samples, targets, meta_info = prefetcher.next()
                    # continue

                # measure error
                try:
                    stats = measure_error(data, args.eval_metrics)
                except:
                    print('Fail to mesure the data of last iteration.')
                    break
                
                for k,v in stats.items():
                    not_non_idx = ~np.isnan(stats[k])
                    replace_value = float(stats[k][not_non_idx].mean())
                    # If all values are nan, drop that key.
                    if replace_value != replace_value:
                        stats = stats.rm(k)
                    else:
                        stats.overwrite(k, replace_value)

                pbar.set_postfix(stat_round(**stats))
            metric_logger.update(**stats)

        if args.debug == True:
            if args.num_debug == _:
                break

        if args.dataset_file == 'arctic':
            samples, targets, meta_info = prefetcher.next()
        else:
            samples, targets = prefetcher.next()

    if args.extract:
        return 0

    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if args.distributed:
        if utils.get_local_rank() != 0:
            return stats
    
    save_dir = os.path.join(f'{args.output_dir}/results.txt')
    epoch = extract_epoch(args.resume) if epoch is None else epoch
    with open(save_dir, 'a') as f:
        if args.test_viewpoint is not None:
            f.write(f"{'='*10} {args.test_viewpoint} {'='*10}\n")
        f.write(f"{'='*10} epoch : {epoch} {'='*10}\n\n")
        f.write(f"{'='*9} {args.val_batch_size}*{args.window_size}, {args.iter}iter {'='*9}\n")
        for key, val in stats.items():
            f.write(f'{key:30} : {val}\n')
        f.write('\n\n')

    if args is not None and args.wandb:
        if args.dataset_file == 'arctic':
            wandb.log(
                create_arctic_score_dict(stats), step=epoch
            )
        else:
            wandb.log(
                {
                    'mpjpe' : stats['mpjpe'],
                }, step=epoch
            )            
    return stats


def eval_coco(model, criterion, postprocessors, data_loader, device, output_dir, wo_class_error=False, args=None, logger=None):
    assert postprocessors is not None, "If you want to evaluate coco dataset, replace None to postprocessors."

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(data_loader.dataset.coco, 'bbox', useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    _cnt = 0
    output_state_dict = {} # for debug only
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            # outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)
        
        if args.save_results:
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']


            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _ == args.num_debug:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator