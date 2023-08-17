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
from collections import defaultdict
import math
import os
import sys
from typing import Iterable
from cv2 import KeyPoint

import torch
import util.misc as utils
from util.misc import NestedTensor
from datasets.data_prefetcher import data_prefetcher
from datasets.arctic_prefetcher import data_prefetcher as arctic_prefetcher
from tqdm import tqdm
import numpy as np
import copy
from scipy.spatial import procrustes
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
# os.environ["CUB_HOME"] = os.getcwd() + '/cub-1.10.0'
from pytorch3d.ops.knn import knn_points
from AIK import AIK_torch as AIK
import AIK.AIK_config as AIK_config 
import pickle
from manopth.manolayer import ManoLayer
import trimesh
import json
import wandb

from arctic_tools.common.xdict import xdict
from arctic_tools.process import arctic_pre_process, prepare_data, measure_error
from arctic_tools.visualizer import visualize_arctic_result

def make_line(cv_img, img_points, idx_1, idx_2, color, line_thickness=2):
    if -1 not in tuple(img_points[idx_1][:-1]):
        if -1 not in tuple(img_points[idx_2][:-1]):
            cv2.line(cv_img, tuple(img_points[idx_1][:-1]), tuple(
                img_points[idx_2][:-1]), color, line_thickness)    

def visualize(cv_img, img_points, mode='left'):
    if mode == 'left':
        color = (255,0,0)
    else:
        color = (0,0,255)
    
    make_line(cv_img, img_points, 0, 1, color, line_thickness=2)
    make_line(cv_img, img_points, 1, 2, color, line_thickness=2)
    make_line(cv_img, img_points, 2, 3, color, line_thickness=2)

    make_line(cv_img, img_points, 4, 5, color, line_thickness=2)
    make_line(cv_img, img_points, 5, 6, color, line_thickness=2)
    make_line(cv_img, img_points, 6, 7, color, line_thickness=2)

    make_line(cv_img, img_points, 8, 9, color, line_thickness=2)
    make_line(cv_img, img_points, 9, 10, color, line_thickness=2)
    make_line(cv_img, img_points, 10, 11, color, line_thickness=2)

    make_line(cv_img, img_points, 12, 13, color, line_thickness=2)
    make_line(cv_img, img_points, 13, 14, color, line_thickness=2)
    make_line(cv_img, img_points, 14, 15, color, line_thickness=2)

    make_line(cv_img, img_points, 16, 17, color, line_thickness=2)
    make_line(cv_img, img_points, 17, 18, color, line_thickness=2)
    make_line(cv_img, img_points, 18, 19, color, line_thickness=2)

    make_line(cv_img, img_points, 20, 3, color, line_thickness=2)
    make_line(cv_img, img_points, 20, 7, color, line_thickness=2)
    make_line(cv_img, img_points, 20, 11, color, line_thickness=2)
    make_line(cv_img, img_points, 20, 15, color, line_thickness=2)
    make_line(cv_img, img_points, 20, 19, color, line_thickness=2)

    # plt.imshow(cv_img)
    return cv_img

def visualize_obj(cv_img, img_points):
    cv2.line(cv_img, tuple(img_points[1][:-1]), tuple(
        img_points[2][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[2][:-1]), tuple(
        img_points[3][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[3][:-1]), tuple(
        img_points[4][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[4][:-1]), tuple(
        img_points[1][:-1]), (0, 255, 0), 5)

    cv2.line(cv_img, tuple(img_points[1][:-1]), tuple(
        img_points[5][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[2][:-1]), tuple(
        img_points[6][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[3][:-1]), tuple(
        img_points[7][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[4][:-1]), tuple(
        img_points[8][:-1]), (0, 255, 0), 5)

    cv2.line(cv_img, tuple(img_points[5][:-1]), tuple(
        img_points[6][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[6][:-1]), tuple(
        img_points[7][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[7][:-1]), tuple(
        img_points[8][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[8][:-1]), tuple(
        img_points[5][:-1]), (0, 255, 0), 5)

    return cv_img

def get_NN(src_xyz, trg_xyz, k=1):
    '''
    :param src_xyz: [B, N1, 3]
    :param trg_xyz: [B, N2, 3]
    :return: nn_dists, nn_dix: all [B, 3000] tensor for NN distance and index in N2
    '''
    B = src_xyz.size(0)
    src_lengths = torch.full(
        (src_xyz.shape[0],), src_xyz.shape[1], dtype=torch.int64, device=src_xyz.device
    )  # [B], N for each num
    trg_lengths = torch.full(
        (trg_xyz.shape[0],), trg_xyz.shape[1], dtype=torch.int64, device=trg_xyz.device
    )
    src_nn = knn_points(src_xyz, trg_xyz, lengths1=src_lengths, lengths2=trg_lengths, K=k)  # [dists, idx]
    nn_dists = src_nn.dists ## (x-x')**2 + (y-y')**2
    nn_idx = src_nn.idx
    # nn_dists = src_nn.dists[..., 0] ## (x-x')**2 + (y-y')**2
    # nn_idx = src_nn.idx[..., 0]
    return nn_dists#, nn_idx

def get_pseudo_cmap(nn_dists):
    '''
    calculate pseudo contactmap: 0~3cm mapped into value 1~0
    :param nn_dists: object nn distance [B, N] or [N,] in meter**2
    :return: pseudo contactmap [B,N] or [N,] range in [0,1]
    '''
    # nn_dists = 100.0 * torch.sqrt(nn_dists)  # turn into center-meter
    nn_dists = torch.sqrt(nn_dists) / 10.0  # turn into center-meter
    cmap = 1.0 - 2 * (torch.sigmoid(nn_dists*2) -0.5)
    return cmap

def rigid_transform_3D_numpy(A, B):
    batch, n, dim = A.shape
    # tmp_A = A.detach().cpu().numpy()
    # tmp_B = B.detach().cpu().numpy()
    tmp_A = A.copy()
    tmp_B = B.copy()
    centroid_A = np.mean(tmp_A, axis = 1)
    centroid_B = np.mean(tmp_B, axis = 1)
    H = np.matmul((tmp_A - centroid_A[:,None]).transpose(0,2,1), tmp_B - centroid_B[:,None]) / n
    U, s, V = np.linalg.svd(H)
    R = np.matmul(V.transpose(0,2,1), U.transpose(0, 2, 1))

    negative_det = np.linalg.det(R) < 0
    s[negative_det, -1] = -s[negative_det, -1]
    V[negative_det, :, 2] = -V[negative_det, :, 2]
    R[negative_det] = np.matmul(V[negative_det].transpose(0,2,1), U[negative_det].transpose(0, 2, 1))

    varP = np.var(tmp_A, axis=1).sum(-1)
    c = 1/varP * np.sum(s, axis=-1) 

    t = -np.matmul(c[:,None,None]*R, centroid_A[...,None])[...,-1] + centroid_B
    return c, R, t

def vis(data_loader, targets, FPHA=False):
    filename = data_loader.dataset.coco.loadImgs(targets[0]['image_id'][0].item())[0]['file_name']
    if FPHA:
        filepath = data_loader.dataset.root / 'Video_files'/ filename
    else:
        filepath = data_loader.dataset.root / filename
    cv_img = np.array(Image.open(filepath))
    img_points = targets[0]['keypoints'][0].cpu().detach().numpy().astype(np.int32)
    color = (0,0,255)
    line_thickness = 2
    cv2.line(cv_img, tuple(img_points[1][:-1]), tuple(
        img_points[2][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[2][:-1]), tuple(
        img_points[3][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[3][:-1]), tuple(
        img_points[4][:-1]), color, line_thickness)

    cv2.line(cv_img, tuple(img_points[5][:-1]), tuple(
        img_points[6][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[6][:-1]), tuple(
        img_points[7][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[7][:-1]), tuple(
        img_points[8][:-1]), color, line_thickness)

    cv2.line(cv_img, tuple(img_points[9][:-1]), tuple(
        img_points[10][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[10][:-1]), tuple(
        img_points[11][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[11][:-1]), tuple(
        img_points[12][:-1]), color, line_thickness)

    cv2.line(cv_img, tuple(img_points[13][:-1]), tuple(
        img_points[14][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[14][:-1]), tuple(
        img_points[15][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[15][:-1]), tuple(
        img_points[16][:-1]), color, line_thickness)

    cv2.line(cv_img, tuple(img_points[17][:-1]), tuple(
        img_points[18][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[18][:-1]), tuple(
        img_points[19][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[19][:-1]), tuple(
        img_points[20][:-1]), color, line_thickness)

    cv2.line(cv_img, tuple(img_points[0][:-1]), tuple(
        img_points[1][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[0][:-1]), tuple(
        img_points[5][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[0][:-1]), tuple(
        img_points[9][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[0][:-1]), tuple(
        img_points[13][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[0][:-1]), tuple(
        img_points[17][:-1]), color, line_thickness)

    return cv_img

def keep_valid(outputs, is_valid):
    outputs['pred_logits'] = outputs['pred_logits'][is_valid]
    outputs['pred_mano_params'][0] = outputs['pred_mano_params'][0][is_valid]
    outputs['pred_mano_params'][1] = outputs['pred_mano_params'][1][is_valid]
    outputs['pred_obj_params'][0] = outputs['pred_obj_params'][0][is_valid]
    outputs['pred_obj_params'][1] = outputs['pred_obj_params'][1][is_valid]
    outputs['pred_cams'][0] = outputs['pred_cams'][0][is_valid]
    outputs['pred_cams'][1] = outputs['pred_cams'][1][is_valid]    
    for idx, aux in enumerate(outputs['aux_outputs']):
        outputs['aux_outputs'][idx]['pred_logits'] = aux['pred_logits'][is_valid]
        outputs['aux_outputs'][idx]['pred_mano_params'][0] = aux['pred_mano_params'][0][is_valid]
        outputs['aux_outputs'][idx]['pred_mano_params'][1] = aux['pred_mano_params'][1][is_valid]
        outputs['aux_outputs'][idx]['pred_obj_params'][0] = aux['pred_obj_params'][0][is_valid]
        outputs['aux_outputs'][idx]['pred_obj_params'][1] = aux['pred_obj_params'][1][is_valid]
        outputs['aux_outputs'][idx]['pred_cams'][0] = aux['pred_cams'][0][is_valid]
        outputs['aux_outputs'][idx]['pred_cams'][1] = aux['pred_cams'][1][is_valid]      
    return outputs


def train_pose(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, args=None, cfg=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print(header)
    print_freq = 10

    if args.dataset_file == 'arctic':
        prefetcher = arctic_prefetcher(data_loader, device, prefetch=True)
        samples, targets, meta_info = prefetcher.next()
    else:
        prefetcher = data_prefetcher(data_loader, device, prefetch=True)
        samples, targets = prefetcher.next()

    pbar = tqdm(range(len(data_loader)))

    for _ in pbar:
        if args.dataset_file == 'arctic':
            targets, meta_info = arctic_pre_process(args, targets, meta_info)
        outputs = model(samples)

        # check validation
        is_valid = targets['is_valid'].type(torch.bool)
        for k,v in targets.items():
            if k == 'labels':
                targets[k] = [v for idx, v in enumerate(targets[k]) if is_valid[idx] == True]
            else:
                targets[k] = v[is_valid]
        for k,v in meta_info.items():
            if k in ['imgname', 'query_names']:
                meta_info[k] = [v for idx, v in enumerate(meta_info[k]) if is_valid[idx] == True]
            elif 'mano.faces' in k:
                continue
            else:
                meta_info[k] = v[is_valid]                
        outputs = keep_valid(outputs, is_valid)
        data = prepare_data(args, outputs, targets, meta_info, cfg)

        # calc loss
        loss_dict = criterion(args, outputs, targets, meta_info, data)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

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
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        pbar.set_postfix({
            'loss' : loss_value,
            'ce_loss' : loss_dict_reduced_scaled['loss_ce'].item(),
            'CDev' : loss_dict_reduced_scaled['loss/cd'].item(),
            'loss_mano' : round(
                loss_dict_reduced_scaled["loss/mano/pose/r"].item() + \
                loss_dict_reduced_scaled["loss/mano/beta/r"].item() + \
                loss_dict_reduced_scaled["loss/mano/pose/l"].item() + \
                loss_dict_reduced_scaled["loss/mano/beta/l"].item(), 2
            ),
            'loss_rot' : round(
                loss_dict_reduced_scaled["loss/object/radian"].item() + \
                loss_dict_reduced_scaled["loss/object/rot"].item(), 2
            ),
            'loss_transl' : round(
                loss_dict_reduced_scaled["loss/mano/transl/l"].item() + \
                loss_dict_reduced_scaled["loss/object/transl"].item(), 2
            ),
            'loss_kp' : round(
                loss_dict_reduced_scaled["loss/mano/kp2d/r"].item() + \
                loss_dict_reduced_scaled["loss/mano/kp3d/r"].item() + \
                loss_dict_reduced_scaled["loss/mano/kp2d/l"].item() + \
                loss_dict_reduced_scaled["loss/mano/kp3d/l"].item() + \
                loss_dict_reduced_scaled["loss/object/kp2d"].item() + \
                loss_dict_reduced_scaled["loss/object/kp3d"].item(), 2
            ),
            'loss_cam' : round(
                loss_dict_reduced_scaled["loss/mano/cam_t/r"].item() + \
                loss_dict_reduced_scaled["loss/mano/cam_t/l"].item() + \
                loss_dict_reduced_scaled["loss/object/cam_t"].item(), 2
            ),            
            })

        # print(f'{_} : {utils.get_local_rank()} done')
        if args.debug:
            if args.num_debug == _:
                break
        samples, targets, meta_info = prefetcher.next()       

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    train_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    if args is not None and args.wandb:
        if args.distributed:
            if utils.get_local_rank() != 0:
                return train_stat
        wandb.log({
            'loss' : loss_value,
            'ce_loss' : train_stat['loss_ce'],
            'loss_CDev' : train_stat['loss/cd'],
            'loss_mano' : (
                train_stat["loss/mano/pose/r"] + \
                train_stat["loss/mano/beta/r"] + \
                train_stat["loss/mano/pose/l"] + \
                train_stat["loss/mano/beta/l"]
            ),
            'loss_rot' : (
                train_stat["loss/object/radian"] + \
                train_stat["loss/object/rot"]
            ),
            'loss_transl' : (
                train_stat["loss/mano/transl/l"] + \
                train_stat["loss/object/transl"]
            ),
            'loss_kp' : (
                train_stat["loss/mano/kp2d/r"] + \
                train_stat["loss/mano/kp3d/r"] + \
                train_stat["loss/mano/kp2d/l"] + \
                train_stat["loss/mano/kp3d/l"] + \
                train_stat["loss/object/kp2d"] + \
                train_stat["loss/object/kp3d"]
            ),
            'loss_cam' : (
                train_stat["loss/mano/cam_t/r"] + \
                train_stat["loss/mano/cam_t/l"] + \
                train_stat["loss/object/cam_t"]
            )
        }, step=epoch)
    return train_stat


def cam2pixel(cam_coord, f, c):
    x = cam_coord[..., 0] / (cam_coord[..., 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[..., 1] / (cam_coord[..., 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[..., 2]
    try:
        img_coord = np.concatenate((x[...,None], y[...,None], z[...,None]), -1)
    except:
        img_coord = torch.cat((x[...,None], y[...,None], z[...,None]), -1)
    return img_coord

def pixel2cam(pixel_coord, f, c, T_=None):
    x = (pixel_coord[..., 0] - c[0]) / f[0] * pixel_coord[..., 2]
    y = (pixel_coord[..., 1] - c[1]) / f[1] * pixel_coord[..., 2]
    z = pixel_coord[..., 2]
    try:
        cam_coord = np.concatenate((x[...,None], y[...,None], z[...,None]), -1)
    except:
        cam_coord = torch.cat((x[...,None], y[...,None], z[...,None]), -1)
        
    if T_ is not None: # MANO space와 scale과 wrist를 맞추고자
        # New
        # if pixel_coord.shape[1] == 1:
        #     T_ = T_[1][None]
        # else:
        #     T_ = torch.stack(T_)
        ratio = torch.linalg.norm(T_[:,9] - T_[:,0], dim=-1) / torch.linalg.norm(cam_coord[:,:,9] - cam_coord[:,:,0], dim=-1)
        cam_coord = cam_coord * ratio[:,:,None,None]  # template, m
        cam_coord = cam_coord - cam_coord[:, :, :1] + T_[:,:1]
    return cam_coord

def test_pose(model, criterion, data_loader, device, cfg, args=None, vis=False, save_pickle=False, epoch=None):
    model.eval()
    criterion.eval()

    if args.dataset_file == 'arctic':
        prefetcher = arctic_prefetcher(data_loader, device, prefetch=True)
        samples, targets, meta_info = prefetcher.next()
    else:
        prefetcher = data_prefetcher(data_loader, device, prefetch=True)
        samples, targets = prefetcher.next()

    metric_logger = utils.MetricLogger(delimiter="  ")
    pbar = tqdm(range(len(data_loader)))

    for _ in pbar:
        if args.dataset_file == 'arctic':
            targets, meta_info = arctic_pre_process(args, targets, meta_info)

        with torch.no_grad():
            outputs = model(samples.to(device))
            
            # # check validation
            # is_valid = targets['is_valid'].type(torch.bool)
            # for k,v in targets.items():
            #     if k == 'labels':
            #         targets[k] = [v for idx, v in enumerate(targets[k]) if is_valid[idx] == True]
            #     else:
            #         targets[k] = v[is_valid]
            # outputs = keep_valid(outputs, is_valid)

            # prepare data
            data = prepare_data(args, outputs, targets, meta_info, cfg)

            if args.visualization:
                assert samples.tensors.shape[0] == 1
                visualize_arctic_result(args, data, 'pred')
            else:
                # measure error
                assert samples.tensors.shape[0] != 1
                stats = measure_error(data, args.eval_metrics)
                for k,v in stats.items():
                    not_non_idx = ~np.isnan(stats[k])
                    stats.overwrite(k, float(stats[k][not_non_idx].mean()))

                pbar.set_postfix({
                    'CDev':round(stats['cdev/ho'],2),
                    'MRRPE_rl/ro':f"{round(stats['mrrpe/r/l'],2)} / {round(stats['mrrpe/r/o'],2)}",
                    'MPJPE': round(stats['mpjpe/ra/h'],2),
                    'AAE': round(stats['aae'],2),
                    'S_R_0.05': round(stats['success_rate/0.05'],2),
                    })
                metric_logger.update(**stats)

            if args.debug == True:
                if args.num_debug == _:
                    break
        samples, targets, meta_info = prefetcher.next()

    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    save_dir = os.path.join(f'exps/{args.dataset_file}/results.txt')
    if utils.get_local_rank() == 0:
        with open(save_dir, 'a') as f:
            if args.test_viewpoint is not None:
                f.write(f"{'='*10} {args.test_viewpoint} {'='*10}\n")
            f.write(f"{'='*10} epoch : {epoch} {'='*10}\n\n")
            for key, val in stats.items():
                f.write(f'{key:30} : {val}\n')
            f.write('\n\n')

    if args is not None and args.wandb:
        if args.distributed:
            if utils.get_local_rank() != 0:
                return stats
        wandb.log(
            {
                'score_CDev' : stats['cdev/ho'],
                'score_MRRPE_rl': stats['mrrpe/r/l'],
                'score_MRRPE_ro' : stats['mrrpe/r/o'],
                'score_MPJPE' : stats['mpjpe/ra/h'],
                'score_AAE' : stats['aae'],
                'score_S_R_0.05' : stats['success_rate/0.05'],
            }, step=epoch
        )
    return stats


def train_contact(temporal_model, mano_left, mano_right, GT_3D_bbox_dict, GT_obj_vertices_dict, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, cfg=None):

    temporal_model.train()
    header = 'Epoch: [{}]'.format(epoch)
    print(header)

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    pbar = tqdm(range(len(data_loader)))

    fx,fy,cx,cy = cfg.cam_param
    for i, _ in enumerate(pbar):
        cam = targets[0]['cam'][0,:,0]
        obj_6D = targets[0]['6D'][0,:,0]
        label = targets[0]['label'][0,:,0]
        mano_param = targets[0]['mano'][0,:,0]
        action_label = targets[0]['action']
        dataset = 'H2O' if len(cfg.hand_idx) ==2 else 'FPHA'
        num_frame = cam.shape[0]
        obj_label = label[:,:-2].sum(0).argmax(dim=-1).item()
        R = obj_6D[:,:9].reshape(-1,3,3).to(torch.float32)
        t = obj_6D[:,9:].to(torch.float32)
        obj_vertices = torch.tensor(GT_obj_vertices_dict[obj_label], dtype=torch.float32)[None].repeat(R.shape[0], 1, 1).cuda()
        obj_vertices = torch.matmul(R, obj_vertices.permute(0,2,1)*1000).permute(0,2,1) + t[:,None]

        hand_cam = cam[:,:-63].reshape(num_frame, -1,21,3)
        
        opt_tensor_shape = torch.zeros(cam.shape[0], 10).to(cam.device)
        
        MANO_LAYER= [mano_left, mano_right] if dataset == 'H2O' else [mano_right]
        mano_results = [mano_layer(mano_param[:,48*i:48*(i+1)], opt_tensor_shape) for i, mano_layer in enumerate(MANO_LAYER)]
        hand_verts = torch.stack([m[0] for m in mano_results], dim=1)
        j3d_recon = torch.stack([m[1] for m in mano_results], dim=1)
        hand_verts = hand_verts - j3d_recon[:,:,:1] + hand_cam[:,:,:1]
        
        obj_nn_dist_affordance = get_NN(obj_vertices.to(torch.float32), hand_verts.reshape(num_frame,-1,3).to(torch.float32))
        hand_nn_dist_affordance = torch.stack([get_NN(hand_verts[:,idx].to(torch.float32), obj_vertices.to(torch.float32)) for idx in range(hand_verts.shape[1])], dim=1)
        obj_cmap_affordance = get_pseudo_cmap(obj_nn_dist_affordance)
        hand_cmap_affordance = torch.stack([get_pseudo_cmap(hand_nn_dist_affordance[:,idx]) for idx in range(hand_verts.shape[1])], dim=1)

        ##################### vis #############
        # idx = 0
        # v_color = np.array([[v.item()*255, 255-v.item()*255, 0] for v in obj_cmap_affordance[idx]], dtype=np.int64)
        # obj_mesh = trimesh.Trimesh(vertices=obj_vertices.detach().cpu().numpy()[idx], vertex_colors=v_color)
        # # left_hand = trimesh.Trimesh(vertices=hand_verts.detach().cpu().numpy()[idx,0], faces=(mano_left.th_faces).detach().cpu().numpy())
        # right_hand = trimesh.Trimesh(vertices=hand_verts.detach().cpu().numpy()[idx,-1], faces=(mano_right.th_faces).detach().cpu().numpy())
        # # obj_mesh = trimesh.Trimesh(vertices=obj_vertices.detach().cpu().numpy()[0])
        # save_path = './'
        # # trimesh.exchange.export.export_mesh(left_hand,f'{save_path}_left.obj')
        # trimesh.exchange.export.export_mesh(right_hand,f'{save_path}_right.obj')
        # trimesh.exchange.export.export_mesh(obj_mesh,f'{save_path}_obj.obj')
        #########################################
        ############### all mesh (dim=308) ###########
        hand_info = torch.cat([hand_verts/1000, hand_cmap_affordance], dim=-1).to(torch.float32)
        obj_info = torch.cat([obj_vertices/1000, obj_cmap_affordance], dim=-1).to(torch.float32)

        outputs = temporal_model(hand_info.detach(), obj_info.detach(), label[:,1:cfg.hand_idx[0]].detach())
        # outputs = temporal_model(left_info.reshape(64,-1), right_info.reshape(64,-1), obj_info.reshape(64,-1), label[:,1:]) ## h+o ##########################################3
        if torch.isnan(outputs.sum()):
            samples, targets = prefetcher.next()
            continue

        loss = criterion(outputs, action_label)
        if torch.isnan(loss):
            print('this')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({
            'ce_loss' : loss.item(),
            })
    
        samples, targets = prefetcher.next()

    return 0

# def test_contact(temporal_model, mano_left, mano_right, GT_3D_bbox_dict, GT_obj_vertices_dict, criterion: torch.nn.Module,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, cfg=None, save_json=None, save_confusion=None):
#     temporal_model.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Epoch: [{}]'.format(epoch)
#     print(header)

#     prefetcher = data_prefetcher(data_loader, device, prefetch=True)
#     samples, targets = prefetcher.next()
#     pbar = tqdm(range(len(data_loader)))

#     cnt = 0
    
#     preds = []
#     gts = []
#     json_action_dict = defaultdict(int)
#     fx,fy,cx,cy = cfg.cam_param
#     for i, _ in enumerate(pbar):
#         cam = targets[0]['cam'][0,:,0]
#         obj_6D = targets[0]['6D'][0,:,0]
#         label = targets[0]['label'][0,:,0]
#         mano_param = targets[0]['mano'][0,:,0]
#         action_label = targets[0]['action']
#         dataset = 'H2O' if len(cfg.hand_idx) ==2 else 'FPHA'
#         num_frame = cam.shape[0]
#         obj_label = label[:,:-2].sum(0).argmax(dim=-1).item()
#         R = obj_6D[:,:9].reshape(-1,3,3).to(torch.float32)
#         t = obj_6D[:,9:].to(torch.float32)
#         obj_vertices = torch.tensor(GT_obj_vertices_dict[obj_label], dtype=torch.float32)[None].repeat(R.shape[0], 1, 1).cuda()
#         obj_vertices = torch.matmul(R, obj_vertices.permute(0,2,1)*1000).permute(0,2,1) + t[:,None]

#         hand_cam = cam[:,:-63].reshape(num_frame, -1,21,3)
        
#         opt_tensor_shape = torch.zeros(cam.shape[0], 10).to(cam.device)
        
#         MANO_LAYER= [mano_left, mano_right] if dataset == 'H2O' else [mano_right]
#         mano_results = [mano_layer(mano_param[:,48*i:48*(i+1)], opt_tensor_shape) for i, mano_layer in enumerate(MANO_LAYER)]
#         hand_verts = torch.stack([m[0] for m in mano_results], dim=1)
#         j3d_recon = torch.stack([m[1] for m in mano_results], dim=1)
#         hand_verts = hand_verts - j3d_recon[:,:,:1] + hand_cam[:,:,:1]
        
#         obj_nn_dist_affordance = get_NN(obj_vertices.to(torch.float32), hand_verts.reshape(num_frame,-1,3).to(torch.float32))
#         hand_nn_dist_affordance = torch.stack([get_NN(hand_verts[:,idx].to(torch.float32), obj_vertices.to(torch.float32)) for idx in range(hand_verts.shape[1])], dim=1)
#         obj_cmap_affordance = get_pseudo_cmap(obj_nn_dist_affordance)
#         hand_cmap_affordance = torch.stack([get_pseudo_cmap(hand_nn_dist_affordance[:,idx]) for idx in range(hand_verts.shape[1])], dim=1)

#         ##################### vis #############
#         '''
#         idx = 32
#         v_color = np.array([[v.item()*255, 255-v.item()*255, 0] for v in obj_cmap_affordance[idx]], dtype=np.int64)
#         obj_mesh = trimesh.Trimesh(vertices=obj_vertices.detach().cpu().numpy()[idx], vertex_colors=v_color)
#         left_hand = trimesh.Trimesh(vertices=hand_verts.detach().cpu().numpy()[idx,0], faces=(mano_left.th_faces).detach().cpu().numpy())
#         right_hand = trimesh.Trimesh(vertices=hand_verts.detach().cpu().numpy()[idx,1], faces=(mano_right.th_faces).detach().cpu().numpy())
#         # obj_mesh = trimesh.Trimesh(vertices=obj_vertices.detach().cpu().numpy()[0])
#         save_path = './'
#         trimesh.exchange.export.export_mesh(left_hand,f'{save_path}_left.obj')
#         trimesh.exchange.export.export_mesh(right_hand,f'{save_path}_right.obj')
#         trimesh.exchange.export.export_mesh(obj_mesh,f'{save_path}_obj.obj')
#         '''
#         #########################################
#         ############### all mesh (dim=308) ###########
#         hand_info = torch.cat([hand_verts/1000, hand_cmap_affordance], dim=-1).to(torch.float32)
#         obj_info = torch.cat([obj_vertices/1000, obj_cmap_affordance], dim=-1).to(torch.float32)

#         with torch.no_grad():
#             outputs = temporal_model(hand_info.detach(), obj_info.detach(), label[:,1:cfg.hand_idx[0]].detach())

#         if outputs[0].argmax() == action_label[0]:
#             cnt+=1
#         preds.append(outputs[0].argmax().item())
#         gts.append(action_label[0].item())
#         metric_logger.update(**{'action_acc':cnt})
#             # reduce losses over all GPUs for logging purposes
#         json_action_dict[str(i+1)] = outputs[0].argmax().item()
#         samples, targets = prefetcher.next()
#     # gather the stats from all processes

#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", cnt/len(data_loader.dataset))
    
#     if save_json :
#         with open('action_labels.json', 'w') as f:
#             json.dump(json_action_dict, f)
#     if save_confusion:
#         cf_mat = confusion_matrix(preds, gts)
#         df_cm = pd.DataFrame(cf_mat, index = [i for i in interaction_label],
#                     columns = [i for i in interaction_label])
#         plt.figure(figsize = (19,15))
#         heatmap = sn.heatmap(df_cm, annot=True, annot_kws={'fontsize' : 20})
#         plt.savefig(f'./confusion/{epoch}.png')
        
#     return cnt/len(data_loader.dataset)

# def old_test_pose(model, criterion, data_loader, device, cfg, args=None, vis=False, save_pickle=False):
    
#     model.eval()
#     criterion.eval()
#     dataset = args.dataset_file

#     ## old script ##
#     def trash():
#         pass
#         # if dataset == 'arctic' and args.visualization == True:
#         #     from arctic_tools.extract_predicts import main
#         #     main(args, model, data_loader)    

#         # try:
#         #     idx2obj = {v:k for k, v in cfg.obj2idx.items()}
#         #     GT_obj_vertices_dict = {}
#         #     GT_3D_bbox_dict = {}        
#         #     for i in range(1,cfg.hand_idx[0]):
#         #         with open(os.path.join(data_loader.dataset.root, 'obj_pkl', f'{idx2obj[i]}_2000.pkl'), 'rb') as f:
#         #             vertices = pickle.load(f)
#         #             GT_obj_vertices_dict[i] = vertices
#         #         with open(os.path.join(data_loader.dataset.root, 'obj_pkl', f'{idx2obj[i]}_bbox.pkl'), 'rb') as f:
#         #             bbox = pickle.load(f)
#         #             GT_3D_bbox_dict[i] = bbox
#         # except:
#         #     dataset = 'AssemblyHands'
#         #     print('Not exist obj pkl')

#         # _mano_root = 'mano/models'
#         # mano_left = ManoLayer(flat_hand_mean=True,
#         #                 side="left",
#         #                 mano_root=_mano_root,
#         #                 use_pca=False,
#         #                 root_rot_mode='axisang',
#         #                 joint_rot_mode='axisang').to(device)

#         # mano_right = ManoLayer(flat_hand_mean=True,
#         #                 side="right",
#         #                 mano_root=_mano_root,
#         #                 use_pca=False,
#         #                 root_rot_mode='axisang',
#         #                 joint_rot_mode='axisang').to(device)
#     ## old script ##

#     if args.dataset_file == 'arctic':
#         prefetcher = arctic_prefetcher(data_loader, device, prefetch=True)
#         samples, targets, meta_info = prefetcher.next()
#     else:
#         prefetcher = data_prefetcher(data_loader, device, prefetch=True)
#         samples, targets = prefetcher.next()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     pbar = tqdm(range(len(data_loader)))

#     for _ in pbar:
#         ## old script ##
#         def trash():
#             pass
#             # samples, targets = prefetcher.next()
            
#             # try:
#             #     gt_keypoints = [t['keypoints'] for t in targets]
#             # except:
#             #     print('no gts')
#             #     continue

#             # if 'labels' in targets[0].keys():
#             #     gt_labels = [t['labels'].detach().cpu().numpy() for t in targets]

#             # try:
#             #     filename = data_loader.dataset.coco.loadImgs(targets[0]['image_id'][0].item())[0]['file_name']
#             # except:
#             #     filename = meta[0]['imgname']
            
#             # if args.test_viewpoint is not None:
#             #     if args.test_viewpoint != '/'.join(filename.split('/')[:-1]):
#             #         continue

#             # if vis:
#             #     assert data_loader.batch_size == 1  
#             #     if args.dataset_file=='arctic':
#             #         filepath = os.path.join(args.coco_path, args.dataset_file) + filename[1:]
#             #     elif dataset == 'H2O' or dataset == 'AssemblyHands':
#             #         filepath = data_loader.dataset.root / filename
#             #     else:
#             #         filepath = data_loader.dataset.root / 'Video_files'/ filename
#             #     source_img = np.array(Image.open(filepath))

#             # if os.path.exists(os.path.join(f'./pickle/{dataset}_aug45/{data_loader.dataset.mode}', f'{filename[:-4]}_data.pkl')):
#             #     samples, targets = prefetcher.next()
#             #     # continue

#             # if filename != 'ego_images_rectified/val/nusar-2021_action_both_9081-c11b_9081_user_id_2021-02-12_161433/HMC_21176623_mono10bit/006667.jpg':
#             #     continue
#         ## old script ##

#         if args.dataset_file == 'arctic':
#             targets, meta_info = arctic_pre_process(args, targets, meta_info)

#         with torch.no_grad():
#             outputs = model(samples.to(device))
            
#             # check validation
#             is_valid = targets['is_valid'].type(torch.bool)
#             for k,v in targets.items():
#                 if k == 'labels':
#                     targets[k] = [v for idx, v in enumerate(targets[k]) if is_valid[idx] == True]
#                 else:
#                     targets[k] = v[is_valid]
#             outputs = keep_valid(outputs, is_valid)

#             # prepare data
#             data = prepare_data(args, outputs, targets, meta_info, cfg)

#             ## old script ##
#             def trash():
#                 # calc loss
#                 loss_dict = criterion(outputs, targets)
#                 loss_dict_reduced = utils.reduce_dict(loss_dict)
#                 weight_dict = criterion.weight_dict
#                 losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

#                 # reduce losses over all GPUs for logging purposes
#                 loss_dict_reduced = utils.reduce_dict(loss_dict)
#                 loss_dict_reduced_unscaled = {f'{k}_unscaled': v
#                                             for k, v in loss_dict_reduced.items()}
#                 loss_dict_reduced_scaled = {k: v * weight_dict[k]
#                                             for k, v in loss_dict_reduced.items() if k in weight_dict}
#                 losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

#                 loss_value = losses_reduced_scaled.item()
#                 metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
#                 ####################################################

#                 # model output
#                 # out_logits, pred_keypoints, pred_obj_keypoints = outputs['pred_logits'], outputs['pred_keypoints'], outputs['pred_obj_keypoints']
#                 out_logits, out_mano_pose, out_mano_beta = outputs['pred_logits'], outputs['pred_manoparams'][0], outputs['pred_manoparams'][1]

#                 prob = out_logits.sigmoid()
#                 B, num_queries, num_classes = prob.shape

#                 # query index select
#                 best_score = torch.zeros(B).to(device)
#                 # if dataset != 'AssemblyHands':
#                 obj_idx = torch.zeros(B).to(device).to(torch.long)
#                 for i in range(1, cfg.hand_idx[0]):
#                     score, idx = torch.max(prob[:,:,i], dim=-1)
#                     obj_idx[best_score < score] = idx[best_score < score]
#                     best_score[best_score < score] = score[best_score < score]

#                 left_hand_idx = []
#                 right_hand_idx = []
#                 for i in cfg.hand_idx:
#                     hand_idx.append(torch.argmax(prob[:,:,i], dim=-1)) 
#                 hand_idx = torch.stack(hand_idx, dim=-1)   
#                 if dataset != 'AssemblyHands':
#                     keep = torch.cat([hand_idx, obj_idx[:,None]], dim=-1)
#                 else:
#                     keep = hand_idx
#                 hand_kp = torch.gather(pred_keypoints, 1, hand_idx.unsqueeze(-1).repeat(1,1,63)).reshape(B, -1 ,21, 3)
#                 obj_kp = torch.gather(pred_obj_keypoints, 1, obj_idx.unsqueeze(1).unsqueeze(1).repeat(1,1,63)).reshape(B, 21, 3)

#                 continue

#                 im_h, im_w, _ = source_img.shape
#                 hand_kp = targets[0]['keypoints'][1] * 1000
#                 visualize(source_img, hand_kp.detach().cpu().numpy().astype(np.int32), 'left')

#                 orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
#                 im_h, im_w = orig_target_sizes[:,0], orig_target_sizes[:,1]
#                 target_sizes = torch.cat([im_w.unsqueeze(-1), im_h.unsqueeze(-1)], dim=-1)
#                 target_sizes =target_sizes.cuda()

#                 labels = torch.gather(out_logits, 1, keep.unsqueeze(2).repeat(1,1,num_classes)).softmax(dim=-1)
#                 hand_kp[...,:2] *=  target_sizes.unsqueeze(1).unsqueeze(1); hand_kp[...,2] *= 1000
#                 obj_kp[...,:2] *=  target_sizes.unsqueeze(1); obj_kp[...,2] *= 1000
#                 key_points = torch.cat([hand_kp, obj_kp.unsqueeze(1)], dim=1)
#                 key_points = hand_kp
                
#                 if args.debug:
#                     if vis:
#                         batch, js, _, _ = key_points.shape
#                         for b in range(batch):
#                             for j in range(js):
#                                 pred_kp = key_points[b][j]

#                                 target_keys = targets[0]['keypoints']
#                                 target_keys[...,:2] *=  target_sizes.unsqueeze(1)
#                                 target_keys = target_keys[0]
#                                 if j ==0:
#                                     # gt = visualize(source_img, target_keys.detach().cpu().numpy().astype(np.int32), 'left')
#                                     pred = visualize(source_img, pred_kp.detach().cpu().numpy().astype(np.int32), 'left')
#                                 elif j == 1:
#                                     source_img = visualize(source_img, pred_kp.detach().cpu().numpy().astype(np.int32), 'right')
#                                 else:
#                                     source_img = visualize_obj(source_img, pred_kp.detach().cpu().numpy().astype(np.int32))
#                     continue

#                 # measure
#                 if dataset != 'AssemblyHands':
#                     tmp = []
#                     for gt_label in gt_labels:
#                         tmp.append([i for i in cfg.hand_idx if i in gt_label])
#                     gt_labels = tmp

#                 for i, batch in enumerate(gt_labels):
#                     cam_fx, cam_fy, cam_cx, cam_cy, _, _ = targets[i]['cam_param']
#                     for k, label in enumerate(batch):
#                         if dataset == 'H2O':
#                             if label == cfg.hand_idx[0]: j=0
#                             elif label == cfg.hand_idx[1]: j=1
#                             else: j=2
#                         else:
#                             if label == cfg.hand_idx[0]: j=0
#                             else: j=1
                                
#                         is_correct_class = int(labels[i][j].argmax().item() == gt_labels[i][k])
#                         pred_kp = key_points[i][j]

#                         x, y = target_sizes[0].detach().cpu().numpy()
#                         gt_scaled_keypoints = gt_keypoints[i][k] * torch.tensor([x, y, 1000]).cuda()
#                         gt_joint_cam = pixel2cam(gt_scaled_keypoints, (cam_fx.item(), cam_fy.item()), (cam_cx.item(), cam_cy.item()))

#                         # uvd to xyz
#                         if dataset == 'AssemblyHands':
#                             pred_kp[gt_scaled_keypoints==-1] = -1
#                             pred_kp[..., 2] = 1000
#                         pred_joint_cam = pixel2cam(pred_kp, (cam_fx.item(), cam_fy.item()), (cam_cx.item(), cam_cy.item()))

#                         if args.eval_method=='EPE':
#                             gt_relative = gt_scaled_keypoints[:,2:] - gt_scaled_keypoints[0,2:]
#                             pred_relative = pred_kp[:,2:] - pred_kp[0,2:]
                            
#                             xy_epe = torch.mean(torch.norm(gt_scaled_keypoints[:,:2] - pred_kp[:,:2], dim=-1))
#                             z_epe = torch.mean(torch.norm(gt_scaled_keypoints[:,2:] - pred_kp[:,2:], dim=-1))
#                             relative_depth_error = torch.mean(torch.norm(gt_relative - pred_relative, dim=-1))
            
#                             ###################################################################################
#                             # if j==2:
#                             #     pred_joint_cam = rigid_align(world_objcoord[0,:,:3], pred_joint_cam/1000)*1000
#                             ###################################################################################

#                             error = torch.mean(torch.norm(gt_joint_cam - pred_joint_cam, dim=-1))

#                         elif args.eval_method=='MPJPE':
#                             error = torch.sqrt(((pred_joint_cam - gt_joint_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

#                         # for visualization
#                         if dataset == 'FPHA': j+=1
#                         if vis:
#                             if j ==0:
#                                 source_img = visualize(source_img, pred_kp.detach().cpu().numpy().astype(np.int32), 'left')
#                             elif j == 1:
#                                 source_img = visualize(source_img, pred_kp.detach().cpu().numpy().astype(np.int32), 'right')
#                             else:
#                                 source_img = visualize_obj(source_img, pred_kp.detach().cpu().numpy().astype(np.int32))
#                         if j==1:
#                             metric_logger.update(**{'left': float(error)})
#                             # metric_logger.update(**{'uv_error': float(xy_epe)})
#                             # metric_logger.update(**{'d_error': float(z_epe)})
#                             # metric_logger.update(**{'relative_d_error': float(relative_depth_error)})
#                         elif j==0:
#                             metric_logger.update(**{'right': float(error)})
#                             # metric_logger.update(**{'uv_error': float(xy_epe)})
#                             # metric_logger.update(**{'d_error': float(z_epe)})
#                             # metric_logger.update(**{'relative_d_error': float(relative_depth_error)})
#                         else:
#                             metric_logger.update(**{'obj': float(error)})
#                             # metric_logger.update(**{'obj_uv_error': float(xy_epe)})
#                             # metric_logger.update(**{'obj_d_error': float(z_epe)})
#                             # metric_logger.update(**{'obj_relative_d_error': float(relative_depth_error)})
#                         metric_logger.update(**{'class_error':is_correct_class})
                    
#                 pbar.set_postfix({
#                     'left' : metric_logger.meters['left'].global_avg,
#                     'right' : metric_logger.meters['right'].global_avg,
#                     'obj' : metric_logger.meters['obj'].global_avg,
#                     'class_error' : metric_logger.meters['class_error'].global_avg,
#                     })

#                 if vis or save_pickle:
#                     assert data_loader.batch_size == 1
#                     save_path = os.path.join(f'./pickle/{dataset}_aug45/{data_loader.dataset.mode}', filename)
#                     # obj_label = labels[:,-1,:cfg.hand_idx[0]].argmax(-1)
#                     # GT_3D_bbox = GT_3D_bbox_dict[obj_label.item()][None]
#                     # pred_obj_cam = pixel2cam(obj_kp,  (cam_fx.item(), cam_fy.item()), (cam_cx.item(), cam_cy.item())).detach().cpu().numpy()
#                     # c, R, t = rigid_transform_3D_numpy(GT_3D_bbox*1000, pred_obj_cam)
#                     # c = torch.from_numpy(c).cuda(); R = torch.from_numpy(R).cuda(); t = torch.from_numpy(t).cuda()

#                     T_keypoints_left, T_keypoints_right = AIK_config.T_keypoints_left.cuda(), AIK_config.T_keypoints_right.cuda()
#                     T_ = torch.stack([T_keypoints_left, T_keypoints_right]) if hand_kp.shape[1] == 2 else T_keypoints_right[None]
#                     hand_cam_align = pixel2cam(hand_kp, (cam_fx.item(),cam_fy.item()), (cam_cx.item(),cam_cy.item()), T_)

#                     pose_params = [AIK.adaptive_IK(t, hand_cam_align[:,i]) for i, t in enumerate(T_)]            
#                     pose_params = torch.cat(pose_params, dim=-1)
                    
#                     if save_pickle:
#                         all_uvd = key_points.reshape(1, -1)
#                         all_cam = pixel2cam(key_points, (cam_fx.item(),cam_fy.item()), (cam_cx.item(),cam_cy.item())).reshape(1, -1)
#                         # obj_6D = torch.cat([R.reshape(-1,9), t], dim=-1)
#                         label_prob = labels[:,-1]

#                         # data={'uvd':all_uvd.detach().cpu().numpy(), 'cam':all_cam.detach().cpu().numpy(), '6D':obj_6D.detach().cpu().numpy(), 'label':label_prob.detach().cpu().numpy(), 'mano':pose_params.detach().cpu().numpy()}
#                         data={'uvd':all_uvd.detach().cpu().numpy(), 'cam':all_cam.detach().cpu().numpy(), 'label':label_prob.detach().cpu().numpy(), 'mano':pose_params.detach().cpu().numpy()}
                        
#                         if not os.path.exists(os.path.dirname(save_path)):
#                             os.makedirs(os.path.dirname(save_path))     
#                         with open(f'{save_path[:-4]}_data.pkl', 'wb') as f:
#                             pickle.dump(data, f)
                
#                 if vis:    
#                     ################# 2D vis #####################
#                     img_path = os.path.join(args.output_dir, filename)
#                     if not os.path.exists(os.path.dirname(img_path)):
#                         os.makedirs(os.path.dirname(img_path))
#                     cv2.imwrite(img_path, source_img[...,::-1])
#                     ###############################################
#                     ###### contact vis #####
#                     save_contact_vis_path = os.path.join(f'./contact_vis/{dataset}', filename)
#                     opt_tensor_shape = torch.zeros(prob.shape[0], 10).to(prob.device)
#                     MANO_LAYER= [mano_left, mano_right] if hand_kp.shape[1] == 2 else [mano_right]

#                     mano_results = [mano_layer(pose_params[:,48*i:48*(i+1)], opt_tensor_shape) for i, mano_layer in enumerate(MANO_LAYER)]
#                     hand_verts = torch.stack([m[0] for m in mano_results], dim=1)
#                     j3d_recon = torch.stack([m[1] for m in mano_results], dim=1)

#                     hand_cam = pixel2cam(hand_kp, (cam_fx.item(),cam_fy.item()), (cam_cx.item(),cam_cy.item()))
#                     hand_verts = hand_verts - j3d_recon[:,:,:1] + hand_cam[:,:,:1]

#                     # obj_name = idx2obj[obj_label.item()]
#                     # if dataset=='H2O':
#                     #     obj_mesh = trimesh.load(f'{cfg.object_model_path}/{obj_name}/{obj_name}.obj')
#                     # else:
#                     #     obj_mesh = trimesh.load(f'{cfg.object_model_path}/{obj_name}_model/{obj_name}_model.ply')
#                     # obj_mesh.vertices = (torch.matmul(R[0].detach().cpu().to(torch.float32), torch.tensor(obj_mesh.vertices, dtype=torch.float32).permute(1,0)*1000).permute(1,0) + t[0,None].detach().cpu()).numpy()
#                     # obj_vertices = torch.tensor(obj_mesh.vertices)[None].repeat(labels.shape[0], 1, 1).to(torch.float32).cuda()
                    
#                     # obj_nn_dist_affordance = get_NN(obj_vertices.to(torch.float32), hand_verts.reshape(1,-1,3).to(torch.float32))
#                     # hand_nn_dist_affordance = torch.stack([get_NN(hand_verts[:,idx].to(torch.float32), obj_vertices.to(torch.float32)) for idx in range(hand_verts.shape[1])], dim=1)
#                     # hand_nn_dist_affordance = torch.stack([get_NN(hand_verts[:,idx].to(torch.float32)) for idx in range(hand_verts.shape[1])], dim=1)
#                     # obj_cmap_affordance = get_pseudo_cmap(obj_nn_dist_affordance)
#                     # hand_cmap_affordance = torch.stack([get_pseudo_cmap(hand_nn_dist_affordance[:,idx]) for idx in range(hand_verts.shape[1])], dim=1)

#                     # cmap = plt.cm.get_cmap('plasma')
#                     # obj_v_color = (cmap(obj_cmap_affordance[0].detach().cpu().numpy())[:,0,:-1] * 255).astype(np.int64)
#                     # hand_v_color = [(cmap(hand_cmap_affordance[0, idx].detach().cpu().numpy())[:,0,:-1] * 255).astype(np.int64) for idx in range(hand_verts.shape[1])]

#                     # obj_mesh = trimesh.Trimesh(vertices=obj_vertices[0].detach().cpu().numpy(), vertex_colors=obj_v_color, faces = obj_mesh.faces)
#                     # hand_mesh = [trimesh.Trimesh(vertices=hand_verts[:,i].detach().cpu().numpy()[0], faces=(mano_layer.th_faces).detach().cpu().numpy(), vertex_colors=hand_v_color[i]) 
#                     #              for i, mano_layer in enumerate(MANO_LAYER)]

#                     # if not os.path.exists(os.path.dirname(save_contact_vis_path)):
#                     #     os.makedirs(os.path.dirname(save_contact_vis_path))

#                     # if len(hand_mesh) == 2:
#                     #     trimesh.exchange.export.export_mesh(hand_mesh[0],f'{save_contact_vis_path[:-4]}_left.obj')
#                     #     trimesh.exchange.export.export_mesh(hand_mesh[1],f'{save_contact_vis_path[:-4]}_right.obj')
#                     #     # trimesh.exchange.export.export_mesh(obj_mesh,f'{save_contact_vis_path[:-4]}_obj.obj')
#                     # else:
#                     #     trimesh.exchange.export.export_mesh(hand_mesh[0],f'{save_contact_vis_path[:-4]}_right.obj')
#                         # trimesh.exchange.export.export_mesh(obj_mesh,f'{save_contact_vis_path[:-4]}_obj.obj')
#                     ######################
#             ## old script ##

#         samples, targets = prefetcher.next()

#     metric_logger.synchronize_between_processes()
#     stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

#     save_dir = os.path.join(args.output_dir, 'results.txt')
#     with open(save_dir, 'a') as f:
#         if args.test_viewpoint is not None:
#             f.write(f"{'='*10} {args.test_viewpoint} {'='*10}\n")
#         f.write(f"{'='*10} {args.resume} {'='*10}\n\n")
#         for key, val in stats.items():
#             f.write(f'{key:30} : {val}\n')
#         f.write('\n\n')

#     return stats