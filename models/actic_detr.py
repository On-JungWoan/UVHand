# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .arctic_transformer import build_deforamble_transformer, _get_activation_fn
import copy
import numpy as np
from arctic_tools.process import prepare_data
from arctic_tools.src.callbacks.loss.loss_arctic_sf import compute_loss

import sys

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, 
                 aux_loss=True, with_box_refine=False, two_stage=False, cfg=None,
                 method=None, window_size=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.method = method
        self.window_size = window_size

        self.cls_embed = nn.Linear(self.hidden_dim, num_classes)
        self.mano_pose_embed = nn.Linear(self.hidden_dim, 48)
        self.mano_beta_embed = nn.Linear(self.hidden_dim, 10)
        self.hand_cam = nn.Linear(self.hidden_dim, 3)
        self.obj_cam = nn.Linear(self.hidden_dim, 3)
        self.obj_rot = nn.Linear(self.hidden_dim, 3)
        self.obj_rad = nn.Linear(self.hidden_dim, 1)

        self.num_feature_levels = num_feature_levels
        self.cfg = cfg

        self.query_embed = nn.Embedding(num_queries, self.hidden_dim*2)
        if self.method == 'arctic_sf':
            if num_feature_levels > 1:
                num_backbone_outs = len(backbone.strides)
                input_proj_list = []
                for _ in range(num_backbone_outs):
                    in_channels = backbone.num_channels[_]
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, self.hidden_dim),
                    ))
                for _ in range(num_feature_levels - num_backbone_outs):   # multi-scale feature는 4개, backbone output feature가 3개.
                    input_proj_list.append(nn.Sequential(                 # backbone의 last featrue에 kernel_3, stride 2인 conv를 추가하여 4개로 만들어줌
                        nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, self.hidden_dim),
                    ))
                    in_channels = self.hidden_dim
                self.input_proj = nn.ModuleList(input_proj_list)
            else:
                self.input_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], self.hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, self.hidden_dim),
                    )])
        else:
            self.lstm = nn.LSTM(
                input_size=2048,
                hidden_size=1024,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
            )
            self.feature_proj = nn.Sequential(
                nn.Linear(2048, 4096),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(4096, 256*256),
                nn.Dropout(0.1),
                nn.LayerNorm(256*256)
            )
            input_proj_list = [
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, padding=6),
                    nn.GroupNorm(32, self.hidden_dim)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3),
                    nn.GroupNorm(32, self.hidden_dim)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=2),
                    nn.GroupNorm(32, self.hidden_dim)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=4, stride=4),
                    nn.GroupNorm(32, self.hidden_dim)
                )
            ]
            self.input_proj = nn.ModuleList(input_proj_list)
            for idx, proj in enumerate(self.feature_proj):
                if idx in [0, 3]:
                    nn.init.xavier_uniform_(proj.weight, gain=1)
                    nn.init.constant_(proj.bias, 0)

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.cls_embed.bias.data = torch.ones(num_classes) * bias_value

        nn.init.xavier_uniform_(self.mano_pose_embed.weight, gain=1)
        nn.init.constant_(self.mano_pose_embed.bias.data, 0)
        nn.init.xavier_uniform_(self.mano_beta_embed.weight, gain=1)
        nn.init.constant_(self.mano_beta_embed.bias.data, 0)
        nn.init.xavier_uniform_(self.hand_cam.weight, gain=1)
        nn.init.constant_(self.hand_cam.bias.data, 0)
        nn.init.xavier_uniform_(self.obj_cam.weight, gain=1)
        nn.init.constant_(self.obj_cam.bias.data, 0)
        nn.init.xavier_uniform_(self.obj_rot.weight, gain=1)
        nn.init.constant_(self.obj_rot.bias.data, 0)                                        
        nn.init.xavier_uniform_(self.obj_rad.weight, gain=1)
        nn.init.constant_(self.obj_rad.bias.data, 0)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
    
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            raise Exception('Not implemented yet')
            self.cls_embed = _get_clones(self.cls_embed, num_pred)
            self.transformer.decoder.cls_embed = self.cls_embed

        else:
            self.cls_embed = nn.ModuleList([self.cls_embed for _ in range(num_pred)])
            self.mano_pose_embed = nn.ModuleList([self.mano_pose_embed for _ in range(num_pred)])
            self.mano_beta_embed = nn.ModuleList([self.mano_beta_embed for _ in range(num_pred)])
            self.hand_cam = nn.ModuleList([self.hand_cam for _ in range(num_pred)])
            self.obj_cam = nn.ModuleList([self.obj_cam for _ in range(num_pred)])
            self.obj_rot = nn.ModuleList([self.obj_rot for _ in range(num_pred)])
            self.obj_rad = nn.ModuleList([self.obj_rad for _ in range(num_pred)])
        if two_stage:
            raise Exception('Not implemented yet')
            # hack implementation for two-stage
            self.transformer.decoder.cls_embed = self.cls_embed
        self.transformer.decoder.get_reference_point = self.get_reference_point 

    def forward(self, samples, is_extract=False):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if self.method == 'arctic_lstm':
            samples = samples.view(-1, self.window_size, 2048)
            B, W, C = samples.shape
            device = samples.device
            
            # bidirectional
            h0 = torch.randn(2 * 2, B, C // 2, device=device)
            c0 = torch.randn(2 * 2, B, C // 2, device=device)
            feat_vec, (hn, cn) = self.lstm(samples, (h0, c0))  # batch, seq, 2*dim
            feat_vec = feat_vec.reshape(B*W, C)
            feat_vec = self.feature_proj(feat_vec)
            feat_vec = feat_vec.view(B*W, -1, 16, 16)

            srcs = []
            masks = []
            pos = []
            for i in range(self.num_feature_levels):
                src = self.input_proj[i](feat_vec)
                B, C, W, H = src.shape
                mask = torch.cuda.FloatTensor(B,W,H).uniform_() > 0.3
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)

                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        
        else:
            ### backbone ###
            if not isinstance(samples, NestedTensor):
                samples = nested_tensor_from_tensor_list(samples)
            features, pos = self.backbone(samples)  # output feature, output feature size에 해당하는 positional embedding
            srcs = []
            masks = []
            for l, feat in enumerate(features):
                src, mask = feat.decompose() # Nested tensor -> return (tensor, mask)
                srcs.append(self.input_proj[l](src))  # 모든 feature의 output dim -> hidden dim으로 projection
                masks.append(mask)
                assert mask is not None
            if self.num_feature_levels > len(srcs): # output last feature map 이후
                _len_srcs = len(srcs)
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:
                        src = self.input_proj[l](features[-1].tensors) # 
                    else:
                        src = self.input_proj[l](srcs[-1])
                    m = samples.mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src) # B, C, (28 & 14 & 7 & 4)
                    masks.append(mask)
                    pos.append(pos_l)

        if is_extract:
            return srcs, pos

        query_embeds = None
        query_embeds = self.query_embed.weight ############## two_stage
        if not self.two_stage:
            query_embeds = self.query_embed.weight  # (num_query, self.hidden_dim)

        ### backbone ###
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_hand_coord_unact, enc_outputs_obj_coord_unact = self.transformer(srcs, masks, pos, query_embeds)
        # hs : result include intermeditate feature (num_decoder_layer, B, num_queries, hidden_dim)
        # dataset = 'H2O' if len(self.cfg.hand_idx) == 2 else 'FPHA'

        dataset = self.cfg.dataset_file
        outputs_classes = []
        outputs_mano_pose = []
        outputs_mano_beta = []
        outputs_hand_cams = []
        outputs_obj_cams = []
        outputs_obj_radians = []
        outputs_obj_rotations = []

        for lvl in range(hs.shape[0]):
            outputs_class = self.cls_embed[lvl](hs[lvl])
            out_mano_pose = self.mano_pose_embed[lvl](hs[lvl])
            out_mano_beta = self.mano_beta_embed[lvl](hs[lvl])
            out_hand_cam = self.hand_cam[lvl](hs[lvl])
            out_obj_cam = self.obj_cam[lvl](hs[lvl])
            out_obj_rot = self.obj_rot[lvl](hs[lvl])
            out_obj_rad = self.obj_rad[lvl](hs[lvl])
            
            outputs_classes.append(outputs_class)
            outputs_mano_pose.append(out_mano_pose)
            outputs_mano_beta.append(out_mano_beta)
            outputs_hand_cams.append(out_hand_cam)
            outputs_obj_cams.append(out_obj_cam)
            outputs_obj_radians.append(out_obj_rad)
            outputs_obj_rotations.append(out_obj_rot)

        outputs_class = torch.stack(outputs_classes)
        outputs_mano_params = [torch.stack(outputs_mano_pose), torch.stack(outputs_mano_beta)]
        outputs_obj_params = [torch.stack(outputs_obj_radians), torch.stack(outputs_obj_rotations)]
        outputs_cams = [torch.stack(outputs_hand_cams), torch.stack(outputs_obj_cams)]

        out = {
            'pred_logits': outputs_class[-1], 'pred_mano_params': [outputs_mano_params[0][-1], outputs_mano_params[1][-1]],
            'pred_obj_params': [outputs_obj_params[0][-1], outputs_obj_params[1][-1]],
            'pred_cams': [outputs_cams[0][-1], outputs_cams[1][-1]]
        }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_mano_params, outputs_obj_params, outputs_cams)

        if self.two_stage:
            raise Exception('Not implemeted.')
            enc_outputs_hand_coord = enc_outputs_hand_coord_unact.sigmoid()
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_mano_params, outputs_obj_params, outputs_cams): 
        return [
                {'pred_logits': c, 'pred_mano_params': [s, p], 'pred_obj_params': [ra, ro], 'pred_cams': [hc, oc]} \
                for c, s, p, ra, ro, hc, oc in \
                    zip(
                        outputs_class[:-1], 
                        outputs_mano_params[0][:-1], outputs_mano_params[1][:-1],
                        outputs_obj_params[0][:-1], outputs_obj_params[1][:-1],
                        outputs_cams[0][:-1], outputs_cams[1][:-1]
                    )
            ] 

    def get_reference_point(self, bs:int, mano_params:list, class_indices:torch.Tensor, reference_points:torch.Tensor):
        out_mano_pose, out_mano_beta = mano_params

        hand_lr_idx = [class_indices == self.cfg.hand_idx[0], class_indices == self.cfg.hand_idx[1]]
        MANO_LAYER = [self.mano_left,self.mano_right]
        tmp = torch.zeros_like(reference_points)
        
        for b in range(bs):
            for mano, idx in zip(MANO_LAYER, hand_lr_idx):
                _, out_key = mano(out_mano_pose[b][idx[b]], out_mano_beta[b][idx[b]])
                tmp[b][idx[b]] = out_key[...,:2]

        return tmp


class SetArcticCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, cfg=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.cfg = cfg

    def loss_labels(self, outputs, targets, idx, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).cuda()
        # target_classes_o = torch.cat([t[0][J] for t, (_, J) in zip(targets['labels'], indices)]).cuda()
        try:
            target_classes_o = torch.cat([t[0] for idx, t in enumerate(targets['labels']) if targets['is_valid'][idx] == 1]).cuda()
        except:
            target_classes_o = torch.cat([torch.tensor(t) for idx, t in enumerate(targets['labels']) if targets['is_valid'][idx] == 1]).cuda()
            
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    # @torch.no_grad()
    # def loss_cardinality(self, outputs, targets, indices, num_boxes):
    #     """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
    #     This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
    #     """
    #     pred_logits = outputs['pred_logits']
    #     device = pred_logits.device
    #     # tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
    #     tgt_lengths = torch.as_tensor([v.shape[-1] for v in targets['labels']], device=device)
    #     # Count the number of predictions that are NOT "no-object" (which is the last class)
    #     # card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
    #     card_pred = (pred_logits.argmax(-1) != 0).sum(1)
    #     card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
    #     losses = {'cardinality_error': card_err}
    #     return losses

    def loss_cam(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_cams' in outputs
        # assert sum([t["mano_pose"].sum() for t in targets]).item() != 0
        left_valid = targets['left_valid'].type(torch.bool)
        right_valid = targets['right_valid'].type(torch.bool)

        pred_hand_cam, pred_obj_cam = outputs['pred_cams']
        idx = self._get_src_permutation_idx(indices)
        assert sum(left_valid) + sum(right_valid) + len(targets['is_valid']) == len(idx[0])

        src_hand = pred_hand_cam[idx]
        src_obj = pred_obj_cam[idx]

        # target_labels = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_labels = torch.cat([t[0][J] for t, (_, J) in zip(targets['labels'], indices)], dim=0)
        left_idx = target_labels == self.cfg.hand_idx[0]
        right_idx = target_labels == self.cfg.hand_idx[1]
        obj_idx = torch.tensor([label not in self.cfg.hand_idx for label in target_labels])

        src_left_cam = src_hand[left_idx]
        src_right_cam = src_hand[right_idx]
        src_obj_cam = src_obj[obj_idx]

        # target_mano_pose = torch.cat([t['mano_pose'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_left_cam = targets['mano.cam_t.wp.l'][left_valid]
        target_right_cam = targets['mano.cam_t.wp.l'][right_valid]
        target_obj_cam = targets['object.cam_t.wp']

        loss_left_cam = F.l1_loss(src_left_cam, target_left_cam, reduction='mean')
        loss_right_cam = F.l1_loss(src_right_cam, target_right_cam, reduction='mean')
        loss_obj_cam = F.l1_loss(src_obj_cam, target_obj_cam, reduction='mean')

        loss_hand_cam = (loss_left_cam + loss_right_cam) / 2

        losses = {}
        losses['loss_cam'] = loss_hand_cam + loss_obj_cam
        return losses

    def loss_obj_rotations(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_obj_params' in outputs
        left_valid = targets['left_valid'].type(torch.bool)
        right_valid = targets['right_valid'].type(torch.bool)
        bs = left_valid.shape[0]

        pred_obj_radian, pred_obj_rot = outputs['pred_obj_params']
        idx = self._get_src_permutation_idx(indices)
        assert sum(left_valid) + sum(right_valid) + len(targets['is_valid']) == len(idx[0])

        src_rad = pred_obj_radian[idx]
        src_rot = pred_obj_rot[idx]
    
        target_labels = torch.cat([t[0][J] for t, (_, J) in zip(targets['labels'], indices)], dim=0)
        obj_idx = torch.tensor([label not in self.cfg.hand_idx for label in target_labels])

        src_obj_rad = src_rad[obj_idx]
        src_obj_rot = src_rot[obj_idx]

        # target_mano_pose = torch.cat([t['mano_pose'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_rad = targets['object.radian']
        target_obj_rot = targets['object.rot']

        loss_rad = F.l1_loss(src_obj_rad, target_obj_rad.unsqueeze(-1), reduction='mean')
        loss_rot = F.l1_loss(src_obj_rot, target_obj_rot.view(bs, -1), reduction='mean')

        losses = {}
        losses['loss_rad_rot'] = loss_rad + loss_rot
        return losses

    def loss_mano_poses(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_mano_params' in outputs
        # assert sum([t["mano_pose"].sum() for t in targets]).item() != 0
        left_valid = targets['left_valid'].type(torch.bool)
        right_valid = targets['right_valid'].type(torch.bool)

        pred_mano_pose, pred_mano_beta = outputs['pred_mano_params']
        idx = self._get_src_permutation_idx(indices)
        assert sum(left_valid) + sum(right_valid) + len(targets['is_valid']) == len(idx[0])

        src_pose = pred_mano_pose[idx]

        # target_labels = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_labels = torch.cat([t[0][J] for t, (_, J) in zip(targets['labels'], indices)], dim=0)
        left_idx = target_labels == self.cfg.hand_idx[0]
        right_idx = target_labels == self.cfg.hand_idx[1]

        src_left_pose = src_pose[left_idx]
        src_right_pose = src_pose[right_idx]

        # target_mano_pose = torch.cat([t['mano_pose'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_left_mano_pose = targets['mano.pose.l'][left_valid]
        target_right_mano_pose = targets['mano.pose.r'][right_valid]
        
        loss_left_pose = F.l1_loss(src_left_pose, target_left_mano_pose, reduction='mean')
        loss_right_pose = F.l1_loss(src_right_pose, target_right_mano_pose, reduction='mean')

        loss_pose = (loss_left_pose + loss_right_pose) / 2

        losses = {}
        losses['loss_mano_pose'] = loss_pose
        return losses
    
    def loss_mano_betas(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_mano_params' in outputs
        # assert sum([t["mano_pose"].sum() for t in targets]).item() != 0
        left_valid = targets['left_valid'].type(torch.bool)
        right_valid = targets['right_valid'].type(torch.bool)

        pred_mano_pose, pred_mano_beta = outputs['pred_mano_params']
        idx = self._get_src_permutation_idx(indices)
        assert sum(left_valid) + sum(right_valid) + len(targets['is_valid']) == len(idx[0])

        src_beta = pred_mano_beta[idx]

        # target_labels = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_labels = torch.cat([t[0][J] for t, (_, J) in zip(targets['labels'], indices)], dim=0)
        left_idx = target_labels == self.cfg.hand_idx[0]
        right_idx = target_labels == self.cfg.hand_idx[1]

        src_left_beta = src_beta[left_idx]
        src_right_beta = src_beta[right_idx]

        # target_mano_beta = torch.cat([t['mano_beta'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_left_mano_beta = targets['mano.beta.l'][left_valid]
        target_right_mano_beta = targets['mano.beta.r'][right_valid]
        
        loss_left_beta = F.l1_loss(src_left_beta, target_left_mano_beta, reduction='mean')
        loss_right_beta = F.l1_loss(src_right_beta, target_right_mano_beta, reduction='mean')

        loss_beta = (loss_left_beta + loss_right_beta) / 2

        losses = {}
        losses['loss_mano_beta'] = loss_beta
        return losses    

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            # 'cardinality': self.loss_cardinality,
            # 'mano_params': self.loss_mano_params,

            # 'mano_poses': self.loss_mano_poses,
            # 'mano_betas': self.loss_mano_betas,
            # 'cam': self.loss_cam,
            # 'obj_rotation': self.loss_obj_rotations,

            # 'obj_keypoint': self.loss_obj_keypoints,
            # 'hand_keypoint': self.loss_hand_keypoints,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def select_indices(self, cfg, outputs, targets, device):
        prob = outputs['pred_logits']
        B = prob.shape[0]

        is_valid = targets['is_valid']
        left_valid = targets['left_valid'] * is_valid
        right_valid = targets['right_valid'] * is_valid

        # query index select
        best_score = torch.zeros(B).to(device)
        # if dataset != 'AssemblyHands':
        obj_idx = torch.zeros(B).to(device).to(torch.long)
        for i in range(1, cfg.hand_idx[0]):
            score, idx = torch.max(prob[:,:,i], dim=-1)
            obj_idx[best_score < score] = idx[best_score < score]
            best_score[best_score < score] = score[best_score < score]

        hand_idx = []
        for i in cfg.hand_idx:
            hand_idx.append(torch.argmax(prob[:,:,i], dim=-1)) 
        # hand_idx = torch.stack(hand_idx, dim=-1) 
        left_hand_idx, right_hand_idx = hand_idx

        indicies = []
        batch = []

        for idx, item in enumerate(zip(obj_idx, left_hand_idx, right_hand_idx)):
            obj, left, right = item

            obj_item = obj[is_valid.to(torch.bool)[idx]]
            left_item = left[left_valid.to(torch.bool)[idx]]
            right_item = right[right_valid.to(torch.bool)[idx]]
            for item in [obj_item, left_item, right_item]:
                if len(item) > 0:
                    indicies.append(item[0])
                    batch.append(idx)
        indicies = torch.stack(indicies)
        batch = torch.as_tensor(batch, dtype=torch.int64).to(device)

        return batch, indicies

    def forward(self, outputs, targets, data, args, meta_info, cfg):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        device = args.device
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(outputs_without_aux, targets)
        idx = self.select_indices(cfg, outputs_without_aux, targets, device)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # num_boxes = sum(len(t["labels"]) for t in targets)
        try:
            num_boxes = sum([v.shape[-1] for v in targets['labels']])
        except:
            num_boxes = sum([len(v) for v in targets['labels']])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            # losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))
            losses.update(self.get_loss(loss, outputs, targets, idx, num_boxes, **kwargs))

        arctic_pred = data.search('pred.', replace_to='')
        arctic_gt = data.search('targets.', replace_to='')
        losses.update(compute_loss(arctic_pred, arctic_gt, meta_info))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                idx = self.select_indices(cfg, aux_outputs, targets, device)

                aux_data = prepare_data(args, aux_outputs, targets, meta_info, self.cfg)
                aux_arctic_pred = aux_data.search('pred.', replace_to='')
                aux_arctic_gt = aux_data.search('targets.', replace_to='')

                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, idx, num_boxes, **kwargs)
                    l_dict.update(compute_loss(aux_arctic_pred, aux_arctic_gt, meta_info))
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build(args, cfg):
    num_classes = cfg.num_obj_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args, cfg)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        cfg = cfg,
        method = args.method,
        window_size= args.window_size
    )
    matcher = build_matcher(args, cfg)
    # weight_dict = {
    #     'loss_ce': args.cls_loss_coef,
    #     'loss_cam': 1.0,
    #     'loss_rad_rot' : 1.0,
    #     'loss_mano_pose' : 10.0,
    #     'loss_mano_beta' : 0.001,
    #     "loss/mano/kp2d/r":5.0,
    #     "loss/mano/kp3d/r":5.0,
    #     "loss/mano/kp2d/l":5.0,
    #     "loss/mano/kp3d/l":5.0,
    #     # "loss/cd":1.0,
    #     "loss/cd":2.0,
    #     "loss/mano/transl/l":1.0,
    #     "loss/object/kp2d":1.0,
    #     "loss/object/kp3d":5.0,
    #     "loss/object/transl":1.0,
    # }
    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'class_error':1,
        'cardinality_error':1,
        "loss/mano/cam_t/r":1.0,
        "loss/mano/cam_t/l":1.0,
        "loss/object/cam_t":1.0,
        "loss/mano/kp2d/r":5.0,
        "loss/mano/kp3d/r":5.0,
        "loss/mano/pose/r":10.0,
        "loss/mano/beta/r":0.001,
        "loss/mano/kp2d/l":5.0,
        "loss/mano/kp3d/l":5.0,
        "loss/mano/pose/l":10.0,
        # "loss/cd":1.0,
        "loss/cd":10.0,
        "loss/mano/transl/l":10.0,
        "loss/mano/beta/l":0.001,
        "loss/object/kp2d":1.0,
        "loss/object/kp3d":5.0,
        "loss/object/radian":1.0,
        "loss/object/rot":1.0,
        "loss/object/transl":10.0,
        "loss/penetr": 0.1,
        # 'loss_cam': args.cls_loss_coef,
        # 'loss_mano_params': args.keypoint_loss_coef, 'loss_rad_rot': args.keypoint_loss_coef
        # 'loss_mano_params': args.cls_loss_coef, 'loss_rad_rot': args.cls_loss_coef
    }

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # losses = ['labels', 'cardinality', 'mano_poses', 'mano_betas', 'cam', 'obj_rotation']
    losses = ['labels',]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetArcticCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha, cfg=cfg)
    criterion.to(device)

    return model, criterion
