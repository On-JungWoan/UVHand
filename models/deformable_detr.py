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
from .deformable_transformer import build_deforamble_transformer, _get_activation_fn
import copy
import numpy as np
from manopth.manolayer import ManoLayer

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, 
                 aux_loss=True, with_box_refine=False, two_stage=False, cfg=None):
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
        self.cls_embed = nn.Linear(self.hidden_dim, num_classes)

        # self.keypoint_embed = MLP(self.hidden_dim, self.hidden_dim, 63, 3) 
        # self.obj_keypoint_embed = MLP(self.hidden_dim, self.hidden_dim, 63, 3)
        self.mano_pose_embed = MLP(self.hidden_dim, self.hidden_dim, 48, 3)
        self.mano_beta_embed = MLP(self.hidden_dim, self.hidden_dim, 10, 3)
        self.obj_keypoint_embed = MLP(self.hidden_dim, self.hidden_dim, 48, 3)
        self.obj_ref_embed = nn.Linear(48, 63)
        self.ref_obj_embed = nn.Linear(63, 48)
        self.obj_rad = nn.Linear(self.hidden_dim, 1)

        self.num_feature_levels = num_feature_levels
        self.cfg = cfg

        self.query_embed = nn.Embedding(num_queries, self.hidden_dim*2) 
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
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.cls_embed.bias.data = torch.ones(num_classes) * bias_value

        # nn.init.constant_(self.keypoint_embed.layers[-1].weight.data, 0)      
        # nn.init.constant_(self.keypoint_embed.layers[-1].bias.data, 0)          
        # nn.init.constant_(self.obj_keypoint_embed.layers[-1].weight.data, 0)          
        # nn.init.constant_(self.obj_keypoint_embed.layers[-1].bias.data, 0)

        # xavier & uniform initialization
        nn.init.xavier_uniform_(self.mano_pose_embed.layers[-1].weight, gain=1)
        nn.init.constant_(self.mano_pose_embed.layers[-1].bias.data, 0)
        nn.init.xavier_uniform_(self.mano_beta_embed.layers[-1].weight, gain=1)
        nn.init.constant_(self.mano_beta_embed.layers[-1].bias.data, 0)
        nn.init.xavier_uniform_(self.obj_keypoint_embed.layers[-1].weight, gain=1)
        nn.init.constant_(self.obj_keypoint_embed.layers[-1].bias.data, 0)
        nn.init.xavier_uniform_(self.obj_ref_embed.weight, gain=1)
        nn.init.constant_(self.obj_ref_embed.bias.data, 0)
        nn.init.xavier_uniform_(self.ref_obj_embed.weight, gain=1)
        nn.init.constant_(self.ref_obj_embed.bias.data, 0)        
        nn.init.xavier_uniform_(self.obj_rad.weight, gain=1)
        nn.init.constant_(self.obj_rad.bias.data, 0)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        _mano_root = 'mano/models'
        self.mano_left = ManoLayer(flat_hand_mean=True,
                        side="left",
                        mano_root=_mano_root,
                        use_pca=False,
                        root_rot_mode='axisang',
                        joint_rot_mode='axisang')
        self.mano_right = ManoLayer(flat_hand_mean=True,
                        side="right",
                        mano_root=_mano_root,
                        use_pca=False,
                        root_rot_mode='axisang',
                        joint_rot_mode='axisang')
        self.transformer.decoder.mano_left = self.mano_left
        self.transformer.decoder.mano_right = self.mano_right
    
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.cls_embed = _get_clones(self.cls_embed, num_pred)
            # self.keypoint_embed = _get_clones(self.keypoint_embed, num_pred)
            # self.obj_keypoint_embed = _get_clones(self.obj_keypoint_embed, num_pred)
            self.mano_pose_embed = _get_clones(self.mano_pose_embed, num_pred)
            self.mano_beta_embed = _get_clones(self.mano_beta_embed, num_pred)
            self.obj_keypoint_embed = _get_clones(self.obj_keypoint_embed, num_pred)
            self.obj_ref_embed = _get_clones(self.obj_ref_embed, num_pred)
            self.ref_obj_embed = _get_clones(self.ref_obj_embed, num_pred)
            self.obj_rad = _get_clones(self.obj_rad, num_pred)

            # hack implementation for iterative bounding box refinement
            # self.transformer.decoder.keypoint_embed = self.keypoint_embed           
            # self.transformer.decoder.obj_keypoint_embed = self.obj_keypoint_embed
            self.transformer.decoder.cls_embed = self.cls_embed
            self.transformer.decoder.mano_pose_embed = self.mano_pose_embed
            self.transformer.decoder.mano_beta_embed = self.mano_beta_embed
            self.transformer.decoder.obj_keypoint_embed = self.obj_keypoint_embed
            self.transformer.decoder.obj_ref_embed = self.obj_ref_embed
            # self.transformer.decoder.obj_rad = self.obj_rad

        else:
            self.cls_embed = nn.ModuleList([self.cls_embed for _ in range(num_pred)])
            # self.keypoint_embed = nn.ModuleList([self.keypoint_embed for _ in range(num_pred)]) 
            # self.obj_keypoint_embed = nn.ModuleList([self.obj_keypoint_embed for _ in range(num_pred)]) 
            self.mano_pose_embed = nn.ModuleList([self.mano_pose_embed for _ in range(num_pred)])
            self.mano_beta_embed = nn.ModuleList([self.mano_beta_embed for _ in range(num_pred)])
            self.obj_keypoint_embed = nn.ModuleList([self.obj_keypoint_embed for _ in range(num_pred)])
            self.obj_ref_embed = nn.ModuleList([self.obj_ref_embed for _ in range(num_pred)])
            self.ref_obj_embed = nn.ModuleList([self.ref_obj_embed for _ in range(num_pred)])
            self.obj_rad = nn.ModuleList([self.obj_rad for _ in range(num_pred)])

            # self.transformer.decoder.keypoint_embed = None      
            # self.transformer.decoder.obj_keypoint_embed = None
            self.transformer.decoder.mano_pose_embed = None
            self.transformer.decoder.mano_beta_embed = None
            self.transformer.decoder.obj_keypoint_embed = None
            self.transformer.decoder.obj_ref_embed = None
            # self.transformer.decoder.obj_rad = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.cls_embed = self.cls_embed
        self.transformer.decoder.get_reference_point = self.get_reference_point 

    def forward(self, samples: NestedTensor):
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
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

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
        outputs_manopose = []
        outputs_manobeta = []
        outputs_obj_radians = []
        outputs_obj_keypoints = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
                # reference = inverse_sigmoid(reference)
            else:
                reference = inter_references[lvl - 1]
                # if dataset == 'H2O':
                # reference = inverse_sigmoid(reference)
                # else:
                # reference = inverse_sigmoid((reference+ 0.5)/2)

            outputs_class = self.cls_embed[lvl](hs[lvl])
            mano_pose = self.mano_pose_embed[lvl](hs[lvl])
            mano_beta = self.mano_beta_embed[lvl](hs[lvl])
            obj_key = self.obj_keypoint_embed[lvl](hs[lvl])
            obj_key = self.obj_ref_embed[lvl](obj_key)
            obj_rad = self.obj_rad[lvl](hs[lvl])
                
            if reference.shape[-1] == 42:
                ref_x = reference[...,0::2].mean(-1).unsqueeze(-1)
                ref_y = reference[...,1::2].mean(-1).unsqueeze(-1)

                # key = key.reshape(key.shape[0], key.shape[1], 21, 3)
                # key[..., :2] += torch.cat([ref_x, ref_y], dim=-1)[:,:,None,:] 
                # key = key.reshape(key.shape[0], key.shape[1], -1)

                obj_key = obj_key.reshape(obj_key.shape[0], obj_key.shape[1], 21, 3)
                obj_key[..., :2] += torch.cat([ref_x, ref_y], dim=-1)[:,:,None,:] 
                obj_key = obj_key.reshape(obj_key.shape[0], obj_key.shape[1], -1)

            else:
                assert reference.shape[-1] == 2
                # key = key.reshape(key.shape[0], key.shape[1], 21, 3)
                # key[..., :2] += reference[:,:,None,:] 
                # key = key.reshape(key.shape[0], key.shape[1], -1)
                
                obj_key = obj_key.reshape(obj_key.shape[0], obj_key.shape[1], 21, 3)
                obj_key[..., :2] += reference[:,:,None,:] 
                obj_key = obj_key.reshape(obj_key.shape[0], obj_key.shape[1], -1)

            # if dataset == 'H2O':
            #     outputs_keypoint = key.sigmoid() 
            #     outputs_obj_keypoint = obj_key.sigmoid() 
            # else:
            # outputs_obj_keypoint = obj_key.sigmoid()*2 - 0.5
            outputs_obj_keypoint = self.ref_obj_embed[lvl](obj_key)

            outputs_classes.append(outputs_class)
            outputs_manopose.append(mano_pose)
            outputs_manobeta.append(mano_beta)
            outputs_obj_radians.append(obj_rad)
            outputs_obj_keypoints.append(outputs_obj_keypoint)
            # outputs_keypoints.append(outputs_keypoint) 
            # outputs_obj_keypoints.append(outputs_obj_keypoint) 
        outputs_class = torch.stack(outputs_classes)
        outputs_manopose = torch.stack(outputs_manopose)
        outputs_manobeta = torch.stack(outputs_manobeta)
        outputs_obj_radians = torch.stack(outputs_obj_radians)
        # outputs_keypoints = torch.stack(outputs_keypoints) 
        outputs_obj_keypoints = torch.stack(outputs_obj_keypoints) 

        out = {
            'pred_logits': outputs_class[-1], 'pred_manoparams': [outputs_manopose[-1], outputs_manobeta[-1]],
            'pred_obj_params': [outputs_obj_keypoints[-1], outputs_obj_radians[-1]]
        }
        # out = {'pred_logits': outputs_class[-1], 'pred_keypoints': outputs_keypoints[-1]} 
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_manopose, outputs_manobeta, outputs_obj_keypoints, outputs_obj_radians)
            # out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_keypoints) 

        if self.two_stage:
            enc_outputs_hand_coord = enc_outputs_hand_coord_unact.sigmoid()
            raise Exception('Not implemeted.')
            # enc_outputs_obj_coord = enc_outputs_obj_coord_unact.sigmoid()
            # out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_keypoints': enc_outputs_hand_coord, 'pred_obj_keypoints': enc_outputs_obj_coord}
            # out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_keypoints': enc_outputs_hand_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_manopose, outputs_manobeta, outputs_obj_keypoints, outputs_obj_radians) : 
    # def _set_aux_loss(self, outputs_class, outputs_keypoints) : 
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # return [{'pred_logits': a, 'pred_keypoints': b}
        #         for a, b in zip(outputs_class[:-1], outputs_keypoints[:-1])]
        return [
                {'pred_logits': c, 'pred_manoparams': [p, b], 'pred_obj_params': [k, r]} \
                for c, p, b, k, r in \
                    zip(outputs_class[:-1], outputs_manopose[:-1], outputs_manobeta[:-1], outputs_obj_keypoints[:-1], outputs_obj_radians[:-1])
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

class SetCriterion(nn.Module):
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

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).cuda()
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

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses
    
    def loss_hand_keypoints(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_keypoints' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_keypoints = outputs['pred_keypoints'][idx]

        target_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        target_labels = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        hand_cal_idx = torch.zeros_like(target_labels, dtype=torch.bool)
        for idx in self.cfg.hand_idx:
            hand_cal_idx |= (target_labels == idx)

        # target_keypoints = target_keypoints[hand_cal_idx].view(-1, 63)
        # src_keypoints = src_keypoints[hand_cal_idx]

        # occlusion_mask = target_keypoints<0
        # for idx, mask in enumerate(occlusion_mask):
        #     target_keypoints[idx][mask] = 0
        #     src_keypoints[idx][mask] = 0

        # loss_handkey = F.l1_loss(src_keypoints, target_keypoints, reduction='none')

        losses = {}
        
        if len(src_keypoints[hand_cal_idx]) == 0:
            losses['loss_hand_keypoint'] = torch.tensor(0)
        else:
            loss_handkey = F.l1_loss(src_keypoints[hand_cal_idx], target_keypoints[hand_cal_idx].view(-1, 63).cuda(), reduction='none')
            losses['loss_hand_keypoint'] = (loss_handkey.sum() / hand_cal_idx.sum().item()) / 21

        return losses
    
    def loss_obj_keypoints(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_obj_params' in outputs
        pred_obj_keypoints, _ = outputs['pred_obj_params']
        idx = self._get_src_permutation_idx(indices)
        src_objkeys = pred_obj_keypoints[idx]

        target_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        target_labels = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        obj_cal_idx = torch.ones_like(target_labels, dtype=torch.bool)
        for idx in self.cfg.hand_idx:
            obj_cal_idx &= (target_labels != idx)

        dt_obj_key = src_objkeys[obj_cal_idx]
        gt_obj_key = target_keypoints[obj_cal_idx][:, :16, :]
        
        loss_objkey = F.l1_loss(dt_obj_key, gt_obj_key.view(-1, 48), reduction='none')
        
        losses = {}
        if obj_cal_idx.sum().item() == 0:
            losses['loss_obj_key'] = loss_objkey.sum()
        else:
            losses['loss_obj_key'] = (loss_objkey.sum()/ obj_cal_idx.sum().item()) / 21

        return losses    

    def loss_obj_radians(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_obj_params' in outputs
        _, pred_obj_radian = outputs['pred_obj_params']
        idx = self._get_src_permutation_idx(indices)
        src_objrad = pred_obj_radian[idx]
    
        target_labels = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        obj_cal_idx = torch.ones_like(target_labels, dtype=torch.bool)
        for idx in self.cfg.hand_idx:
            obj_cal_idx &= (target_labels != idx)

        dt_obj_rad = src_objrad[obj_cal_idx]
        gt_obj_rad = torch.stack([t['object.radian'] for t in targets]).unsqueeze(1)
        loss_obj_rad = F.l1_loss(dt_obj_rad, gt_obj_rad, reduction='mean')

        losses = {}
        losses['loss_obj_rad'] = loss_obj_rad
        return losses

    def loss_mano_params(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_manoparams' in outputs
        assert sum([t["mano_pose"].sum() for t in targets]).item() != 0

        pred_mano_pose, pred_mano_beta = outputs['pred_manoparams']
        idx = self._get_src_permutation_idx(indices)
        src_pose = pred_mano_pose[idx]
        src_beta = pred_mano_beta[idx]

        target_labels = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        left_idx = target_labels == self.cfg.hand_idx[0]
        right_idx = target_labels == self.cfg.hand_idx[1]

        src_left_pose = src_pose[left_idx]
        src_left_beta = src_beta[left_idx]
        src_right_pose = src_pose[right_idx]
        src_right_beta = src_beta[right_idx]

        target_mano_pose = torch.cat([t['mano_pose'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_left_mano_pose = target_mano_pose[left_idx]
        target_right_mano_pose = target_mano_pose[right_idx]
        target_mano_beta = torch.cat([t['mano_beta'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_left_mano_beta = target_mano_beta[left_idx]
        target_right_mano_beta = target_mano_beta[right_idx]
        
        loss_left_pose = F.l1_loss(src_left_pose, target_left_mano_pose, reduction='none')
        loss_right_pose = F.l1_loss(src_right_pose, target_right_mano_pose, reduction='none')
        loss_left_beta = F.l1_loss(src_left_beta, target_left_mano_beta, reduction='none')
        loss_right_beta = F.l1_loss(src_right_beta, target_right_mano_beta, reduction='none')

        loss_pose = torch.cat([loss_left_pose, loss_right_pose])
        loss_beta = torch.cat([loss_left_beta, loss_right_beta])
        loss_pose = loss_pose.sum()/len(loss_pose)
        loss_beta = loss_beta.sum()/len(loss_beta)

        losses = {}
        losses['loss_mano_params'] = loss_pose + loss_beta
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
            'cardinality': self.loss_cardinality,
            'mano_params': self.loss_mano_params,
            'obj_keypoint': self.loss_obj_keypoints,
            'obj_radian': self.loss_obj_radians,
            'hand_keypoint': self.loss_hand_keypoints,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
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
        cfg = cfg
    )
    matcher = build_matcher(args, cfg)
    # weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_hand_keypoint': args.keypoint_loss_coef, 'loss_obj_keypoint': args.keypoint_loss_coef}
    weight_dict = {
        'loss_ce': args.cls_loss_coef, 'loss_obj_rad': args.cls_loss_coef,
        'loss_mano_params': args.keypoint_loss_coef, 'loss_obj_key': args.keypoint_loss_coef
    }

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'cardinality', 'mano_params', 'obj_keypoint', 'obj_radian']
    # losses = ['labels', 'cardinality', 'hand_keypoint']

    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha, cfg=cfg)
    criterion.to(device)

    return model, criterion
