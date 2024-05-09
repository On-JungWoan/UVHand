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
import math
import copy
import torch
from torch import nn
import torch.nn.functional as F

from .matcher import build_matcher
from .backbone import build_backbone
from .segmentation import sigmoid_focal_loss
from .arctic_transformer import build_deforamble_transformer, _get_activation_fn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from arctic_tools.common.body_models import build_mano_aa
from arctic_tools.common.object_tensors import ObjectTensors
from arctic_tools.process import prepare_data, get_arctic_item
from arctic_tools.src.callbacks.loss.loss_arctic_sf import compute_loss, compute_small_loss


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, 
                 aux_loss=True, with_box_refine=False, two_stage=False, cfg=None,
                 method=None, window_size=None, feature_type='local_fm'):
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
        self.feature_type = feature_type

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
        if self.feature_type == 'origin':
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
            for proj in self.input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)
            self.backbone = backbone
        else:
            # only positional encoding
            self.backbone = backbone[1]
            # self.gru = nn.GRU(
            #         input_size=self.hidden_dim, hidden_size=int(self.hidden_dim/2),
            #         num_layers=1, bidirectional=True, batch_first=True,
            #     )

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
    
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            assert two_stage, "Not implemented! You should use 'with_box_refine' and 'two_stage' option simultaneously."

            self.cls_embed = _get_clones(self.cls_embed, num_pred)

            # for only ref points
            self.key_embed = MLP(self.hidden_dim, self.hidden_dim, 42, 3)
            self.obj_key_embed = MLP(self.hidden_dim, self.hidden_dim, 42, 3)    
            nn.init.xavier_uniform_(self.key_embed.layers[-1].weight.data, gain=1)
            nn.init.constant_(self.key_embed.layers[-1].bias.data, 0)
            nn.init.xavier_uniform_(self.obj_key_embed.layers[-1].weight.data, gain=1)
            nn.init.constant_(self.obj_key_embed.layers[-1].bias.data, 0)

            self.key_embed = _get_clones(self.key_embed, num_pred)
            self.obj_key_embed = _get_clones(self.obj_key_embed, num_pred)

            self.transformer.decoder.cls_embed = self.cls_embed
            self.transformer.decoder.key_embed = self.key_embed
            self.transformer.decoder.obj_key_embed = self.obj_key_embed

        else:
            self.cls_embed = nn.ModuleList([self.cls_embed for _ in range(num_pred)])
            # if self.method == 'arctic_lstm':
            #     self.gru = nn.ModuleList([self.gru for _ in range(num_pred)])
        self.mano_pose_embed = nn.ModuleList([self.mano_pose_embed for _ in range(num_pred)])
        self.mano_beta_embed = nn.ModuleList([self.mano_beta_embed for _ in range(num_pred)])
        self.hand_cam = nn.ModuleList([self.hand_cam for _ in range(num_pred)])
        self.obj_cam = nn.ModuleList([self.obj_cam for _ in range(num_pred)])
        self.obj_rot = nn.ModuleList([self.obj_rot for _ in range(num_pred)])
        self.obj_rad = nn.ModuleList([self.obj_rad for _ in range(num_pred)])
        # if two_stage:
        #     raise Exception('Not implemented yet')
        #     # hack implementation for two-stage
        #     self.transformer.decoder.cls_embed = self.cls_embed

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

        if self.feature_type == 'origin':
            ### backbone ###
            if not isinstance(samples, NestedTensor):
                samples = nested_tensor_from_tensor_list(samples)
            features, pos = self.backbone(samples)  # output feature, output feature size에 해당하는 positional embedding

            if is_extract:
                return features

            srcs = []
            masks = []
            for l, feat in enumerate(features):
                src, mask = feat.decompose() # Nested tensor -> return (tensor, mask)

                # encoder input의 30% masking
                src_input = self.input_proj[l](src) # 모든 feature의 output dim -> hidden dim으로 projection
                if self.training:
                    src_mask = torch.cuda.FloatTensor(src_input.shape).uniform_() > 0.3
                    srcs.append(src_input*src_mask)
                else:
                    srcs.append(src_input)
                
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

                    # encoder input의 30% masking
                    if self.training:
                        src_mask = torch.cuda.FloatTensor(src.shape).uniform_() > 0.3
                        srcs.append(src*src_mask) # B, C, (28 & 14 & 7 & 4)
                    else:
                        srcs.append(src)

                    masks.append(mask)
                    pos.append(pos_l)

        else:
            srcs = []
            masks = []
            pos = []
            for i in range(self.num_feature_levels):
                B, N, C, W, H = samples[i].shape
                device = samples[i].device

                src = samples[i].view(B*N, C, W, H)
                # mask = torch.cuda.FloatTensor(B*N,W,H).uniform_() > 0.2
                mask = torch.zeros(B*N,W,H).to(device).type(torch.bool)
                pos_l = self.backbone(NestedTensor(src, mask)).to(src.dtype)

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

        levels = hs.shape[0]
        outputs_hand_coord = torch.zeros(levels)
        outputs_obj_coord = torch.zeros(levels)

        if self.two_stage:
            outputs_hand_coord_list = []
            outputs_obj_coord_list = []

        outputs_classes = []
        outputs_mano_pose = []
        outputs_mano_beta = []
        outputs_hand_cams = []
        outputs_obj_cams = []
        outputs_obj_radians = []
        outputs_obj_rotations = []

        for lvl in range(levels):
            # if self.method == 'arctic_sf':
            hs_lvl = hs[lvl]
            # else:
            #     Q, C = self.num_queries, self.hidden_dim
            #     # time step을 고려하기 위해 GRU 통과
            #     hs_lvl = hs[lvl]
            #     hs_lvl = hs_lvl.reshape(B, N, Q, C) # split batch & frame
            #     hs_lvl = hs_lvl.permute(0,2,1,3).reshape(-1, N, C) # B*Q, N, C
            #     hs_lvl, _ = self.gru[lvl](hs_lvl) # GRU
            #     hs_lvl = hs_lvl.reshape(B, Q, N, C).permute(0,2,1,3).reshape(B*N, Q, C) # B*N, Q, C
            
            if self.two_stage:
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                
                layer_key_delta_unsig = self.key_embed[lvl](hs_lvl)
                layer_obj_delta_unsig = self.obj_key_embed[lvl](hs_lvl)
                layer_key_outputs_unsig = (layer_key_delta_unsig  + reference).sigmoid() * 2 - 1
                layer_obj_outputs_unsig = (layer_obj_delta_unsig  + reference).sigmoid() * 2 - 1

                outputs_hand_coord_list.append(layer_key_outputs_unsig)
                outputs_obj_coord_list.append(layer_obj_outputs_unsig)

            outputs_class = self.cls_embed[lvl](hs_lvl)
            out_mano_pose = self.mano_pose_embed[lvl](hs_lvl)
            out_mano_beta = self.mano_beta_embed[lvl](hs_lvl)
            out_hand_cam = self.hand_cam[lvl](hs_lvl)
            out_obj_cam = self.obj_cam[lvl](hs_lvl)
            out_obj_rot = self.obj_rot[lvl](hs_lvl)
            out_obj_rad = self.obj_rad[lvl](hs_lvl)
            
            outputs_classes.append(outputs_class.to(torch.float32))
            outputs_mano_pose.append(out_mano_pose)
            outputs_mano_beta.append(out_mano_beta)
            outputs_hand_cams.append(out_hand_cam)
            outputs_obj_cams.append(out_obj_cam)
            outputs_obj_radians.append(out_obj_rad)
            outputs_obj_rotations.append(out_obj_rot)

        if self.two_stage:
            outputs_hand_coord = torch.stack(outputs_hand_coord_list)
            outputs_obj_coord = torch.stack(outputs_obj_coord_list)

        outputs_class = torch.stack(outputs_classes)
        outputs_mano_params = [torch.stack(outputs_mano_pose), torch.stack(outputs_mano_beta)]
        outputs_obj_params = [torch.stack(outputs_obj_radians), torch.stack(outputs_obj_rotations)]
        outputs_cams = [torch.stack(outputs_hand_cams), torch.stack(outputs_obj_cams)]

        out = {
            'pred_logits': outputs_class[-1], 'pred_hand_key': outputs_hand_coord[-1], 'pred_obj_key': outputs_obj_coord[-1],
            'pred_mano_params': [outputs_mano_params[0][-1], outputs_mano_params[1][-1]],
            'pred_obj_params': [outputs_obj_params[0][-1], outputs_obj_params[1][-1]],
            'pred_cams': [outputs_cams[0][-1], outputs_cams[1][-1]]
        }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_hand_coord, outputs_obj_coord,
                outputs_mano_params, outputs_obj_params, outputs_cams
            )

        if self.two_stage:
            enc_outputs_hand_coord = enc_outputs_hand_coord_unact.sigmoid() * 2 - 1
            enc_outputs_obj_coord = enc_outputs_obj_coord_unact.sigmoid() * 2 - 1

            out['interm_outputs'] = {
                'pred_logits': enc_outputs_class,
                'pred_hand_key': enc_outputs_hand_coord,
                'pred_obj_key': enc_outputs_obj_coord
            }
        return out

    @torch.jit.unused
    def _set_aux_loss(
            self, outputs_class, outputs_hand_coord, outputs_obj_coord,
            outputs_mano_params, outputs_obj_params, outputs_cams
        ): 
        return [
                {
                    'pred_logits': c, 'pred_hand_key': hk, 'pred_obj_key': ok,
                    'pred_mano_params': [s, p], 'pred_obj_params': [ra, ro], 'pred_cams': [hc, oc]
                } \
                for c, hk, ok, s, p, ra, ro, hc, oc in \
                    zip(
                        outputs_class[:-1], outputs_hand_coord[:-1], outputs_obj_coord[:-1],
                        outputs_mano_params[0][:-1], outputs_mano_params[1][:-1],
                        outputs_obj_params[0][:-1], outputs_obj_params[1][:-1],
                        outputs_cams[0][:-1], outputs_cams[1][:-1]
                    )
            ] 


class SetArcticCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, cfg=None, pre_process_models=None):
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
        self.pre_process_models = pre_process_models

    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        labels = [t for i, t in enumerate(targets['labels']) if targets['is_valid'][i] == 1]
        target_classes_o = torch.cat([torch.tensor(t)[indices[i][1]] for i, t in enumerate(labels)]).to('cuda')

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
        # tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # tgt_lengths = torch.as_tensor([v.shape[-1] for v in targets['labels']], device=device)
        tgt_lengths = torch.as_tensor([len(l) for l in targets['labels']], device=device)
        
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        # card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_pred = (pred_logits.argmax(-1) != 0).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_hand_key' in outputs and 'pred_obj_key' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_hand_key = outputs['pred_hand_key'][idx]
        src_obj_key = outputs['pred_obj_key'][idx]

        labels = [t for i, t in enumerate(targets['labels']) if targets['is_valid'][i] == 1]
        keypoints = [t for i, t in enumerate(targets['keypoints']) if targets['is_valid'][i] == 1]

        target_keypoints = torch.cat([t[indices[i][1]] for i, t in enumerate(keypoints)]).to('cuda')
        target_labels = torch.cat([torch.tensor(t)[indices[i][1]] for i, t in enumerate(labels)]).to('cuda')

        #
        hand_cal_idx = (target_labels == 12) + (target_labels == 13)
        obj_cal_idx = ~hand_cal_idx
        
        loss_handkey = F.l1_loss(src_hand_key[hand_cal_idx], target_keypoints[hand_cal_idx], reduction='none')
        loss_objkey = F.l1_loss(src_obj_key[obj_cal_idx], target_keypoints[obj_cal_idx], reduction='none')

        losses = {}
        if len(loss_handkey) != 0:
            losses['loss_hand_keypoint'] = (loss_handkey.sum() / hand_cal_idx.sum().item()) / 21
        else:
            losses['loss_hand_keypoint'] = torch.tensor(0).cuda()
        losses['loss_obj_keypoint'] = (loss_objkey.sum() / obj_cal_idx.sum().item()) / 21

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
            'boxes' : self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, args, meta_info):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'interm_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        # idx = self.select_indices(cfg, outputs_without_aux, targets, device)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # num_boxes = sum(len(t["labels"]) for t in targets)
        losses = {}
        if isinstance(indices, list):
            num_boxes = sum([len(v) for v in targets['labels']])
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

            # Compute all the requested losses
            for loss in self.losses:
                kwargs = {}
                losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))
                # losses.update(self.get_loss(loss, outputs, targets, idx, num_boxes, **kwargs))
        else:
            assert isinstance(indices, int)

        pred = get_arctic_item(outputs, self.cfg, args.device)
        losses.update(compute_small_loss(pred, targets, meta_info, self.pre_process_models, args.img_res))
        # arctic_pred = data.search('pred.', replace_to='')
        # arctic_gt = data.search('targets.', replace_to='')
        # losses.update(compute_loss(arctic_pred, arctic_gt, meta_info, args))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                # idx = self.select_indices(cfg, aux_outputs, targets, device)

                if isinstance(indices, list):
                    for loss in self.losses:
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs['log'] = False
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
                    # l_dict = compute_loss(aux_arctic_pred, aux_arctic_gt, meta_info, args)
                else:
                    assert isinstance(indices, int)
                    
                # aux_data = prepare_data(args, aux_outputs, targets, meta_info, self.cfg)
                # aux_arctic_pred = aux_data.search('pred.', replace_to='')
                # aux_arctic_gt = aux_data.search('targets.', replace_to='')
                aux_pred = get_arctic_item(aux_outputs, self.cfg, args.device)
                l_dict = compute_small_loss(aux_pred, targets, meta_info, self.pre_process_models, args.img_res)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
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

    transformer = build_deforamble_transformer(args)
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
        window_size= args.window_size,
        feature_type=args.feature_type
    )
    matcher = build_matcher(args, cfg)

    # TODO this is a hack
    # loss_weights = {
    #     'loss_ce': args.cls_loss_coef,
    #     # 'loss_ce': 1,
    #     # 'class_error':1,
    #     # 'cardinality_error':1,
    #     "loss/mano/cam_t/r":1.0,
    #     "loss/mano/cam_t/l":1.0,
    #     "loss/object/cam_t":1.0,
    #     # "loss/mano/kp2d/r":5.0,
    #     "loss/mano/kp3d/r":5.0,
    #     "loss/mano/pose/r":10.0,
    #     "loss/mano/beta/r":0.001,
    #     # "loss/mano/kp2d/l":5.0,
    #     "loss/mano/kp3d/l":5.0,
    #     "loss/mano/pose/l":10.0,
    #     "loss/cd":1.0,
    #     # "loss/cd":10.0,
    #     "loss/mano/transl/l":1.0,
    #     # "loss/mano/transl/l":10.0,
    #     "loss/mano/beta/l":0.001,
    #     # "loss/object/kp2d":1.0,
    #     "loss/object/kp3d":5.0,
    #     "loss/object/radian":1.0,
    #     "loss/object/rot":1.0,
    #     # "loss/object/rot":10.0,
    #     "loss/object/transl":1.0,
    #     # "loss/object/transl":10.0,
    #     # "loss/penetr": 0.1,
    #     "loss/penetr": 0.005,
    #     # "loss/smooth/2d": 10.0,
    #     # "loss/smooth/3d": 10.0,
    # }        
    loss_weights = {
        'loss_ce': args.cls_loss_coef,
        # 'loss_hand_keypoint': args.keypoint_loss_coef,
        # 'loss_obj_keypoint': args.keypoint_loss_coef,        
        # # 'loss_ce': 1,
        "loss/object/v3d_smoothing":0.0005,
        
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
        # # "loss/cd":1.0,
        "loss/cd":10.0,
        "loss/mano/transl/l":10.0,
        "loss/mano/beta/l":0.001,
        "loss/object/kp2d":1.0,
        "loss/object/kp3d":5.0,
        "loss/object/radian":1.0,
        "loss/object/rot":1000.0,
        "loss/object/transl":10.0,
        # # "loss/object/transl":10.0,
        # # "loss/penetr": 0.1,
        # "loss/penetr": 0.05,
        # "loss/smooth/2d": 1.0,
        # "loss/smooth/3d": 1.0,
    }
    if args.two_stage:
        loss_weights['loss_hand_keypoint'] = args.keypoint_loss_coef
        loss_weights['loss_obj_keypoint'] = args.keypoint_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in loss_weights.items()})
        aux_weight_dict.update({k + f'_interm': v for k, v in loss_weights.items()})
        loss_weights.update(aux_weight_dict)

    if args.two_stage:
        losses = ['labels', 'boxes']
    else:
        losses = ['labels']

    # losses = ['labels', 'cardinality', 'mano_poses', 'mano_betas', 'cam', 'obj_rotation']
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25

    obj_tensor = ObjectTensors()
    obj_tensor.to(args.device)
    pre_process_models = {
        "mano_r": build_mano_aa(is_rhand=True).to(args.device),
        "mano_l": build_mano_aa(is_rhand=False).to(args.device),
        "arti_head": obj_tensor
    }    
    criterion = SetArcticCriterion(num_classes, matcher, loss_weights, losses, focal_alpha=args.focal_alpha, cfg=cfg,
                                   pre_process_models=pre_process_models)
    criterion.to(device)

    return model, criterion