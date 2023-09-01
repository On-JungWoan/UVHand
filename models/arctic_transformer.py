# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300, cfg=None):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.n_levels=num_feature_levels
        self.n_points=dec_n_points
        self.cfg = cfg

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec, cfg)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask,
                                     spatial_shapes):
        """Generate proposals from encoded memory.

        Args:
            memory (Tensor) : The output of encoder,
                has shape (bs, num_key, embed_dim).  num_key is
                equal the number of points on feature map from
                all level.
            memory_padding_mask (Tensor): Padding mask for memory.
                has shape (bs, num_key).
            spatial_shapes (Tensor): The shape of all feature maps.
                has shape (num_level, 2).

        Returns:
            tuple: A tuple of feature map and bbox prediction.

                - output_memory (Tensor): The input of decoder,  \
                    has shape (bs, num_key, embed_dim).  num_key is \
                    equal the number of points on feature map from \
                    all levels.
                - output_proposals (Tensor): The normalized proposal \
                    after a inverse sigmoid, has shape \
                    (bs, num_keys, 4).
        """

        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H * W)].view(N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1),valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            proposal = grid.view(N, -1, 2)
            proposals.append(proposal)
            _cur += (H * W)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) &(output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2) # (B, hw, d)
            mask = mask.flatten(1) # (B, hw)
            pos_embed = pos_embed.flatten(2).transpose(1, 2) # (B, hw, d)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1) # (batch, num_featrues, 2), 2 -> h,w ratio

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory[:,level_start_index[-1]:], mask_flatten[:,level_start_index[-1]:], spatial_shapes[-1:])
            # output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.cls_embed[self.decoder.num_layers](output_memory)
            # enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
            enc_outputs_hand_coord_unact = self.decoder.keypoint_embed[self.decoder.num_layers](output_memory)#.reshape(bs, _, 21, 3) + torch.cat([output_proposals, torch.zeros(bs, _, 1).cuda()], dim=-1).unsqueeze(2)
            enc_outputs_obj_coord_unact = self.decoder.obj_keypoint_embed[self.decoder.num_layers](output_memory)#.reshape(bs, _, 21, 3) + torch.cat([output_proposals, torch.zeros(bs, _, 1).cuda()], dim=-1).unsqueeze(2)
            enc_outputs_hand_coord_unact[..., 0::3] += output_proposals[..., 0:1]
            enc_outputs_hand_coord_unact[..., 1::3] += output_proposals[..., 1:2]
            enc_outputs_obj_coord_unact[..., 0::3] += output_proposals[..., 0:1]
            enc_outputs_obj_coord_unact[..., 1::3] += output_proposals[..., 1:2]

            # topk = self.two_stage_num_proposals
            # left_proposals = torch.topk(enc_outputs_class[..., 9], 1, dim=1)[1]
            # right_proposals = torch.topk(enc_outputs_class[..., 10], 1, dim=1)[1]
            # obj_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            # topk_coords_unact = torch.gather(enc_outputs_hand_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))

            best_score = torch.zeros(bs).to(memory.device)
            obj_idx = torch.zeros(bs).to(memory.device).to(torch.long)
            for i in range(1, 9):
                score, idx = torch.max(enc_outputs_class[:,:,i], dim=-1)
                obj_idx[best_score < score] = idx[best_score < score]
                best_score[best_score < score] = score[best_score < score]

            left_idx = torch.argmax(enc_outputs_class[:,:,9], dim=-1)
            right_idx = torch.argmax(enc_outputs_class[:,:,10], dim=-1)

            # left_coords_unact = torch.gather(
            #     enc_outputs_hand_coord_unact, 1,
            #     left_proposals.unsqueeze(-1).repeat(
            #         1, 1, enc_outputs_hand_coord_unact.size(-1)))

            left_kp = torch.gather(enc_outputs_hand_coord_unact, 1, left_idx.unsqueeze(1).unsqueeze(1).repeat(1,1,63))
            right_kp = torch.gather(enc_outputs_hand_coord_unact, 1, right_idx.unsqueeze(1).unsqueeze(1).repeat(1,1,63))
            obj_kp = torch.gather(enc_outputs_obj_coord_unact, 1, obj_idx.unsqueeze(1).unsqueeze(1).repeat(1,1,63))

            topk_coords_unact = torch.cat([left_kp, right_kp, obj_kp], dim=1).detach()
            reference_points = topk_coords_unact.sigmoid()
            ref_x = reference_points[...,0::3].mean(-1).unsqueeze(-1) ########################################################### one ref-point
            ref_y = reference_points[...,1::3].mean(-1).unsqueeze(-1)
            reference_points = torch.cat([ref_x, ref_y], dim=-1)
            init_reference_out = reference_points

            # pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            # query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
            
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
 
            reference_points = self.reference_points(query_embed).sigmoid()
            # reference_points = self.reference_points(query_embed)
            init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)
        # hs : result include intermeditate feature (num_decoder_layer, B, num_queries, hidden_dim)
        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_hand_coord_unact, enc_outputs_obj_coord_unact
        return hs, init_reference_out, inter_references_out, None, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),  # [0.5, 1.5, ... , H_-0.5] -> shape:(H_)
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))  # [0.5, 1.5, ... , W_-0.5] -> shape:(W_)
                                          # -> ref_y, ref_x's shape:(H_, W_)
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_) # (H_ x W_) 로 flatten 후 [0, 1]로 normalize
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.inter_rp = nn.ReLU()
        self.attn_matrix = nn.ReLU()

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        self.inter_rp(reference_points)
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2, attn_matrix = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))
        self.attn_matrix(attn_matrix)
        tgt2 = tgt2.transpose(0,1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, cfg=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        # self.keypoint_embed = None # modify
        # self.obj_keypoint_embed = None
        self.mano_pose_embed = None
        self.mano_beta_embed = None
        self.obj_keypoint_embed = None
        self.obj_ref_embed = None
        # self.obj_rad = None
        self.cls_embed = None
        self.cfg = cfg

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        
        for lid, layer in enumerate(self.layers):          
            if reference_points.shape[-1] == 42:
                reference_points_input = reference_points[:, :, None] * src_valid_ratios.repeat(1, 1, 21)[:, None]

            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            ## hack implementation for iterative bounding box refinement ##
            if self.cls_embed is not None:
                assert sum(
                    [ebd is not None for ebd in [self.mano_pose_embed, self.mano_beta_embed, self.obj_keypoint_embed, self.obj_ref_embed]]
                ) == 4

                #################
                ## class embed ##
                #################
                cls_out = self.cls_embed[lid](output)    
                class_indices = cls_out.argmax(dim=-1)
                # hand_idx = class_indices != 0 # 1과 2는 각각 left, right hand
                    
                # obj_idx : class 예측결과 중 hand idx가 아닌 것만 추출
                # hand_idx : class 예측결과 중 hand idx만 추출
                obj_idx = torch.ones_like(class_indices, dtype=torch.bool)
                hand_idx = torch.zeros_like(class_indices, dtype=torch.bool)
                for idx in [0] + self.cfg.hand_idx:
                    obj_idx &= (class_indices != idx)
                    if idx != 0:
                        hand_idx |= (class_indices == idx)


                ######################
                ## reference points ##
                ######################
                if reference_points.shape[-1] == 2:
                    ref = inverse_sigmoid(reference_points).unsqueeze(2)
                    # ref = reference_points.unsqueeze(2)
                    new_reference_points = ref.repeat(1,1,21,1).clone()
        
                elif reference_points.shape[-1] == 42:
                    ref_x = reference_points[...,0::2].mean(-1).unsqueeze(-1)
                    ref_y = reference_points[...,1::2].mean(-1).unsqueeze(-1)
                    # if len(self.cfg.hand_idx) == 2:
                    #     new_reference_points = inverse_sigmoid(torch.cat([ref_x, ref_y], dim=-1)).unsqueeze(2).repeat(1,1,21,1).clone()
                    # else:
                    # new_reference_points = torch.cat([ref_x, ref_y], dim=-1).unsqueeze(2).repeat(1,1,21,1).clone()
                    new_reference_points = inverse_sigmoid(torch.cat([ref_x, ref_y], dim=-1)).unsqueeze(2).repeat(1,1,21,1).clone()
                    # new_reference_points = inverse_sigmoid((torch.cat([ref_x, ref_y], dim=-1)+0.5)/2).unsqueeze(2).repeat(1,1,21,1).clone()


                ################
                ## mano embed ##
                ################
                bs = class_indices.shape[0]

                out_mano_pose = self.mano_pose_embed[lid](output)
                out_mano_beta = self.mano_beta_embed[lid](output)

                tmp = self.get_reference_point(bs, [out_mano_pose,out_mano_beta], class_indices, new_reference_points) # <- 이건 구현 해야 함
                new_reference_points[hand_idx] += tmp[hand_idx]
                # new_reference_points[hand_idx] += tmp.reshape(tmp.shape[0], tmp.shape[1], -1, 3)[hand_idx][...,:2] 


                ########################
                ## obj keypoint embed ##
                ########################
                tmp = self.obj_keypoint_embed[lid](output)
                tmp = self.obj_ref_embed[lid](tmp)

                new_reference_points[obj_idx] += tmp.reshape(tmp.shape[0], tmp.shape[1], -1, 3)[obj_idx][...,:2]
                new_reference_points = new_reference_points.reshape(tmp.shape[0], tmp.shape[1], -1)
                # if len(self.cfg.hand_idx) == 2:
                new_reference_points = new_reference_points.sigmoid()
                # else:
                #     new_reference_points = new_reference_points.sigmoid()*2 -0.5
                    
                reference_points = new_reference_points.detach()
            ## modify ##

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args, cfg):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        cfg = cfg)


