# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]


import torch
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
# from .DABDETR import sigmoid_focal_loss
from util import box_ops
import torch.nn.functional as F


def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim, label_enc):
    """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
    if training:
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args

        new_targets = []        
        tmp_label = [torch.tensor(l).cuda() for l in targets['labels']]
        tmp_key = [key for key in targets['keypoints']]
        for l, k in zip(tmp_label, tmp_key):
            new_t = {}
            new_t['labels'] = l
            new_t['keys'] = k
            new_targets.append(new_t)
        targets = new_targets

        # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1
        unmask_key = unmask_label = torch.cat(known)
        labels = torch.cat([t['labels'] for t in targets])
        keys = torch.cat([t['keys'] for t in targets])
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_key)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_keys = keys.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_key_expand = known_keys.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = torch.tensor(range(len(keys))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(keys) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(keys)
        if box_noise_scale > 0:
            # known_bbox_ = torch.zeros_like(known_bboxs)
            # known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            # known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            # diff = torch.zeros_like(known_bboxs)
            # diff[:, :2] = known_bboxs[:, 2:] / 2
            # diff[:, 2:] = known_bboxs[:, 2:] / 2
            known_key_ = known_keys
            diff = known_key_expand

            rand_sign = torch.randint_like(known_keys, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_keys)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_key_ = known_key_ + torch.mul(rand_part,
                                                  diff).cuda() * box_noise_scale
            known_key_ = known_key_.clamp(min=0.0, max=1.0)
            # known_key_expand[:, :2] = (known_key_[:, :2] + known_key_[:, 2:]) / 2
            # known_key_expand[:, 2:] = known_key_[:, 2:] - known_key_[:, :2]

        m = known_labels_expaned.long().to('cuda')
        input_label_embed = label_enc(m)
        input_key_embed = inverse_sigmoid(known_key_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_key = torch.zeros(pad_size, 42).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_key = padding_key.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_key[(known_bid.long(), map_known_indice)] = input_key_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': dn_number,
        }
    else:

        input_query_label = None
        input_query_key = None
        attn_mask = None
        dn_meta = None

    return input_query_label, input_query_key, attn_mask, dn_meta


def dn_post_process(
        outputs_class, hand_coord, obj_coord, mano_param, cam_param, obj_param,
        dn_meta, aux_loss, _set_aux_loss
    ):
    """
        post process of dn after output from the transformer
        put the dn part in the dn_meta
    """
    if dn_meta and dn_meta['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :dn_meta['pad_size'], :]
        output_known_hand_coord = hand_coord[:, :, :dn_meta['pad_size'], :]
        output_known_obj_coord = obj_coord[:, :, :dn_meta['pad_size'], :]
        mano_known_param = [
            mano_param[0][:, :, :dn_meta['pad_size'], :], mano_param[1][:, :, :dn_meta['pad_size'], :]
        ]
        cam_known_param = [
            cam_param[0][:, :, :dn_meta['pad_size'], :], cam_param[1][:, :, :dn_meta['pad_size'], :]
        ]
        obj_known_param = [
            obj_param[0][:, :, :dn_meta['pad_size'], :], obj_param[1][:, :, :dn_meta['pad_size'], :]
        ]

        outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
        outputs_hand_coord = hand_coord[:, :, dn_meta['pad_size']:, :]
        outputs_obj_coord = obj_coord[:, :, dn_meta['pad_size']:, :]
        mano_param = [
            mano_param[0][:, :, dn_meta['pad_size']:, :], mano_param[1][:, :, dn_meta['pad_size']:, :]
        ]
        cam_param = [
            cam_param[0][:, :, dn_meta['pad_size']:, :], cam_param[1][:, :, dn_meta['pad_size']:, :]
        ]
        obj_param = [
            obj_param[0][:, :, dn_meta['pad_size']:, :], obj_param[1][:, :, dn_meta['pad_size']:, :]
        ]        

        out = {
            'pred_logits': output_known_class[-1], 'pred_hand_key': output_known_hand_coord[-1], 'pred_obj_key': output_known_obj_coord[-1],
            'pred_mano_params': [mano_known_param[0][-1], mano_known_param[1][-1]],
            'pred_cams': [cam_known_param[0][-1], cam_known_param[1][-1]],
            'pred_obj_params': [obj_known_param[0][-1], obj_known_param[1][-1]],
        }
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(
                output_known_class, output_known_hand_coord, output_known_obj_coord,
                mano_known_param, cam_known_param, obj_known_param
            )
        dn_meta['output_known_lbs_bboxes'] = out
    return outputs_class, outputs_hand_coord, outputs_obj_coord, mano_param, cam_param, obj_param


