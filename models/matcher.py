# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from copy import copy

class ArcticMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                cost_class: float = 1,
                cost_keypoint: float = 1,
                cfg = None):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_keypoint: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_keypoint = cost_keypoint
        self.cfg = cfg
        assert cost_class != 0 or cost_keypoint != 0, "all costs cant be 0"     

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_mano_pose, out_mano_beta = outputs['pred_mano_params']
            out_obj_rad, out_obj_rot = outputs['pred_obj_params']
            out_hand_cam, out_obj_cam = outputs['pred_cams']

            tgt_ids = torch.cat(targets['labels'], dim=1)[0]
            left_valid = targets['left_valid'].type(torch.bool)
            right_valid = targets['right_valid'].type(torch.bool)

            tgt_mano_pose_l = targets['mano.pose.l'][left_valid]
            tgt_mano_pose_r = targets['mano.pose.r'][right_valid]
            tgt_mano_beta_l = targets['mano.beta.l'][left_valid]
            tgt_mano_beta_r = targets['mano.beta.r'][right_valid]
            tgt_obj_rad = targets['object.radian']
            tgt_obj_rot = targets['object.rot']
            tgt_hand_cam_l = targets['mano.cam_t.wp.l'][left_valid]
            tgt_hand_cam_r = targets['mano.cam_t.wp.r'][right_valid]
            tgt_obj_cam = targets['object.cam_t.wp']

            
            # hand_idx = torch.zeros_like(tgt_ids, dtype=torch.bool)
            obj_idx = torch.ones_like(tgt_ids, dtype=torch.bool)
            for idx in [0] + self.cfg.hand_idx:
                obj_idx &= (tgt_ids != idx)
                # if idx != 0:
                #     hand_idx |= (tgt_ids == idx)
            left_idx = tgt_ids == self.cfg.hand_idx[0]
            right_idx = tgt_ids == self.cfg.hand_idx[1]
            # hand_idx = tgt_ids != 0
            
            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            cost_cam = torch.zeros_like(cost_class)
            cost_rad_pose = torch.zeros_like(cost_class)
            cost_rot_shape = torch.zeros_like(cost_class)

            cost_cam_left = torch.cdist(out_hand_cam, tgt_hand_cam_l.reshape(-1, 3), p=1).view(bs*num_queries, -1)
            cost_cam_right = torch.cdist(out_hand_cam, tgt_hand_cam_r.reshape(-1, 3), p=1).view(bs*num_queries, -1)
            cost_cam_obj = torch.cdist(out_obj_cam, tgt_obj_cam.reshape(-1, 3), p=1).view(bs*num_queries, -1)
            cost_cam[:, left_idx] = cost_cam_left
            cost_cam[:, right_idx] = cost_cam_right
            cost_cam[:, obj_idx] = cost_cam_obj

            cost_pose_left = torch.cdist(out_mano_pose, tgt_mano_pose_l.reshape(-1, 48), p=1).view(bs*num_queries, -1)
            cost_pose_right = torch.cdist(out_mano_pose, tgt_mano_pose_r.reshape(-1, 48), p=1).view(bs*num_queries, -1)
            cost_rad = torch.cdist(out_obj_rad, tgt_obj_rad.unsqueeze(-1), p=1).view(bs*num_queries, -1)
            cost_rad_pose[:, left_idx] = cost_pose_left
            cost_rad_pose[:, right_idx] = cost_pose_right
            cost_rad_pose[:, obj_idx] = cost_rad

            cost_beta_left = torch.cdist(out_mano_beta, tgt_mano_beta_l.reshape(-1, 10), p=1).view(bs*num_queries, -1)
            cost_beta_right = torch.cdist(out_mano_beta, tgt_mano_beta_r.reshape(-1, 10), p=1).view(bs*num_queries, -1)
            cost_rot = torch.cdist(out_obj_rot, tgt_obj_rot.reshape(-1, 3), p=1).view(bs*num_queries, -1)
            cost_rot_shape[:, left_idx] = cost_beta_left
            cost_rot_shape[:, right_idx] = cost_beta_right
            cost_rot_shape[:, obj_idx] = cost_rot

            C = self.cost_keypoint * (cost_rot_shape + cost_cam) + self.cost_class * (cost_class + cost_rad_pose)
            # C = self.cost_keypoint * cost_keypoints + self.cost_class * cost_class
            C = C.view(bs, num_queries, -1).cpu()
            # sizes = [len(v["keypoints"]) for idx, v in enumerate(targets)]
            sizes = [v.shape[-1] for v in targets['labels']]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class AssemblyMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                cost_class: float = 1,
                cost_keypoint: float = 1,
                cfg = None):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_keypoint: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_keypoint = cost_keypoint
        self.cfg = cfg
        assert cost_class != 0 or cost_keypoint != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_kp = outputs["pred_keypoints"].flatten(0,1)
            # out_objkp = outputs["pred_obj_keypoints"].flatten(0,1)

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_kp = torch.cat([v["keypoints"] for v in targets])

            # hand_idx = torch.zeros_like(tgt_ids, dtype=torch.bool)
            # obj_idx = torch.ones_like(tgt_ids, dtype=torch.bool)
            # for idx in [0] + self.cfg.hand_idx:
            #     obj_idx &= (tgt_ids != idx)
            #     if idx != 0:
            #         hand_idx |= (tgt_ids == idx)
            hand_idx = tgt_ids != 0
            
            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_keypoints = torch.zeros_like(cost_class)

            # tgt_kp = tgt_kp.reshape(-1, 63)[hand_idx]
            # occlusion_mask = tgt_kp>0

            # cost = []
            # for idx, mask in enumerate(occlusion_mask):
            #     tmp_tgt = tgt_kp[idx][mask]
            #     tmp_out = None
            #     for out in out_kp:
            #         if tmp_out is None:
            #             tmp_out = out[mask][None]
            #         else:
            #             tmp_out = torch.cat([tmp_out, out[mask][None]])
            #     cost.append(torch.cdist(tmp_out, tmp_tgt.unsqueeze(0), p=1))
            # cost_hand = torch.cat(cost, dim=1)

            cost_hand = torch.cdist(out_kp, tgt_kp.reshape(-1, 63)[hand_idx], p=1)
            # cost_obj = torch.cdist(out_objkp, tgt_kp.reshape(-1, 63)[obj_idx], p=1)

            cost_keypoints[:,hand_idx] = cost_hand
            # cost_keypoints[:,obj_idx] = cost_obj

            C = self.cost_keypoint * cost_keypoints + self.cost_class * cost_class
            C = C.view(bs, num_queries, -1).cpu()
            sizes = [len(v["keypoints"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args, cfg):
    if args.dataset_file == 'arctic':
        return ArcticMatcher(cost_class=args.set_cost_class,
                                cost_keypoint=args.set_cost_keypoint,
                                cfg = cfg)
    elif args.dataset_file == 'AssemblyHands':
        return AssemblyMatcher(cost_class=args.set_cost_class,
                                cost_keypoint=args.set_cost_keypoint,
                                cfg = cfg)
    else:
        raise Exception('Not implemeted!')