import torch
import numpy as np
import torch.nn as nn
from pytorch3d.structures import Meshes
from pytorch3d.ops.knn import knn_gather, knn_points

from common.torch_utils import nanmean
import common.torch_utils as torch_utils

l1_loss = nn.L1Loss(reduction="none")
mse_loss = nn.MSELoss(reduction="none")

def compute_smooth_loss(
        args, dim,
        right_pred, right_gt, left_pred, left_gt, obj_pred, obj_gt,
        right_valid, left_valid, is_valid
    ):
    B, N = args.batch_size, args.window_size

    # right hand
    rs = compute_acc_vel_loss(
        right_pred.reshape(B, N, -1),
        right_gt.reshape(B, N, -1),
        criterion=mse_loss,
        valid=right_valid.reshape(B, N, -1).repeat(1,1,dim),
    )

    # left hand
    ls = compute_acc_vel_loss(
        left_pred.reshape(B, N, -1),
        left_gt.reshape(B, N, -1),
        criterion=mse_loss,
        valid=left_valid.reshape(B, N, -1).repeat(1,1,dim),
    )

    # object
    _, C, D = obj_pred.shape
    is_valid = is_valid.unsqueeze(-1).unsqueeze(-1).repeat(1, C, D)
    obj_pred = obj_pred * is_valid
    obj_gt = obj_gt * is_valid
    os = compute_acc_vel_loss(
        obj_pred.reshape(B, N, -1),
        obj_gt.reshape(B, N, -1),
        criterion=mse_loss,
    )

    return rs + ls + os


def compute_acc_vel_loss(pred, gt, criterion, valid=None):
    pred = pred.permute(0,2,1) # N, C, T
    gt = gt.permute(0,2,1) # N, C, T

    # check validation
    if valid is not None:
        valid = valid.permute(0,2,1)
        pred = pred * valid
        gt = gt * valid

    # velocity
    vel_pred = pred[..., 1:] - pred[..., :-1]
    vel_gt = gt[..., 1:] - gt[..., :-1]

    # accel
    acc_pred = vel_pred[..., 1:] - vel_pred[..., :-1]
    acc_gt = vel_gt[..., 1:] - vel_gt[..., :-1]

    loss_vel = criterion(vel_pred, vel_gt).mean()
    loss_acc = criterion(acc_pred, acc_gt).mean()
    return 1.0 * loss_vel + 1.0 * loss_acc


def compute_penetration_loss(pred, gt, meta_info):
    is_valid = gt['is_valid']
    left_valid = gt['left_valid']
    right_valid = gt['right_valid']
    B = is_valid.size(0)

    hand_face_r = torch.tensor(meta_info['mano.faces.r'].astype(np.int64)).unsqueeze(0).repeat(B,1,1)
    hand_face_l = torch.tensor(meta_info['mano.faces.l'].astype(np.int64)).unsqueeze(0).repeat(B,1,1)
    pred_obj_xyz = pred["object.v.cam"]

    pl_or = penetration_loss(
        hand_face_r,
        pred["mano.v3d.cam.r"],
        pred_obj_xyz,
        pred['nn_dist_r'],
        pred['nn_idx_r'],
        is_valid,
        right_valid
    )
    pl_ol = penetration_loss(
        hand_face_l,
        pred["mano.v3d.cam.l"],
        pred_obj_xyz,
        pred['nn_dist_l'],
        pred['nn_idx_l'],
        is_valid,
        left_valid
    )

    return pl_or, pl_ol


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
    nn_dists = src_nn.dists[..., 0]
    nn_idx = src_nn.idx[..., 0]
    return nn_dists, nn_idx


def penetration_loss(
        hand_face, pred_hand_xyz, pred_obj_xyz, nn_dist, nn_idx, is_valid, hand_valid
    ):
    # batch size
    B = hand_face.size(0)

    # validation check
    valid_info = hand_valid.clone() * is_valid
    invalid_idx = (1 - valid_info).nonzero()[:, 0]
    
    # construct meshes
    mesh = Meshes(verts=pred_hand_xyz, faces=hand_face)
    hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)

    # [B,778,3] -> [B,3947,3] or 3947?
    NN_src_xyz = batched_index_select(pred_hand_xyz, nn_idx)
    NN_src_normal = batched_index_select(hand_normal, nn_idx)

    # get interior
    NN_vector = NN_src_xyz - pred_obj_xyz  # [B, 3000, 3]
    interior = (NN_vector * NN_src_normal).sum(dim=-1) > 0

    # validation check
    interior[invalid_idx] = False

    # get penetration loss
    penetr_dist = 120 * nn_dist[interior].sum() / B

    return penetr_dist


def batched_index_select(input, index, dim=1):
    '''
    :param input: [B, N1, *]
    :param dim: the dim to be selected
    :param index: [B, N2]
    :return: [B, N2, *] selected result
    '''
    views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim=dim, index=index)


def subtract_root_batch(joints: torch.Tensor, root_idx: int):
    assert len(joints.shape) == 3
    assert joints.shape[2] == 3
    joints_ra = joints.clone()
    root = joints_ra[:, root_idx : root_idx + 1].clone()
    joints_ra = joints_ra - root
    return joints_ra


def compute_contact_devi_loss(pred, targets):
    cd_ro = contact_deviation(
        pred["object.v.cam"],
        pred["mano.v3d.cam.r"],
        targets["dist.ro"],
        targets["idx.ro"],
        targets["is_valid"],
        targets["right_valid"],
    )

    cd_lo = contact_deviation(
        pred["object.v.cam"],
        pred["mano.v3d.cam.l"],
        targets["dist.lo"],
        targets["idx.lo"],
        targets["is_valid"],
        targets["left_valid"],
    )
    cd_ro = nanmean(cd_ro)
    cd_lo = nanmean(cd_lo)
    cd_ro = torch.nan_to_num(cd_ro)
    cd_lo = torch.nan_to_num(cd_lo)
    return cd_ro, cd_lo


def contact_deviation(pred_v3d_o, pred_v3d_r, dist_ro, idx_ro, is_valid, _right_valid):
    right_valid = _right_valid.clone() * is_valid
    contact_dist = 3 * 1e-3  # 3mm considered in contact
    vo_r_corres = torch.gather(pred_v3d_o, 1, idx_ro[:, :, None].repeat(1, 1, 3))

    # displacement vector H->O
    disp_ro = vo_r_corres - pred_v3d_r  # batch, num_v, 3
    invalid_ridx = (1 - right_valid).nonzero()[:, 0]
    disp_ro[invalid_ridx] = float("nan")
    disp_ro[dist_ro > contact_dist] = float("nan")
    cd = (disp_ro**2).sum(dim=2).sqrt()
    err_ro = torch_utils.nanmean(cd, axis=1)  # .cpu().numpy()  # m
    return err_ro


def keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, criterion, jts_valid):
    """
    Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
    The loss is weighted by the confidence.
    """

    gt_root = gt_keypoints_3d[:, :1, :]
    gt_keypoints_3d = gt_keypoints_3d - gt_root
    pred_root = pred_keypoints_3d[:, :1, :]
    pred_keypoints_3d = pred_keypoints_3d - pred_root

    return joints_loss(pred_keypoints_3d, gt_keypoints_3d, criterion, jts_valid)


def object_kp3d_loss(pred_3d, gt_3d, criterion, is_valid):
    num_kps = pred_3d.shape[1] // 2
    pred_3d_ra = subtract_root_batch(pred_3d, root_idx=num_kps)
    gt_3d_ra = subtract_root_batch(gt_3d, root_idx=num_kps)
    loss_kp = vector_loss(
        pred_3d_ra,
        gt_3d_ra,
        criterion=criterion,
        is_valid=is_valid,
    )
    return loss_kp


def hand_kp3d_loss(pred_3d, gt_3d, criterion, jts_valid):
    pred_3d_ra = subtract_root_batch(pred_3d, root_idx=0)
    gt_3d_ra = subtract_root_batch(gt_3d, root_idx=0)
    loss_kp = keypoint_3d_loss(
        pred_3d_ra, gt_3d_ra, criterion=criterion, jts_valid=jts_valid
    )
    return loss_kp


def vector_loss(pred_vector, gt_vector, criterion, is_valid=None):
    dist = criterion(pred_vector, gt_vector)
    if is_valid.sum() == 0:
        return torch.zeros((1)).to(gt_vector.device)
    if is_valid is not None:
        valid_idx = is_valid.long().bool()
        dist = dist[valid_idx]
    loss = dist.mean().view(-1)
    return loss


def joints_loss(pred_vector, gt_vector, criterion, jts_valid):
    dist = criterion(pred_vector, gt_vector)
    if jts_valid is not None:
        dist = dist * jts_valid[:, :, None]
    loss = dist.mean().view(-1)
    return loss


def mano_loss(pred_rotmat, pred_betas, gt_rotmat, gt_betas, criterion, is_valid=None):
    loss_regr_pose = vector_loss(pred_rotmat, gt_rotmat, criterion, is_valid)
    loss_regr_betas = vector_loss(pred_betas, gt_betas, criterion, is_valid)
    return loss_regr_pose, loss_regr_betas
