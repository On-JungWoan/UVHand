import torch
import torch.nn as nn
import arctic_tools.common.camera as camera
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix

from src.utils.loss_modules import (
    compute_contact_devi_loss,
    hand_kp3d_loss,
    joints_loss,
    mano_loss,
    object_kp3d_loss,
    vector_loss,
    compute_penetration_loss,
    compute_smooth_loss,
)

l1_loss = nn.L1Loss(reduction="none")
mse_loss = nn.MSELoss(reduction="none")


def compute_loss(pred, gt, meta_info, args, device='cuda'):
    # unpacking pred and gt
    pred_betas_r = pred["mano.beta.r"]
    pred_rotmat_r = pred["mano.pose.r"]
    pred_joints_r = pred["mano.j3d.cam.r"]
    pred_projected_keypoints_2d_r = pred["mano.j2d.norm.r"]
    pred_betas_l = pred["mano.beta.l"]
    pred_rotmat_l = pred["mano.pose.l"]
    pred_joints_l = pred["mano.j3d.cam.l"]
    pred_projected_keypoints_2d_l = pred["mano.j2d.norm.l"]
    pred_kp2d_o = pred["object.kp2d.norm"]
    pred_kp3d_o = pred["object.kp3d.cam"]
    pred_rot = pred["object.rot"].view(-1, 3).float()
    pred_radian = pred["object.radian"].view(-1).float()

    gt_pose_r = gt["mano.pose.r"]
    gt_betas_r = gt["mano.beta.r"]
    gt_joints_r = gt["mano.j3d.cam.r"]
    gt_keypoints_2d_r = gt["mano.j2d.norm.r"]
    gt_pose_l = gt["mano.pose.l"]
    gt_betas_l = gt["mano.beta.l"]
    gt_joints_l = gt["mano.j3d.cam.l"]
    gt_keypoints_2d_l = gt["mano.j2d.norm.l"]
    gt_kp2d_o = torch.cat((gt["object.kp2d.norm.t"], gt["object.kp2d.norm.b"]), dim=1)
    gt_kp3d_o = gt["object.kp3d.cam"]
    gt_rot = gt["object.rot"].view(-1, 3).float()
    gt_radian = gt["object.radian"].view(-1).float()

    is_valid = gt["is_valid"]
    right_valid = gt["right_valid"]
    left_valid = gt["left_valid"]
    joints_valid_r = gt["joints_valid_r"]
    joints_valid_l = gt["joints_valid_l"]

    # reshape
    gt_pose_r = axis_angle_to_matrix(gt_pose_r.reshape(-1, 3)).reshape(-1, 16, 3, 3)
    gt_pose_l = axis_angle_to_matrix(gt_pose_l.reshape(-1, 3)).reshape(-1, 16, 3, 3)
    pred_rotmat_r = axis_angle_to_matrix(pred_rotmat_r.reshape(-1, 3)).reshape(-1, 16, 3, 3)
    pred_rotmat_l = axis_angle_to_matrix(pred_rotmat_l.reshape(-1, 3)).reshape(-1, 16, 3, 3)

    # Compute loss on MANO parameters
    loss_regr_pose_r, loss_regr_betas_r = mano_loss(
        pred_rotmat_r,
        pred_betas_r,
        gt_pose_r,
        gt_betas_r,
        criterion=mse_loss,
        is_valid=right_valid,
    )
    loss_regr_pose_l, loss_regr_betas_l = mano_loss(
        pred_rotmat_l,
        pred_betas_l,
        gt_pose_l,
        gt_betas_l,
        criterion=mse_loss,
        is_valid=left_valid,
    )

    # Compute 2D reprojection loss for the keypoints
    loss_keypoints_r = joints_loss(
        pred_projected_keypoints_2d_r,
        gt_keypoints_2d_r,
        criterion=mse_loss,
        jts_valid=joints_valid_r,
    )
    loss_keypoints_l = joints_loss(
        pred_projected_keypoints_2d_l,
        gt_keypoints_2d_l,
        criterion=mse_loss,
        jts_valid=joints_valid_l,
    )

    loss_keypoints_o = vector_loss(
        pred_kp2d_o, gt_kp2d_o, criterion=mse_loss, is_valid=is_valid
    )

    # Compute 3D keypoint loss
    loss_keypoints_3d_r = hand_kp3d_loss(
        pred_joints_r, gt_joints_r, mse_loss, joints_valid_r
    )
    loss_keypoints_3d_l = hand_kp3d_loss(
        pred_joints_l, gt_joints_l, mse_loss, joints_valid_l
    )
    loss_keypoints_3d_o = object_kp3d_loss(pred_kp3d_o, gt_kp3d_o, mse_loss, is_valid)

    loss_radian = vector_loss(pred_radian, gt_radian, mse_loss, is_valid)
    loss_rot = vector_loss(pred_rot, gt_rot, mse_loss, is_valid)
    loss_transl_l = vector_loss(
        pred["mano.cam_t.wp.l"] - pred["mano.cam_t.wp.r"],
        gt["mano.cam_t.wp.l"] - gt["mano.cam_t.wp.r"],
        mse_loss,
        right_valid * left_valid,
    )
    loss_transl_o = vector_loss(
        pred["object.cam_t.wp"] - pred["mano.cam_t.wp.r"],
        gt["object.cam_t.wp"] - gt["mano.cam_t.wp.r"],
        mse_loss,
        right_valid * is_valid,
    )

    loss_cam_t_r = vector_loss(
        pred["mano.cam_t.wp.r"],
        gt["mano.cam_t.wp.r"],
        mse_loss,
        right_valid,
    )
    loss_cam_t_l = vector_loss(
        pred["mano.cam_t.wp.l"],
        gt["mano.cam_t.wp.l"],
        mse_loss,
        left_valid,
    )
    loss_cam_t_o = vector_loss(
        pred["object.cam_t.wp"], gt["object.cam_t.wp"], mse_loss, is_valid
    )

    # cdev loss
    cd_ro, cd_lo = compute_contact_devi_loss(pred, gt)

    # penetraion loss
    pl_or, pl_ol = compute_penetration_loss(pred, gt, meta_info)

    # # motion smooth loss
    # smooth_2d_loss = compute_smooth_loss(args, 2,
    #     pred_projected_keypoints_2d_r, gt_keypoints_2d_r,
    #     pred_projected_keypoints_2d_l, gt_keypoints_2d_l,
    #     pred_kp2d_o, gt_kp2d_o,
    #     joints_valid_r, joints_valid_l, is_valid
    # )
    # smooth_3d_loss = compute_smooth_loss(args, 3,
    #     pred_joints_r, gt_joints_r,
    #     pred_joints_l, gt_joints_l,
    #     pred_kp3d_o, gt_kp3d_o,
    #     joints_valid_r, joints_valid_l, is_valid
    # )

    loss_dict = {
        "loss/mano/cam_t/r": loss_cam_t_r.to(device),
        "loss/mano/cam_t/l": loss_cam_t_l.to(device),
        "loss/object/cam_t": loss_cam_t_o.to(device),
        "loss/mano/kp2d/r": loss_keypoints_r.to(device),
        "loss/mano/kp3d/r": loss_keypoints_3d_r.to(device),
        "loss/mano/pose/r": loss_regr_pose_r.to(device),
        "loss/mano/beta/r": loss_regr_betas_r.to(device),
        "loss/mano/kp2d/l": loss_keypoints_l.to(device),
        "loss/mano/kp3d/l": loss_keypoints_3d_l.to(device),
        "loss/mano/pose/l": loss_regr_pose_l.to(device),
        "loss/cd": cd_ro.to(device) + cd_lo.to(device),
        "loss/mano/transl/l": loss_transl_l.to(device),
        "loss/mano/beta/l": loss_regr_betas_l.to(device),
        "loss/object/kp2d": loss_keypoints_o.to(device),
        "loss/object/kp3d": loss_keypoints_3d_o.to(device),
        "loss/object/radian": loss_radian.to(device),
        "loss/object/rot": loss_rot.to(device),
        "loss/object/transl": loss_transl_o.to(device),
        "loss/penetr": pl_or.to(device) + pl_ol.to(device),
        # "loss/smooth/2d": smooth_2d_loss.to(device),
        # "loss/smooth/3d": smooth_3d_loss.to(device),
    }
    
    return loss_dict


def compute_small_loss(pred, gt, meta_info, pre_process_models, img_res, device='cuda'):
    # unpacking pred and gt
    root, mano_pose, mano_shape, obj_angle = pred
    root_l, root_r, root_o = root
    pred_betas_l, pred_betas_r = mano_shape
    pred_rotmat_l, pred_rotmat_r = mano_pose
    pred_rot, pred_radian = obj_angle

    gt_pose_r = gt["mano.pose.r"]
    gt_betas_r = gt["mano.beta.r"]
    gt_pose_l = gt["mano.pose.l"]
    gt_betas_l = gt["mano.beta.l"]
    gt_joints_l = gt["mano.j3d.cam.l"]
    gt_joints_r = gt["mano.j3d.cam.r"]
    gt_kp3d_o = gt["object.kp3d.cam"]
    gt_rot = gt["object.rot"].view(-1, 3).float()
    gt_radian = gt["object.radian"].view(-1).float()
    joints_valid_r = gt["joints_valid_r"]
    joints_valid_l = gt["joints_valid_l"]    

    is_valid = gt["is_valid"]
    right_valid = gt["right_valid"]
    left_valid = gt["left_valid"]

    K = meta_info["intrinsics"]
    query_names = meta_info["query_names"]
    avg_focal_length = (K[:, 0, 0] + K[:, 1, 1]) / 2.0
    cam_t_r = camera.weak_perspective_to_perspective_torch(root_r, focal_length=avg_focal_length, img_res=img_res, min_s=0.1)
    cam_t_l = camera.weak_perspective_to_perspective_torch(root_l, focal_length=avg_focal_length, img_res=img_res, min_s=0.1)
    cam_t_o = camera.weak_perspective_to_perspective_torch(root_o, focal_length=avg_focal_length, img_res=img_res, min_s=0.1)


    # pre process
    mano_output_r = pre_process_models['mano_r'](
        betas=pred_betas_r,
        hand_pose=pred_rotmat_r[:, 3:],
        global_orient=pred_rotmat_r[:, :3],
    )
    mano_output_l = pre_process_models['mano_l'](
        betas=pred_betas_l,
        hand_pose=pred_rotmat_l[:, 3:],
        global_orient=pred_rotmat_l[:, :3],
    )
    obj_output = pre_process_models['arti_head'].forward(
        pred_radian.view(-1, 1), pred_rot, None, query_names
    )
    
    # r hand
    joints3d_cam_r = mano_output_r.joints + cam_t_r[:, None, :]
    v3d_cam_r = mano_output_r.vertices + cam_t_r[:, None, :]

    # l hand
    joints3d_cam_l = mano_output_l.joints + cam_t_l[:, None, :]
    v3d_cam_l = mano_output_l.vertices + cam_t_l[:, None, :]

    # obj
    kp3d_cam_o = obj_output["kp3d"] + cam_t_o[:, None, :]
    v3d_cam_o = obj_output["v"] + cam_t_o[:, None, :]

    tmp_pred = {}
    tmp_pred["mano.v3d.cam.l"] = v3d_cam_l
    tmp_pred["mano.v3d.cam.r"] = v3d_cam_r
    tmp_pred["object.v.cam"] = v3d_cam_o

    # reshape
    gt_pose_r = axis_angle_to_matrix(gt_pose_r.reshape(-1, 3)).reshape(-1, 16, 3, 3)
    gt_pose_l = axis_angle_to_matrix(gt_pose_l.reshape(-1, 3)).reshape(-1, 16, 3, 3)
    pred_rotmat_r = axis_angle_to_matrix(pred_rotmat_r.reshape(-1, 3)).reshape(-1, 16, 3, 3)
    pred_rotmat_l = axis_angle_to_matrix(pred_rotmat_l.reshape(-1, 3)).reshape(-1, 16, 3, 3)

    # Compute loss on MANO parameters
    loss_regr_pose_r, loss_regr_betas_r = mano_loss(
        pred_rotmat_r,
        pred_betas_r,
        gt_pose_r,
        gt_betas_r,
        criterion=mse_loss,
        is_valid=right_valid,
    )
    loss_regr_pose_l, loss_regr_betas_l = mano_loss(
        pred_rotmat_l,
        pred_betas_l,
        gt_pose_l,
        gt_betas_l,
        criterion=mse_loss,
        is_valid=left_valid,
    )

    # rotation
    loss_radian = vector_loss(pred_radian, gt_radian, mse_loss, is_valid)
    loss_rot = vector_loss(pred_rot, gt_rot, mse_loss, is_valid)
    loss_transl_l = vector_loss(
        root_l - root_r,
        gt["mano.cam_t.wp.l"] - gt["mano.cam_t.wp.r"],
        mse_loss,
        right_valid * left_valid,
    )
    loss_transl_o = vector_loss(
        root_o - root_r,
        gt["object.cam_t.wp"] - gt["mano.cam_t.wp.r"],
        mse_loss,
        right_valid * is_valid,
    )

    loss_cam_t_r = vector_loss(
        root_r,
        gt["mano.cam_t.wp.r"],
        mse_loss,
        right_valid,
    )
    loss_cam_t_l = vector_loss(
        root_l,
        gt["mano.cam_t.wp.l"],
        mse_loss,
        left_valid,
    )
    loss_cam_t_o = vector_loss(
        root_o, gt["object.cam_t.wp"], mse_loss, is_valid
    )

    # Compute 3D keypoint loss
    loss_keypoints_3d_r = hand_kp3d_loss(
        joints3d_cam_r, gt_joints_r, mse_loss, joints_valid_r
    )
    loss_keypoints_3d_l = hand_kp3d_loss(
        joints3d_cam_l, gt_joints_l, mse_loss, joints_valid_l
    )
    loss_keypoints_3d_o = object_kp3d_loss(kp3d_cam_o, gt_kp3d_o, mse_loss, is_valid)    

    # cdev
    cd_ro, cd_lo = compute_contact_devi_loss(tmp_pred, gt)

    loss_dict = {
        "loss/mano/cam_t/r": loss_cam_t_r.to(device),
        "loss/mano/cam_t/l": loss_cam_t_l.to(device),
        "loss/object/cam_t": loss_cam_t_o.to(device),
        "loss/mano/pose/r": loss_regr_pose_r.to(device),
        "loss/mano/beta/r": loss_regr_betas_r.to(device),
        "loss/mano/pose/l": loss_regr_pose_l.to(device),
        "loss/mano/transl/l": loss_transl_l.to(device),
        "loss/mano/beta/l": loss_regr_betas_l.to(device),
        "loss/object/radian": loss_radian.to(device),
        "loss/object/rot": loss_rot.to(device),
        "loss/object/transl": loss_transl_o.to(device),
        "loss/cd": cd_ro.to(device) + cd_lo.to(device),
        "loss/mano/kp3d/r": loss_keypoints_3d_r.to(device),
        "loss/mano/kp3d/l": loss_keypoints_3d_l.to(device),
        "loss/object/kp3d": loss_keypoints_3d_o.to(device),
    }
    
    return loss_dict