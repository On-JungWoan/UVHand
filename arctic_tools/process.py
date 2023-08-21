
import torch

from arctic_tools.src.nets.obj_heads.obj_head import ArtiHead
from arctic_tools.common.body_models import MANODecimator, build_mano_aa
from arctic_tools.src.callbacks.process.process_arctic import process_data

import arctic_tools.src.utils.interfield as inter
import arctic_tools.common.data_utils as data_utils
from arctic_tools.common.xdict import xdict
from pytorch3d.transforms import matrix_to_axis_angle
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix
import arctic_tools.common.ld_utils as ld_utils
from arctic_tools.src.nets.obj_heads.obj_head import ArtiHead
from arctic_tools.src.nets.hand_heads.mano_head import MANOHead
from arctic_tools.common.body_models import build_layers
from arctic_tools.src.utils.eval_modules import eval_fn_dict

def arctic_pre_process(args, targets, meta_info):
    pre_process_models = {
        "mano_r": build_mano_aa(is_rhand=True).to(args.device),
        "mano_l": build_mano_aa(is_rhand=False).to(args.device),
        "arti_head": ArtiHead(focal_length=args.focal_length, img_res=args.img_res, device=args.device).to(args.device)
    }
    with torch.no_grad():
        inputs, targets, meta_info = process_data(
            pre_process_models, None, targets, meta_info, 'extract', args
        )

    move_keys = ["object.v_len"]
    for key in move_keys:
        meta_info[key] = targets[key]
    meta_info["mano.faces.r"] = pre_process_models['mano_r'].faces
    meta_info["mano.faces.l"] = pre_process_models['mano_l'].faces

    meta_info = xdict(meta_info)
    meta_info.overwrite("part_ids", targets["object.parts_ids"])
    meta_info.overwrite("diameter", targets["object.diameter"])    

    return targets, meta_info

def post_process_arctic_output(outputs, meta_info, args, cfg):
    # model settings
    mano_r_head = MANOHead(is_rhand=True, focal_length=args.focal_length, img_res=args.img_res).to(args.device)
    mano_l_head = MANOHead(is_rhand=False, focal_length=args.focal_length, img_res=args.img_res).to(args.device)
    arti_head = ArtiHead(focal_length=args.focal_length, img_res=args.img_res, device=args.device)

    # variable settings
    query_names = meta_info["query_names"]
    out_logits = outputs['pred_logits']
    hand_cam, obj_cam = outputs['pred_cams']
    mano_pose, mano_shape = outputs['pred_mano_params']
    out_obj_rad, out_obj_rot = outputs['pred_obj_params']
    prob = out_logits.sigmoid()
    B, num_queries, num_classes = prob.shape
    K = meta_info["intrinsics"]

    # query index select
    best_score = torch.zeros(B).to(args.device)
    # if dataset != 'AssemblyHands':
    obj_idx = torch.zeros(B).to(args.device).to(torch.long)
    for i in range(1, cfg.hand_idx[0]):
        score, idx = torch.max(prob[:,:,i], dim=-1)
        obj_idx[best_score < score] = idx[best_score < score]
        best_score[best_score < score] = score[best_score < score]

    hand_idx = []
    for i in cfg.hand_idx:
        hand_idx.append(torch.argmax(prob[:,:,i], dim=-1)) 
    # hand_idx = torch.stack(hand_idx, dim=-1) 
    left_hand_idx, right_hand_idx = hand_idx

    # extract cam
    root_r=root_l=root_o=mano_pose_l=mano_pose_r=mano_shape_l=mano_shape_r=obj_rot=obj_rad = torch.tensor([]).to(args.device)
    for b in range(B):
        root_r = torch.cat([root_r, hand_cam[b, left_hand_idx[b], :].unsqueeze(0)])
        root_l = torch.cat([root_l, hand_cam[b, right_hand_idx[b], :].unsqueeze(0)])
        root_o = torch.cat([root_o, obj_cam[b, obj_idx[b], :].unsqueeze(0)])
        # extract mano param
        mano_pose_l = torch.cat([mano_pose_l, mano_pose[b, left_hand_idx[b], :].unsqueeze(0)])
        mano_pose_r = torch.cat([mano_pose_r, mano_pose[b, right_hand_idx[b], :].unsqueeze(0)])
        mano_shape_l = torch.cat([mano_shape_l, mano_shape[b, left_hand_idx[b], :].unsqueeze(0)])
        mano_shape_r = torch.cat([mano_shape_r, mano_shape[b, right_hand_idx[b], :].unsqueeze(0)])
        # extract rotation
        obj_rot = torch.cat([obj_rot, out_obj_rot[b, obj_idx[b], :].unsqueeze(0)])
        obj_rad = torch.cat([obj_rad, out_obj_rad[b, obj_idx[b], :].unsqueeze(0)])

    mano_pose_r = axis_angle_to_matrix(mano_pose_r.reshape(-1, 3)).reshape(-1, 16, 3, 3)
    mano_pose_l = axis_angle_to_matrix(mano_pose_l.reshape(-1, 3)).reshape(-1, 16, 3, 3)

    mano_output_r = mano_r_head(
        rotmat=mano_pose_r,
        shape=mano_shape_r,
        K=K,
        cam=root_r,
    )
    mano_output_l = mano_l_head(
        rotmat=mano_pose_l,
        shape=mano_shape_l,
        K=K,
        cam=root_l,
    )
    arti_output = arti_head(
        rot=obj_rot,
        angle=obj_rad,
        query_names=query_names,
        cam=root_o,
        K=K,
    )

    mano_output_r = ld_utils.prefix_dict(mano_output_r, "mano.")
    mano_output_l = ld_utils.prefix_dict(mano_output_l, "mano.")
    arti_output = ld_utils.prefix_dict(arti_output, "object.")
    output = xdict()
    output.merge(mano_output_r)
    output.merge(mano_output_l)
    output.merge(arti_output)  

    return output

def fk_params_batch(batch, layers, meta_info, device):
    mano_r = layers["right"]
    mano_l = layers["left"]
    object_tensors = layers["object_tensors"]
    pose_r = batch[f"mano.pose.r"].reshape(-1, 48)
    pose_l = batch[f"mano.pose.l"].reshape(-1, 48)
    cam_r = batch[f"mano.cam_t.r"].view(-1, 1, 3)
    cam_l = batch[f"mano.cam_t.l"].view(-1, 1, 3)
    cam_o = batch[f"object.cam_t"].view(-1, 1, 3)

    out_r = mano_r(
        global_orient=pose_r[:, :3].reshape(-1, 3),
        hand_pose=pose_r[:, 3:].reshape(-1, 45),
        betas=batch["mano.beta.r"].view(-1, 10),
    )
    out_l = mano_l(
        global_orient=pose_l[:, :3].reshape(-1, 3),
        hand_pose=pose_l[:, 3:].reshape(-1, 45),
        betas=batch["mano.beta.l"].view(-1, 10),
    )
    query_names = meta_info["query_names"]
    out_o = object_tensors.forward(
        batch["object.radian"].view(-1, 1).to(device),
        batch["object.rot"].view(-1, 3).to(device),
        None,
        query_names,
    )
    v3d_r = out_r.vertices + cam_r
    v3d_l = out_l.vertices + cam_l
    v3d_o = out_o["v"] + cam_o
    j3d_r = out_r.joints + cam_r
    j3d_l = out_l.joints + cam_l
    out = {
        "mano.v3d.cam.r": v3d_r,
        "mano.v3d.cam.l": v3d_l,
        "mano.j3d.cam.r": j3d_r,
        "mano.j3d.cam.l": j3d_l,
        "object.v.cam": v3d_o,
        "object.v_len": out_o["v_len"],
        "object.diameter": out_o["diameter"],
        "object.parts_ids": out_o["parts_ids"],
    }
    for k,v in out.items():
        if k in batch.keys():
            batch.overwrite(k, v)
        else:
            batch[k] = v
    # batch.merge(out)

    return batch

def prepare_interfield(targets, max_dist):
    test_key = ["dist.ro","dist.lo","dist.or","dist.ol","idx.ro","idx.lo","idx.or","idx.ol"]
    assert len([k for k in targets.keys() if k in test_key]) == 8
    return targets

    dist_min = 0.0
    dist_max = max_dist
    dist_ro, dist_ro_idx = inter.compute_dist_mano_to_obj(
        targets["mano.v3d.cam.r"],
        targets["object.v.cam"],
        targets["object.v_len"],
        dist_min,
        dist_max,
    )
    dist_lo, dist_lo_idx = inter.compute_dist_mano_to_obj(
        targets["mano.v3d.cam.l"],
        targets["object.v.cam"],
        targets["object.v_len"],
        dist_min,
        dist_max,
    )
    dist_or, dist_or_idx = inter.compute_dist_obj_to_mano(
        targets["mano.v3d.cam.r"],
        targets["object.v.cam"],
        targets["object.v_len"],
        dist_min,
        dist_max,
    )
    dist_ol, dist_ol_idx = inter.compute_dist_obj_to_mano(
        targets["mano.v3d.cam.l"],
        targets["object.v.cam"],
        targets["object.v_len"],
        dist_min,
        dist_max,
    )

    targets["dist.ro"] = dist_ro
    targets["dist.lo"] = dist_lo
    targets["dist.or"] = dist_or
    targets["dist.ol"] = dist_ol

    targets["idx.ro"] = dist_ro_idx
    targets["idx.lo"] = dist_lo_idx
    targets["idx.or"] = dist_or_idx
    targets["idx.ol"] = dist_ol_idx
    return targets

def prepare_data(args, outputs, targets, meta_info, cfg):
    targets = xdict(targets)
    meta_info = xdict(meta_info)

    pred = post_process_arctic_output(outputs, meta_info, args, cfg)
    keys = list(pred.keys())
    for key in keys:
        # denormalize 2d keypoints
        if "2d.norm" in key:
            denorm_key = key.replace(".norm", "")
            assert key in targets.keys(), f"Do not have key {key}"

            val_pred = pred[key]
            val_gt = targets[key]

            val_denorm_pred = data_utils.unormalize_kp2d(
                val_pred, args.img_res
            )
            val_denorm_gt = data_utils.unormalize_kp2d(val_gt, args.img_res)

            pred[denorm_key] = val_denorm_pred
            targets[denorm_key] = val_denorm_gt
    
    layers = build_layers(args.device)
    pred.overwrite(
        "mano.pose.r", matrix_to_axis_angle(pred["mano.pose.r"])
    )
    pred.overwrite(
        "mano.pose.l", matrix_to_axis_angle(pred["mano.pose.l"])
    )

    # fk_params_batch
    # targets = fk_params_batch(xdict(targets), layers, meta_info, args.device)
    pred = fk_params_batch(pred, layers, meta_info, args.device)
    # meta_info.overwrite("part_ids", targets["object.parts_ids"])
    # meta_info.overwrite("diameter", targets["object.diameter"])
    # targets = prepare_interfield(targets, max_dist=0.1)

    data = xdict()
    data.merge(pred.prefix("pred."))
    data.merge(targets.prefix("targets."))
    data.merge(meta_info.prefix("meta_info."))
    data = data.to("cpu")

    return data

def measure_error(data, metrics):
    pred = data.search("pred.", replace_to="")
    targets = data.search("targets.", replace_to="")
    meta_info = data.search("meta_info.", replace_to="")

    metric_dict = xdict()
    for metric in metrics:
        # each metric returns a tensor with shape (N, )
        if metric in ['mdev', 'acc_err_pose']:
            continue
        out = eval_fn_dict[metric](pred, targets, meta_info)
        metric_dict.merge(out)
    metric_dict = metric_dict.to_np()
    return metric_dict