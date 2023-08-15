import numpy as np
import torch

import common.data_utils as data_utils
import common.ld_utils as ld_utils
import src.callbacks.process.process_generic as generic
from common.abstract_pl import AbstractPL
from common.body_models import MANODecimator, build_mano_aa
from common.comet_utils import push_images
from common.rend_utils import Renderer
from common.xdict import xdict
from src.utils.eval_modules import eval_fn_dict


def mul_loss_dict(loss_dict):
    for key, val in loss_dict.items():
        loss, weight = val
        loss_dict[key] = loss * weight
    return loss_dict


class GenericWrapper(AbstractPL):
    def __init__(self, args):
        super().__init__(
            args,
            push_images,
            "loss__val",
            float("inf"),
            high_loss_val=float("inf"),
        )
        self.args = args
        self.mano_r = build_mano_aa(is_rhand=True)
        self.mano_l = build_mano_aa(is_rhand=False)
        self.add_module("mano_r", self.mano_r)
        self.add_module("mano_l", self.mano_l)
        self.renderer = Renderer(img_res=args.img_res)
        self.object_sampler = np.load(
            "./data/arctic_data/data/meta/downsamplers.npy", allow_pickle=True
        ).item()

    def set_flags(self, mode):
        self.model.mode = mode
        if mode == "train":
            self.train()
        else:
            self.eval()

    def inference_pose(self, inputs, meta_info):
        pred = self.model(inputs, meta_info)
        mydict = xdict()
        mydict.merge(xdict(inputs).prefix("inputs."))
        mydict.merge(pred.prefix("pred."))
        mydict.merge(xdict(meta_info).prefix("meta_info."))
        mydict = mydict.detach()
        return mydict

    def inference_field(self, inputs, meta_info):
        meta_info = xdict(meta_info)

        models = {
            "mano_r": self.mano_r,
            "mano_l": self.mano_l,
            "arti_head": self.model.arti_head,
            "mesh_sampler": MANODecimator(),
            "object_sampler": self.object_sampler,
        }

        batch_size = meta_info["intrinsics"].shape[0]

        (
            v0_r,
            v0_l,
            v0_o,
            pidx,
            v0_r_full,
            v0_l_full,
            v0_o_full,
            mask,
            cams,
            bottom_anchor,
        ) = generic.prepare_templates(
            batch_size,
            models["mano_r"],
            models["mano_l"],
            models["mesh_sampler"],
            models["arti_head"],
            meta_info["query_names"],
        )

        meta_info["v0.r"] = v0_r
        meta_info["v0.l"] = v0_l
        meta_info["v0.o"] = v0_o

        pred = self.model(inputs, meta_info)
        mydict = xdict()
        mydict.merge(xdict(inputs).prefix("inputs."))
        mydict.merge(pred.prefix("pred."))
        mydict.merge(meta_info.prefix("meta_info."))
        mydict = mydict.detach()
        return mydict

    def forward(self, inputs, targets, meta_info, mode):
        models = {
            "mano_r": self.mano_r,
            "mano_l": self.mano_l,
            "arti_head": self.model.arti_head,
            "mesh_sampler": MANODecimator(),
            "object_sampler": self.object_sampler,
        }

        self.set_flags(mode)
        inputs = xdict(inputs)
        targets = xdict(targets)
        meta_info = xdict(meta_info)
        with torch.no_grad():
            inputs, targets, meta_info = self.process_fn(
                models, inputs, targets, meta_info, mode, self.args
            )

        move_keys = ["object.v_len"]
        for key in move_keys:
            meta_info[key] = targets[key]
        meta_info["mano.faces.r"] = self.mano_r.faces
        meta_info["mano.faces.l"] = self.mano_l.faces
        pred = self.model(inputs, meta_info)
        loss_dict = self.loss_fn(
            pred=pred, gt=targets, meta_info=meta_info, args=self.args
        )
        loss_dict = {k: (loss_dict[k][0].mean(), loss_dict[k][1]) for k in loss_dict}
        loss_dict = mul_loss_dict(loss_dict)
        loss_dict["loss"] = sum(loss_dict[k] for k in loss_dict)

        # conversion for vis and eval
        keys = list(pred.keys())
        for key in keys:
            # denormalize 2d keypoints
            if "2d.norm" in key:
                denorm_key = key.replace(".norm", "")
                assert key in targets.keys(), f"Do not have key {key}"

                val_pred = pred[key]
                val_gt = targets[key]

                val_denorm_pred = data_utils.unormalize_kp2d(
                    val_pred, self.args.img_res
                )
                val_denorm_gt = data_utils.unormalize_kp2d(val_gt, self.args.img_res)

                pred[denorm_key] = val_denorm_pred
                targets[denorm_key] = val_denorm_gt

        if mode == "train":
            return {"out_dict": (inputs, targets, meta_info, pred), "loss": loss_dict}

        if mode == "vis":
            vis_dict = xdict()
            vis_dict.merge(inputs.prefix("inputs."))
            vis_dict.merge(pred.prefix("pred."))
            vis_dict.merge(targets.prefix("targets."))
            vis_dict.merge(meta_info.prefix("meta_info."))
            vis_dict = vis_dict.detach()
            return vis_dict

        # evaluate metrics
        metrics_all = self.evaluate_metrics(
            pred, targets, meta_info, self.metric_dict
        ).to_torch()
        out_dict = xdict()
        out_dict["imgname"] = meta_info["imgname"]
        out_dict.merge(ld_utils.prefix_dict(metrics_all, "metric."))

        if mode == "extract":
            mydict = xdict()
            mydict.merge(inputs.prefix("inputs."))
            mydict.merge(pred.prefix("pred."))
            mydict.merge(targets.prefix("targets."))
            mydict.merge(meta_info.prefix("meta_info."))
            mydict = mydict.detach()
            return mydict
        return out_dict, loss_dict

    def evaluate_metrics(self, pred, targets, meta_info, specs):
        metric_dict = xdict()
        for key in specs:
            metrics = eval_fn_dict[key](pred, targets, meta_info)
            metric_dict.merge(metrics)

        return metric_dict

import common.camera as camera
import common.data_utils as data_utils
import common.transforms as tf
import src.callbacks.process.process_generic as generic


def process_data(
    models, inputs, targets, meta_info, mode, args, field_max=float("inf")
):
    img_res = 224
    K = meta_info["intrinsics"]
    gt_pose_r = targets["mano.pose.r"]  # MANO pose parameters
    gt_betas_r = targets["mano.beta.r"]  # MANO beta parameters

    gt_pose_l = targets["mano.pose.l"]  # MANO pose parameters
    gt_betas_l = targets["mano.beta.l"]  # MANO beta parameters

    gt_kp2d_b = targets["object.kp2d.norm.b"]  # 2D keypoints for object base
    gt_object_rot = targets["object.rot"].view(-1, 3)

    # pose the object without translation (call it object cano space)
    out = models["arti_head"].object_tensors.forward(
        angles=targets["object.radian"].view(-1, 1),
        global_orient=gt_object_rot,
        transl=None,
        query_names=meta_info["query_names"],
    )
    diameters = out["diameter"]
    parts_idx = out["parts_ids"]
    meta_info["part_ids"] = parts_idx
    meta_info["diameter"] = diameters

    # targets keypoints of hand and objects are in camera coord (full resolution image) space
    # map all entities from camera coord to object cano space based on the rigid-transform
    # between the object base keypoints in camera coord and object cano space
    # since R, T is used, relative distance btw hand and object is preserved
    num_kps = out["kp3d"].shape[1] // 2
    kp3d_b_cano = out["kp3d"][:, num_kps:]
    R0, T0 = tf.batch_solve_rigid_tf(targets["object.kp3d.full.b"], kp3d_b_cano)
    joints3d_r0 = tf.rigid_tf_torch_batch(targets["mano.j3d.full.r"], R0, T0)
    joints3d_l0 = tf.rigid_tf_torch_batch(targets["mano.j3d.full.l"], R0, T0)

    # pose MANO in MANO canonical space
    gt_out_r = models["mano_r"](
        betas=gt_betas_r,
        hand_pose=gt_pose_r[:, 3:],
        global_orient=gt_pose_r[:, :3],
        transl=None,
    )
    gt_model_joints_r = gt_out_r.joints
    gt_vertices_r = gt_out_r.vertices
    gt_root_cano_r = gt_out_r.joints[:, 0]

    gt_out_l = models["mano_l"](
        betas=gt_betas_l,
        hand_pose=gt_pose_l[:, 3:],
        global_orient=gt_pose_l[:, :3],
        transl=None,
    )
    gt_model_joints_l = gt_out_l.joints
    gt_vertices_l = gt_out_l.vertices
    gt_root_cano_l = gt_out_l.joints[:, 0]

    # map MANO mesh to object canonical space
    Tr0 = (joints3d_r0 - gt_model_joints_r).mean(dim=1)
    Tl0 = (joints3d_l0 - gt_model_joints_l).mean(dim=1)
    gt_model_joints_r = joints3d_r0
    gt_model_joints_l = joints3d_l0
    gt_vertices_r += Tr0[:, None, :]
    gt_vertices_l += Tl0[:, None, :]

    # now that everything is in the object canonical space
    # find camera translation for rendering relative to the object

    # unnorm 2d keypoints
    gt_kp2d_b_cano = data_utils.unormalize_kp2d(gt_kp2d_b, img_res)

    # estimate camera translation by solving 2d to 3d correspondence
    gt_transl = camera.estimate_translation_k(
        kp3d_b_cano,
        gt_kp2d_b_cano,
        meta_info["intrinsics"].cpu().numpy(),
        use_all_joints=True,
        pad_2d=True,
    )

    # move to camera coord
    gt_vertices_r = gt_vertices_r + gt_transl[:, None, :]
    gt_vertices_l = gt_vertices_l + gt_transl[:, None, :]
    gt_model_joints_r = gt_model_joints_r + gt_transl[:, None, :]
    gt_model_joints_l = gt_model_joints_l + gt_transl[:, None, :]

    ####
    gt_kp3d_o = out["kp3d"] + gt_transl[:, None, :]
    gt_bbox3d_o = out["bbox3d"] + gt_transl[:, None, :]

    # roots
    gt_root_cam_patch_r = gt_model_joints_r[:, 0]
    gt_root_cam_patch_l = gt_model_joints_l[:, 0]
    gt_cam_t_r = gt_root_cam_patch_r - gt_root_cano_r
    gt_cam_t_l = gt_root_cam_patch_l - gt_root_cano_l
    gt_cam_t_o = gt_transl

    targets["mano.cam_t.r"] = gt_cam_t_r
    targets["mano.cam_t.l"] = gt_cam_t_l
    targets["object.cam_t"] = gt_cam_t_o

    avg_focal_length = (K[:, 0, 0] + K[:, 1, 1]) / 2.0
    gt_cam_t_wp_r = camera.perspective_to_weak_perspective_torch(
        gt_cam_t_r, avg_focal_length, img_res
    )

    gt_cam_t_wp_l = camera.perspective_to_weak_perspective_torch(
        gt_cam_t_l, avg_focal_length, img_res
    )

    gt_cam_t_wp_o = camera.perspective_to_weak_perspective_torch(
        gt_cam_t_o, avg_focal_length, img_res
    )

    targets["mano.cam_t.wp.r"] = gt_cam_t_wp_r
    targets["mano.cam_t.wp.l"] = gt_cam_t_wp_l
    targets["object.cam_t.wp"] = gt_cam_t_wp_o

    # cam coord of patch
    targets["object.cam_t.kp3d.b"] = gt_transl

    targets["mano.v3d.cam.r"] = gt_vertices_r
    targets["mano.v3d.cam.l"] = gt_vertices_l
    targets["mano.j3d.cam.r"] = gt_model_joints_r
    targets["mano.j3d.cam.l"] = gt_model_joints_l
    targets["object.kp3d.cam"] = gt_kp3d_o
    targets["object.bbox3d.cam"] = gt_bbox3d_o

    out = models["arti_head"].object_tensors.forward(
        angles=targets["object.radian"].view(-1, 1),
        global_orient=gt_object_rot,
        transl=None,
        query_names=meta_info["query_names"],
    )

    # GT vertices relative to right hand root
    targets["object.v.cam"] = out["v"] + gt_transl[:, None, :]
    targets["object.v_len"] = out["v_len"]

    targets["object.f"] = out["f"]
    targets["object.f_len"] = out["f_len"]

    targets = generic.prepare_interfield(targets, field_max)

    return inputs, targets, meta_info
