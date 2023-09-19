import json
import os.path as op

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

import common.data_utils as data_utils
import common.rot as rot
import common.transforms as tf
import src.datasets.dataset_utils as dataset_utils
from common.data_utils import read_img
from common.object_tensors import ObjectTensors
from src.datasets.dataset_utils import get_valid, pad_jts2d
from arctic_tools.common.data_utils import unormalize_kp2d
from arctic_tools.common.data_utils import transform, get_transform

from cfg import Config as cfg

class ArcticDataset(Dataset):
    def __getitem__(self, index):
        imgname = self.imgnames[index]
        data = self.getitem(imgname)
        return data

    def getitem(self, imgname, load_rgb=True):
        args = self.args
        root = op.join(args.coco_path, args.dataset_file)

        # LOADING START
        speedup = args.speedup
        sid, seq_name, view_idx, image_idx = imgname.split("/")[-4:]
        obj_name = seq_name.split("_")[0]
        view_idx = int(view_idx)

        seq_data = self.data[f"{sid}/{seq_name}"]

        data_cam = seq_data["cam_coord"]
        data_2d = seq_data["2d"]
        data_bbox = seq_data["bbox"]
        data_params = seq_data["params"]

        vidx = int(image_idx.split(".")[0]) - self.ioi_offset[sid]
        vidx, is_valid, right_valid, left_valid = get_valid(
            data_2d, data_cam, vidx, view_idx, imgname
        )

        if view_idx == 0:
            intrx = data_params["K_ego"][vidx].copy()
        else:
            intrx = np.array(self.intris_mat[sid][view_idx - 1])

        # hands
        joints2d_r = pad_jts2d(data_2d["joints.right"][vidx, view_idx].copy())
        joints3d_r = data_cam["joints.right"][vidx, view_idx].copy()

        joints2d_l = pad_jts2d(data_2d["joints.left"][vidx, view_idx].copy())
        joints3d_l = data_cam["joints.left"][vidx, view_idx].copy()

        pose_r = data_params["pose_r"][vidx].copy()
        betas_r = data_params["shape_r"][vidx].copy()
        pose_l = data_params["pose_l"][vidx].copy()
        betas_l = data_params["shape_l"][vidx].copy()

        # distortion parameters for egocam rendering
        dist = data_params["dist"][vidx].copy()
        # NOTE:
        # kp2d, kp3d are in undistored space
        # thus, results for evaluation is in the undistorted space (non-curved)
        # dist parameters can be used for rendering in visualization

        # objects
        bbox2d = pad_jts2d(data_2d["bbox3d"][vidx, view_idx].copy())
        bbox3d = data_cam["bbox3d"][vidx, view_idx].copy()
        bbox2d_t = bbox2d[:8]
        bbox2d_b = bbox2d[8:]
        bbox3d_t = bbox3d[:8]
        bbox3d_b = bbox3d[8:]

        kp2d = pad_jts2d(data_2d["kp3d"][vidx, view_idx].copy())
        kp3d = data_cam["kp3d"][vidx, view_idx].copy()
        kp2d_t = kp2d[:16]
        kp2d_b = kp2d[16:]
        kp3d_t = kp3d[:16]
        kp3d_b = kp3d[16:]

        obj_radian = data_params["obj_arti"][vidx].copy()

        image_size = self.image_sizes[sid][view_idx]
        image_size = {"width": image_size[0], "height": image_size[1]}

        bbox = data_bbox[vidx, view_idx]  # original bbox
        is_egocam = "/0/" in imgname

        # LOADING END

        # SPEEDUP PROCESS
        (
            joints2d_r,
            joints2d_l,
            kp2d_b,
            kp2d_t,
            bbox2d_b,
            bbox2d_t,
            bbox,
        ) = dataset_utils.transform_2d_for_speedup(
            speedup,
            is_egocam,
            joints2d_r,
            joints2d_l,
            kp2d_b,
            kp2d_t,
            bbox2d_b,
            bbox2d_t,
            bbox,
            args.ego_image_scale,
        )
        img_status = True
        if load_rgb:
            if speedup:
                imgname = imgname.replace("/images/", "/cropped_images/")
            imgname = imgname.replace(
                "/arctic_data/", "/data/arctic_data/data/"
            ).replace("/data/data/", "/data/")
            # imgname = imgname.replace("/arctic_data/", "/data/arctic_data/")
            cv_img, img_status = read_img(op.join(root, imgname[2:]), (2800, 2000, 3))

            if img_status==False:
                is_valid == 0            
        else:
            norm_img = None

        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        # augment parameters
        augm_dict = data_utils.augm_params(
            self.aug_data,
            args.flip_prob,
            args.noise_factor,
            90, #args.rot_factor,
            args.scale_factor,
        )

        use_gt_k = args.use_gt_k
        if is_egocam:
            # no scaling for egocam to make intrinsics consistent
            use_gt_k = True
            augm_dict["sc"] = 1.0

        # visualization #
        # import matplotlib.pyplot as plt
        # import cv2
        # cv_img = cv_img.astype(np.uint8)
        # for bbox in bbox2d_t:
        #     x = int(bbox[0])
        #     y = int(bbox[1])
        #     cv2.line(cv_img, (x, y), (x,y), (255,0,0), 3)

        # for bbox in bbox2d_b:
        #     x = int(bbox[0])
        #     y = int(bbox[1])
        #     cv2.line(cv_img, (x, y), (x,y), (0,255,0), 3)            
        # plt.imshow(cv_img)
        # from arctic_tools.common.data_utils import unnormalize_2d_kp
        # test = unnormalize_2d_kp(bbox2d_t, args.img_res)
        # test[:, 0] = (test[:, 0] * (840/224)).astype(np.int64)
        # (test[:, 0] * (840/224)).astype(np.int64)

        joints2d_r = data_utils.j2d_processing(
            joints2d_r, center, scale, augm_dict, args.img_res
        )
        joints2d_l = data_utils.j2d_processing(
            joints2d_l, center, scale, augm_dict, args.img_res
        )
        kp2d_b = data_utils.j2d_processing(
            kp2d_b, center, scale, augm_dict, args.img_res
        )
        kp2d_t = data_utils.j2d_processing(
            kp2d_t, center, scale, augm_dict, args.img_res
        )
        bbox2d_b = data_utils.j2d_processing(
            bbox2d_b, center, scale, augm_dict, args.img_res
        )
        bbox2d_t = data_utils.j2d_processing(
            bbox2d_t, center, scale, augm_dict, args.img_res
        )
        bbox2d = np.concatenate((bbox2d_t, bbox2d_b), axis=0)

        # from arctic_tools.common.data_utils import unnormalize_2d_kp
        # # test = unnormalize_2d_kp(targets['object.cam_t.wp'].cpu(), args.img_res)
        # cv_img = cv_img.astype(np.int32)
        # test = unnormalize_2d_kp(bbox2d, args.img_res)
        # import cv2

        # for i in range(16):
        #     x = int((test[i][0]) / (224/840))
        #     y = int((test[i][1]-32) / (224/840))
        #     cv2.line(cv_img, (x, y), (x, y), (0,255,0), 10)     
        # plt.imshow(cv_img)   

        kp2d = np.concatenate((kp2d_t, kp2d_b), axis=0)

        # data augmentation: image
        if load_rgb:
            img = data_utils.rgb_processing(
                self.aug_data,
                cv_img,
                center,
                scale,
                augm_dict,
                img_res=args.img_res,
            )
            img = torch.from_numpy(img).float()
            norm_img = self.normalize_img(img)

        # exporting starts
        # inputs = {}
        inputs = norm_img
        targets = {}
        meta_info = {}
        # inputs["img"] = norm_img
        meta_info["imgname"] = '/'.join(imgname.split('/')[-4:])
        rot_r = data_cam["rot_r_cam"][vidx, view_idx]
        rot_l = data_cam["rot_l_cam"][vidx, view_idx]

        pose_r = np.concatenate((rot_r, pose_r), axis=0)
        pose_l = np.concatenate((rot_l, pose_l), axis=0)

        # hands
        targets["mano.pose.r"] = torch.from_numpy(
            data_utils.pose_processing(pose_r, augm_dict)
        ).float()
        targets["mano.pose.l"] = torch.from_numpy(
            data_utils.pose_processing(pose_l, augm_dict)
        ).float()
        targets["mano.beta.r"] = torch.from_numpy(betas_r).float()
        targets["mano.beta.l"] = torch.from_numpy(betas_l).float()
        targets["mano.j2d.norm.r"] = torch.from_numpy(joints2d_r[:, :2]).float()
        targets["mano.j2d.norm.l"] = torch.from_numpy(joints2d_l[:, :2]).float()

        # object
        targets["object.kp3d.full.b"] = torch.from_numpy(kp3d_b[:, :3]).float()
        targets["object.kp2d.norm.b"] = torch.from_numpy(kp2d_b[:, :2]).float()
        targets["object.kp3d.full.t"] = torch.from_numpy(kp3d_t[:, :3]).float()
        targets["object.kp2d.norm.t"] = torch.from_numpy(kp2d_t[:, :2]).float()

        targets["object.bbox3d.full.b"] = torch.from_numpy(bbox3d_b[:, :3]).float()
        targets["object.bbox2d.norm.b"] = torch.from_numpy(bbox2d_b[:, :2]).float()
        targets["object.bbox3d.full.t"] = torch.from_numpy(bbox3d_t[:, :3]).float()
        targets["object.bbox2d.norm.t"] = torch.from_numpy(bbox2d_t[:, :2]).float()
        targets["object.radian"] = torch.FloatTensor(np.array(obj_radian))

        targets["object.kp2d.norm"] = torch.from_numpy(kp2d[:, :2]).float()
        targets["object.bbox2d.norm"] = torch.from_numpy(bbox2d[:, :2]).float()

        # compute RT from cano space to augmented space
        # this transform match j3d processing
        obj_idx = self.obj_names.index(obj_name)
        meta_info["kp3d.cano"] = self.kp3d_cano[obj_idx] / 1000  # meter
        kp3d_cano = meta_info["kp3d.cano"].numpy()
        kp3d_target = targets["object.kp3d.full.b"][:, :3].numpy()

        # rotate canonical kp3d to match original image
        R, _ = tf.solve_rigid_tf_np(kp3d_cano, kp3d_target)
        obj_rot = (
            rot.batch_rot2aa(torch.from_numpy(R).float().view(1, 3, 3)).view(3).numpy()
        )

        # multiply rotation from data augmentation
        obj_rot_aug = rot.rot_aa(obj_rot, augm_dict["rot"])
        targets["object.rot"] = torch.FloatTensor(obj_rot_aug).view(1, 3)

        # full image camera coord
        targets["mano.j3d.full.r"] = torch.FloatTensor(joints3d_r[:, :3])
        targets["mano.j3d.full.l"] = torch.FloatTensor(joints3d_l[:, :3])
        targets["object.kp3d.full.b"] = torch.FloatTensor(kp3d_b[:, :3])

        meta_info["query_names"] = obj_name
        meta_info["window_size"] = torch.LongTensor(np.array([args.window_size]))

        # scale and center in the original image space
        scale_original = max([image_size["width"], image_size["height"]]) / 200.0
        center_original = [image_size["width"] / 2.0, image_size["height"] / 2.0]
        intrx = data_utils.get_aug_intrix(
            intrx,
            args.focal_length,
            args.img_res,
            use_gt_k,
            center_original[0],
            center_original[1],
            augm_dict["sc"] * scale_original,
        )

        if is_egocam and self.egocam_k is None:
            self.egocam_k = intrx
        elif is_egocam and self.egocam_k is not None:
            intrx = self.egocam_k

        meta_info["intrinsics"] = torch.FloatTensor(intrx)
        if not is_egocam:
            dist = dist * float("nan")
        meta_info["dist"] = torch.FloatTensor(dist)
        meta_info["center"] = torch.tensor(center, dtype=torch.float32)
        meta_info["is_flipped"] = torch.tensor(augm_dict["flip"])
        meta_info["rot_angle"] = torch.tensor(augm_dict["rot"], dtype=torch.float32)
        # meta_info["sample_index"] = index

        # root and at least 3 joints inside image
        targets["is_valid"] = torch.tensor(is_valid, dtype=torch.float32)
        targets["left_valid"] = float(left_valid) * targets["is_valid"]
        targets["right_valid"] = float(right_valid) * targets["is_valid"]
        targets["joints_valid_r"] = torch.ones(21) * targets["right_valid"]
        targets["joints_valid_l"] = torch.ones(21) * targets["left_valid"]

        # save keypoints & labels for reference points
        label = []
        obj2idx = cfg(args).obj2idx
        hand_idx = cfg(args).hand_idx
        label.append(obj2idx[meta_info['query_names']])

        small_obj_idx = [idx for idx in range(32) if idx %3 != 0]
        keypoints = targets['object.kp2d.norm'][small_obj_idx].unsqueeze(0)
        
        assert isinstance(left_valid, np.int64)
        assert isinstance(right_valid, np.int64)
        # l hand
        if left_valid == 1:
            label.append(hand_idx[0])
            keypoints = torch.cat([targets["mano.j2d.norm.l"].unsqueeze(0), keypoints])
        # r hand
        if right_valid == 1:
            label.append(hand_idx[1])
            keypoints = torch.cat([targets["mano.j2d.norm.r"].unsqueeze(0), keypoints])
        keypoints = unormalize_kp2d(keypoints, args.img_res)

        # re-normalize
        # import cv2
        # import copy
        # import matplotlib.pyplot as plt
        # from arctic_tools.common.data_utils import denormalize_images, transform

        # test_img = copy.deepcopy(img)
        # test_img = denormalize_images(test_img)[0].permute(1,2,0).cpu().numpy()
        # test_img = (test_img*255).astype(np.uint8)
        # test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        # color = [(255,0,0), (0,255,0), (0,0,255)]
        # t = get_transform(center, augm_dict['sc']*scale, [args.img_res, args.img_res], rot=0)
        # for b in range(len(keypoints)):
        #     for k_idx in range(len(keypoints[b])):
        #         tmp = torch.ones(3)
        #         tmp[:2] = keypoints[b][k_idx]
        #         xy = np.dot(np.linalg.inv(t), tmp)                
        #         # xy = transform(keypoints[b][k_idx], center, augm_dict['sc']*scale, [args.img_res, args.img_res], invert=1, rot=0)

        #         x = int((xy[0]/840) * 224)
        #         # y = int((160*xy[1]/(600*224)) + 32/224)
        #         y = int(
        #             224*((xy[1]/(600*224)*160) + 32/224)
        #         )
        #         cv2.line(test_img, (x, y), (x, y), color[b], 5)
        # plt.imshow(test_img)

        for b in range(
            len(keypoints)):
            for k_idx in range(len(keypoints[b])):
                xy = transform(keypoints[b][k_idx], center, augm_dict['sc']*scale, [args.img_res, args.img_res], invert=1, rot=0)

                x = (xy[0]/840)
                y = (
                    160*xy[1]/(600*224) + 32/224
                )
                keypoints[b][k_idx][0] = x
                keypoints[b][k_idx][1] = y
        keypoints = keypoints.view(-1, 42)

        if args.method == 'arctic_lstm':
            targets["labels"] = [(label)]
            targets['keypoints'] = [keypoints]
        else:
            targets["labels"] = torch.tensor(label)
            targets['keypoints'] = keypoints
        # targets["obj_hand_valid"] = torch.cat([targets["is_valid"].unsqueeze(0), targets["left_valid"].unsqueeze(0), targets["right_valid"].unsqueeze(0)])

        return inputs, targets, meta_info

    def _process_imgnames(self, seq, split):
        imgnames = self.imgnames
        if seq is not None:
            imgnames = [imgname for imgname in imgnames if "/" + seq + "/" in imgname]
        assert len(imgnames) == len(set(imgnames))
        imgnames = dataset_utils.downsample(imgnames, split)
        self.imgnames = imgnames

    def _load_data(self, args, split, seq):
        self.args = args
        self.split = split
        self.aug_data = split.endswith("train")
        # during inference, turn off
        if seq is not None:
            self.aug_data = False
        self.normalize_img = Normalize(mean=args.img_norm_mean, std=args.img_norm_std)

        if "train" in split:
            self.mode = "train"
        elif "val" in split:
            self.mode = "val"
        elif "test" in split:
            self.mode = "test"

        short_split = split.replace("mini", "").replace("tiny", "").replace("small", "")
        root = op.join(args.coco_path, args.dataset_file)
        data_p = op.join(
            root, f"data/arctic_data/data/splits/{args.setup}_{short_split}.npy"
        )
        logger.info(f"Loading {data_p}")
        data = np.load(data_p, allow_pickle=True).item()

        self.data = data["data_dict"]
        self.imgnames = data["imgnames"]

        with open(op.join(root, "data/arctic_data/data/meta/misc.json"), "r") as f:
            misc = json.load(f)

        # unpack
        subjects = list(misc.keys())
        intris_mat = {}
        world2cam = {}
        image_sizes = {}
        ioi_offset = {}
        for subject in subjects:
            world2cam[subject] = misc[subject]["world2cam"]
            intris_mat[subject] = misc[subject]["intris_mat"]
            image_sizes[subject] = misc[subject]["image_size"]
            ioi_offset[subject] = misc[subject]["ioi_offset"]

        self.world2cam = world2cam
        self.intris_mat = intris_mat
        self.image_sizes = image_sizes
        self.ioi_offset = ioi_offset

        object_tensors = ObjectTensors()
        self.kp3d_cano = object_tensors.obj_tensors["kp_bottom"]
        self.obj_names = object_tensors.obj_tensors["names"]
        self.egocam_k = None

    def __init__(self, args, split, seq=None):
        self._load_data(args, split, seq)
        self._process_imgnames(seq, split)
        logger.info(
            f"ImageDataset Loaded {self.split} split, num samples {len(self.imgnames)}"
        )

    def __len__(self):
        return len(self.imgnames)

    def getitem_eval(self, imgname, load_rgb=True):
        args = self.args
        imgname = op.join(args.coco_path, args.dataset_file, imgname[2:])
        
        # LOADING START
        speedup = args.speedup
        sid, seq_name, view_idx, image_idx = imgname.split("/")[-4:]
        obj_name = seq_name.split("_")[0]
        view_idx = int(view_idx)

        seq_data = self.data[f"{sid}/{seq_name}"]

        data_bbox = seq_data["bbox"]
        data_params = seq_data["params"]

        vidx = int(image_idx.split(".")[0]) - self.ioi_offset[sid]

        if view_idx == 0:
            intrx = data_params["K_ego"][vidx].copy()
        else:
            intrx = np.array(self.intris_mat[sid][view_idx - 1])

        # distortion parameters for egocam rendering
        dist = data_params["dist"][vidx].copy()

        bbox = data_bbox[vidx, view_idx]  # original bbox
        is_egocam = "/0/" in imgname

        image_size = self.image_sizes[sid][view_idx]
        image_size = {"width": image_size[0], "height": image_size[1]}

        # SPEEDUP PROCESS
        bbox = dataset_utils.transform_bbox_for_speedup(
            speedup,
            is_egocam,
            bbox,
            args.ego_image_scale,
        )
        img_status = True
        if load_rgb:
            if speedup:
                imgname = imgname.replace("/images/", "/cropped_images/")
            imgname = imgname.replace("/arctic_data/", "/data/arctic_data/data/").replace("/data/data/", "/data/")
            cv_img, img_status = read_img(imgname, (2800, 2000, 3))
        else:
            norm_img = None

        center = [bbox[0], bbox[1]]
        scale = bbox[2]
        self.aug_data = False

        # augment parameters
        augm_dict = data_utils.augm_params(
            self.aug_data,
            args.flip_prob,
            args.noise_factor,
            args.rot_factor,
            args.scale_factor,
        )

        use_gt_k = args.use_gt_k
        if is_egocam:
            # no scaling for egocam to make intrinsics consistent
            use_gt_k = True
            augm_dict["sc"] = 1.0

        # data augmentation: image
        if load_rgb:
            img = data_utils.rgb_processing(
                self.aug_data,
                cv_img,
                center,
                scale,
                augm_dict,
                img_res=args.img_res,
            )
            img = torch.from_numpy(img).float()
            norm_img = self.normalize_img(img)

        # exporting starts
        inputs = {}
        targets = {}
        meta_info = {}
        inputs["img"] = norm_img
        meta_info["imgname"] = imgname

        meta_info["query_names"] = obj_name
        meta_info["window_size"] = torch.LongTensor(np.array([args.window_size]))

        # scale and center in the original image space
        scale_original = max([image_size["width"], image_size["height"]]) / 200.0
        center_original = [image_size["width"] / 2.0, image_size["height"] / 2.0]
        intrx = data_utils.get_aug_intrix(
            intrx,
            args.focal_length,
            args.img_res,
            use_gt_k,
            center_original[0],
            center_original[1],
            augm_dict["sc"] * scale_original,
        )

        if is_egocam and self.egocam_k is None:
            self.egocam_k = intrx
        elif is_egocam and self.egocam_k is not None:
            intrx = self.egocam_k

        meta_info["intrinsics"] = torch.FloatTensor(intrx)
        if not is_egocam:
            dist = dist * float("nan")

        meta_info["dist"] = torch.FloatTensor(dist)
        meta_info["center"] = np.array(center, dtype=np.float32)
        meta_info["is_flipped"] = augm_dict["flip"]
        meta_info["rot_angle"] = np.float32(augm_dict["rot"])
        return inputs, targets, meta_info
