import json
import os.path as op
import sys
from pprint import pformat
import pickle

import torch
from loguru import logger
from tqdm import tqdm

from util.tools import arctic_smoothing
from arctic_tools.common.torch_utils import nanmean
from arctic_tools.visualizer import visualize_arctic_result
from arctic_tools.process import prepare_data, measure_error

# LSTM models are trained using image features from single-frame models
# this specify the single-frame model features that the LSTM model was trained on
# model_dependencies[lstm_model_id] = single_frame_model_id
model_dependencies = {
    "423c6057b": "3558f1342",
    "40ae50712": "28bf3642f",
    "546c1e997": "1f9ac0b15",
    "701a72569": "58e200d16",
    "fdc34e6c3": "66417ff6e",
    "49abdaee9": "7d09884c6",
    "5e6f6aeb9": "fb59bac27",
    "ec90691f8": "782c39821",
}


def make_stat(res, cnt):
    mean_list = []
    bound_list = []

    for i in range(cnt):
        stack_tensor = torch.cat([r[i] for r in res])
        mean_list.append(stack_tensor.mean(0))

        q1 = torch.quantile(stack_tensor, 0.25, dim=0)
        q3 = torch.quantile(stack_tensor, 0.75, dim=0)
        upper = q1 - 1.5 * (q3 - q1)
        under = q3 + 1.5 * (q3 - q1)
        bound_list.append((upper, under))
    return mean_list, bound_list


def main(args=None, wrapper=None, cfg=None):
    import origin_arctic.common.thing as thing
    import origin_arctic.src.extraction.interface as interface
    import origin_arctic.src.factory as factory
    from origin_arctic.common.xdict import xdict
    from origin_arctic.src.parsers.parser import construct_args
    import origin_arctic.common.camera as camera
    from arctic_tools.process import get_arctic_item, arctic_pre_process, make_output
    from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix

    args.experiment = None
    args.exp_key = "xxxxxxx"

    device = "cuda:0"
    # wrapper.metric_dict = []

    exp_key = op.abspath(args.load_ckpt).split("/")[-3]
    if exp_key in model_dependencies.keys():
        assert (
            args.img_feat_version == model_dependencies[exp_key]
        ), f"Image features used for training ({model_dependencies[exp_key]}) do not match the ones used for the current inference ({args.img_feat_version})"

    out_dir = op.join(args.output_dir, "eval")

    with open(
        op.join(args.coco_path, args.dataset_file,f"data/arctic_data/data/splits_json/protocol_{args.setup}.json"), "r"
    ) as f:
        seqs = json.load(f)[args.run_on]

    logger.info(f"Hyperparameters: \n {pformat(args)}")
    logger.info(f"Seqs to process ({len(seqs)}): {seqs}")

    if args.extraction_mode in ["eval_pose"]:
        from origin_arctic.src.extraction.keys.eval_pose import KEYS
    elif args.extraction_mode in ["eval_field"]:
        from origin_arctic.src.extraction.keys.eval_field import KEYS
    elif args.extraction_mode in ["submit_pose"]:
        from origin_arctic.src.extraction.keys.submit_pose import KEYS
    elif args.extraction_mode in ["submit_field"]:
        from origin_arctic.src.extraction.keys.submit_field import KEYS
    elif args.extraction_mode in ["feat_pose"]:
        from origin_arctic.src.extraction.keys.feat_pose import KEYS
    elif args.extraction_mode in ["feat_field"]:
        from origin_arctic.src.extraction.keys.feat_field import KEYS
    elif args.extraction_mode in ["vis_pose"]:
        from origin_arctic.src.extraction.keys.vis_pose import KEYS
    elif args.extraction_mode in ["vis_field"]:
        from origin_arctic.src.extraction.keys.vis_field import KEYS
    else:
        assert False, f"Invalid extract ({args.extraction_mode})"

    for seq_idx, seq in enumerate(seqs):
        logger.info(f"Processing seq {seq} {seq_idx + 1}/{len(seqs)}")
        out_list = []
        val_loader = factory.fetch_dataloader(args, "val", seq)
        # val_loader.dataset[0]

        roots = []
        mano_poses = []
        mano_shapes = []
        objes = []

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                bs = args.test_batch_size
                batch = thing.thing2dev(batch, device)
                inputs, targets, meta_info = batch

                # meta info
                query_names = meta_info["query_names"]
                K = meta_info["intrinsics"]
                avg_focal_length = (K[:, 0, 0] + K[:, 1, 1]) / 2.0

                # extract outputs
                outputs = wrapper(inputs['img'])

                # ####
                # import cv2
                # import matplotlib.pyplot as plt
                # (h, w) = inputs['img'].shape[-2:]
                # (cX, cY) = (w // 2, h // 2)
                # M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
                # for i in range(len(inputs['img'])):
                #     inputs['img'][i] = torch.tensor(cv2.warpAffine(inputs['img'][i].permute(1,2,0).cpu().numpy(), M, (w, h))).cuda().permute(2,0,1)
                # ####

                #### req ####
                targets, meta_info = arctic_pre_process(args, targets, meta_info)
                origin_data = prepare_data(args, outputs, targets, meta_info, cfg)
                origin_stats = measure_error(origin_data, args.eval_metrics)
                cdev = nanmean(torch.tensor(origin_stats['cdev/ho'])).item()
                print(f'cdev : {cdev}')
                #### req ####

                # select query
                root, mano_pose, mano_shape, obj = get_arctic_item(outputs, cfg, args.device)
                pred = make_output(args, root, mano_pose, mano_shape, obj, query_names, K)
                visualize_arctic_result(args, origin_data, 'pred')

                # if cdev >= 100:
                    # out = root, mano_pose, mano_shape, obj
                    # global_avg = [
                    #     make_stat([root], 3)[0],
                    #     make_stat([mano_pose], 2)[0],
                    #     make_stat([mano_shape], 2)[0],
                    #     make_stat([obj], 2)[0]
                    # ]
                    
                    # for list_idx, avg_list in enumerate(global_avg):
                    #     for item_idx, avg in enumerate(avg_list):
                    #         out[list_idx][item_idx] = avg.view(1, -1).repeat(bs, 1)

                    # pred = make_output(args, root, mano_pose, mano_shape, obj, query_names, K)
                    # data = prepare_data(args, None, targets, meta_info, cfg, pred=pred)
                    # stats = measure_error(data, args.eval_metrics)
                    # print(nanmean(torch.tensor(stats['cdev/ho'])).item())
                    # visualize_arctic_result(args, data, 'pred')

                # #### for test ####
                # data = xdict()
                # data.merge(pred.prefix("pred."))
                # data.merge(xdict(targets).prefix("targets."))
                # data.merge(xdict(meta_info).prefix("meta_info."))
                # visualize_arctic_result(args, data.to('cpu'), 'pred')
                # continue
                # #### for test ####

                # ####
                # if cdev < 100:
                #     roots.append(root)
                #     mano_poses.append(mano_pose)
                #     mano_shapes.append(mano_shape)
                #     objes.append(obj)
                # continue
                # ####

                # if not (targets['right_valid'].sum() == 0 and targets['left_valid'].sum() == 0):
                #     if True:
                #     # if seq not in ['s05/box_use_02', 's05/espressomachine_use_01', 's05/notebook_grab_01', 's05/phone_use_03', 's05/phone_use_04', 's05/waffleiron_grab_01']:
                #     # if seq in ['s05/box_grab_01', 's05/laptop_grab_01']:

                #         # #### req ####
                #         with open(op.join(args.output_dir, 'pkl', f"{seq.replace('/', '_')}.pkl"), 'rb') as f:
                #             global_avg = pickle.load(f)
                #         root_stat, pose_stat, shape_stat, obj_stat = global_avg
                #         root_mean, root_bound = root_stat
                #         pose_mean, pose_bound = pose_stat
                #         shape_mean, shape_bound = shape_stat
                #         obj_mean, obj_bound = obj_stat
                #         # #### req ####

                #         def find_outlier(res, bound, targets, out_idx, no_valid=False):
                #             assert len(res) == len(bound)

                #             for i, r in enumerate(res):
                #                 b, c = r.shape

                #                 low, up = bound[i]
                #                 if no_valid:
                #                     valid = torch.ones_like(targets['left_valid']).type(torch.bool)
                #                 else:
                #                     valid = targets['left_valid'] if i == 0 else targets['right_valid'].type(torch.bool)
                #                     # valid = valid.view(-1, 1).repeat(1, c)
                #                     valid = valid.type(torch.bool)

                #                 r = r.mean(-1)
                #                 low = low.mean(-1)
                #                 up = up.mean(-1)

                #                 out_idx += (((r>up) + (r<low)) * valid)
                #             return out_idx

                #         # #### req ####
                #         bs = targets['right_valid'].shape[0]
                #         out_idx = torch.zeros(bs).type(torch.bool).cuda()
                #         out_idx = find_outlier(root, root_bound, targets, out_idx)
                #         out_idx = find_outlier(mano_pose, pose_bound, targets, out_idx)
                #         out_idx = find_outlier(mano_shape, shape_bound, targets, out_idx)
                #         out_idx = find_outlier(obj, obj_bound, targets, out_idx, no_valid=True)
                #         num_outlier = out_idx.sum()
                #         # print(f'num_outlier ; {num_outlier.item()}')

                #         p_l = pred['mano.v3d.cam.l'].view(bs, 1, -1).permute(1,2,0)
                #         p_r = pred['mano.v3d.cam.r'].view(bs, 1, -1).permute(1,2,0)
                #         p_o = pred['object.v.cam'].view(bs, 1, -1).permute(1,2,0)
                #         acc_l = p_l[:,:,:-2] - 2 * p_l[:,:,1:-1] + p_l[:,:,2:]
                #         acc_r = p_r[:,:,:-2] - 2 * p_r[:,:,1:-1] + p_r[:,:,2:]
                #         acc_o = p_o[:,:,:-2] - 2 * p_o[:,:,1:-1] + p_o[:,:,2:]
                #         acc_l = acc_l.view(-1, bs-2)
                #         acc_r = acc_r.view(-1, bs-2)
                #         acc_o = acc_o.view(-1, bs-2)
                #         max_l = abs(acc_l.mean(0)).max().item()
                #         max_r = abs(acc_r.mean(0)).max().item()
                #         max_o = abs(acc_o.mean(0)).max().item()
                #         # print(
                #         #     round(max_l, 4),
                #         #     round(max_r, 4),
                #         #     round(max_o, 4),
                #         #     end = '\n'
                #         # )
                #         #### req ####

                #         # if False:
                #         # if num_outlier >= 1:
                #         # if (max_l>0.04) or (max_r>0.04) or (max_o>0.04):
                #         if cdev >= 100:
                #         # if (num_outlier >= 20 and ((max_l>0.04) or (max_r>0.04) or (max_o>0.04))):
                #             # or ((max_l>0.075) or (max_r>0.075) or (max_o>0.075)) or \
                #             #     cdev >= 85:

                #             #### req ####
                #             global_avg = root_mean, pose_mean, shape_mean, obj_mean
                #             out = root, mano_pose, mano_shape, obj
                            
                #             for list_idx, avg_list in enumerate(global_avg):
                #                 for item_idx, avg in enumerate(avg_list):
                #                     # out[list_idx][item_idx][out_idx] = avg.view(1, -1).repeat(num_outlier, 1)
                #                     out[list_idx][item_idx] = avg.view(1, -1).repeat(bs, 1)
                #             #### req ####

                #             #### req ####
                #             pred = make_output(args, root, mano_pose, mano_shape, obj, query_names, K)
                #             data = prepare_data(args, None, targets, meta_info, cfg, pred=pred)
                #             stats = measure_error(data, args.eval_metrics)
                #             print(nanmean(torch.tensor(stats['cdev/ho'])).item())
                #             #### req ####
                #             # visualize_arctic_result(args, data, 'pred')
                #             # visualize_arctic_result(args, origin_data, 'pred')

                # ####
                # # root
                # if cdev > 100:
                #     iter = 40
                #     root[0] = arctic_smoothing(root[0].view(1, -1, 3), iter)
                #     root[1] = arctic_smoothing(root[1].view(1, -1, 3), iter)
                #     root[2] = arctic_smoothing(root[2].view(1, -1, 3), iter)
                #     # mano
                #     mano_pose[0] = arctic_smoothing(mano_pose[0].view(1, -1, 48), iter)
                #     mano_pose[1] = arctic_smoothing(mano_pose[1].view(1, -1, 48), iter)
                #     mano_shape[0] = arctic_smoothing(mano_shape[0].view(1, -1, 10), iter)
                #     mano_shape[1] = arctic_smoothing(mano_shape[1].view(1, -1, 10), iter)
                #     # obj
                #     obj[0] = arctic_smoothing(obj[0].view(1, -1, 3), iter)
                #     obj[1] = arctic_smoothing(obj[1].view(1, -1, 1), iter)

                #     #### req ####
                #     pred = make_output(args, root, mano_pose, mano_shape, obj, query_names, K)
                #     data = prepare_data(args, None, targets, meta_info, cfg, pred=pred)
                #     stats = measure_error(data, args.eval_metrics)
                #     print(nanmean(torch.tensor(stats['cdev/ho'])).item())
                #     #### req ####
                #     # visualize_arctic_result(args, data, 'pred')
                #     # visualize_arctic_result(args, origin_data, 'pred')                    
                # ####

                continue
                
                root_l, root_r, root_o = root
                mano_pose_l, mano_pose_r = mano_pose
                mano_shape_l, mano_shape_r = mano_shape
                obj_rot, obj_rad = obj

                cam_t_l = camera.weak_perspective_to_perspective_torch(
                    root_l, focal_length=avg_focal_length, img_res=args.img_res, min_s=0.1
                )
                cam_t_r = camera.weak_perspective_to_perspective_torch(
                    root_r, focal_length=avg_focal_length, img_res=args.img_res, min_s=0.1
                )
                cam_t_o = camera.weak_perspective_to_perspective_torch(
                    root_o, focal_length=avg_focal_length, img_res=args.img_res, min_s=0.1
                )

                mano_pose_r = axis_angle_to_matrix(mano_pose_r.reshape(-1, 3)).reshape(-1, 16, 3, 3)
                mano_pose_l = axis_angle_to_matrix(mano_pose_l.reshape(-1, 3)).reshape(-1, 16, 3, 3)

                # save
                out_dict = {
                    'pred.mano.cam_t.l' : cam_t_l.cpu(),
                    'pred.mano.beta.l' : mano_shape_l.cpu(),
                    'pred.mano.pose.l' : mano_pose_l.cpu(),
                    'pred.mano.cam_t.r' : cam_t_r.cpu(),
                    'pred.mano.beta.r' : mano_shape_r.cpu(),
                    'pred.mano.pose.r' : mano_pose_r.cpu(),
                    'pred.object.rot' : obj_rot.cpu(),
                    'pred.object.cam_t' : cam_t_o.cpu(),
                    'pred.object.radian' : obj_rad.cpu(),
                    'meta_info.imgname' : meta_info['imgname']
                }

                if "submit_" not in args.extraction_mode:
                    targets, meta_info = arctic_pre_process(args, targets, meta_info)

                    out_dict['targets.mano.pose.r'] = targets['mano.pose.r']
                    out_dict['targets.mano.pose.l'] = targets['mano.pose.l']
                    out_dict['targets.mano.beta.r'] = targets['mano.beta.r']
                    out_dict['targets.mano.beta.l'] = targets['mano.beta.l']
                    out_dict['targets.object.radian'] = targets['object.radian']
                    out_dict['targets.object.rot'] = targets['object.rot']
                    out_dict['targets.is_valid'] = targets['is_valid']
                    out_dict['targets.left_valid'] = targets['left_valid']
                    out_dict['targets.right_valid'] = targets['right_valid']
                    out_dict['targets.joints_valid_r'] = targets['joints_valid_r']
                    out_dict['targets.joints_valid_l'] = targets['joints_valid_l']
                    out_dict['targets.mano.cam_t.r'] = targets['mano.cam_t.r']
                    out_dict['targets.mano.cam_t.l'] = targets['mano.cam_t.l']
                    out_dict['targets.object.cam_t'] = targets['object.cam_t']
                    out_dict['targets.object.bbox3d.cam'] = targets['object.bbox3d.cam']
                    
                    out_dict['meta_info.imgname'] = meta_info['imgname']
                    out_dict['meta_info.query_names'] = meta_info['query_names']
                    out_dict['meta_info.window_size'] = meta_info['window_size']
                    out_dict['meta_info.center'] = meta_info['center']
                    out_dict['meta_info.is_flipped'] = meta_info['is_flipped']
                    out_dict['meta_info.rot_angle'] = meta_info['rot_angle']
                    out_dict['meta_info.diameter'] = meta_info['diameter']

                out_dict = xdict(out_dict)
                out_dict = out_dict.subset(KEYS)
                out_list.append(out_dict)

        # ####

        # root = make_stat(roots, 3)
        # pose = make_stat(mano_poses, 2)
        # shape = make_stat(mano_shapes, 2)
        # obj = make_stat(objes, 2)

        # # roots_0_mean = torch.cat([r[0] for r in roots]).mean(0)
        # # roots_1_mean = torch.cat([r[1] for r in roots]).mean(0)
        # # roots_2_mean = torch.cat([r[2] for r in roots]).mean(0)
        # # poses_0_mean = torch.cat([r[0] for r in mano_poses]).mean(0)
        # # poses_1_mean = torch.cat([r[1] for r in mano_poses]).mean(0)
        # # shapes_0_mean = torch.cat([r[0] for r in mano_shapes]).mean(0)
        # # shapes_1_mean = torch.cat([r[1] for r in mano_shapes]).mean(0)
        # # obj_0_mean = torch.cat([r[0] for r in objes]).mean(0)
        # # obj_1_mean = torch.cat([r[1] for r in objes]).mean(0)
        
        # # root = [roots_0_mean, roots_1_mean, roots_2_mean]
        # # pose = [poses_0_mean, poses_1_mean]
        # # shape = [shapes_0_mean, shapes_1_mean]
        # # obj = [obj_0_mean, obj_1_mean]  

        # from pathlib import Path
        # save_path = op.join(args.output_dir, 'best_pkl')
        # Path(save_path).mkdir(parents=True, exist_ok=True)
        # with open(save_path+f"/{seq.replace('/', '_')}.pkl", 'wb') as f:
        #     pickle.dump([root, pose, shape, obj], f)
        # ####

        continue
        
        out = interface.std_interface(out_list)
        interface.save_results(out, out_dir)
        logger.info("Done")


if __name__ == "__main__":
    main()
