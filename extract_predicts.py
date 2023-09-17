import json
import os.path as op
import sys
from pprint import pformat

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
        val_loader.dataset[0]

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                batch = thing.thing2dev(batch, device)
                inputs, targets, meta_info = batch

                # meta info
                query_names = meta_info["query_names"]
                K = meta_info["intrinsics"]
                avg_focal_length = (K[:, 0, 0] + K[:, 1, 1]) / 2.0

                # extract outputs
                outputs = wrapper(inputs['img'])

                #
                bs = args.test_batch_size
                ws = args.window_size

                # targets, meta_info = arctic_pre_process(args, targets, meta_info)
                # origin_data = prepare_data(args, outputs, targets, meta_info, cfg)
                # origin_stats = measure_error(origin_data, args.eval_metrics)
                # print(nanmean(torch.tensor(origin_stats['cdev/ho'])).item())

                # select query
                root, mano_pose, mano_shape, obj = get_arctic_item(outputs, cfg, args.device)

                # root
                if args.iter != 0:
                    assert bs == 2
                    root[0] = arctic_smoothing(root[0].view(2, -1, 3), args.iter)
                    root[1] = arctic_smoothing(root[1].view(2, -1, 3), args.iter)
                    root[2] = arctic_smoothing(root[2].view(2, -1, 3), args.iter)
                    # mano
                    mano_pose[0] = arctic_smoothing(mano_pose[0].view(2, -1, 48), args.iter)
                    mano_pose[1] = arctic_smoothing(mano_pose[1].view(2, -1, 48), args.iter)
                    mano_shape[0] = arctic_smoothing(mano_shape[0].view(2, -1, 10), args.iter)
                    mano_shape[1] = arctic_smoothing(mano_shape[1].view(2, -1, 10), args.iter)
                    # obj
                    obj[0] = arctic_smoothing(obj[0].view(2, -1, 3), args.iter)
                    obj[1] = arctic_smoothing(obj[1].view(2, -1, 1), args.iter)

                # root = [targets['mano.cam_t.wp.l'], targets['mano.cam_t.wp.r'], targets['object.cam_t.wp']]
                # mano_pose = [targets['mano.pose.l'], targets['mano.pose.r']]
                # mano_shape = [targets['mano.beta.l'], targets['mano.beta.r']]
                # obj_angle = [targets['object.rot'].view(-1, 3), targets['object.radian'].view(-1, 1)]
                # pred = make_output(args, root, mano_pose, mano_shape, obj, query_names, K)

                # data = prepare_data(args, None, targets, meta_info, cfg, pred=pred)
                # stats = measure_error(data, args.eval_metrics)
                # print(nanmean(torch.tensor(stats['cdev/ho'])).item())

                # # visualize_arctic_result(args, data, 'pred')
                # continue
                
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

        out = interface.std_interface(out_list)
        interface.save_results(out, out_dir)
        logger.info("Done")


if __name__ == "__main__":
    main()
