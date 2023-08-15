
import torch

from arctic_tools.src.nets.obj_heads.obj_head import ArtiHead
from arctic_tools.common.body_models import MANODecimator, build_mano_aa
from arctic_tools.src.callbacks.process.process_arctic import process_data

def arctic_pre_process(args, targets, meta_info):
    pre_process_models = {
        "mano_r": build_mano_aa(is_rhand=True).to(args.device),
        "mano_l": build_mano_aa(is_rhand=False).to(args.device),
        "arti_head": ArtiHead(focal_length=args.focal_length, img_res=args.img_res).to(args.device)
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

    return targets, meta_info