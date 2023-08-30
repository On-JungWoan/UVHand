import os
import os.path as op

import numpy as np
import torch
import trimesh

import common.viewer as viewer_utils
from common.body_models import build_layers, seal_mano_mesh
from common.xdict import xdict
from src.extraction.interface import prepare_data
from src.extraction.keys.vis_pose import KEYS as keys
from arctic_tools.common.viewer import ARCTICViewer, ViewerData

def construct_meshes(data, flag, device):
    # load object faces
    obj_name = data['meta_info.query_names'][0]
    f3d_o = trimesh.load(
        f"./data/arctic_data/data/meta/object_vtemplates/{obj_name}/mesh.obj",
        process=False,
    ).faces
    layers = build_layers(device)

    # center verts
    v3d_r = data[f"{flag}.mano.v3d.cam.r"] 
    v3d_l = data[f"{flag}.mano.v3d.cam.l"] 
    v3d_o = data[f"{flag}.object.v.cam"] 
    cam_t = data[f"{flag}.object.cam_t"] 
    v3d_r -= cam_t[:, None, :] 
    v3d_l -= cam_t[:, None, :] 
    v3d_o -= cam_t[:, None, :] 

    # seal MANO mesh
    f3d_r = torch.LongTensor(layers["right"].faces.astype(np.int64))
    f3d_l = torch.LongTensor(layers["left"].faces.astype(np.int64))
    v3d_r, f3d_r = seal_mano_mesh(v3d_r, f3d_r, True)
    v3d_l, f3d_l = seal_mano_mesh(v3d_l, f3d_l, False)

    # AIT meshes
    hand_color = "white"
    object_color = "light-blue"
    right = {
        "v3d": v3d_r.numpy(),
        "f3d": f3d_r.numpy(),
        "vc": None,
        "name": "right",
        "color": hand_color,
    }
    left = {
        "v3d": v3d_l.numpy(),
        "f3d": f3d_l.numpy(),
        "vc": None,
        "name": "left",
        "color": hand_color,
    }
    obj = {
        "v3d": v3d_o.numpy(),
        "f3d": f3d_o,
        "vc": None,
        "name": "object",
        "color": object_color,
    }

    meshes = viewer_utils.construct_viewer_meshes(
        {
            "right": right,
            "left": left,
            "object": obj,
        },
        draw_edges=False,
        flat_shading=True,
    )
    return meshes, data

def visualize_arctic_result(args, data, flag):
    args.headless = False
    viewer = ARCTICViewer(
        interactive=not args.headless,
        size=(2048, 2048),
        render_types=["rgb", "video"],
    )

    meshes_all = xdict()
    meshes, data = construct_meshes(data, flag, args.device)
    meshes_all.merge(meshes)

    root = op.join(args.coco_path, args.dataset_file)
    if args.method == 'arctic_sf':
        # imgnames = [op.join(root, data['meta_info.imgname'][0][2:])]
        imgnames = [op.join(root, img[2:]) for img in data['meta_info.imgname']]
    else: #lstm
        imgnames = [op.join(root, 'data/arctic_data/data/cropped_images', img) for img in data['meta_info.imgname']]

    num_frames = min(len(imgnames), data[f"{flag}.object.cam_t"].shape[0])

    # setup camera
    focal = 1000.0
    rows = 224
    cols = 224
    K = np.array([[focal, 0, rows / 2.0], [0, focal, cols / 2.0], [0, 0, 1]])
    cam_t = data[f"{flag}.object.cam_t"]
    cam_t = cam_t[:num_frames]
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :, 3] = cam_t
    Rt[:, :3, :3] = np.eye(3)
    Rt[:, 1:3, :3] *= -1.0

    data = ViewerData(Rt=Rt, K=K, cols=cols, rows=rows, imgnames=imgnames)
    batch = meshes_all, data

    save_foler = op.join(f'results/{args.dataset_file}/{args.setup}')
    if not op.isdir(save_foler):
        os.mkdir(save_foler)

    viewer.check_format(batch)
    viewer.render_seq(batch, out_folder=save_foler)