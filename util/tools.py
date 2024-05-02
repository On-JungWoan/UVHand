import os
import cv2
import wandb
import torch
import pickle
import numpy as np
import os.path as op
from PIL import Image
import util.misc as utils
import matplotlib.pyplot as plt
from pytorch3d.ops.knn import knn_points


def prefetcher_next(prefetcher, dataset_name='arctic'):
    if dataset_name == 'arctic':
        return prefetcher.next()
    else:
        t1, t2 = prefetcher.next()
        return t1, t2, 0


def arctic_smoothing(target, count):
    # B, x1, x2 =target.shape
    B, f, x = target.shape
    target = target.permute(0,2,1)
    # target = target[None].view(1,B,-1).permute(0,2,1)

    for _ in range(count):
        for b in range(B):
            for i in range(f-1):
                target[b, :, i+1] = target[b, :, i] = (target[b, :, i+1] + target[b, :, i]) / 2
    return target.permute(0,2,1).view(-1, x)
    # target[..., i+1] = target[..., i] = (target[..., i+1] + target[..., i])/2
    # return target[0].reshape(x1,x2,B).permute(2,0,1)


def create_loss_dict(loss_value, loss_out, flag='', round_value=False, mode='baseline'):
    if not 'loss' in loss_out:
        res_dict = {'loss' : loss_value}
    else:
        res_dict = {'loss' : loss_out['loss']}

    loss_keys = {
        # arctic
        'loss_obj_smt' : ["loss/object/v3d_smoothing"],
        'loss_ce' : ['loss_ce'],
        'loss_CDev' : ['loss/cd'],
        'loss_smooth' : ['loss/smooth/2d', 'loss/smooth/3d'],
        'loss_smooth_2d' : ['loss/smooth/2d'],
        'loss_smooth_3d' : ['loss/smooth/3d'],
        'loss_penetr' : ['loss/penetr'],
        'loss_mano' : ['loss/mano/pose/r', 'loss/mano/beta/r', 'loss/mano/pose/l', 'loss/mano/beta/l'],
        'loss_rot' : ['loss/object/radian', 'loss/object/rot'],
        'loss_transl' : ['loss/mano/transl/l', 'loss/object/transl'],
        'loss_kp' : [
            'loss/mano/kp2d/r', 'loss/mano/kp3d/r', 'loss/mano/kp2d/l', 'loss/mano/kp3d/l',
            'loss/object/kp2d', 'loss/object/kp3d'
        ],
        'loss_3d_kp' : [
            'loss/mano/kp3d/r', 'loss/mano/kp3d/l', 'loss/object/kp3d'
        ],
        'loss_2d_kp' : [
            'loss/mano/kp2d/r', 'loss/mano/kp2d/l', 'loss/object/kp2d'
        ],        
        'loss_cam' : ['loss/mano/cam_t/r', 'loss/mano/cam_t/l', 'loss/object/cam_t'],

        # as hands
        'loss_left' : ['loss_left'],
        'loss_right' : ['loss_right'],
        'loss_obj' : ['loss_obj'],

        # dino
        'loss_hand_key' : ['loss_hand_keypoint'],
        'loss_obj_key' : ['loss_obj_keypoint'],

        # acc
        'loss_acc_h' : ['acc/h'],
        'loss_acc_o' : ['acc/o'],
    }

    # select item
    if mode == 'dino':
        items = [
            'loss_ce', 'loss_CDev', 'loss_penetr', 'loss_mano', 'loss_rot', 'loss_transl',
            'loss_kp', 'loss_cam', 'loss_hand_key', 'loss_obj_key'
        ]
    elif mode == 'small':
        items = [
            'loss_obj_smt',
            'loss_ce', 'loss_CDev', 'loss_mano', 'loss_rot', 'loss_transl',
            # 'loss_cam', 'loss_3d_kp', 'loss_2d_kp'
            'loss_cam', 'loss_3d_kp', 'loss_2d_kp', 'loss_hand_key', 'loss_obj_key',
            # 'loss_smooth_2d', 'loss_smooth_3d'
        ]        
    elif mode == 'baseline':
        items = [
            'loss_obj_smt',
            'loss_ce', 'loss_CDev', 'loss_penetr', 'loss_mano', 'loss_rot', 'loss_transl',
            'loss_kp', 'loss_cam', 'loss_smooth'
        ]
    elif mode == 'smoothnet':
        items = [
            # 'loss_left', 'loss_right', 'loss_obj'
            'loss_transl', 'loss_acc_h', 'loss_acc_o',
            'loss_CDev', 'loss_mano', 'loss_rot', 'loss_cam',
        ]
    elif mode == 'all':
        return dict((f'{flag}_{k}', float(v)) for k,v in loss_out.items())
    else:
        raise Exception('Not existed mode')
    
    # make results
    for item in items:
        value = 0
        for loss_key in loss_keys[item]:
            try:
                value += float(loss_out[loss_key])
            except:
                pass
        if round_value:
            value = round(value, 2)
        res_dict[f'{flag}_{item}'] = value
    
    return res_dict


def create_arctic_score_dict(stats):
    return {
        'score_CDev' : stats['cdev/ho'],
        'score_MRRPE_rl': stats['mrrpe/r/l'],
        'score_MRRPE_ro' : stats['mrrpe/r/o'],
        'score_MPJPE' : stats['mpjpe/ra/h'],
        'score_AAE' : stats['aae'],
        'score_S_R_0.05' : stats['success_rate/0.05'],
    }


def test(meta_info, targets, data_loader):
    from util.tools import cam2pixel
    from PIL import Image
    import cv2

    B = 60
    # testing_idx = 0
    meta_info['intrinsics'][B]

    fx = meta_info['intrinsics'][B][0,0]
    fy = meta_info['intrinsics'][B][1,1]
    cx = meta_info['intrinsics'][B][0,2]
    cy = meta_info['intrinsics'][B][1,2]

    f = [fx, fy]
    c = [cx, cy]

    imgname = meta_info['imgname'][B]
    img = Image.open('/home/unist/Desktop/hdd/arctic/data/arctic_data/data/cropped_images/' + imgname)
    img = np.array(img)

    # test = targets['object.bbox3d.full.b'][B][testing_idx]
    # test_bbox = cam2pixel(test, f, c).type(torch.uint8).cpu().numpy()

    test = targets['mano.j3d.cam.r'][B]
    # test[:, 0] *= 600
    # test[:, 1] *= 840

    p_test = cam2pixel(test, f, c)
    for t in p_test:
        # x = int(t[0] * 600)
        # y = int(t[1] * 840)
        x = int(t[0])
        y = int(t[1])
        cv2.line(img, (x, y), (x, y), (255,0,0), 3)
    # plt.imshow(cv2.line(img, (test_bbox[0], test_bbox[1]), (test_bbox[0], test_bbox[1]), (255,0,0), 3))
    # test_bbox = cam2pixel(targets['object.cam_t.wp'][B], f, c).type(torch.uint8).cpu().numpy()
    # cam2pixel(targets['mano.j3d.cam.r'][B], f, c)

    plt.imshow(img)


def cam2pixel(cam_coord, f, c):
    x = cam_coord[..., 0] / (cam_coord[..., 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[..., 1] / (cam_coord[..., 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[..., 2]
    try:
        img_coord = np.concatenate((x[...,None], y[...,None], z[...,None]), -1)
    except:
        img_coord = torch.cat((x[...,None], y[...,None], z[...,None]), -1)
    return img_coord


def pixel2cam(pixel_coord, f, c, T_=None):
    x = (pixel_coord[..., 0] - c[0]) / f[0] * pixel_coord[..., 2]
    y = (pixel_coord[..., 1] - c[1]) / f[1] * pixel_coord[..., 2]
    z = pixel_coord[..., 2]
    try:
        cam_coord = np.concatenate((x[...,None], y[...,None], z[...,None]), -1)
    except:
        cam_coord = torch.cat((x[...,None], y[...,None], z[...,None]), -1)
        
    if T_ is not None: # MANO space와 scale과 wrist를 맞추고자
        # New
        # if pixel_coord.shape[1] == 1:
        #     T_ = T_[1][None]
        # else:
        #     T_ = torch.stack(T_)
        ratio = torch.linalg.norm(T_[:,9] - T_[:,0], dim=-1) / torch.linalg.norm(cam_coord[:,:,9] - cam_coord[:,:,0], dim=-1)
        cam_coord = cam_coord * ratio[:,:,None,None]  # template, m
        cam_coord = cam_coord - cam_coord[:, :, :1] + T_[:,:1]
    return cam_coord


def stat_round(round_num=2, **kwargs):
    result = {}
    for k,v in kwargs.items():
        result[k] = round(v, round_num)
    return result


def eval_assembly_result(outputs, targets, cfg, data_loader):
    key_points, target_sizes = extract_assembly_output(outputs, targets, cfg)
    gt_keypoints = [t['keypoints'] for t in targets]

    if 'labels' in targets[0].keys():
        gt_labels = [t['labels'].detach().cpu().numpy() for t in targets]

    # measure
    mpjpe_ra_list = []
    for i, batch in enumerate(gt_labels):
        target = targets[i]
        img_id = target['image_id'].item()
        joint_valid = data_loader.dataset.coco.loadAnns(img_id)[0]['joint_valid']
        joint_valid = torch.stack([torch.tensor(joint_valid[:21]), torch.tensor(joint_valid[21:])]).type(torch.bool)

        cam_fx, cam_fy, cam_cx, cam_cy, _, _ = target['cam_param']

        for k, label in enumerate(batch):
            if label == cfg.hand_idx[0]: j=0
            else: j=1
            
            pred_kp = key_points[i][j]
            pred_joint_cam = pixel2cam(pred_kp, (cam_fx.item(), cam_fy.item()), (cam_cx.item(), cam_cy.item()))

            x, y = target_sizes[0].detach().cpu().numpy()
            gt_scaled_keypoints = gt_keypoints[i][k] * torch.tensor([x, y, 1000]).cuda()
            gt_joint_cam = pixel2cam(gt_scaled_keypoints, (cam_fx.item(), cam_fy.item()), (cam_cx.item(), cam_cy.item()))

            joints3d_cam_gt_ra = gt_joint_cam - gt_joint_cam[:1, :]
            joints3d_cam_pred_ra = pred_joint_cam - pred_joint_cam[:1, :]

            mpjpe_ra = ((joints3d_cam_gt_ra - joints3d_cam_pred_ra) ** 2)[joint_valid[j]].sum(dim=-1).sqrt()
            mpjpe_ra = mpjpe_ra.cpu().numpy()
            mpjpe_ra_list.append(mpjpe_ra.mean())
    return {
        'mpjpe':float(np.array(mpjpe_ra_list).mean())
    }


def visualize_assembly_result(args, cfg, outputs, targets, data_loader, ):
    filename = data_loader.dataset.coco.loadImgs(targets[0]['image_id'][0].item())[0]['file_name']
    filepath = data_loader.dataset.root / filename
    source_img = np.array(Image.open(filepath))


    # model output
    out_logits,  pred_keypoints = outputs['pred_logits'], outputs['pred_keypoints']
    prob = out_logits.sigmoid()
    B, num_queries, num_classes = prob.shape

    # hand index select
    thold = 0.1
    hand_idx = []
    hand_score = []
    for i in cfg.hand_idx:
        score, idx = torch.max(prob[:,:,i], dim=-1)
        hand_idx.append(idx)
        hand_score.append(score)
    hand_idx = torch.stack(hand_idx, dim=-1)

    # de-normalize
    hand_kp = torch.gather(pred_keypoints, 1, hand_idx.unsqueeze(-1).repeat(1,1,63)).reshape(B, -1 ,21, 3)
    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    im_h, im_w = orig_target_sizes[:,0], orig_target_sizes[:,1]
    target_sizes = torch.cat([im_w.unsqueeze(-1), im_h.unsqueeze(-1)], dim=-1)
    target_sizes =target_sizes.cuda()

    hand_kp[...,:2] *=  target_sizes.unsqueeze(1).unsqueeze(1); hand_kp[...,2] *= 1000

    batch, js, _, _ = hand_kp.shape
    for b in range(batch):
        for j in range(js):
            pred_kp = hand_kp[b][j]
            pred_score = hand_score[j]

            if pred_score < thold:
                continue

            if j ==0:
                source_img = visualize(source_img, pred_kp.detach().cpu().numpy().astype(np.int32), 'left')
            elif j == 1:
                source_img = visualize(source_img, pred_kp.detach().cpu().numpy().astype(np.int32), 'right')

    save_name = '_'.join(filename.split('/')[-2:])
    plt.imsave(f'exps/{args.dataset_file}/{save_name}.png', source_img)


def make_line(cv_img, img_points, idx_1, idx_2, color, line_thickness=2):
    if -1 not in tuple(img_points[idx_1][:-1]):
        if -1 not in tuple(img_points[idx_2][:-1]):
            cv2.line(cv_img, tuple(img_points[idx_1][:-1]), tuple(
                img_points[idx_2][:-1]), color, line_thickness)    


def visualize(cv_img, img_points, mode='left'):
    if mode == 'left':
        color = (255,0,0)
    else:
        color = (0,0,255)
    
    make_line(cv_img, img_points, 0, 1, color, line_thickness=2)
    make_line(cv_img, img_points, 1, 2, color, line_thickness=2)
    make_line(cv_img, img_points, 2, 3, color, line_thickness=2)

    make_line(cv_img, img_points, 4, 5, color, line_thickness=2)
    make_line(cv_img, img_points, 5, 6, color, line_thickness=2)
    make_line(cv_img, img_points, 6, 7, color, line_thickness=2)

    make_line(cv_img, img_points, 8, 9, color, line_thickness=2)
    make_line(cv_img, img_points, 9, 10, color, line_thickness=2)
    make_line(cv_img, img_points, 10, 11, color, line_thickness=2)

    make_line(cv_img, img_points, 12, 13, color, line_thickness=2)
    make_line(cv_img, img_points, 13, 14, color, line_thickness=2)
    make_line(cv_img, img_points, 14, 15, color, line_thickness=2)

    make_line(cv_img, img_points, 16, 17, color, line_thickness=2)
    make_line(cv_img, img_points, 17, 18, color, line_thickness=2)
    make_line(cv_img, img_points, 18, 19, color, line_thickness=2)

    make_line(cv_img, img_points, 20, 3, color, line_thickness=2)
    make_line(cv_img, img_points, 20, 7, color, line_thickness=2)
    make_line(cv_img, img_points, 20, 11, color, line_thickness=2)
    make_line(cv_img, img_points, 20, 15, color, line_thickness=2)
    make_line(cv_img, img_points, 20, 19, color, line_thickness=2)

    # plt.imshow(cv_img)
    return cv_img


def visualize_obj(cv_img, img_points):
    cv2.line(cv_img, tuple(img_points[1][:-1]), tuple(
        img_points[2][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[2][:-1]), tuple(
        img_points[3][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[3][:-1]), tuple(
        img_points[4][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[4][:-1]), tuple(
        img_points[1][:-1]), (0, 255, 0), 5)

    cv2.line(cv_img, tuple(img_points[1][:-1]), tuple(
        img_points[5][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[2][:-1]), tuple(
        img_points[6][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[3][:-1]), tuple(
        img_points[7][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[4][:-1]), tuple(
        img_points[8][:-1]), (0, 255, 0), 5)

    cv2.line(cv_img, tuple(img_points[5][:-1]), tuple(
        img_points[6][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[6][:-1]), tuple(
        img_points[7][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[7][:-1]), tuple(
        img_points[8][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[8][:-1]), tuple(
        img_points[5][:-1]), (0, 255, 0), 5)

    return cv_img


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
    nn_dists = src_nn.dists ## (x-x')**2 + (y-y')**2
    nn_idx = src_nn.idx
    # nn_dists = src_nn.dists[..., 0] ## (x-x')**2 + (y-y')**2
    # nn_idx = src_nn.idx[..., 0]
    return nn_dists#, nn_idx


def get_pseudo_cmap(nn_dists):
    '''
    calculate pseudo contactmap: 0~3cm mapped into value 1~0
    :param nn_dists: object nn distance [B, N] or [N,] in meter**2
    :return: pseudo contactmap [B,N] or [N,] range in [0,1]
    '''
    # nn_dists = 100.0 * torch.sqrt(nn_dists)  # turn into center-meter
    nn_dists = torch.sqrt(nn_dists) / 10.0  # turn into center-meter
    cmap = 1.0 - 2 * (torch.sigmoid(nn_dists*2) -0.5)
    return cmap


def rigid_transform_3D_numpy(A, B):
    batch, n, dim = A.shape
    # tmp_A = A.detach().cpu().numpy()
    # tmp_B = B.detach().cpu().numpy()
    tmp_A = A.copy()
    tmp_B = B.copy()
    centroid_A = np.mean(tmp_A, axis = 1)
    centroid_B = np.mean(tmp_B, axis = 1)
    H = np.matmul((tmp_A - centroid_A[:,None]).transpose(0,2,1), tmp_B - centroid_B[:,None]) / n
    U, s, V = np.linalg.svd(H)
    R = np.matmul(V.transpose(0,2,1), U.transpose(0, 2, 1))

    negative_det = np.linalg.det(R) < 0
    s[negative_det, -1] = -s[negative_det, -1]
    V[negative_det, :, 2] = -V[negative_det, :, 2]
    R[negative_det] = np.matmul(V[negative_det].transpose(0,2,1), U[negative_det].transpose(0, 2, 1))

    varP = np.var(tmp_A, axis=1).sum(-1)
    c = 1/varP * np.sum(s, axis=-1) 

    t = -np.matmul(c[:,None,None]*R, centroid_A[...,None])[...,-1] + centroid_B
    return c, R, t


def vis(data_loader, targets, FPHA=False):
    filename = data_loader.dataset.coco.loadImgs(targets[0]['image_id'][0].item())[0]['file_name']
    if FPHA:
        filepath = data_loader.dataset.root / 'Video_files'/ filename
    else:
        filepath = data_loader.dataset.root / filename
    cv_img = np.array(Image.open(filepath))
    img_points = targets[0]['keypoints'][0].cpu().detach().numpy().astype(np.int32)
    color = (0,0,255)
    line_thickness = 2
    cv2.line(cv_img, tuple(img_points[1][:-1]), tuple(
        img_points[2][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[2][:-1]), tuple(
        img_points[3][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[3][:-1]), tuple(
        img_points[4][:-1]), color, line_thickness)

    cv2.line(cv_img, tuple(img_points[5][:-1]), tuple(
        img_points[6][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[6][:-1]), tuple(
        img_points[7][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[7][:-1]), tuple(
        img_points[8][:-1]), color, line_thickness)

    cv2.line(cv_img, tuple(img_points[9][:-1]), tuple(
        img_points[10][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[10][:-1]), tuple(
        img_points[11][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[11][:-1]), tuple(
        img_points[12][:-1]), color, line_thickness)

    cv2.line(cv_img, tuple(img_points[13][:-1]), tuple(
        img_points[14][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[14][:-1]), tuple(
        img_points[15][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[15][:-1]), tuple(
        img_points[16][:-1]), color, line_thickness)

    cv2.line(cv_img, tuple(img_points[17][:-1]), tuple(
        img_points[18][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[18][:-1]), tuple(
        img_points[19][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[19][:-1]), tuple(
        img_points[20][:-1]), color, line_thickness)

    cv2.line(cv_img, tuple(img_points[0][:-1]), tuple(
        img_points[1][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[0][:-1]), tuple(
        img_points[5][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[0][:-1]), tuple(
        img_points[9][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[0][:-1]), tuple(
        img_points[13][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[0][:-1]), tuple(
        img_points[17][:-1]), color, line_thickness)

    return cv_img


def keep_valid(outputs, is_valid):
    outputs['pred_logits'] = outputs['pred_logits'][is_valid]
    outputs['pred_mano_params'][0] = outputs['pred_mano_params'][0][is_valid]
    outputs['pred_mano_params'][1] = outputs['pred_mano_params'][1][is_valid]
    outputs['pred_obj_params'][0] = outputs['pred_obj_params'][0][is_valid]
    outputs['pred_obj_params'][1] = outputs['pred_obj_params'][1][is_valid]
    outputs['pred_cams'][0] = outputs['pred_cams'][0][is_valid]
    outputs['pred_cams'][1] = outputs['pred_cams'][1][is_valid]    
    for idx, aux in enumerate(outputs['aux_outputs']):
        outputs['aux_outputs'][idx]['pred_logits'] = aux['pred_logits'][is_valid]
        outputs['aux_outputs'][idx]['pred_mano_params'][0] = aux['pred_mano_params'][0][is_valid]
        outputs['aux_outputs'][idx]['pred_mano_params'][1] = aux['pred_mano_params'][1][is_valid]
        outputs['aux_outputs'][idx]['pred_obj_params'][0] = aux['pred_obj_params'][0][is_valid]
        outputs['aux_outputs'][idx]['pred_obj_params'][1] = aux['pred_obj_params'][1][is_valid]
        outputs['aux_outputs'][idx]['pred_cams'][0] = aux['pred_cams'][0][is_valid]
        outputs['aux_outputs'][idx]['pred_cams'][1] = aux['pred_cams'][1][is_valid]      
    return outputs


def extract_assembly_output(outputs, targets, cfg):
    # model output
    out_logits,  pred_keypoints = outputs['pred_logits'], outputs['pred_keypoints']
    prob = out_logits.sigmoid()
    B, num_queries, num_classes = prob.shape

    # hand index select
    hand_idx = []
    for i in cfg.hand_idx:
        hand_idx.append(torch.argmax(prob[:,:,i], dim=-1))
    hand_idx = torch.stack(hand_idx, dim=-1)

    # de-normalize
    hand_kp = torch.gather(pred_keypoints, 1, hand_idx.unsqueeze(-1).repeat(1,1,63)).reshape(B, -1 ,21, 3)
    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    im_h, im_w = orig_target_sizes[:,0], orig_target_sizes[:,1]
    target_sizes = torch.cat([im_w.unsqueeze(-1), im_h.unsqueeze(-1)], dim=-1)
    target_sizes =target_sizes.cuda()

    hand_kp[...,:2] *=  target_sizes.unsqueeze(1).unsqueeze(1); hand_kp[...,2] *= 1000
    return hand_kp, target_sizes


def extract_feature(
        args, model, samples, targets, meta_info, data_loader, cfg,
        check_mode = False
    ):
    # dir settings
    B = samples.tensors.size(0)
    root = op.join(args.coco_path, args.dataset_file)

    # for arctic
    if args.dataset_file == 'arctic':
        # check missing files
        if check_mode:
            for name in data_loader.dataset.imgnames:
                save_name = '+'.join(name.split('/')[-4:])
                mode = 'val' if args.eval else 'train'
                save_dir = f'{root}/data/pickle/{args.setup}_2048/{mode}/{op.splitext(save_name)[0]}.pkl'
                if not op.isfile(save_dir):
                    print(save_name)
                    img = data_loader.dataset.getitem(name, load_rgb=True)[0]
                    srcs = model(img.unsqueeze(0).cuda(), is_extract=True)
                    with open(save_dir, 'wb') as f:
                        pickle.dump(
                            [
                                srcs[0].tensors[0].cpu().detach(),
                                srcs[1].tensors[0].cpu().detach(),
                                srcs[2].tensors[0].cpu().detach()
                            ], f)
                        
        # save results
        else:
            srcs = model(samples, is_extract=True)
            for i in range(B):
                save_name = '+'.join(meta_info['imgname'][i].split('/')[-4:])
                mode = 'val' if args.eval else 'train'
                save_dir = f'{root}/data/pickle/{args.setup}_2048/{mode}/{op.splitext(save_name)[0]}.pkl'
                with open(save_dir, 'wb') as f:
                    pickle.dump(
                        [
                            srcs[0].tensors[i].cpu().detach(),
                            srcs[1].tensors[i].cpu().detach(),
                            srcs[2].tensors[i].cpu().detach()
                        ], f)

        # # just debug
        # with open(f'{root}/data/pickle/{args.setup}/s02+ketchup_grab_01+0+00434.pkl', 'rb') as f:
        #     test = pickle.load(f)


    # for assembly hands
    elif args.dataset_file == 'AssemblyHands':
        outputs = model(samples)
        
        # post-process output
        out_key = extract_assembly_output(outputs, targets, cfg)[0]
        B = out_key.size(0)

        # store result
        res = {}
        for idx in range(B):
            key = data_loader.dataset.coco.loadImgs(targets[idx]['image_id'].item())[0]['file_name']
            value = out_key[idx]

            res[key] = value
            key = key.replace('/', '+')
            key = key.replace('.jpg', '')
            with open(f'results/Assemblyhands/train/{key}.pkl', 'wb') as f:
                pickle.dump(res, f)


def save_results(args, epoch, result, stats, flag):
    # for wandb
    if args is not None:
        if args.distributed:
            if utils.get_local_rank() != 0:
                return stats
        
        # save results
        epoch = extract_epoch(args.resume) if epoch is None else epoch

        if flag == 'train':
            save_dir = os.path.join(f'{args.output_dir}/loss.txt')
            with open(save_dir, 'a') as f:
                if args.test_viewpoint is not None:
                    f.write(f"{'='*10} {args.test_viewpoint} {'='*10}\n")
                f.write(f"{'='*10} epoch : {epoch} {'='*10}\n\n")
                f.write(f"{'='*9} {args.val_batch_size}*{args.window_size}, {args.iter}iter {'='*9}\n")
                for key, val in stats.items():
                    res = f'{key:35} : {round(val, 8)}\n'
                    f.write(res)
                    print(res, end='')
                f.write('\n\n')
        elif flag == 'eval':
            save_dir = os.path.join(f'{args.output_dir}/results.txt')
            with open(save_dir, 'a') as f:
                if args.test_viewpoint is not None:
                    f.write(f"{'='*10} {args.test_viewpoint} {'='*10}\n")

                f.write(f"{'='*10} epoch : {epoch} {'='*10}\n\n")
                f.write(f"{'='*9} {args.val_batch_size}*{args.window_size}, {args.iter}iter {'='*9}\n")

                for key, val in stats.items():
                    f.write(f'{key:30} : {val}\n')
                f.write('\n\n')

        if args.wandb:
            wandb.log(result, step=epoch)    


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def extract_epoch(file_path):
    file_name = file_path.split('/')[-1]
    epoch = op.splitext(file_name)[0]

    return int(epoch)


# def old_test_pose(model, criterion, data_loader, device, cfg, args=None, vis=False, save_pickle=False):
    
#     model.eval()
#     criterion.eval()
#     dataset = args.dataset_file

#     ## old script ##
#     def trash():
#         pass
#         # if dataset == 'arctic' and args.visualization == True:
#         #     from arctic_tools.extract_predicts import main
#         #     main(args, model, data_loader)    

#         # try:
#         #     idx2obj = {v:k for k, v in cfg.obj2idx.items()}
#         #     GT_obj_vertices_dict = {}
#         #     GT_3D_bbox_dict = {}        
#         #     for i in range(1,cfg.hand_idx[0]):
#         #         with open(os.path.join(data_loader.dataset.root, 'obj_pkl', f'{idx2obj[i]}_2000.pkl'), 'rb') as f:
#         #             vertices = pickle.load(f)
#         #             GT_obj_vertices_dict[i] = vertices
#         #         with open(os.path.join(data_loader.dataset.root, 'obj_pkl', f'{idx2obj[i]}_bbox.pkl'), 'rb') as f:
#         #             bbox = pickle.load(f)
#         #             GT_3D_bbox_dict[i] = bbox
#         # except:
#         #     dataset = 'AssemblyHands'
#         #     print('Not exist obj pkl')

#         # _mano_root = 'mano/models'
#         # mano_left = ManoLayer(flat_hand_mean=True,
#         #                 side="left",
#         #                 mano_root=_mano_root,
#         #                 use_pca=False,
#         #                 root_rot_mode='axisang',
#         #                 joint_rot_mode='axisang').to(device)

#         # mano_right = ManoLayer(flat_hand_mean=True,
#         #                 side="right",
#         #                 mano_root=_mano_root,
#         #                 use_pca=False,
#         #                 root_rot_mode='axisang',
#         #                 joint_rot_mode='axisang').to(device)
#     ## old script ##

#     if args.dataset_file == 'arctic':
#         prefetcher = arctic_prefetcher(data_loader, device, prefetch=True)
#         samples, targets, meta_info = prefetcher.next()
#     else:
#         prefetcher = data_prefetcher(data_loader, device, prefetch=True)
#         samples, targets = prefetcher.next()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     pbar = tqdm(range(len(data_loader)))

#     for _ in pbar:
#         ## old script ##
#         def trash():
#             pass
#             # samples, targets = prefetcher.next()
            
#             # try:
#             #     gt_keypoints = [t['keypoints'] for t in targets]
#             # except:
#             #     print('no gts')
#             #     continue

#             # if 'labels' in targets[0].keys():
#             #     gt_labels = [t['labels'].detach().cpu().numpy() for t in targets]

#             # try:
#             #     filename = data_loader.dataset.coco.loadImgs(targets[0]['image_id'][0].item())[0]['file_name']
#             # except:
#             #     filename = meta[0]['imgname']
            
#             # if args.test_viewpoint is not None:
#             #     if args.test_viewpoint != '/'.join(filename.split('/')[:-1]):
#             #         continue

#             # if vis:
#             #     assert data_loader.batch_size == 1  
#             #     if args.dataset_file=='arctic':
#             #         filepath = os.path.join(args.coco_path, args.dataset_file) + filename[1:]
#             #     elif dataset == 'H2O' or dataset == 'AssemblyHands':
#             #         filepath = data_loader.dataset.root / filename
#             #     else:
#             #         filepath = data_loader.dataset.root / 'Video_files'/ filename
#             #     source_img = np.array(Image.open(filepath))

#             # if os.path.exists(os.path.join(f'./pickle/{dataset}_aug45/{data_loader.dataset.mode}', f'{filename[:-4]}_data.pkl')):
#             #     samples, targets = prefetcher.next()
#             #     # continue

#             # if filename != 'ego_images_rectified/val/nusar-2021_action_both_9081-c11b_9081_user_id_2021-02-12_161433/HMC_21176623_mono10bit/006667.jpg':
#             #     continue
#         ## old script ##

#         if args.dataset_file == 'arctic':
#             targets, meta_info = arctic_pre_process(args, targets, meta_info)

#         with torch.no_grad():
#             outputs = model(samples.to(device))
            
#             # check validation
#             is_valid = targets['is_valid'].type(torch.bool)
#             for k,v in targets.items():
#                 if k == 'labels':
#                     targets[k] = [v for idx, v in enumerate(targets[k]) if is_valid[idx] == True]
#                 else:
#                     targets[k] = v[is_valid]
#             outputs = keep_valid(outputs, is_valid)

#             # prepare data
#             data = prepare_data(args, outputs, targets, meta_info, cfg)

#             ## old script ##
#             def trash():
#                 # calc loss
#                 loss_dict = criterion(outputs, targets)
#                 loss_dict_reduced = utils.reduce_dict(loss_dict)
#                 weight_dict = criterion.weight_dict
#                 losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

#                 # reduce losses over all GPUs for logging purposes
#                 loss_dict_reduced = utils.reduce_dict(loss_dict)
#                 loss_dict_reduced_unscaled = {f'{k}_unscaled': v
#                                             for k, v in loss_dict_reduced.items()}
#                 loss_dict_reduced_scaled = {k: v * weight_dict[k]
#                                             for k, v in loss_dict_reduced.items() if k in weight_dict}
#                 losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

#                 loss_value = losses_reduced_scaled.item()
#                 metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
#                 ####################################################

#                 # model output
#                 # out_logits, pred_keypoints, pred_obj_keypoints = outputs['pred_logits'], outputs['pred_keypoints'], outputs['pred_obj_keypoints']
#                 out_logits, out_mano_pose, out_mano_beta = outputs['pred_logits'], outputs['pred_manoparams'][0], outputs['pred_manoparams'][1]

#                 prob = out_logits.sigmoid()
#                 B, num_queries, num_classes = prob.shape

#                 # query index select
#                 best_score = torch.zeros(B).to(device)
#                 # if dataset != 'AssemblyHands':
#                 obj_idx = torch.zeros(B).to(device).to(torch.long)
#                 for i in range(1, cfg.hand_idx[0]):
#                     score, idx = torch.max(prob[:,:,i], dim=-1)
#                     obj_idx[best_score < score] = idx[best_score < score]
#                     best_score[best_score < score] = score[best_score < score]

#                 left_hand_idx = []
#                 right_hand_idx = []
#                 for i in cfg.hand_idx:
#                     hand_idx.append(torch.argmax(prob[:,:,i], dim=-1)) 
#                 hand_idx = torch.stack(hand_idx, dim=-1)   
#                 if dataset != 'AssemblyHands':
#                     keep = torch.cat([hand_idx, obj_idx[:,None]], dim=-1)
#                 else:
#                     keep = hand_idx
#                 hand_kp = torch.gather(pred_keypoints, 1, hand_idx.unsqueeze(-1).repeat(1,1,63)).reshape(B, -1 ,21, 3)
#                 obj_kp = torch.gather(pred_obj_keypoints, 1, obj_idx.unsqueeze(1).unsqueeze(1).repeat(1,1,63)).reshape(B, 21, 3)

#                 continue

#                 im_h, im_w, _ = source_img.shape
#                 hand_kp = targets[0]['keypoints'][1] * 1000
#                 visualize(source_img, hand_kp.detach().cpu().numpy().astype(np.int32), 'left')

#                 orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
#                 im_h, im_w = orig_target_sizes[:,0], orig_target_sizes[:,1]
#                 target_sizes = torch.cat([im_w.unsqueeze(-1), im_h.unsqueeze(-1)], dim=-1)
#                 target_sizes =target_sizes.cuda()

#                 labels = torch.gather(out_logits, 1, keep.unsqueeze(2).repeat(1,1,num_classes)).softmax(dim=-1)
#                 hand_kp[...,:2] *=  target_sizes.unsqueeze(1).unsqueeze(1); hand_kp[...,2] *= 1000
#                 obj_kp[...,:2] *=  target_sizes.unsqueeze(1); obj_kp[...,2] *= 1000
#                 key_points = torch.cat([hand_kp, obj_kp.unsqueeze(1)], dim=1)
#                 key_points = hand_kp
                
#                 if args.debug:
#                     if vis:
#                         batch, js, _, _ = key_points.shape
#                         for b in range(batch):
#                             for j in range(js):
#                                 pred_kp = key_points[b][j]

#                                 target_keys = targets[0]['keypoints']
#                                 target_keys[...,:2] *=  target_sizes.unsqueeze(1)
#                                 target_keys = target_keys[0]
#                                 if j ==0:
#                                     # gt = visualize(source_img, target_keys.detach().cpu().numpy().astype(np.int32), 'left')
#                                     pred = visualize(source_img, pred_kp.detach().cpu().numpy().astype(np.int32), 'left')
#                                 elif j == 1:
#                                     source_img = visualize(source_img, pred_kp.detach().cpu().numpy().astype(np.int32), 'right')
#                                 else:
#                                     source_img = visualize_obj(source_img, pred_kp.detach().cpu().numpy().astype(np.int32))
#                     continue

#                 # measure
#                 if dataset != 'AssemblyHands':
#                     tmp = []
#                     for gt_label in gt_labels:
#                         tmp.append([i for i in cfg.hand_idx if i in gt_label])
#                     gt_labels = tmp

#                 for i, batch in enumerate(gt_labels):
#                     cam_fx, cam_fy, cam_cx, cam_cy, _, _ = targets[i]['cam_param']
#                     for k, label in enumerate(batch):
#                         if dataset == 'H2O':
#                             if label == cfg.hand_idx[0]: j=0
#                             elif label == cfg.hand_idx[1]: j=1
#                             else: j=2
#                         else:
#                             if label == cfg.hand_idx[0]: j=0
#                             else: j=1
                                
#                         is_correct_class = int(labels[i][j].argmax().item() == gt_labels[i][k])
#                         pred_kp = key_points[i][j]

#                         x, y = target_sizes[0].detach().cpu().numpy()
#                         gt_scaled_keypoints = gt_keypoints[i][k] * torch.tensor([x, y, 1000]).cuda()
#                         gt_joint_cam = pixel2cam(gt_scaled_keypoints, (cam_fx.item(), cam_fy.item()), (cam_cx.item(), cam_cy.item()))

#                         # uvd to xyz
#                         if dataset == 'AssemblyHands':
#                             pred_kp[gt_scaled_keypoints==-1] = -1
#                             pred_kp[..., 2] = 1000
#                         pred_joint_cam = pixel2cam(pred_kp, (cam_fx.item(), cam_fy.item()), (cam_cx.item(), cam_cy.item()))

#                         if args.eval_method=='EPE':
#                             gt_relative = gt_scaled_keypoints[:,2:] - gt_scaled_keypoints[0,2:]
#                             pred_relative = pred_kp[:,2:] - pred_kp[0,2:]
                            
#                             xy_epe = torch.mean(torch.norm(gt_scaled_keypoints[:,:2] - pred_kp[:,:2], dim=-1))
#                             z_epe = torch.mean(torch.norm(gt_scaled_keypoints[:,2:] - pred_kp[:,2:], dim=-1))
#                             relative_depth_error = torch.mean(torch.norm(gt_relative - pred_relative, dim=-1))
            
#                             ###################################################################################
#                             # if j==2:
#                             #     pred_joint_cam = rigid_align(world_objcoord[0,:,:3], pred_joint_cam/1000)*1000
#                             ###################################################################################

#                             error = torch.mean(torch.norm(gt_joint_cam - pred_joint_cam, dim=-1))

#                         elif args.eval_method=='MPJPE':
#                             error = torch.sqrt(((pred_joint_cam - gt_joint_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

#                         # for visualization
#                         if dataset == 'FPHA': j+=1
#                         if vis:
#                             if j ==0:
#                                 source_img = visualize(source_img, pred_kp.detach().cpu().numpy().astype(np.int32), 'left')
#                             elif j == 1:
#                                 source_img = visualize(source_img, pred_kp.detach().cpu().numpy().astype(np.int32), 'right')
#                             else:
#                                 source_img = visualize_obj(source_img, pred_kp.detach().cpu().numpy().astype(np.int32))
#                         if j==1:
#                             metric_logger.update(**{'left': float(error)})
#                             # metric_logger.update(**{'uv_error': float(xy_epe)})
#                             # metric_logger.update(**{'d_error': float(z_epe)})
#                             # metric_logger.update(**{'relative_d_error': float(relative_depth_error)})
#                         elif j==0:
#                             metric_logger.update(**{'right': float(error)})
#                             # metric_logger.update(**{'uv_error': float(xy_epe)})
#                             # metric_logger.update(**{'d_error': float(z_epe)})
#                             # metric_logger.update(**{'relative_d_error': float(relative_depth_error)})
#                         else:
#                             metric_logger.update(**{'obj': float(error)})
#                             # metric_logger.update(**{'obj_uv_error': float(xy_epe)})
#                             # metric_logger.update(**{'obj_d_error': float(z_epe)})
#                             # metric_logger.update(**{'obj_relative_d_error': float(relative_depth_error)})
#                         metric_logger.update(**{'class_error':is_correct_class})
                    
#                 pbar.set_postfix({
#                     'left' : metric_logger.meters['left'].global_avg,
#                     'right' : metric_logger.meters['right'].global_avg,
#                     'obj' : metric_logger.meters['obj'].global_avg,
#                     'class_error' : metric_logger.meters['class_error'].global_avg,
#                     })

#                 if vis or save_pickle:
#                     assert data_loader.batch_size == 1
#                     save_path = os.path.join(f'./pickle/{dataset}_aug45/{data_loader.dataset.mode}', filename)
#                     # obj_label = labels[:,-1,:cfg.hand_idx[0]].argmax(-1)
#                     # GT_3D_bbox = GT_3D_bbox_dict[obj_label.item()][None]
#                     # pred_obj_cam = pixel2cam(obj_kp,  (cam_fx.item(), cam_fy.item()), (cam_cx.item(), cam_cy.item())).detach().cpu().numpy()
#                     # c, R, t = rigid_transform_3D_numpy(GT_3D_bbox*1000, pred_obj_cam)
#                     # c = torch.from_numpy(c).cuda(); R = torch.from_numpy(R).cuda(); t = torch.from_numpy(t).cuda()

#                     T_keypoints_left, T_keypoints_right = AIK_config.T_keypoints_left.cuda(), AIK_config.T_keypoints_right.cuda()
#                     T_ = torch.stack([T_keypoints_left, T_keypoints_right]) if hand_kp.shape[1] == 2 else T_keypoints_right[None]
#                     hand_cam_align = pixel2cam(hand_kp, (cam_fx.item(),cam_fy.item()), (cam_cx.item(),cam_cy.item()), T_)

#                     pose_params = [AIK.adaptive_IK(t, hand_cam_align[:,i]) for i, t in enumerate(T_)]            
#                     pose_params = torch.cat(pose_params, dim=-1)
                    
#                     if save_pickle:
#                         all_uvd = key_points.reshape(1, -1)
#                         all_cam = pixel2cam(key_points, (cam_fx.item(),cam_fy.item()), (cam_cx.item(),cam_cy.item())).reshape(1, -1)
#                         # obj_6D = torch.cat([R.reshape(-1,9), t], dim=-1)
#                         label_prob = labels[:,-1]

#                         # data={'uvd':all_uvd.detach().cpu().numpy(), 'cam':all_cam.detach().cpu().numpy(), '6D':obj_6D.detach().cpu().numpy(), 'label':label_prob.detach().cpu().numpy(), 'mano':pose_params.detach().cpu().numpy()}
#                         data={'uvd':all_uvd.detach().cpu().numpy(), 'cam':all_cam.detach().cpu().numpy(), 'label':label_prob.detach().cpu().numpy(), 'mano':pose_params.detach().cpu().numpy()}
                        
#                         if not os.path.exists(os.path.dirname(save_path)):
#                             os.makedirs(os.path.dirname(save_path))     
#                         with open(f'{save_path[:-4]}_data.pkl', 'wb') as f:
#                             pickle.dump(data, f)
                
#                 if vis:    
#                     ################# 2D vis #####################
#                     img_path = os.path.join(args.output_dir, filename)
#                     if not os.path.exists(os.path.dirname(img_path)):
#                         os.makedirs(os.path.dirname(img_path))
#                     cv2.imwrite(img_path, source_img[...,::-1])
#                     ###############################################
#                     ###### contact vis #####
#                     save_contact_vis_path = os.path.join(f'./contact_vis/{dataset}', filename)
#                     opt_tensor_shape = torch.zeros(prob.shape[0], 10).to(prob.device)
#                     MANO_LAYER= [mano_left, mano_right] if hand_kp.shape[1] == 2 else [mano_right]

#                     mano_results = [mano_layer(pose_params[:,48*i:48*(i+1)], opt_tensor_shape) for i, mano_layer in enumerate(MANO_LAYER)]
#                     hand_verts = torch.stack([m[0] for m in mano_results], dim=1)
#                     j3d_recon = torch.stack([m[1] for m in mano_results], dim=1)

#                     hand_cam = pixel2cam(hand_kp, (cam_fx.item(),cam_fy.item()), (cam_cx.item(),cam_cy.item()))
#                     hand_verts = hand_verts - j3d_recon[:,:,:1] + hand_cam[:,:,:1]

#                     # obj_name = idx2obj[obj_label.item()]
#                     # if dataset=='H2O':
#                     #     obj_mesh = trimesh.load(f'{cfg.object_model_path}/{obj_name}/{obj_name}.obj')
#                     # else:
#                     #     obj_mesh = trimesh.load(f'{cfg.object_model_path}/{obj_name}_model/{obj_name}_model.ply')
#                     # obj_mesh.vertices = (torch.matmul(R[0].detach().cpu().to(torch.float32), torch.tensor(obj_mesh.vertices, dtype=torch.float32).permute(1,0)*1000).permute(1,0) + t[0,None].detach().cpu()).numpy()
#                     # obj_vertices = torch.tensor(obj_mesh.vertices)[None].repeat(labels.shape[0], 1, 1).to(torch.float32).cuda()
                    
#                     # obj_nn_dist_affordance = get_NN(obj_vertices.to(torch.float32), hand_verts.reshape(1,-1,3).to(torch.float32))
#                     # hand_nn_dist_affordance = torch.stack([get_NN(hand_verts[:,idx].to(torch.float32), obj_vertices.to(torch.float32)) for idx in range(hand_verts.shape[1])], dim=1)
#                     # hand_nn_dist_affordance = torch.stack([get_NN(hand_verts[:,idx].to(torch.float32)) for idx in range(hand_verts.shape[1])], dim=1)
#                     # obj_cmap_affordance = get_pseudo_cmap(obj_nn_dist_affordance)
#                     # hand_cmap_affordance = torch.stack([get_pseudo_cmap(hand_nn_dist_affordance[:,idx]) for idx in range(hand_verts.shape[1])], dim=1)

#                     # cmap = plt.cm.get_cmap('plasma')
#                     # obj_v_color = (cmap(obj_cmap_affordance[0].detach().cpu().numpy())[:,0,:-1] * 255).astype(np.int64)
#                     # hand_v_color = [(cmap(hand_cmap_affordance[0, idx].detach().cpu().numpy())[:,0,:-1] * 255).astype(np.int64) for idx in range(hand_verts.shape[1])]

#                     # obj_mesh = trimesh.Trimesh(vertices=obj_vertices[0].detach().cpu().numpy(), vertex_colors=obj_v_color, faces = obj_mesh.faces)
#                     # hand_mesh = [trimesh.Trimesh(vertices=hand_verts[:,i].detach().cpu().numpy()[0], faces=(mano_layer.th_faces).detach().cpu().numpy(), vertex_colors=hand_v_color[i]) 
#                     #              for i, mano_layer in enumerate(MANO_LAYER)]

#                     # if not os.path.exists(os.path.dirname(save_contact_vis_path)):
#                     #     os.makedirs(os.path.dirname(save_contact_vis_path))

#                     # if len(hand_mesh) == 2:
#                     #     trimesh.exchange.export.export_mesh(hand_mesh[0],f'{save_contact_vis_path[:-4]}_left.obj')
#                     #     trimesh.exchange.export.export_mesh(hand_mesh[1],f'{save_contact_vis_path[:-4]}_right.obj')
#                     #     # trimesh.exchange.export.export_mesh(obj_mesh,f'{save_contact_vis_path[:-4]}_obj.obj')
#                     # else:
#                     #     trimesh.exchange.export.export_mesh(hand_mesh[0],f'{save_contact_vis_path[:-4]}_right.obj')
#                         # trimesh.exchange.export.export_mesh(obj_mesh,f'{save_contact_vis_path[:-4]}_obj.obj')
#                     ######################
#             ## old script ##

#         samples, targets = prefetcher.next()

#     metric_logger.synchronize_between_processes()
#     stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

#     save_dir = os.path.join(args.output_dir, 'results.txt')
#     with open(save_dir, 'a') as f:
#         if args.test_viewpoint is not None:
#             f.write(f"{'='*10} {args.test_viewpoint} {'='*10}\n")
#         f.write(f"{'='*10} {args.resume} {'='*10}\n\n")
#         for key, val in stats.items():
#             f.write(f'{key:30} : {val}\n')
#         f.write('\n\n')

#     return stats    