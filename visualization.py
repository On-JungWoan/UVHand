import os
import sys
import torch
import util.misc as utils
from datasets.data_prefetcher import data_prefetcher
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from manopth.manolayer import ManoLayer
from engine import visualize, visualize_obj
from torchvision import transforms as T

def make_coco_transforms(image_set, img_size, make_pickle):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train' and make_pickle==False:
        return T.Compose([
            T.Resize(img_size),
            T.CollorJitter(),
            T.RandomRotation(45), #15
            normalize,
        ])
    else:
        return T.Compose([
            T.Resize(img_size),
            normalize,
        ])

def vis(model, device, cfg, vis=True):
    
    model.eval()
    
    filepath = input("Path(enter 'x' to exit) : ")
    if filepath == 'x':
        sys.exit(0)
    source_img = Image.open(filepath)
    samples = make_coco_transforms('val', (960, 540), False)(source_img).to(device)
    source_img = np.array(source_img)
    orig_target_sizes = torch.tensor([source_img.shape[:-1]])

    with torch.no_grad():
        outputs = model(samples[None])
                        
        # model output
        out_logits,  pred_keypoints, pred_obj_keypoints = outputs['pred_logits'], outputs['pred_keypoints'], outputs['pred_obj_keypoints']

        prob = out_logits.sigmoid()
        B, num_queries, num_classes = prob.shape

        # query index select
        best_score = torch.zeros(B).to(device)
        obj_idx = torch.zeros(B).to(device).to(torch.long)
        for i in range(1, cfg.hand_idx[0]):
            score, idx = torch.max(prob[:,:,i], dim=-1)
            obj_idx[best_score < score] = idx[best_score < score]
            best_score[best_score < score] = score[best_score < score]

        hand_idx = []
        for i in cfg.hand_idx:
            hand_idx.append(torch.argmax(prob[:,:,i], dim=-1)) 
        hand_idx = torch.stack(hand_idx, dim=-1)   
        hand_kp = torch.gather(pred_keypoints, 1, hand_idx.unsqueeze(-1).repeat(1,1,63)).reshape(B, -1 ,21, 3)
        obj_kp = torch.gather(pred_obj_keypoints, 1, obj_idx.unsqueeze(1).unsqueeze(1).repeat(1,1,63)).reshape(B, 21, 3)

        im_h, im_w = orig_target_sizes[:,0], orig_target_sizes[:,1]
        target_sizes = torch.cat([im_w.unsqueeze(-1), im_h.unsqueeze(-1)], dim=-1)
        target_sizes =target_sizes.cuda()

        hand_kp[...,:2] *=  target_sizes.unsqueeze(1).unsqueeze(1); hand_kp[...,2] *= 1000
        obj_kp[...,:2] *=  target_sizes.unsqueeze(1); obj_kp[...,2] *= 1000
        key_points = torch.cat([hand_kp, obj_kp.unsqueeze(1)], dim=1)
        
        if vis:
            batch, js, _, _ = key_points.shape
            for b in range(batch):
                for j in range(js):
                    pred_kp = key_points[b][j]
                    if j ==0:
                        source_img = visualize(source_img, pred_kp.detach().cpu().numpy().astype(np.int32), 'left')
                    elif j == 1:
                        source_img = visualize(source_img, pred_kp.detach().cpu().numpy().astype(np.int32), 'right')
                    else:
                        source_img = visualize_obj(source_img, pred_kp.detach().cpu().numpy().astype(np.int32))