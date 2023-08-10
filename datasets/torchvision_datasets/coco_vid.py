# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from torchvision
# ------------------------------------------------------------------------

"""
Copy-Paste from torchvision, but add utility of caching images on memory
"""
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import tqdm
from io import BytesIO
from collections import defaultdict
import numpy as np
import multiprocessing as mp
import ctypes
import torch
import math
import trimesh
import pickle

class CocoDetection_vid(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1, mode=None, args=None):
        super(CocoDetection_vid, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.mode = mode
        self.vid = defaultdict(list)
        self.imgid2vidid = {}
        for img_id in self.coco.imgs.keys():
            vid_id = self.coco.loadImgs(img_id)[0]['vid_id']
            self.vid[vid_id].append(img_id)
            self.imgid2vidid[img_id] = vid_id
        self.all_vid_len = len(self.vid)
        self.per_vid_len = [len(self.vid[i]) for i in self.vid.keys()]

        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        if cache_mode:
            self.cache = {}
            self.cache_images()

        self.coco.imgs.keys()
        self.num_frame = args.num_frame
        self.dataset = args.dataset_file

    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path = self.coco.loadImgs(img_id)[0]['file_name']
            with open(os.path.join(self.root, path), 'rb') as f:
                self.cache[path] = f.read()

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert('RGB')
        return Image.open(os.path.join(self.root, path)).convert('RGB')
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_ids = self.vid[index]
        if self.mode == 'train':
            img_ids = np.array(img_ids)[self.sampling(len(img_ids), self.num_frame)].tolist()
        else:
            img_ids = np.array(img_ids)[self.uniform_sampling(len(img_ids), self.num_frame)].tolist()

        uvd_list = []
        cam_list = []
        _6D_list = []
        label_list = []
        mano_list = []
        path_list = []
        
        action_id = -1

        for img_id, infor in zip(img_ids, coco.loadImgs(img_ids)):
            
            # file_path = os.path.join(f'/home/user/hoseong/H2OTR/pickle/FPHA/{self.mode}', infor['file_name'])#.replace('H2O', f'H2O_results_aug_{self.mode}')
            file_path = os.path.join(os.getcwd(),f'pickle/{self.dataset}/{self.mode}', infor['file_name'])#.replace('H2O', f'H2O_results_aug_{self.mode}')
            data_path = file_path[:-4] + '_data.pkl'
  
            with open(data_path, 'rb') as f:
                pred = pickle.load(f)

            uvd_list.append(torch.tensor(pred['uvd']))
            cam_list.append(torch.tensor(pred['cam']))
            _6D_list.append(torch.tensor(pred['6D']))
            label_list.append(torch.tensor(pred['label']))
            mano_list.append(torch.tensor(pred['mano']))
            
            action_id = coco.loadImgs(img_id)[0]['action']
            path_list.append(data_path)
        
        uvd_list = torch.stack(uvd_list)
        cam_list = torch.stack(cam_list)
        _6D_list = torch.stack(_6D_list)
        label_list = torch.stack(label_list)
        mano_list = torch.stack(mano_list)
        return uvd_list, cam_list, _6D_list, label_list, mano_list, action_id, img_ids, data_path
    
    
    def __len__(self):
        return self.all_vid_len

    def sampling(self, total_frame_num, sample_frame_num):
        
        if total_frame_num > sample_frame_num:
            idxs = np.arange(0, sample_frame_num + 1) * total_frame_num/(sample_frame_num + 1)
            idxs = np.unique(np.trunc(idxs))
            idxs = np.array([np.random.choice(range(idxs[i].astype(np.int32), idxs[i+1].astype(np.int32))) for i in range(sample_frame_num)])
            
        else:
            idxs = np.arange(0, sample_frame_num ) * total_frame_num/(sample_frame_num )
            idxs = np.trunc(idxs)
        return list(idxs.astype(np.int32))

    #전체 frame중 sampling
    def uniform_sampling(self, total_frame_num, sample_frame_num):
        idxs = np.arange(0, sample_frame_num) * total_frame_num/sample_frame_num
        if total_frame_num >= sample_frame_num:
            idxs = np.unique(np.trunc(idxs))
        else:
            idxs = np.trunc(idxs)
        return list(idxs.astype(np.int32))