# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .actic_detr import build as build_arctic
from .assembly_detr import build as build_assembly
# from .dn_dab_dino_deformable_detr import build_dab_dino_deformable_detr
from .dino import build_dino

def build_model(args, cfg):
    if args.modelname == 'dino':
        return build_dino(args, cfg)
    
    # for dn-dino-deformable-detr
    # if args.modelname == 'dn_detr':
    #     return build_dab_dino_deformable_detr(args, cfg)
    
    # for deformable detr
    else:
        if args.dataset_file == 'arctic':
            return build_arctic(args, cfg)
        elif args.dataset_file == 'AssemblyHands':
            return build_assembly(args, cfg)
        else:
            raise Exception('Not implemented!')