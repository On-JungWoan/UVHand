import torch
import pickle
import argparse
import numpy as np
import os.path as op


# general arguments
def get_general_args_parser():
    parser = argparse.ArgumentParser('General args', add_help=False)

    # general
    parser.add_argument('--modelname', default='deformable_detr', choices=['deformable_detr', 'dn_detr'])    
    parser.add_argument('--dataset_file', default='arctic')

    # for eval
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--visualization', default=False, action='store_true')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume_dir', default='', help='resume dir from checkpoint')
    parser.add_argument('--val_batch_size', default=4, type=int)
    parser.add_argument('--eval_metrics', default=["aae","mpjpe.ra","mrrpe","success_rate","cdev","mdev","acc_err_pose"], nargs='+', \
                        help='Choose evaluation metrics.')
    parser.add_argument('--full_validation', default=False, action='store_true', \
                        help='Use full size of validation dataset to evaluate the model.')
    parser.add_argument('--test_viewpoint', default=None, type=str, \
                        help='If you want to evaluate a specific viewpoint, then you can simply put the viewpoint name.\n \
                            e.g) --test_viewpoint nusar-2021_action_both_9081-c11b_9081_user_id_2021-02-12_161433/HMC_21110305_mono10bit')
    parser.add_argument('--extract', default=False, action='store_true',
                        help='Save pred_keypoints to json format.')

    # for train
    parser.add_argument('--not_use_params', default=[], nargs='+', help='The params, including this keywords, are ignored when the model imports the checkpoint.')
    parser.add_argument('--wandb', default=False, action='store_true', help='Use wandb')
    parser.add_argument('--dist_backend', default=None, help='Choose backend of distribtion mode.')
    parser.add_argument('--sgd', action='store_true')

    # for debug
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--num_debug', default=3, type=int)

    # for custom arctic
    parser.add_argument('--seq', default=None, type=str)
    parser.add_argument('--split_window', default=False, action='store_true')
    parser.add_argument('--feature_type', default='local_fm', choices=['origin', 'global_fm', 'local_fm'])
    parser.add_argument('--train_smoothnet', default=False, action='store_true')
    parser.add_argument('--iter', default=20, type=int)

    # for coco
    parser.add_argument('--img_size', default=(960, 540), type=tuple)
    parser.add_argument('--make_pickle', default=False, action='store_true')
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser


# deformable detr arguments
def get_deformable_detr_args_parser(parser):
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    # parser.add_argument('--set_cost_class', default=2, type=float,            
    #                     help="Class coefficient in the matching cost")
    # parser.add_argument('--set_cost_keypoint', default=5, type=float,
    #                     help="L1 box coefficient in the matching cost") 
    parser.add_argument('--set_cost_class', default=1.5, type=float,            
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_keypoint', default=4, type=float,
                        help="L1 box coefficient in the matching cost") 

    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=2, type=float) 
    parser.add_argument('--keypoint_loss_coef', default=5, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--train_stage', default='pose')
    parser.add_argument('--coco_path', required=True, type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./weights/local_test',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)

    return parser


# dn detr arguments
def get_dn_detr_args_parser(parser):
    # about dn args
    parser.add_argument('--use_dn', action="store_true",
                        help="use denoising training.")
    parser.add_argument('--box_noise_scale', default=0.4, type=float,
                        help="box noise scale to shift and scale")
    parser.add_argument('--contrastive', action="store_true",
                        help="use contrastive training.")
    parser.add_argument('--use_mqs', action="store_true",
                        help="use mixed query selection from DINO.")
    parser.add_argument('--use_lft', action="store_true",
                        help="use look forward twice from DINO.")

    # about lr
    parser.add_argument('--lr', default=1e-4, type=float, 
                        help='learning rate')
    parser.add_argument('--lr_backbone', default=1e-5, type=float, 
                        help='learning rate for backbone')

    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--override_resumed_lr_drop', default=False, action='store_true')
    parser.add_argument('--drop_lr_now', action="store_true", help="load checkpoint and drop for 12epoch setting")
    parser.add_argument('--save_checkpoint_interval', default=10, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--pe_temperatureH', default=20, type=int, 
                        help="Temperature for height positional encoding.")
    parser.add_argument('--pe_temperatureW', default=20, type=int, 
                        help="Temperature for width positional encoding.")
    parser.add_argument('--batch_norm_type', default='FrozenBatchNorm2d', type=str, 
                        choices=['SyncBatchNorm', 'FrozenBatchNorm2d', 'BatchNorm2d'], help="batch norm type for backbone")

    # * Transformer
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--backbone_freeze_keywords', nargs="+", type=str, 
                        help='freeze some layers in backbone. for catdet5.')
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true', 
                        help="Using pre-norm in the Transformer blocks.")    
    parser.add_argument('--num_select', default=300, type=int, 
                        help='the number of predictions selected for evaluation')
    parser.add_argument('--num_results', default=300, type=int,help="Number of detection results")
    # parser.add_argument('--transformer_activation', default='prelu', type=str)
    # parser.add_argument('--num_patterns', default=0, type=int, help='number of pattern embeddings. See Anchor DETR for more details.')
    # parser.add_argument('--num_queries', default=300, type=int,help="Number of query slots")
    # parser.add_argument('--scalar', default=5, type=int,help="number of dn groups")
    # parser.add_argument('--label_noise_scale', default=0.2, type=float,help="label noise ratio to flip")
    parser.add_argument('--transformer_activation', default='relu', type=str)
    parser.add_argument('--num_patterns', default=3, type=int, help='number of pattern embeddings. See Anchor DETR for more details.')
    parser.add_argument('--num_queries', default=900, type=int,help="Number of query slots")
    parser.add_argument('--scalar', default=200, type=int,help="number of dn groups")
    parser.add_argument('--label_noise_scale', default=0.5, type=float,help="label noise ratio to flip")
    
    parser.add_argument('--random_refpoints_xy', action='store_true', 
                        help="Random init the x,y of anchor boxes and freeze them.")

    # for DAB-Deformable-DETR
    parser.add_argument('--two_stage', default=False, action='store_true', 
                        help="Using two stage variant for DAB-Deofrmable-DETR")
    parser.add_argument('--num_feature_levels', default=4, type=int, 
                        help='number of feature levels')
    parser.add_argument('--dec_n_points', default=4, type=int, 
                        help="number of deformable attention sampling points in decoder layers")
    parser.add_argument('--enc_n_points', default=4, type=int, 
                        help="number of deformable attention sampling points in encoder layers")


    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float, 
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=1, type=float, 
                        help="loss coefficient for cls")
    parser.add_argument('--mask_loss_coef', default=1, type=float, 
                        help="loss coefficient for mask")
    parser.add_argument('--dice_loss_coef', default=1, type=float, 
                        help="loss coefficient for dice")
    parser.add_argument('--bbox_loss_coef', default=5, type=float, 
                        help="loss coefficient for bbox L1 loss")
    parser.add_argument('--giou_loss_coef', default=2, type=float, 
                        help="loss coefficient for bbox GIOU loss")
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--focal_alpha', type=float, default=0.25, 
                        help="alpha for focal loss")


    # dataset parameters
    parser.add_argument('--coco_path', type=str, required=True)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true', 
                        help="Using for debug only. It will fix the size of input images to the maximum.")


    # Traing utils
    parser.add_argument('--output_dir', default='./weights/local_test', help='path where to save, empty for no saving')
    parser.add_argument('--note', default='', help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+', 
                        help="A list of keywords to ignore when loading pretrained models.")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--find_unused_params', default=False, action='store_true')

    parser.add_argument('--save_results', action='store_true', 
                        help="For eval only. Save the outputs for all images.")
    parser.add_argument('--save_log', action='store_true', 
                        help="If save the training prints to the log file.")

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    return parser


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def set_training_scheduler(args, model, general_lr=None):
    if general_lr is None:
        general_lr = args.lr

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        }
    ]

    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=general_lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=general_lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    return optimizer, lr_scheduler


def load_resume(args, model, resume):
    checkpoint = torch.load(resume, map_location='cpu')
    ckpt = checkpoint['model'].copy()
    for key in ckpt.keys():
        if len(args.not_use_params) != 0:
            for ig_key in args.not_use_params:
                if ig_key in key:
                    print(f'ignored params : {key}')
                    checkpoint['model'].pop(key)
    del ckpt

    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))
    
    return model


def extract_epoch(file_path):
    file_name = file_path.split('/')[-1]
    epoch = op.splitext(file_name)[0]

    return int(epoch)


def make_arctic_environments(args):
    check_dir = 'datasets/arctic/common/environments.py'
    env_dir = op.join(args.coco_path, args.dataset_file)

    with open(check_dir, 'w') as f:
        f.write(
            f"DATASET_ROOT = '{env_dir}'"
        )