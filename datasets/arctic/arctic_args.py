def construct_arctic_args(args):
    args.focal_length = 1000.0
    args.img_res = 224
    args.rot_factor = 30.0
    args.noise_factor = 0.4
    args.scale_factor = 0.25
    args.flip_prob = 0.0
    args.img_norm_mean = [0.485, 0.456, 0.406]
    args.img_norm_std = [0.229, 0.224, 0.225]
    args.pin_memory = True
    args.shuffle_train = True
    args.seed = 1
    args.grad_clip = 150.0
    args.use_gt_k = False  # use weak perspective camera or the actual intrinsics
    args.speedup = True  # load cropped images for faster training
    # args.speedup = False # uncomment this to load full images instead
    args.max_dist = 0.10  # distance range the model predicts on
    args.ego_image_scale = 0.3

    if args.method in ["field_sf", "field_lstm"]:
        args.project = "interfield"
    else:
        args.project = "arctic"
    args.interface_p = None

    if args.fast_dev_run:
        args.num_workers = 0
        args.batch_size = 8
        args.trainsplit = "minitrain"
        args.valsplit = "minival"
        args.log_every = 5
        args.window_size = 3
    else:
        args.window_size = 11

    return args