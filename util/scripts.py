import sys
import util.misc as utils

from models import build_model
from extract_predicts import main as submit_main
from engine import train_smoothnet, test_smoothnet
from arctic_tools.common.body_models import build_mano_aa
from arctic_tools.common.object_tensors import ObjectTensors
from models.smoothnet import ArcticSmoother, SmoothCriterion
from util.settings import set_training_scheduler, load_resume


def smoothnet_main(model, data_loader_train, data_loader_val, args, cfg):
    device = args.device
    smoother = ArcticSmoother(args.batch_size, args.window_size).to(device)
    WEIGHT_DICT = {
        "loss/cd":10.0,
        "loss/mano/cam_t/r":1.0,
        "loss/mano/cam_t/l":1.0,
        "loss/object/cam_t":1.0,
        "loss/mano/pose/r":10.0,
        "loss/mano/beta/r":0.001,
        "loss/mano/pose/l":10.0,
        "loss/mano/beta/l":0.001,
        "loss/object/radian":1.0,
        "loss/object/rot":10.0,
        "loss/mano/transl/l":1.0,
        "loss/object/transl":1.0,
        "acc/h":1.0,
        "acc/o":1.0,
    }
    obj_tensor = ObjectTensors()
    obj_tensor.to(device)
    pre_process_models = {
        "mano_r": build_mano_aa(is_rhand=True).to(device),
        "mano_l": build_mano_aa(is_rhand=False).to(device),
        "arti_head": obj_tensor
    }                
    smoother_criterion = SmoothCriterion(args.batch_size, args.window_size, WEIGHT_DICT, pre_process_models).to(device)
    # optimizer, lr_scheduler = set_training_scheduler(args, smoother, general_lr=0.001)
    if data_loader_train is not None:
        optimizer, lr_scheduler = set_training_scheduler(args, smoother, len_data_loader_train=len(data_loader_train))
    else:
        optimizer = lr_scheduler = None

    if args.smooth_resume:
        smoother, optimizer, lr_scheduler = load_resume(args, smoother, args.smooth_resume, optimizer, lr_scheduler)

    # for evaluation
    if args.eval:
        test_smoothnet(model, smoother, data_loader_val, device, cfg, args=args, vis=args.visualization)    
        sys.exit(0)

    # for train
    else:
        for epoch in range(args.start_epoch, args.epochs):         
            train_smoothnet(model, smoother, smoother_criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, args=args, cfg=cfg)
            if not args.onecyclelr:
                lr_scheduler.step()

            utils.save_on_master({
                'model': smoother.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, f'{args.output_dir}/{epoch}.pth')

            test_smoothnet(model, smoother, data_loader_val, device, cfg, args=args, vis=args.visualization, epoch=epoch)


def submit_result(args, cfg):
    # build model
    model, _ = build_model(args, cfg)
    model.to(args.device)
    model_without_ddp = model

    # load ckpt
    load_resume(args, model_without_ddp, args.resume)
    model_without_ddp.eval()

    # submit
    submit_main(args, model_without_ddp, cfg)