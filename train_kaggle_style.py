#!/usr/bin/env python
import argparse
import os
import random
import shutil
import time
import yaml
import logging
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict
from datetime import datetime
from torchvision.utils import save_image
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, SWALR
from src.data import resolve_data_config, get_valid_transforms, get_train_transforms, ODDDataset, Mixup, FastCollateMixup
from src.data.loader import batch_sampler
from src.utils import SimpleCheckpointSaver, seed_everything
from src.loss import *
from src.runnerOOD import Runner
from timm.data.loader import fast_collate
from timm.models import resume_checkpoint, create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, setup_default_logging, get_outdir, update_summary, distribute_bn
from contextlib import suppress
from tensorboardX import SummaryWriter

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
# Device options
parser.add_argument('-g', '--gpu-id', nargs='+', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

if __name__ == "__main__":
    from args_params import _parse_args
    setup_default_logging()
    cfg, args_text = _parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(cfg.gpu_id)

    # cfg = Config(args)
    cfg.distributed = False
    cfg.T_0 = cfg.epochs

    seed_everything(cfg.seed)

    runner = Runner()
    model = create_model(
            cfg.model,
            pretrained=True,
            num_classes=cfg.num_classes,
            drop_rate=cfg.drop,
            drop_path_rate=cfg.drop_path,
            drop_block_rate=cfg.drop_block,
            checkpoint_path=cfg.initial_checkpoint,
            pretrained_cfg=cfg.pretrained_cfg
            )
    logging.info('Model %s created, param count: %d' % (cfg.model, sum([m.numel() for m in model.parameters()])))
    model = model.cuda()

    data_config = resolve_data_config(vars(cfg), model=model, verbose=cfg.local_rank == 0)

    img_size = data_config['input_size'][1:] # [256,256]
    crop_pct = data_config["crop_pct"]
    train_dir = os.path.join(cfg.data, 'Images')
    eval_dir = os.path.join(cfg.data, 'Images')
    dataset_train = ODDDataset(root=train_dir,
                            transform=get_train_transforms(cfg, img_size, resize_type="normal"), 
                            load_type="cv2")
    dataset_eval = ODDDataset(root=eval_dir,
                           transform=get_valid_transforms(img_size, resize_type="normal", crop_pct=crop_pct), 
                           load_type="cv2")
    
    collate_fn = fast_collate if cfg.prefetcher else torch.utils.data.dataloader.default_collate
    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = cfg.mixup > 0 or cfg.cutmix > 0. or cfg.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=cfg.mixup, cutmix_alpha=cfg.cutmix, cutmix_minmax=cfg.cutmix_minmax,
            prob=cfg.mixup_prob, switch_prob=cfg.mixup_switch_prob, mode=cfg.mixup_mode,
            label_smoothing=cfg.smoothing, num_classes=cfg.num_classes)
        if cfg.prefetcher:
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)
    
    if not cfg.imbalance_sampler:
        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=cfg.batch_size,
            pin_memory=True,
            drop_last=False,
            shuffle=True,
            num_workers=cfg.workers
        )
    else:
        print("Use imbalance_sampler")
        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_sampler=batch_sampler(cfg.batch_size, dataset_train.labels),
            pin_memory=True,
            num_workers=cfg.workers
        )
    loader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.workers,
        drop_last=False,
    )

    loss_scaler = NativeScaler()
    amp_autocast = torch.cuda.amp.autocast
    if cfg.val_amp: val_amp = amp_autocast
    else: val_amp = suppress
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    optimizer = create_optimizer(cfg, model)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=CFG['epochs']-1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.T_0, T_mult=1,
                                                                        eta_min=cfg.min_lr, last_epoch=-1)
        
    train_loss_fns = []
    if cfg.fire_ce_loss:
        train_loss_fn = CrossEntropyLossOneHot(smoothing=cfg.smoothing).cuda()
    elif cfg.asl_loss:
        train_loss_fn = AsymmetricLossOptimized(gamma_neg=2, gamma_pos=1, clip=0)
    elif cfg.fire_focal_loss:
        train_loss_fn = FocalLossWithWeight(label_smooth=cfg.smoothing, alpha=0.25, gamma=2.0).cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
    
    if mixup_active:
        train_loss_fn = CrossEntropyLossLogits().cuda()

    train_loss_fns.append(train_loss_fn)    
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    logging.info("Training functions:{}".format(train_loss_fn))

    exp_name = '-'.join([
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        cfg.model,
        str(img_size[0])
    ])
    output_dir = get_outdir(cfg.output, exp_name)
    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    saver = SimpleCheckpointSaver(model=model, model_ema=None, 
                                  amp_scaler=loss_scaler, checkpoint_dir=output_dir)
    # 复制yaml文件
    shutil.copy("configs/odd.yaml", output_dir)
    
    for epoch in range(cfg.epochs):
        train_metrics = runner.train_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, cfg, model_ema=None,
                lr_scheduler=lr_scheduler, timm_scheduler=False, schd_batch_update=False, 
                output_dir=output_dir, mixup_fn=mixup_fn,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, writer_dict=writer_dict)

        eval_metrics = runner.validate(model, loader_eval, validate_loss_fn, cfg, output_dir=output_dir,
                                       amp_autocast=amp_autocast, writer_dict=writer_dict, verbose=False)

        eval_metric_list = cfg.eval_metric.split("-")
        if saver is not None:
            for k, v in eval_metrics.items():
                decreasing = True if k == 'loss' else False
                if k not in eval_metric_list: continue
                save_metric = eval_metrics[k]
                save_prefix = "%.4f-%s" % (eval_metrics[k], k)
                best_metric, best_epoch = saver.save_checkpoint(epoch=epoch,
                                                                save_prefix=save_prefix,
                                                                metric=save_metric,
                                                                metric_name=k,
                                                                decreasing=decreasing)