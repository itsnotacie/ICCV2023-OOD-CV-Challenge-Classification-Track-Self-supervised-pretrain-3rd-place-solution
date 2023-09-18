#!/usr/bin/env python
import argparse
import logging
import yaml
import os
import random
import torch
import torch.nn as nn
import numpy as np

from torch.nn.parallel import DistributedDataParallel as NativeDDP

from src.data import resolve_data_config, get_valid_transforms, get_train_transforms, Mixup
from src.utils import SimpleCheckpointSaver, PrefetchLoader
from src.loss import *
from src.data.loader import batch_sampler
from timm.models import resume_checkpoint, create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, setup_default_logging, get_outdir, update_summary, distribute_bn

from tensorboardX import SummaryWriter
from datetime import datetime
from contextlib import suppress

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Dataset / Model parameters
parser.add_argument('--data', default='/data/guofeng/classification/Thyroid/nodule_cropv4', metavar='DIR',
                    help='path to dataset')
# tf_efficientnet_b0.ns_jft_in1k tf_efficientnet_lite0.in1k tf_efficientnetv2_s.in1k
# tf_efficientnetv2_b0.in1k efficientnet_b0
parser.add_argument('--model', default='tf_efficientnet_b0_ns', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
# parser.add_argument('--model-type', default='timm', type=str, metavar='MODEL', help='type of model to train (default: "timm"')
parser.add_argument('--initial-checkpoint',
                    default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=2, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--head-num', type=int, default=1, metavar='N',
                    help='number of multi head (default: 1)')
parser.add_argument('-i', '--img-size', type=int, default=160, metavar='N', help='Image patch size (default: None => model default)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 32)')
# -1:水平和垂直翻转;0:垂直翻转;1:水平翻转:2:不进行翻转 [-1,0,1] [1]
parser.add_argument('--train-flips', default=[], metavar='N', help='version of multi head (default: v0)')
parser.add_argument('--val-flips', default=[], metavar='N', help='version of multi head (default: v0)')
# parser.add_argument('--att-layer', type=bool, default=False, metavar='N', help='version of multi head (default: v0)')
parser.add_argument('--att-pattern', type=str, default=None, metavar='N', help='version of multi head (default: v0)')
parser.add_argument('--drop', type=float, default=0.7, metavar='PCT', help='Dropout rate (default: 0.)')
parser.add_argument('--multi-drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--focal-loss', action='store_true', default=False,
                    help='whether to use focal loss')
parser.add_argument('--weighted', action='store_true', default=False,
                    help='whether to use focal loss')
parser.add_argument('--soft-ce-loss', action='store_true', default=False,
                    help='whether to use focal loss')
parser.add_argument('--fire-ce-loss', action='store_true', default=False,
                    help='whether to use focal loss')
parser.add_argument('--fire-focal-loss', action='store_true', default=True,
                    help='whether to use focal loss')
parser.add_argument('--asl-loss', action='store_true', default=False,
                    help='whether to use focal loss')
parser.add_argument('--sce-loss', action='store_true', default=False,
                    help='whether to use focal loss')
parser.add_argument('--multi-color-loss', action='store_true', default=False,
                    help='whether to use focal loss')
parser.add_argument('--imbalance-sampler', action='store_true', default=False,
                    help='whether to use imbalance sampler')

# Augmentation & regularization parameters
parser.add_argument('--aa', type=str, default="originalr", metavar='NAME', # rand-m9-mstd0.5 originalr augmix-m5-w4-d2 original-mstd0.5
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--ssrprob', type=float, default=0.5, metavar='PCT', help='prob (default: 0.)')
parser.add_argument('--shift-limit', type=float, nargs='+', default=[-0.0625, 0.0625], help='shift-limit (default: -0.0625 0.0625)')
parser.add_argument('--scale-limit', type=float, nargs='+', default=[-0.1, 0.1], help='scale-limit (default: -0.1 0.1)')
parser.add_argument('--rotate-limit', type=float, nargs='+', default=[-45, 45], help='rotate-limit (default: -45 45)')
parser.add_argument('--border-mode', default='constant', type=str, metavar='NAME', help='border-mode (default constant)')
parser.add_argument('--hflip', type=float, default=0.5, help='Horizontal flip training aug probability')
parser.add_argument('--gdprob', type=float, default=0.5, metavar='PCT', help='Grid drop prob (default: 0.)')
parser.add_argument('--gdratio', type=float, default=0.2, help='Grid drop ratio')
parser.add_argument('--rgsprob', type=float, default=0.5, metavar='PCT', help='RandomGridShuffle prob (default: 0.)')
parser.add_argument('--rgsgrid', type=int, default=2, help='RandomGridShuffle grid')
parser.add_argument('--reprob', type=float, default=0.5, metavar='PCT', help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
parser.add_argument('--min-area', type=float, default=0.02, help='Random erase min area (default: "pixel")')
parser.add_argument('--max-area', type=float, default=0.1, help='Random erase max area (default: "pixel")')
parser.add_argument('--recount', type=int, default=5, help='Random erase count (default: 1)')
parser.add_argument('--ycprob', type=float, default=0.0, metavar='PCT', help='Yolo cutout prob (default: 0.)')
parser.add_argument('--cur-prob', type=float, default=0.0, help='Random paste cur prob (default: 0.)')
parser.add_argument('--dst-size', type=int, default=64, help='Random paste cur dst size (default: 32)')
parser.add_argument('--cur-num-inner', type=int, default=5, help='Random paste cur num (default: 5)')
# mixup and cutmix
parser.add_argument('--mixup', type=float, default=0.5,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.5,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=0.5,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.01,
                    help='label smoothing (default: 0.1)')

# Optimizer parameters lookahead_radam adamw sgdp adamp radam
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Learning rate schedule parameters
# cosine step
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.01)')
# parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
#                     help='learning rate noise on/off epoch percentages')
# parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
#                     help='learning rate noise limit percent (default: 0.67)')
# parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
#                     help='learning rate noise std-dev (default: 1.0)')
# parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
#                     help='learning rate cycle len multiplier (default: 1.0)')
# parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
#                     help='amount to decay each learning rate cycle (default: 0.5)')
# parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
#                     help='learning rate cycle limit, cycles enabled if > 1')
# parser.add_argument('--lr-k-decay', type=float, default=1.0,
#                     help='learning rate k-decay for cosine/poly (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-7, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('-j', '--workers', type=int, default=8, metavar='N', help='how many training processes to use (default: 1)')
parser.add_argument('--num-gpu', type=int, default=1, help='Number of GPUS to use')
parser.add_argument('--save-images', action='store_true', default=True, help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=True, help='use NVIDIA amp for mixed precision training')
parser.add_argument('--val-amp', action='store_true', default=False, help='use NVIDIA amp for mixed precision validation')
parser.add_argument('--prefetcher', action='store_true', default=False, help='disable fast prefetcher')
parser.add_argument('--output', default='output/thyroid', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
# sf1 cf1 mf1 mif1
parser.add_argument('--eval-metric', default='mf1', type=str, metavar='EVAL_METRIC', help='Best metric (default: "acc1"')
parser.add_argument("--local_rank", default=0, type=int)
# Device options
parser.add_argument('-g', '--gpu-id', nargs='+', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False, sort_keys=False)
    return args, args_text