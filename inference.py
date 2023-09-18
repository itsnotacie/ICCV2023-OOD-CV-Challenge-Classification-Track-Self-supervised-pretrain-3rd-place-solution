#!/usr/bin/env python
import argparse
import os
import csv
import glob
import logging
import torch
import torch.nn.parallel
import torch.nn as nn
import numpy as np
from contextlib import suppress
from timm.models import load_checkpoint
from timm.utils import setup_default_logging, AverageMeter
from timm.models import create_model
from src.runnerSingle import Runner
from src.data import ODDDataset, get_valid_transforms, resolve_data_config
from src.models import SingleLabelModel
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--data', default='/data/guofeng/dataset/OODCV2023/phase2-test-images', metavar='DIR', help='path to dataset')
parser.add_argument('--val-flips', default=[], metavar='N', help='version of multi head (default: v0)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--head-num', type=int, default=1, metavar='N', help='number of multi head (default: 1)')
parser.add_argument('--input-size', type=int, default=(3, 448, 448), metavar='N', help='Image patch size (default: None => model default)')
parser.add_argument('--num-classes', type=int, default=10, metavar='N', help='number of label classes (default: 1000)')
ckpt_name = "20230912-132638-eva02_large_patch14_448.mim_m38m_ft_in22k_in1k-448"
parser.add_argument("-c", '--ckpt-name', type=str, default=ckpt_name, help='ckpt name')
parser.add_argument('--checkpoint', default='output', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--amp', action='store_true', default=False, help='Use half precision (fp16)')
parser.add_argument("-s", '--set-name', type=str, default="test", help='train/val/test')
parser.add_argument("-m", '--metric', type=str, default="mf1", help='mf1, mif1')
parser.add_argument('--results-file', default='infer', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
# Device options
parser.add_argument("-g", '--gpu-id', default='1', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')


def validate(args):
    args.att_pattern = None
    args.multi_drop = 0.0
    args.drop = 0.0
    args.drop_path = 0.0
    args.drop_block = None
    args.save_images = False
    
    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    args.model = args.ckpt_name.split("/")[-1].split("-")[-2]
    model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=args.drop_block,
            )
    if args.checkpoint: load_checkpoint(model, args.checkpoint)

    param_count = sum([m.numel() for m in model.parameters()])
    logging.info('Model %s created, param count: %d' % (args.model, param_count))

    model = model.cuda()
    model.eval()
    
    amp_autocast = suppress  # do nothing
    if args.amp: amp_autocast = torch.cuda.amp.autocast
    
    data_config = resolve_data_config(vars(args), model=model, verbose=True)

    img_size = data_config['input_size'][1:]
    # eval_dir = os.path.join(args.data, args.set_name)
    eval_dir = os.path.join(args.data)
    dataset_test = ODDDataset(root=eval_dir, 
                              transform=get_valid_transforms(img_size, resize_type="normal"), load_type="cv2")
        
    loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )
        
    softmax = torch.nn.Softmax(dim=1)

    total_pred_idx = []
    with torch.no_grad():
        for _, (input, _) in enumerate(tqdm(loader)):
            input = input.cuda()
            with amp_autocast():
                output = model(input)

            preds = torch.max(softmax(output), dim=1)[1].cpu().numpy()
            total_pred_idx.extend(preds)
    
    classes = ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train']
    with open(os.path.join(args.results_file, 'results.csv'), 'w', encoding="utf-8") as out_file:
        writer = csv.writer(out_file)
        csv_header = ["imgs", "pred"]
        writer.writerow(csv_header)
        
        filenames = loader.dataset.filenames()
        for i, pred_idx in enumerate(total_pred_idx):
            line_data = [filenames[i].split('/')[-1], classes[pred_idx]]
            writer.writerow(line_data)
            
    os.system("cd infer/{} && zip {}.zip results.csv".format(ckpt_name, ckpt_name))
    

def main():
    setup_default_logging()
    args = parser.parse_args()
    checkpoints = glob.glob(os.path.join(args.checkpoint, args.ckpt_name, '*.pth'))
    result_file_path = os.path.join(args.results_file, args.ckpt_name)
    if not os.path.exists(result_file_path): os.makedirs(result_file_path)
    args.results_file = result_file_path

    for checkpoint in checkpoints:
        if checkpoint.split("/")[-1].split("-")[1] != args.metric: continue
        args.checkpoint = checkpoint
        validate(args)


if __name__ == '__main__':
    main()
