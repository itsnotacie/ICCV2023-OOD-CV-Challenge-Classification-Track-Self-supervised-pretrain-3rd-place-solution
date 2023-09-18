import json
import logging
import time
import os
import torch
import torch.nn as nn
import torchvision.utils

import numpy as np
from timm.utils import dispatch_clip_grad, AverageMeter, reduce_tensor
from timm.models import model_parameters
from torch.optim.swa_utils import AveragedModel, SWALR
from collections import OrderedDict
from contextlib import suppress
from copy import deepcopy
from sklearn.metrics import accuracy_score, f1_score, classification_report

class Runner(object):
    def __init__(self, weights_path=None):
        
        if weights_path is not None:
            self.weights_path = weights_path
            self.weights = self.load_loss_weights()

        self.classes_num = ""
        self.classes_list = ""
        self.classes_dict = ""
            
    def load_loss_weights(self):
        with open(self.weights_path, "r", encoding="utf-8") as jf:
            json_data = json.load(jf)
        
        return json_data
    
    def set_classes_num(self, model, distributed):
        if distributed:
            self.classes_num = model.module.classes_num
            self.classes_list = model.module.classes_list
            self.classes_dict = model.module.classes_dict
        else:
            self.classes_num = model.classes_num
            self.classes_list = model.classes_list
            self.classes_dict = model.classes_dict    
    
    def train_epoch(self, epoch, model, loader, optimizer, loss_fn, args, model_ema=None, model_swa=None,
                    lr_scheduler=None, timm_scheduler=True, schd_batch_update=False, 
                    output_dir='', amp_autocast=suppress, mixup_fn=None,
                    loss_scaler=None, writer_dict=None):
        if mixup_fn is not None and args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
            if args.prefetcher and loader.mixup_enabled:
                loader.mixup_enabled = False
            elif mixup_fn is not None:
                mixup_fn.mixup_enabled = False
    
        second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        batch_time_m = AverageMeter()
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        prec1_m = AverageMeter()
        f1_m = AverageMeter()
        acc_m = AverageMeter()
        
        softmax = nn.Softmax(dim=1)

        model.train()

        end = time.time()
        last_idx = len(loader) - 1
        num_updates = epoch * len(loader)
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            data_time_m.update(time.time() - end)
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
        
                if mixup_fn is not None:
                    input, lam = mixup_fn(input)
                    mixed_target = mixup_fn.gen_target(lam, target, input.device, args.num_classes)

            with amp_autocast():
                # 取第一个结果，output为list
                output = model(input)[0]

            if mixup_fn is not None:
                loss = loss_fn(output, mixed_target.cuda())
            else:
                loss = loss_fn(output, target)
            
            preds = torch.max(softmax(output), dim=1)[1].cpu().numpy()
            targets = target.cpu().detach().numpy()
            # 计算每个类的 f1 然后取平均
            f1_value = f1_score(targets, preds, average="macro") # 'micro', 'macro', 'samples', 'weighted'
            acc_value = accuracy_score(targets, preds)
            
            if not args.distributed:
                losses_m.update(loss.item(), input.size(0))
                f1_m.update(f1_value, input.size(0))
                acc_m.update(acc_value, input.size(0))
            else:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                mean_f1 = reduce_tensor(torch.tensor(f1_value).cuda(), args.world_size)
                
                losses_m.update(reduced_loss.item(), input.size(0))
                f1_m.update(mean_f1.item(), input.size(0))
                acc_m.update(acc_value, input.size(0))

            optimizer.zero_grad()
            if loss_scaler is not None:
                loss_scaler(
                    loss, optimizer,
                    clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                    parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                    create_graph=second_order)
            else:
                loss.backward(create_graph=second_order)
                if args.clip_grad is not None:
                    dispatch_clip_grad(
                        model_parameters(model, exclude_head='agc' in args.clip_mode),
                        value=args.clip_grad, mode=args.clip_mode)
                optimizer.step()

            if model_ema is not None: model_ema.update(model)

            torch.cuda.synchronize()
            num_updates += 1
            batch_time_m.update(time.time() - end)
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)
                      
                logging.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.6f}  '
                    'Acc: {top1.val:>7.4f} ({top1.avg:>7.4f}) '
                    'F1: {f1.val:>7.4f} ({f1.avg:>7.4f}) '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch, batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) / batch_time_m.val,
                        rate_avg=input.size(0) / batch_time_m.avg,
                        lr=lr, top1=acc_m, f1=f1_m,
                        data_time=data_time_m))
                
                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

            if args.local_rank == 0 and writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('scalar/train_loss', losses_m.val, global_steps)
                writer.add_scalar('scalar/train_prec1', prec1_m.val, global_steps)
                writer.add_scalar('learning_rate', lr, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

            if lr_scheduler is not None and schd_batch_update:
                if timm_scheduler:
                    lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)
                else:
                    lr_scheduler.step()

            end = time.time()
            # end for

        if lr_scheduler is not None and not schd_batch_update and not timm_scheduler:
            # swa_scheduler = lr_scheduler[1]
            # lr_scheduler = lr_scheduler[0]
            # if epoch > int(epoch * 0.25):
            #     model_swa.update_parameters(model)
            #     swa_scheduler.step()
            # else:
            lr_scheduler.step()

        if hasattr(optimizer, 'sync_lookahead'):
            optimizer.sync_lookahead()

        return OrderedDict([('loss', losses_m.avg)])

    def validate(self, model, loader, loss_fn, args, log_suffix='', output_dir='', amp_autocast=suppress,
                 writer_dict=None, show_feats=[], verbose=False, save_json=False):
        batch_time_m, losses_m, f1_m, acc_m = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        names = locals()
        for i in range(args.head_num):
            names["pred_idx{}".format(i)] = []
            names["truth_idx{}".format(i)] = []
            names["pred_prob{}".format(i)] = []
            names["Loss{}".format(i)] = AverageMeter()

        softmax = torch.nn.Softmax(dim=1)

        model.eval()

        end = time.time()
        last_idx = len(loader) - 1
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(loader):
                last_batch = batch_idx == last_idx
                if not args.prefetcher:
                    input = input.cuda()
                    target = target.cuda()
                
                with amp_autocast():
                    output = model(input)[0]

                loss = loss_fn(output, target)

                probs = softmax(output).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                # preds = torch.max(probs, dim=1)[1].cpu().numpy()
                targets = target.cpu().detach().numpy()

                names["pred_idx{}".format(i)].extend(preds)
                names["truth_idx{}".format(i)].extend(targets)
                names["pred_prob{}".format(i)].extend(output.cpu().numpy())
                names["Loss{}".format(i)].update(loss.item(), input.size(0))

                # 计算每个类的 f1 然后取平均
                f1_value = f1_score(targets, preds, average="macro") # 'micro', 'macro', 'samples', 'weighted'
                acc_value = accuracy_score(targets, preds)
                
                if args.distributed:
                    reduced_loss = reduce_tensor(loss.data, args.world_size)
                    f1_value = reduce_tensor(torch.tensor(f1_value).cuda(), args.world_size)
                else:
                    reduced_loss = loss.data
                    # mean_f1 = f1_value

                losses_m.update(reduced_loss.item(), input.size(0))
                f1_m.update(f1_value, input.size(0))
                acc_m.update(acc_value, input.size(0))
                
                torch.cuda.synchronize()

                batch_time_m.update(time.time() - end)
                end = time.time()
                if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                    log_name = 'Test' + log_suffix
                    logging.info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc: {top1.val:>7.4f} ({top1.avg:>7.4f}) '
                        'F1: {f1.val:>7.4f} ({f1.avg:>7.4f})'.format(
                            log_name, batch_idx, last_idx,
                            batch_time=batch_time_m, loss=losses_m,
                            top1=acc_m, f1=f1_m))

                    if args.save_images and output_dir:
                        torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'val-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

                if args.local_rank == 0 and writer_dict:
                    writer = writer_dict['writer']
                    global_steps = writer_dict['valid_global_steps']
                    writer.add_scalar('scalar/valid_loss', losses_m.val, global_steps)
                    writer.add_scalar('scalar/valid_f1', f1_m.val, global_steps)
                    writer.add_scalar('scalar/valid_prec1', acc_m.val, global_steps)
                    writer_dict['valid_global_steps'] = global_steps + 1

        mean_acc, mean_f1 = self.compute_metrics(args, names, verbose=verbose, return_weights=False)
        
        metrics = OrderedDict([('loss', losses_m.avg), ('acc1', mean_acc), ('mf1', mean_f1)])

        return metrics
    
    def compute_metrics(self, args, names, verbose=False, return_weights=False):
        self.weights = {}
        mean_f1, mean_acc = [], []
        for i in range(args.head_num):
            total_pred_idx = np.array(names["pred_idx{}".format(i)], dtype=np.uint8)
            total_truth_idx = np.array(names["truth_idx{}".format(i)], dtype=np.uint8)
    
            tf1 = f1_score(y_true=total_truth_idx, y_pred=total_pred_idx, average="macro")
            acc = accuracy_score(y_true=total_truth_idx, y_pred=total_pred_idx)
            
            target_names = self.classes_dict[self.classes_list[i]]
            labels = list(range(len(target_names)))
            out_dict = classification_report(y_pred=total_pred_idx,
                                             y_true=total_truth_idx,
                                             labels=labels,
                                             target_names=target_names,
                                             digits=4,
                                             zero_division=0,
                                             output_dict=return_weights)
            if (verbose and not return_weights): print(out_dict)
            if return_weights:
                f1_list = [out_dict[name]['f1-score'] for name in target_names]
                
                self.weights[self.classes_list[i]] = {}
                self.weights[self.classes_list[i]]["model_weight"] = tf1
                self.weights[self.classes_list[i]]["label_weight"] = f1_list

            mean_f1.append(tf1)
            mean_acc.append(acc)
            logging.info('Class {:13}: Loss {:.4f} Accuracy:{:.4f} F1-score {:.4f}'.format(
                self.classes_list[i], names["Loss{}".format(i)].avg, np.mean(mean_acc), np.mean(mean_f1)))            

        return np.mean(mean_acc), np.mean(mean_f1)