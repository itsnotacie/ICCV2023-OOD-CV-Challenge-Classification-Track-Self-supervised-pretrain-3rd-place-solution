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
    
    def train_epoch(self, epoch, model, loader, optimizer, loss_fns, args, model_ema=None,
                    lr_scheduler=None, output_dir='', amp_autocast=suppress, mixup_fn=None,
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
        
        names = locals()
        for i in range(args.head_num):
            names["Loss{}".format(i)] = AverageMeter()

        softmax = nn.Softmax(dim=1)
        
        if args.multi_color_loss:
            from src.loss import CrossEntropyLossLogits
            loss_weight = torch.tensor(args.weights[args.classes_list[i]], dtype=torch.half).cuda()
            soft_loss_fn = CrossEntropyLossLogits(weight=loss_weight).cuda()
                    
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
        
                if mixup_fn is not None: input, lam = mixup_fn(input)

            tmp_f1 = []
            total_loss = torch.zeros(1, dtype=torch.float).cuda()
            with amp_autocast():
                output = model(input)
            
            for i, out in enumerate(output):
                if self.classes_num[i] != 11:
                    tmp_target = target[:, i]
                else:
                    if args.feats[i] == 7 or args.feats[i] == 0:
                        target_prob = target[:, i:(i+11)]
                    else:
                        target_prob = target[:, (i+10):]
                    tmp_target = torch.argmax(target_prob, dim=-1)

                if mixup_fn is not None:
                    mixed_target = mixup_fn.gen_target(lam, tmp_target, input.device, self.classes_num[i])
                                    
                if mixup_fn is not None and args.soft_ce_loss:
                    if i < 7: target_prob = mixed_target
                    head_loss = loss_fns[i](out, target_prob.cuda())
                elif mixup_fn is not None and not args.soft_ce_loss:
                    head_loss = loss_fns[i](out, mixed_target.cuda())
                elif mixup_fn is None and args.soft_ce_loss and i >= 7:
                    head_loss = loss_fns[i](out, target_prob.cuda())
                else:
                    head_loss = loss_fns[i](out, tmp_target.long().cuda())
                
                if args.multi_color_loss:
                    soft_loss = soft_loss_fn(out, target_prob.cuda())
                    head_loss = head_loss * 0.5 + 0.5 * soft_loss

                # if i in [0, 2, 5]:
                    # head_loss *= 2
                
                names["Loss{}".format(i)].update(head_loss.item(), input.size(0))
                total_loss += head_loss

                preds = torch.max(softmax(out), dim=1)[1].cpu().numpy()
                targets = tmp_target.cpu().detach().numpy()
                # 计算每个类的 f1 然后取平均
                tmp_f1.append(f1_score(targets, preds, average="macro")) # 'micro', 'macro', 'samples', 'weighted'
            
            loss = torch.div(total_loss, args.head_num)

            if not args.distributed:
                losses_m.update(loss.item(), input.size(0))
                f1_m.update(np.mean(tmp_f1), input.size(0))
            else:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                mean_f1 = reduce_tensor(torch.tensor(np.mean(tmp_f1)).cuda(), args.world_size)
                
                losses_m.update(reduced_loss.item(), input.size(0))
                f1_m.update(mean_f1.item(), input.size(0))

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

            if model_ema is not None:
                model_ema.update(model)

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
                    'F1: {f1.val:>7.4f} ({f1.avg:>7.4f}) '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch, batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) / batch_time_m.val,
                        rate_avg=input.size(0) / batch_time_m.avg,
                        lr=lr, f1=f1_m,
                        data_time=data_time_m))
                
                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

            if args.local_rank == 0 and last_batch:
                tr_log_str = ""                 
                for i in range(args.head_num):
                    tr_log_str += "{}:{:.4f} ".format(self.classes_list[i], names["Loss{}".format(i)].avg)
                logging.info(tr_log_str)

            if args.local_rank == 0 and writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('scalar/train_loss', losses_m.val, global_steps)
                writer.add_scalar('scalar/train_prec1', prec1_m.val, global_steps)
                writer.add_scalar('learning_rate', lr, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

            if lr_scheduler is not None:
                lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

            end = time.time()
            # end for

        if hasattr(optimizer, 'sync_lookahead'):
            optimizer.sync_lookahead()

        return OrderedDict([('loss', losses_m.avg)])

    def validate(self, model, loader, loss_fn, args, log_suffix='', output_dir='', amp_autocast=suppress,
                 writer_dict=None, show_feats=[], verbose=False, save_json=False):
        batch_time_m, losses_m, f1_m = AverageMeter(), AverageMeter(), AverageMeter()

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
                
                tmp_f1 = []
                total_loss = torch.zeros(1, dtype=torch.float32).cuda()
                with amp_autocast():
                    output = model(input)

                for i, out in enumerate(output):
                    if self.classes_num[i] != 11:
                        tmp_target = target[:, i]
                    else:
                        if args.feats[i] == 7 or args.feats[i] == 0:
                            target_prob = target[:, i:(i+11)]
                        else:
                            target_prob = target[:, (i+10):]
                        tmp_target = torch.argmax(target_prob, dim=-1)

                    loss = loss_fn(out, tmp_target.long().cuda())
                    total_loss += loss

                    preds = torch.max(softmax(out), dim=1)[1].cpu().numpy()
                    targets = tmp_target.cpu().detach().numpy()

                    names["pred_idx{}".format(i)].extend(preds)
                    names["truth_idx{}".format(i)].extend(targets)
                    names["pred_prob{}".format(i)].extend(out.cpu().numpy())
                    names["Loss{}".format(i)].update(loss.item(), input.size(0))

                    # 计算每个类的 f1 然后取平均
                    tmp_f1.append(f1_score(targets, preds, average="macro")) # 'micro', 'macro', 'samples', 'weighted'

                loss = torch.div(total_loss, args.head_num)

                if args.distributed:
                    reduced_loss = reduce_tensor(loss.data, args.world_size)
                    mean_f1 = reduce_tensor(torch.tensor(np.mean(tmp_f1)).cuda(), args.world_size)
                else:
                    reduced_loss = loss.data
                    mean_f1 = np.mean(tmp_f1)

                losses_m.update(reduced_loss.item(), input.size(0))
                f1_m.update(mean_f1, input.size(0))
                
                torch.cuda.synchronize()

                batch_time_m.update(time.time() - end)
                end = time.time()
                if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                    log_name = 'Test' + log_suffix
                    logging.info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'F1: {top1.val:>7.4f} ({top1.avg:>7.4f})'.format(
                            log_name, batch_idx, last_idx,
                            batch_time=batch_time_m, loss=losses_m,
                            top1=f1_m))

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
                    writer_dict['valid_global_steps'] = global_steps + 1

        mean_style_f1, mean_color_f1, mean_f1, mean_igf1 = self.compute_metrics(args, names, show_feats=show_feats, 
                                                                     verbose=verbose)
        
        self.output_dict = {}
        if save_json:
            for i in range(args.head_num):
                self.output_dict[self.classes_list[i]] = names["pred_prob{}".format(i)]
            
        metrics = OrderedDict([('loss', losses_m.avg), ('prec1', f1_m.avg),
                               ('sf1', mean_style_f1), ('cf1', mean_color_f1), 
                               ('mf1', mean_f1), ('mif1', mean_igf1)])

        return metrics
    
    def compute_metrics(self, args, names, show_feats=[], verbose=False):
        self.weights = {}
        mean_f1, mean_style_f1, mean_color_f1 = [], [], []
        mean_ig_f1, mean_ig_style_f1 = [], []
        for i in range(args.head_num):
            total_pred_idx = np.array(names["pred_idx{}".format(i)], dtype=np.uint8)
            total_truth_idx = np.array(names["truth_idx{}".format(i)], dtype=np.uint8)
    
            if i == 0: 
                no_sleeve_index = np.where(total_truth_idx == 1)
                ig_total_pred_idx = np.delete(deepcopy(total_pred_idx), no_sleeve_index)
                ig_total_truth_idx = np.delete(deepcopy(total_truth_idx), no_sleeve_index)

                ig_tf1 = f1_score(y_true=ig_total_truth_idx, y_pred=ig_total_pred_idx, average="macro")
                
            tf1 = f1_score(y_true=total_truth_idx, y_pred=total_pred_idx, average="macro")

            target_names = self.classes_dict[self.classes_list[i]]
            labels = list(range(len(target_names)))
            out_dict = classification_report(y_pred=total_pred_idx,
                                                y_true=total_truth_idx,
                                                labels=labels,
                                                target_names=target_names,
                                                digits=4,
                                                zero_division=0,
                                                output_dict=True)
            if (verbose and i in show_feats): print(out_dict)

            f1_list = [out_dict[name]['f1-score'] for name in target_names]
            
            self.weights[self.classes_list[i]] = {}
            self.weights[self.classes_list[i]]["model_weight"] = tf1
            self.weights[self.classes_list[i]]["label_weight"] = f1_list

            if i == 0:
                mean_ig_style_f1.append(ig_tf1)
                mean_ig_f1.append(ig_tf1)
            else:
                mean_ig_style_f1.append(tf1)
                mean_ig_f1.append(tf1) 

            mean_f1.append(tf1)
            if i >= 7:
                mean_color_f1.append(tf1)
                logging.info('Class {:13}: Loss {:.4f} F1-score {:.4f} Mean Color  F1 {:.4f} Mean F1-score {:.4f} Mean IgF1-score {:.4f}'.format(
                    self.classes_list[i], names["Loss{}".format(i)].avg, tf1, np.mean(mean_color_f1), np.mean(mean_f1), np.mean(mean_ig_f1)))            
            else:
                mean_style_f1.append(tf1)
                if i == 0:
                    logging.info('Class {:13}: Loss {:.4f} F1-score {:.4f} IgF1-score     {:.4f} Mean F1-score {:.4f} Mean IgF1-score {:.4f}'.format(
                        self.classes_list[i], names["Loss{}".format(i)].avg, tf1, ig_tf1, np.mean(mean_f1), np.mean(mean_ig_f1)))
                else:
                    logging.info('Class {:13}: Loss {:.4f} F1-score {:.4f} Mean SF1-score {:.4f} Mean F1-score {:.4f} Mean IgF1-score {:.4f}'.format(
                        self.classes_list[i], names["Loss{}".format(i)].avg, tf1, np.mean(mean_style_f1), np.mean(mean_f1), np.mean(mean_ig_f1)))
        
        return np.mean(mean_style_f1), np.mean(mean_color_f1), np.mean(mean_f1), np.mean(mean_ig_f1)