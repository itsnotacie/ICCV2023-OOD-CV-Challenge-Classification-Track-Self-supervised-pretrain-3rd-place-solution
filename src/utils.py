import os
import operator
import logging
import json
import time
import torch
import numpy as np
import random
from collections import OrderedDict
from timm.utils import unwrap_model, get_state_dict

_logger = logging.getLogger(__name__)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # 在所有 GPU 上设定相同的 seed 种子
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        # 使用 cudnn 进行模型自动寻优，改成 TRUE 不可复现，FALSE 可复现
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = True
        
def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                if k.startswith('module'): 
                    name = k[7:]
                elif k.startswith('model'):
                    name = k[6:]
                else:
                    name = k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                # strip `module.` prefix
                if k.startswith('module'): 
                    name = k[7:]
                elif k.startswith('model'):
                    name = k[6:]
                elif k.startswith('classifiers'):
                    continue
                else:
                    name = k
                new_state_dict[name] = v
            state_dict = new_state_dict
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    state_dict = load_state_dict(checkpoint_path, use_ema)
    model.load_state_dict(state_dict, strict=strict)


class SimpleCheckpointSaver:
    def __init__(
            self,
            model,
            model_ema=None,
            amp_scaler=None,
            checkpoint_dir='',
            unwrap_fn=unwrap_model):

        # objects to save state_dicts of
        self.model = model
        self.model_ema = model_ema
        self.amp_scaler = amp_scaler
        
        # state
        self.checkpoint_files = []  # (filename, metric) tuples in order of decreasing betterness
        self.best_epoch = {"loss": np.inf, "mif1": 0, "sf1": 0, "cf1": 0, "mf1": 0, "acc1": 0}
        self.best_metric = {"loss": np.inf, "mif1": 0, "sf1": 0, "cf1": 0, "mf1": 0, "acc1": 0}

        # # config
        self.checkpoint_dir = checkpoint_dir
        self.extension = '.pth'
        self.unwrap_fn = unwrap_fn

    def save_checkpoint(self, epoch, save_prefix, metric=None, metric_name="", decreasing=True):
        cmp = operator.lt if decreasing else operator.gt  # True if lhs better than rhs

        if cmp(metric, self.best_metric[metric_name]):
            filename = '-'.join([save_prefix, str(epoch)]) + self.extension
            save_path = os.path.join(self.checkpoint_dir, filename)

            if epoch >= 1:
                old_filename = "{:.4f}-{}-{}".format(self.best_metric[metric_name], metric_name,
                                                     self.best_epoch[metric_name]) + self.extension
                old_filepath = os.path.join(self.checkpoint_dir, old_filename)
                self.checkpoint_files.remove(old_filepath)
                os.remove(old_filepath)
            self.checkpoint_files.append(save_path)

            self.best_epoch[metric_name] = epoch
            self.best_metric[metric_name] = metric
            # torch.save(get_state_dict(self.model.float()), save_path)
            self._save(save_path, epoch, metric)

            logging.info('Current checkpoints:{}'.format(save_path))

        return self.best_metric, self.best_epoch

    def _save(self, save_path, epoch, metric=None):
        save_state = {
            # 'epoch': epoch,
            # 'arch': type(self.model).__name__.lower(),
            'state_dict': get_state_dict(self.model, self.unwrap_fn),
            # 'optimizer': self.optimizer.state_dict(),
            # 'version': 2,  # version < 2 increments epoch before save
        }
        # if self.args is not None:
            # save_state['arch'] = self.args.model
            # save_state['args'] = self.args
        # if self.amp_scaler is not None:
            # save_state[self.amp_scaler.state_dict_key] = self.amp_scaler.state_dict()
        if self.model_ema is not None:
            save_state['state_dict_ema'] = get_state_dict(self.model_ema, self.unwrap_fn)
        if metric is not None:
            save_state['metric'] = metric
        torch.save(save_state, save_path)
    

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return round(float(obj), 5)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, time):
            return obj.__str__()
        else:
            return super(MyEncoder, self).default(obj)
        

class PrefetchLoader:
    def __init__(self, loader, fp16=True):
        self.loader = loader
        self.fp16 = fp16

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if self.fp16:
                    next_input = next_input.half()
                else:
                    next_input = next_input.float()

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset
    

class JsonSaver(object):
    def __init__(self, root_path, metric, tta_num=1):
        self.pred_dict_path = os.path.join(root_path, "predictions_dict_{}_{}.json".format(metric, tta_num))
        self.weight_path = os.path.join(root_path, "fusion_weights_tta_{}_{}.json".format(metric, tta_num))

        self.predictions_dict, self.weights = self.load_json()
    
    @staticmethod
    def _load_file(path):
        with open(path, 'r') as json_file:
            json_data = json.load(json_file)
        return json_data
        
    def load_json(self):
        if os.path.exists(self.pred_dict_path):
            predictions_dict = self._load_file(self.pred_dict_path)
        else:
            predictions_dict = {}
        
        if os.path.exists(self.pred_dict_path):
            weights = self._load_file(self.weight_path)
        else:
            weights = {}
        
        return predictions_dict, weights
    
    def save_json(self, model_name, output_dict, weight):
        with open(self.pred_dict_path, 'w', encoding="utf-8") as f:
            self.predictions_dict[model_name] = output_dict
            json.dump(self.predictions_dict, f, cls=MyEncoder, indent=2)
        
        with open(self.weight_path, 'w', encoding="utf-8") as f:
            self.weights[model_name] = weight
            json.dump(self.weights, f, cls=MyEncoder, indent=2)    
        