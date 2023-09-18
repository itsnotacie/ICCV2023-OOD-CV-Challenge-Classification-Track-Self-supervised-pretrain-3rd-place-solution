import numpy as np
from itertools import combinations
import torch
import os
from collections import Counter
from torch.nn import functional as F
import json
"""
A:  avg, 均值
B:  model_weight, 按模型acc加权
C:  label_weight, 按类别acc加权
D:  ada_boost, Adaboost方式
P:  prediction_weight, 按预测值本身加权
M:  Max, 
MM: Max with model weight
ML: Max with label weight
"""
POLICIES = ['A', 'B', 'C', 'D', 'E', 'P', 'M', 'MM', 'ML']

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax

def parse_prediction(predictions, top=3, return_with_prob=False):
    result = np.argsort(predictions)
    result = result[:, -top:][:, ::-1]
    if return_with_prob:
        predictions = softmax(np.array(predictions))
        return [[(j, predictions[i][j]) for j in r] for i, r in enumerate(result)]
    else:
        return list(map(lambda x: x.tolist(), result))

def all_combines(data):
    result = []
    for i in range(len(data)):
        combines = list(combinations(data, i + 1))
        result.extend(combines)
    return result

class IntegratedPredictor(object):
    def __init__(self, predictors, predictions, args, policies=POLICIES, standard=False, all_combine=False,):
        self.predictors = predictors
        self.args = args
        self.predictions_dict = predictions[0]
        self.policies = policies
        self.standard = standard

        self.model_weight = predictions[1]
        self.label_weight = predictions[2]
        self.index2combine_name = {}
        self.index2policy = {}
        self.combine_names = []

        self.names = [predictor for predictor in predictors]
        self.predictors_list = all_combines(predictors) if all_combine else [predictors]
        self._parse_predictors_name()

    def _parse_predictors_name(self):
        index = 0
        for predictors in self.predictors_list:
            combine_name = self.get_name_by_predictors(predictors)
            self.combine_names.append(combine_name)
            if len(predictors) == 1:
                self.index2combine_name[index] = combine_name
                self.index2policy[index] = ''
                index += 1
            else:
                for policy in self.policies:
                    self.index2combine_name[index] = combine_name
                    self.index2policy[index] = policy
                    index += 1

    @staticmethod
    def get_name_by_predictors(predictors):
        return '%s' % ('#'.join([predictor for predictor in predictors]))

    def fusion_prediction(self, top=3, return_with_prob=False):
        # get prediction of every predictor
        predictions_dict = self.predictions_dict
        # integrated predictions
        final_predictions_dict = self.integrated_predictions(predictions_dict)
        # parse predictions
        top_predictions = []
        for combine_name in self.combine_names:
            item_predictions = final_predictions_dict[combine_name]
            top_predictions.extend([parse_prediction(item_prediction, top, return_with_prob)
                                    for item_prediction in item_predictions])
        return top_predictions

    def integrated_predictions(self, predictions_dict):
        integrated_predictions_dict = {}
        index = 0
        for predictors, combine_name in zip(self.predictors_list, self.combine_names):
            if len(predictors) == 1:
                integrated_predictions_dict[combine_name] = [predictions_dict[predictors[0]]]
                self.index2combine_name[index] = combine_name
                self.index2policy[index] = ''
                index += 1
            else:
                predictions = [predictions_dict[predictor] for predictor in predictors]
                predictions = np.array(predictions)
                result = []
                for policy in self.policies:
                    result.append(self._perform_integrated(predictors, predictions, policy))
                    self.index2combine_name[index] = combine_name
                    self.index2policy[index] = policy
                    index += 1
                integrated_predictions_dict[combine_name] = result
        return integrated_predictions_dict

    def _perform_integrated(self, predictors, predictions, policy):
        if policy == 'A':
            result = np.mean(predictions, axis=0)
        elif policy == 'B':
            assert self.model_weight, 'The weights is None.'
            c_ns = [self.model_weight[predictor] for predictor in predictors]
            assert len(c_ns) == len(predictions), \
                'The weights length %d is not equal with %d' % (len(c_ns), len(predictions))
            result = np.sum(c_n * p_nj for c_n, p_nj in zip(c_ns, predictions))
        elif policy == 'C':
            c_njs = [self.label_weight[predictor] for predictor in predictors]
            # for c_nj, p_nj in zip(c_njs, predictions):
            #     a = c_nj * p_nj
            result = np.sum(c_nj * p_nj for c_nj, p_nj in zip(c_njs, predictions))
        elif policy == 'D':
            c_ns = [self.model_weight[predictor] for predictor in predictors]
            c_njs = [self.label_weight[predictor] for predictor in predictors]
            alphas = [np.log(c_n / (1 - c_n + 1e-6)) / 2 for c_n in c_ns]
            result = np.sum(c_nj * p_nj * alpha for alpha, c_nj, p_nj in zip(alphas, c_njs, predictions))
        elif policy == 'E':
            c_ns = [self.model_weight[predictor] for predictor in predictors]
            c_njs = [self.label_weight[predictor] for predictor in predictors]
            result = np.sum(c_nj * p_nj * c_n for c_n, c_nj, p_nj in zip(c_ns, c_njs, predictions))
        elif policy == 'P':
            predictions_all = np.sum(predictions, axis=0)
            predictions_weight = predictions / predictions_all
            result = np.sum(predictions_weight * predictions, axis=0)
        elif policy == 'M':
            result = np.max(predictions, axis=0)
        elif policy == 'MM':
            assert self.model_weight, 'The weights is None.'
            model_weight = [self.model_weight[predictor] for predictor in predictors]
            # assert len(model_weight) == len(dcm_list), \
            #     'The weights length %d is not equal with %d' % (len(self.model_weight), len(dcm_list))
            result = np.max([d * w for d, w in zip(predictions, model_weight)], axis=0)
        elif policy == 'ML':
            label_weight = [self.label_weight[predictor] for predictor in predictors]
            result = np.max([c_nj * p_nj for c_nj, p_nj in zip(label_weight, predictions)], axis=0)
        else:
            raise 'Not support for policy named "%s".' % policy
        if self.standard:
            denominators = np.sum(result, axis=1)
            result = [r / denominator for r, denominator in zip(result, denominators)]

        return result

# if __name__ == '__main__':
#     torch.backends.cudnn.benchmark = True
#
#     parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
#     parser.add_argument('--data', metavar='DIR', default='/home/Data/US_ECG/US_ECG_total/valid2',
#                         help='path to dataset')
#     parser.add_argument('--label', default='/home/Data/US_ECG/Fuwai_US', metavar='DIR',
#                         help='path to label')
#     # sk_resnet18 seresnext26_32x4d seres2next26_8cx4wx4 efficientnet_b3 seresnext50_32x4d
#     parser.add_argument('--model-list', '-m', metavar='MODEL', default=['efficientnet_b2', 'seresnext50_32x4d'],
#                         help='model architecture (default: dpn92)')
#     parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
#                         help='number of data loading workers (default: 2)')
#     parser.add_argument('-b', '--batch-size', default=64, type=int,
#                         metavar='N', help='mini-batch size (default: 256)')
#     parser.add_argument('--img-size', default=416, type=int,
#                         metavar='N', help='Input image dimension, uses model default if empty')
#     parser.add_argument('--crop-pct', default=None, type=float,
#                         metavar='N', help='Input image center crop pct')
#     parser.add_argument('--num-classes', type=int, default=26,
#                         help='Number classes in dataset')
#     parser.add_argument('--log-freq', default=50, type=int,
#                         metavar='N', help='batch logging frequency (default: 10)')
#     parser.add_argument('--checkpoint-list',
#                         default=[
#                             '/home/MedicalImage/User/guofeng/data/US_ECG/output/20190923-181044-efficientnet_b2-416',
#                             '/home/MedicalImage/User/guofeng/data/US_ECG/output/20190916-104514-seresnext50_32x4d-416'],
#                         type=str, metavar='PATH',
#                         help='path to latest checkpoint (default: none)')
#     parser.add_argument('--no-prefetcher', action='store_true', default=False,
#                         help='disable fast prefetcher')
#     # Device options
#     parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
#     args = parser.parse_args()
#     args.prefetcher = not args.no_prefetcher
#
#     efficientnet_b2 = create_model('efficientnet_b2',num_classes=26,in_chans=3)
#     load_checkpoint(efficientnet_b2, '/home/MedicalImage/User/guofeng/data/US_ECG/output/20190923-181044-efficientnet_b2-416/model_best.pth.tar')
#     param_count = sum([m.numel() for m in efficientnet_b2.parameters()])
#     print('Model %s created, param count: %d' % ('efficientnet_b2', param_count))
#     efficientnet_b2 = efficientnet_b2.cuda()
#     efficientnet_b2.eval()
#
#     loader = create_loader(
#         Dataset(args.data, args.label),
#         args=args,
#         input_size=args.img_size,
#         batch_size=args.batch_size,
#         use_prefetcher=args.prefetcher,
#         num_workers=args.workers,
#         crop_pct=1.0)
#     INTEGRATED_POLICY = ['A', 'B', 'C', 'D', 'E', 'P', 'M', 'MM', 'ML']
#     # integrated predictor
#     predictor = IntegratedPredictor([efficientnet_b2,seresnext50_32x4d,efficientnet_b4_synbn],loader,
#                                     policies=INTEGRATED_POLICY,all_combine=True)
#     predictions = predictor.fusion_prediction(top=3,return_with_prob=False)
#     print(len(predictions),len(predictions[0]))
#     path_json_dumps = dump_json(predictor)
#     results = []
#     for index, prediction in enumerate(predictions):
#         results[index].extend([item_handler(image_ids[i], prediction[i]) for i in range(end - start)])
#     print(predictions)