import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        # 在指定的dim上，根据index指定的下标，选择元素重组成一个新的tensor，最后输出的out与index的size是一样的
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class LabelSmoothingSigmoidCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingSigmoidCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.sigmoid(x)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


def labelSmooth(one_hot, label_smooth):
    return one_hot*(1-label_smooth)+label_smooth/one_hot.shape[1]


# class CrossEntropyLossOneHot(nn.Module):
#     def __init__(self):
#         super(CrossEntropyLossOneHot, self).__init__()
#         self.log_softmax = nn.LogSoftmax(dim=-1)

#     def forward(self, preds, labels):
#         return torch.mean(torch.sum(-labels * self.log_softmax(preds), -1))


class CrossEntropyLossLogits(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight
        self.epsilon = 1e-7

    def forward(self, x, target_logits):
        # equal below two lines
        y_softmax = F.softmax(x, 1)
        y_softmax = torch.clamp(y_softmax, self.epsilon, 1.0-self.epsilon)  # avoid nan
        y_softmaxlog = torch.log(y_softmax)

        # original CE loss
        loss = -target_logits * y_softmaxlog

        if self.weight is not None:
            loss *= self.weight
        loss = torch.mean(torch.sum(loss, -1))
        return loss


class CrossEntropyLossOneHot(nn.Module):
    def __init__(self, smoothing=0, weight=None):
        super().__init__()
        self.weight = weight
        self.smoothing = smoothing
        self.epsilon = 1e-7

    def forward(self, x, y):
        one_hot_label = F.one_hot(y, x.shape[1])
        if self.smoothing:
            one_hot_label = labelSmooth(one_hot_label, self.smoothing)

        # equal below two lines
        y_softmax = F.softmax(x, 1)
        y_softmax = torch.clamp(y_softmax, self.epsilon, 1.0-self.epsilon)  # avoid nan
        y_softmaxlog = torch.log(y_softmax)

        # original CE loss
        loss = -one_hot_label * y_softmaxlog
        if self.weight is not None:
            loss *= self.weight
        loss = torch.mean(torch.sum(loss, -1))
        return loss


class FocalLossWithWeight(nn.Module):
    def __init__(self, label_smooth=0, alpha=0.25, gamma=2, weight=None, device='cpu'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight  # means alpha
        self.epsilon = 1e-7
        self.label_smooth = label_smooth
        self.device = device

    def forward(self, x, y, sample_weights=0, sample_weight_img_names=None):

        if len(y.shape) == 1:
            one_hot_label = F.one_hot(y, x.shape[1])

            if self.label_smooth:
                one_hot_label = labelSmooth(one_hot_label, self.label_smooth)

            if sample_weights > 0 and sample_weights is not None:
                weigths = [
                    sample_weights if 'yxboard' in img_name else 1 for img_name in sample_weight_img_names]
                weigths = torch.DoubleTensor(weigths).reshape((len(weigths), 1)).to(self.device)
                one_hot_label = one_hot_label*weigths

        else:
            one_hot_label = y

        # equal below two lines
        y_softmax = F.softmax(x, 1)
        y_softmax = torch.clamp(y_softmax, self.epsilon, 1.0-self.epsilon)  # avoid nan
        y_softmaxlog = torch.log(y_softmax)

        # original CE loss
        loss = -one_hot_label * y_softmaxlog
        # loss = 1 * torch.abs(one_hot_label-y_softmax)#my new CE..ok its L1...

        # gamma
        loss = loss*(self.alpha * (torch.abs(one_hot_label-y_softmax))**self.gamma)
        # loss = logpt * self.alpha * torch.pow((1 - y_softmax), self.gamma)

        # alpha
        if self.weight is not None:
            loss *= self.weight

        loss = torch.mean(torch.sum(loss, -1))
        return loss
