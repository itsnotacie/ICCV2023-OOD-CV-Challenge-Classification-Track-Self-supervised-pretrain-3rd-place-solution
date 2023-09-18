from .cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, \
                            LabelSmoothingSigmoidCrossEntropy, CrossEntropyLossLogits, CrossEntropyLossOneHot, \
                            FocalLossWithWeight
from .jsd import JsdCrossEntropy
from .focal_loss import FocalLoss
from .asl_loss import AsymmetricLoss, AsymmetricLossOptimized, ASLSingleLabel
from .sce_loss import SCELoss
from .BalancedSoftmaxLoss import BalancedSoftmax