import torch
import torch.nn as nn
import torch.nn.functional as F
from .AF import AF
from .MNet import MNet


class HP(nn.Module):

    def __init__(self, num_classes=26, att_out=False):
        super(HP, self).__init__()
        self.att_out = att_out
        with torch.no_grad():
            self.MNet = MNet(feat_out=True)
            self.AF1 = AF(att_out=True, feat_out=True, af_name="AF1")
            self.AF2 = AF(att_out=True, feat_out=True, af_name="AF2")
            self.AF3 = AF(att_out=True, feat_out=True, af_name="AF3")

        self.final_fc = nn.Linear(512 * 73, num_classes)

    def forward_features(self, x):
        _, _, _, feat0 = self.MNet(x)

        feat1, att1 = self.AF1(x)
        feat2, att2 = self.AF2(x)
        feat3, att3 = self.AF3(x)

        ret = torch.cat((feat0, feat1, feat2, feat3), dim=1)
        # 9 x 9 x (512x(24x3 + 1))
        return att1, att2, att3, ret
    
    def forward(self, x):
        att1, att2, att3, ret = self.forward_features(x)
        
        ret = F.avg_pool2d(ret, kernel_size=9, stride=1)

        # 1 x 1 x (512 x 73)

        ret = F.dropout(ret, training=self.training)
        # 1 x 1 x (512 x 73)
        ret = ret.view(ret.size(0), -1)
        # 512 x 73

        ret = self.final_fc(ret)
        # (num_classes)
        if self.att_out:
            return att1, att2, att3, ret
        else:
            return ret