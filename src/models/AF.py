from .MNet import *
import pdb


class AF(nn.Module):
    def __init__(self, num_classes=26, att_out=False, feat_out=False, af_name='AF2'):
        super(AF, self).__init__()
        with torch.no_grad():
            self.MNet = MNet(feat_out=True)

        self.att_out = att_out
        self.feat_out = feat_out

        self.att_channel_L = 8

        self.af_idx = int(af_name[2])
        if self.af_idx == 1:
            self.Att = BasicConv2d(256, self.att_channel_L, kernel_size=1)
        elif self.af_idx == 2:
            self.Att = BasicConv2d(502, self.att_channel_L, kernel_size=1)
        elif self.af_idx == 3:
            self.Att = BasicConv2d(512, self.att_channel_L, kernel_size=1)

        self.att_branch_1 = nn.Sequential(InceptBlock1(), InceptBlock2(), InceptBlock3())
        self.att_branch_2 = nn.Sequential(InceptBlock2(), InceptBlock3())
        self.att_branch_3 = InceptBlock3()
        # self.patch = nn.ReflectionPad2d((0, 0, 0, -1))
        self.final_fc = nn.Linear(512 * 3 * self.att_channel_L, num_classes)

    def forward_features(self, x):
        feature_out0, feature_out1, feature_out2, feature_out3 = self.MNet(x)
        # feature_out0: torch.Size([batch_size, 96, 36, 36])
        # feature_out1: torch.Size([batch_size, 256, 18, 18])
        # feature_out2: torch.Size([batch_size, 502, 9, 9])
        # feature_out3: torch.Size([batch_size, 512, 9, 9])

        if self.af_idx == 1:
            att = self.Att(feature_out1)
            att1 = F.interpolate(att, scale_factor=2)
            att2 = att
            att3 = F.avg_pool2d(att, kernel_size=2, stride=2)
        elif self.af_idx == 2:
            att = self.Att(feature_out2)
            att2 = F.interpolate(att, scale_factor=2)
            att1 = F.interpolate(att2, scale_factor=2)
            att3 = att
        elif self.af_idx == 3:
            att = self.Att(feature_out3)
            att2 = F.interpolate(att, scale_factor=2)
            att1 = F.interpolate(att2, scale_factor=2)
            att3 = att

        # attention branch 1
        att1_w, att1_h = att1.size()[2], att1.size()[3]
        for i in range(self.att_channel_L):
            temp = att1[:, i].clone()
            temp = temp.view(-1, 1, att1_w, att1_h).expand(-1, 96, att1_w, att1_h)
            att_feature_out0 = feature_out0 * temp
            att_feature_out3 = self.att_branch_1(att_feature_out0)
            if i == 0:
                ret = att_feature_out3
            else:
                ret = torch.cat((ret, att_feature_out3), dim=1)

        # attention branch 2
        att2_w, att2_h = att2.size()[2], att2.size()[3]
        for i in range(self.att_channel_L):
            temp = att2[:, i].clone()
            temp = temp.view(-1, 1, att2_w, att2_h).expand(-1, 256, att2_w, att2_h)
            att_feature_out1 = feature_out1 * temp
            att_feature_out3 = self.att_branch_2(att_feature_out1)
            # 8 x 8 x 2048
            ret = torch.cat((ret, att_feature_out3), dim=1)

        # attention branch 3

        att3_w, att3_h = att3.size()[2], att3.size()[3]
        for i in range(self.att_channel_L):
            temp = att3[:, i].clone()
            temp = temp.view(-1, 1, att3_w, att3_h).expand(-1, 502, att3_w, att3_h)
            att_feature_out2 = feature_out2 * temp
            att_feature_out3 = self.att_branch_3(att_feature_out2)
            ret = torch.cat((ret, att_feature_out3), dim=1)

        # final feature size: [batch_size, 512 x 3 x L, 19, 33]
        
        return ret, att

    def forward(self, x):
        ret, att = self.forward_features(x)

        x = F.avg_pool2d(ret, kernel_size=9, stride=1)
        # 512*3*8 x1 x1
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 512*3*8
        x = x.view(x.size(0), -1)
        # 512*3*8
        pred_class = self.final_fc(x)
        # num_classes =26

        if self.att_out:
            if self.feat_out:
                # torch.cuda.synchronize()
                return ret, att.cpu()  # todo: compress attention for transfer
            else:
                return pred_class, att.cpu()
        else:
            if self.feat_out:
                return ret
            else:
                return pred_class

    def load_att_brach_weight(self):
        # att_brach_1
        branch1_incept1 = {'0.' + k: v for k, v in self.MNet.incept_block_1.state_dict().items()}
        branch1_incept2 = {'1.' + k: v for k, v in self.MNet.incept_block_2.state_dict().items()}
        branch1_incept3 = {'2.' + k: v for k, v in self.MNet.incept_block_3.state_dict().items()}

        branch1_incept1.update(branch1_incept2)
        branch1_incept1.update(branch1_incept3)

        self.att_branch_1.load_state_dict(branch1_incept1)

        # att_branch_2
        branch2_incept2 = {'0.' + k: v for k, v in self.MNet.incept_block_2.state_dict().items()}
        branch2_incept3 = {'1.' + k: v for k, v in self.MNet.incept_block_3.state_dict().items()}

        branch2_incept2.update(branch2_incept3)

        self.att_branch_2.load_state_dict(branch2_incept2)

        # att_branch_3
        self.att_branch_3.load_state_dict(self.MNet.incept_block_3.state_dict())
