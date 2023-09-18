import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models import create_model
from timm.models import load_checkpoint as load_checkpointv1
from timm.models.layers import SelectAdaptivePool2d, ClassifierHead
from src.models import *
from src.utils import load_checkpoint

class MutiOutputsModel(nn.Module):
    def __init__(self, args, drop_rate=0.5, training=True):
        super(MutiOutputsModel, self).__init__()
                
        self.training = training
        self.drop_rate = drop_rate
        self.model_name = args.model
        self.classes_num = [3, 3, 4, 3, 3, 4, 4, 11, 11]        
        self.classes_list = ["upperLength", "clothesStyles", "hairStyles", "lowerLength", 
                             "lowerStyles", "shoesStyles", "towards", "upperColors", "lowerColors"]
        self.classes_dict = {"upperLength": ["LongSleeve", "NoSleeve", "ShortSleeve"],
                             "clothesStyles": ["Solidcolor", "lattice", "multicolour"],
                             "hairStyles": ["Bald", "Long", "Short", "middle"],
                             "lowerLength": ["Shorts", "Skirt", "Trousers"],
                             "lowerStyles": ["Solidcolor", "lattice", "multicolour"],
                             "shoesStyles": ["LeatherShoes", "Sandals", "Sneaker", "else"],
                             "towards": ["back", "front", "left", "right"],
                             "upperColors": ["upperBlack", "upperBrown", "upperBlue", "upperGreen", "upperGray",
                                             "upperOrange", "upperPink", "upperPurple", "upperRed", "upperWhite", "upperYellow"],
                             "lowerColors": ["lowerBlack", "lowerBrown", "lowerBlue", "lowerGreen", "lowerGray", 
                                             "lowerOrange", "lowerPink", "lowerPurple", "lowerRed", "lowerWhite", "lowerYellow"]}
        if len(args.feats) > 0: 
            self.classes_num = np.array(self.classes_num)[args.feats]
            self.classes_list = np.array(self.classes_list)[args.feats]

        self.model_type = args.model_type
        if self.model_type == "timm":
            # if not str(args.model).startswith("swin"):
            self.model = self.create_timm_model(args)
            # else:
                # self.model = self.create_swin_model(args)
                
            classifier = self.model.default_cfg['classifier']
            if classifier == "fc":
                n_features = self.model.fc.in_features
            elif classifier == "last_linear":
                n_features = self.model.last_linear.in_features
            elif classifier == "head":
                n_features = self.model.head.in_features
            elif classifier == "head.fc":
                n_features = self.model.head.fc.in_features
            else:
                n_features = self.model.classifier.in_features
        else:
            self.model = self.create_hp_model(args)
            n_features = self.model.final_fc.in_features
        
        self.head_drops = nn.ModuleList()
        for i in range(5):
            self.head_drops.append(nn.Dropout(args.multi_drop))

        self.att_layer = False
        self.multi_drop = args.multi_drop
        if args.att_pattern is not None: self.att_layer = True

        if self.att_layer:
            if args.att_pattern == "A":
                self.att_layer = nn.Sequential(
                    nn.Linear(n_features, 256),
                    nn.Tanh(),
                    nn.Linear(256, 1),
                )
            elif args.att_pattern == "B":
                self.att_layer = nn.Linear(n_features, 1)
            else:
                raise ValueError("invalid att pattern")
        
        self.classifiers = nn.ModuleList()
        for i, num_classes in enumerate(self.classes_num):
            # if i == 2 or i == 5: drop_rate = 0.1
            # block = self.add_block(in_chs=n_features, num_bottleneck=n_features // 2, num_classes=num_classes, drop_rate=drop_rate)
            if not str(self.model_name).startswith("swin") and not str(self.model_name).startswith("nat"):
                classifier = ClassifierHead(in_chs=n_features, num_classes=num_classes, drop_rate=drop_rate)
            else:
                classifier = nn.Linear(n_features, num_classes)
            # classifier = self.swin_head(in_chs=n_features, num_bottleneck=n_features // 2, num_classes=num_classes, drop_rate=0.3)
            # self._init_head_weights(classifier)
            self.classifiers.append(classifier)

    @staticmethod
    def _init_head_weights(classifier, init_method="kaiming"):
        for n, m in classifier.named_modules():
            if isinstance(m, nn.Linear):
                if init_method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                else:
                    nn.init.normal_(m.weight, 0., .01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                if init_method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0.0)
    
    def add_block(self, in_chs, num_bottleneck, num_classes, drop_rate=0.0):
        add_block = []
        add_block += [SelectAdaptivePool2d(pool_type='avg', flatten=True)]
        add_block += [nn.Linear(in_chs, num_bottleneck)]
        # add_block += [nn.BatchNorm1d(num_bottleneck)]
        # add_block += [nn.LeakyReLU(0.1)]
        # add_block += [nn.Dropout(p=drop_rate)]
        add_block += [nn.Linear(num_bottleneck, num_classes)]

        add_block = nn.Sequential(*add_block)
        self._init_head_weights(add_block)
        
        return add_block
    
    def swin_head(self, in_chs, num_bottleneck, num_classes, drop_rate=0.0):
        add_block = []
        add_block += [nn.AdaptiveAvgPool1d(1)]
        add_block += [nn.Flatten(1)]
        # add_block += [nn.BatchNorm1d(num_bottleneck)]
        # add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=drop_rate)]
        add_block += [nn.Linear(in_chs, num_classes)]

        add_block = nn.Sequential(*add_block)
        self._init_head_weights(add_block)
        return add_block
    
    def forward(self, x):
        if self.att_layer:
            l = x.shape[2] // 2
            h1 = self.model.forward_features(x[:, :, :l, :l])
            h2 = self.model.forward_features(x[:, :, :l, l:])
            h3 = self.model.forward_features(x[:, :, l:, :l])
            h4 = self.model.forward_features(x[:, :, l:, l:])
            w = F.softmax(torch.cat([
                self.att_layer(h1),
                self.att_layer(h2),
                self.att_layer(h3),
                self.att_layer(h4),
            ], dim=1), dim=1)
            x = h1 * w[:, 0].unsqueeze(-1) + \
                h2 * w[:, 1].unsqueeze(-1) + \
                h3 * w[:, 2].unsqueeze(-1) + \
                h4 * w[:, 3].unsqueeze(-1)
        else:
            if self.model_type == "timm":
                x = self.model.forward_features(x)
            else:
                if self.model_name in ["MNet", "HP"]:
                    index = -1
                elif str(self.model_name).startswith("AF"):
                    index = 0
                else:
                    print("Error mdoel name:{}".format(self.model_name))
                x = self.model.forward_features(x)[index]
        
        outs = []
        for classifier in self.classifiers:
            if self.multi_drop:
                for i, dropout in enumerate(self.head_drops):
                    if i == 0:
                        output = classifier(dropout(x))
                    else:
                        output += classifier(dropout(x))
                output /= len(self.head_drops)
            else:
                output = classifier(x)
            outs.append(output)
        return outs
    
    def create_timm_model(self, args):
        if self.training == True:
            ori_model = create_model(
                        args.model,
                        pretrained=True,
                        num_classes=args.head_num,
                        drop_rate=args.drop,
                        drop_path_rate=args.drop_path,
                        drop_block_rate=args.drop_block,
                        checkpoint_path=args.initial_checkpoint)
        else:
            ori_model = create_model(
                        args.model,
                        pretrained=False,
                        num_classes=args.head_num)
        return ori_model

    def create_swin_model(self, args):
        if args.model == "swin_base_patch4_window7_224_in22k":
            model = swin_base_patch4_window7_224_in22k(pretrained=True,
                                                       num_classes=args.head_num,
                                                       drop_rate=args.drop,
                                                       drop_path_rate=args.drop_path,
                                                       drop_block_rate=args.drop_block)
        return model

    def create_hp_model(self, args):
        if args.model == 'MNet':
            net = MNet()
            if not args.resume:
                # net.apply(self._init_head_weights(net, init_method="xavier"))
                self._init_head_weights(net, init_method="xavier")
        elif 'AF' in args.model:
            net = AF(af_name=args.model)
            if not args.resume:
                load_checkpoint(net.MNet, args.mpath)
            for param in net.MNet.parameters():
                param.requires_grad = False
        elif args.model == 'HP':
            net = HP()
            if not args.resume:
                load_checkpoint(net.MNet, args.mpath)
                load_checkpoint(net.AF1, args.af1path)
                load_checkpoint(net.AF2, args.af2path)
                load_checkpoint(net.AF3, args.af3path)

            for param in net.MNet.parameters():
                param.requires_grad = False
            for param in net.AF1.parameters():
                param.requires_grad = False
            for param in net.AF2.parameters():
                param.requires_grad = False
            for param in net.AF3.parameters():
                param.requires_grad = False
        return net
    
class SingleLabelModel(nn.Module):
    def __init__(self, args, drop_rate=0.5, training=True):
        super(SingleLabelModel, self).__init__()
                
        self.training = training
        self.drop_rate = drop_rate
        self.model_name = args.model
        self.classes_num = [args.num_classes]
        # self.classes_list = ["cervicalCells"]
        # self.classes_dict = {"cervicalCells": ["ASC-H&HSIL", "ASC-US&LSIL", "NILM", "SCC&AdC"]}
        self.classes_list = ["thyroid"]
        self.classes_dict = {"thyroid": ["nodule", "other"]}
        
        self.model_type = args.model_type
        if self.model_type == "timm":
            self.model = self.create_timm_model(args)
                
            classifier = self.model.default_cfg['classifier']
            if classifier == "fc":
                n_features = self.model.fc.in_features
            elif classifier == "last_linear":
                n_features = self.model.last_linear.in_features
            elif classifier == "head":
                n_features = self.model.head.in_features
            elif classifier == "head.fc":
                n_features = self.model.head.fc.in_features
            else:
                n_features = self.model.classifier.in_features
        else:
            self.model = self.create_hp_model(args)
            n_features = self.model.final_fc.in_features
        
        self.head_drops = nn.ModuleList()
        for i in range(5):
            self.head_drops.append(nn.Dropout(args.multi_drop))

        self.att_layer = False
        self.multi_drop = args.multi_drop
        if args.att_pattern is not None: self.att_layer = True

        if self.att_layer:
            if args.att_pattern == "A":
                self.att_layer = nn.Sequential(
                    nn.Linear(n_features, 256),
                    nn.Tanh(),
                    nn.Linear(256, 1),
                )
            elif args.att_pattern == "B":
                self.att_layer = nn.Linear(n_features, 1)
            else:
                raise ValueError("invalid att pattern")
        
        self.classifiers = nn.ModuleList()
        for i, num_classes in enumerate(self.classes_num):
            if not str(self.model_name).startswith("swin") and not str(self.model_name).startswith("nat"):
                classifier = ClassifierHead(in_features=n_features, num_classes=num_classes, drop_rate=drop_rate)
            else:
                classifier = nn.Linear(n_features, num_classes)
        self.classifiers.append(classifier)

    @staticmethod
    def _init_head_weights(classifier, init_method="kaiming"):
        for n, m in classifier.named_modules():
            if isinstance(m, nn.Linear):
                if init_method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                else:
                    nn.init.normal_(m.weight, 0., .01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                if init_method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0.0)
    
    def add_block(self, in_chs, num_bottleneck, num_classes, drop_rate=0.0):
        add_block = []
        add_block += [SelectAdaptivePool2d(pool_type='avg', flatten=True)]
        add_block += [nn.Linear(in_chs, num_bottleneck)]
        # add_block += [nn.BatchNorm1d(num_bottleneck)]
        # add_block += [nn.LeakyReLU(0.1)]
        # add_block += [nn.Dropout(p=drop_rate)]
        add_block += [nn.Linear(num_bottleneck, num_classes)]

        add_block = nn.Sequential(*add_block)
        self._init_head_weights(add_block)
        
        return add_block
    
    def swin_head(self, in_chs, num_bottleneck, num_classes, drop_rate=0.0):
        add_block = []
        add_block += [nn.AdaptiveAvgPool1d(1)]
        add_block += [nn.Flatten(1)]
        # add_block += [nn.BatchNorm1d(num_bottleneck)]
        # add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=drop_rate)]
        add_block += [nn.Linear(in_chs, num_classes)]

        add_block = nn.Sequential(*add_block)
        self._init_head_weights(add_block)
        return add_block
    
    def forward(self, x):
        if self.att_layer:
            l = x.shape[2] // 2
            h1 = self.model.forward_features(x[:, :, :l, :l])
            h2 = self.model.forward_features(x[:, :, :l, l:])
            h3 = self.model.forward_features(x[:, :, l:, :l])
            h4 = self.model.forward_features(x[:, :, l:, l:])
            w = F.softmax(torch.cat([
                self.att_layer(h1),
                self.att_layer(h2),
                self.att_layer(h3),
                self.att_layer(h4),
            ], dim=1), dim=1)
            x = h1 * w[:, 0].unsqueeze(-1) + \
                h2 * w[:, 1].unsqueeze(-1) + \
                h3 * w[:, 2].unsqueeze(-1) + \
                h4 * w[:, 3].unsqueeze(-1)
        else:
            if self.model_type == "timm":
                x = self.model.forward_features(x)
            else:
                if self.model_name in ["MNet", "HP"]:
                    index = -1
                elif str(self.model_name).startswith("AF"):
                    index = 0
                else:
                    print("Error mdoel name:{}".format(self.model_name))
                x = self.model.forward_features(x)[index]
        
        outs = []
        for classifier in self.classifiers:
            if self.multi_drop:
                for i, dropout in enumerate(self.head_drops):
                    if i == 0:
                        output = classifier(dropout(x))
                    else:
                        output += classifier(dropout(x))
                output /= len(self.head_drops)
            else:
                output = classifier(x)
            outs.append(output)
        return outs
    
    def create_timm_model(self, args):
        if self.training == True:
            ori_model = create_model(
                        args.model,
                        pretrained=True,
                        num_classes=args.num_classes,
                        drop_rate=args.drop,
                        drop_path_rate=args.drop_path,
                        drop_block_rate=args.drop_block,
                        checkpoint_path=args.initial_checkpoint)
        else:
            ori_model = create_model(
                        args.model,
                        pretrained=False,
                        num_classes=args.num_classes)
        return ori_model
    
