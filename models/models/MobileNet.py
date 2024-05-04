#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#mobilenetv2 code refer : https://github.com/weiaicunzai/pytorch-cifar100

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, num_classes=100):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.GroupNorm(2,in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.GroupNorm(2,in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.GroupNorm(2,out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual

class MobileNetV2(nn.Module):

    def __init__(self, num_classes=100, *args ,**kwargs):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.GroupNorm(2,32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.GroupNorm(2,1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, num_classes, 1)

    def forward(self, x, *args ,**kwargs):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)




class MobileNetV2_GFLN(nn.Module):

    def __init__(self, num_classes=100, *args ,**kwargs):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.GroupNorm(2,32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.GroupNorm(2,1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, num_classes, 1)

        
    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)        
        
    def forward(self, x, return_feature = False, level = 0, *args ,**kwargs):
        

        if level<=0:
            out0 = self.pre(x)                     
        else:
            out0 = x
            
        
        if level<=1:
            out1 = self.stage1(out0)   
        else:
            out1 = out0
            
        
        if level<=2:
            out2 = self.stage2(out1)        
        else:
            out2 = out1
            
        
        if level<=3:
            out3 = self.stage3(out2)
        else:
            out3 = out2
            
        if level<=4:
            out4 = self.stage4(out3)
        else:
            out4 = out3
        
        if level<=5:
            out5 = self.stage5(out4) 
        else:
            out5 = out4
        
        if level<=6:
            out6 = self.stage6(out5)
        else:
            out6 = out5

        x = self.stage7(out6)
            
        
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        logit = x.view(x.size(0), -1)
        if return_feature:
            return out0,out1,out2,out3,out4,out5,out6,logit
        else:
            return logit


class MobileNetV2_Proc(MobileNetV2_GFLN):

    def __init__(self, num_classes=100, *args, **kwargs):
        super().__init__(num_classes, *args, **kwargs)

    def forward(self, x, return_feature=False, level=0, *args, **kwargs):

        if level <= 0:
            out0 = self.pre(x)
        else:
            out0 = x

        if level <= 1:
            out1 = self.stage1(out0)
        else:
            out1 = out0

        if level <= 2:
            out2 = self.stage2(out1)
        else:
            out2 = out1

        if level <= 3:
            out3 = self.stage3(out2)
        else:
            out3 = out2

        if level <= 4:
            out4 = self.stage4(out3)
        else:
            out4 = out3

        if level <= 5:
            out5 = self.stage5(out4)
        else:
            out5 = out4

        if level <= 6:
            out6 = self.stage6(out5)
        else:
            out6 = out5

        x = self.stage7(out6)

        x = self.conv1(x)
        feat = F.adaptive_avg_pool2d(x, 1)
        logit = self.conv2(feat)
        logit = logit.view(logit.size(0), -1)
        feat = feat.view(feat.size(0), -1)
        if return_feature:
            return feat, logit
        else:
            return logit
        
        
        
        
'''        100,5%,Dir03 : 48.02
    def forward(self, x, return_feature = False, level = 0, *args ,**kwargs):
        

        if level<=0:
            out0 = self.pre(x)
            out0 = self.stage1(out0)
            
        else:
            out0 = x
            
        
        if level<=1:
            out1 = self.stage2(out0)        
            out1 = self.stage3(out1)
        else:
            out1 = out0
            
        
        if level<=2:
            out2 = self.stage4(out1)
            out2 = self.stage5(out2)
        else:
            out2 = out1
            
        
        if level<=3:
        
            out3 = self.stage6(out2)
            out3 = self.stage7(out3)
        else:
            out3 = out2
        
        x = self.conv1(out3)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        logit = x.view(x.size(0), -1)
        if return_feature:
            return out0,out1,out2,out3,logit
        else:
            return logit

'''














'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
# In[1]:
import functools
import torchvision.models as models

class mobilenetv3_GFLN_GN(nn.Module):
    def __init__(self,num_classes = 10,layer_unit=2, *args ,**kwargs):
        super(mobilenetv3_GFLN_GN,self).__init__()       
        Groupnorm_with_2groups = functools.partial(nn.GroupNorm,num_groups = 2)
        Groupnorm = lambda x: Groupnorm_with_2groups(num_channels = x)
        mobilenetbackbone  = models.mobilenet_v3_small(norm_layer = Groupnorm, num_classes = num_classes)
        
        ########## Rearranging features ###########
        arr_feature = []
        tmp = []
        self.layer_unit = layer_unit
        for idx,layer in enumerate(mobilenetbackbone.features):
            tmp.append(layer)
            if idx%layer_unit == 1:
                arr_feature.append(nn.Sequential(*tmp))
                tmp = []

        self.features = nn.ModuleList(arr_feature)
        
        
        ############ avgpool
        self.avgpool = mobilenetbackbone.avgpool
        
        ############ classifier : delete Dropout layer & tuning first linear layer size
        self.first_layer_in_features = 96
        tmp =[]
        first_linear = True
        for idx,layer in enumerate(mobilenetbackbone.classifier):
            if first_linear and isinstance(layer,nn.Linear):
                
                tmp.append(nn.Linear(in_features = self.first_layer_in_features\
                                     ,out_features = layer.out_features, bias = layer.bias is not None))
                first_linear = False
            elif not isinstance(layer,nn.Dropout):
                tmp.append(layer)
        self.classifier = nn.Sequential(*tmp)#nn.ModuleList(tmp)

    def forward(self,x, return_feature = False, level = 0, *args ,**kwargs):
        tmp = []
        for idx,layer in enumerate(self.features):
            if level <= idx:
                x = layer(x)
            else:
                x = x
            
            if return_feature:
                tmp.append(x)
                
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logit = self.classifier(x)
        
        if return_feature:
            return (*tmp,logit)
        else:
            return logit
'''