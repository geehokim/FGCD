#Code and refer from FedMA 


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
import torch.nn.functional as F
import copy
# In[1]:
'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init
import torchinfo



class VGG9(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, num_classes=10, *args ,**kwargs):
        super(VGG9, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1) , nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1) , nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1) , nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2) ,  nn.Dropout(p = 0.05))
        self.layer5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1) , nn.ReLU(inplace=True) )
        self.layer6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1) , nn.ReLU(inplace=True) , nn.MaxPool2d(kernel_size=2, stride=2) , nn.Dropout(p = 0.1))
                
        self.classifier = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Dropout(p = 0.1),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x, *args ,**kwargs):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGG9_GFLN(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, num_classes=10, *args ,**kwargs):
        super(VGG9_GFLN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1) , nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1) , nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1) , nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2) ,  nn.Dropout(p = 0.05))
        self.layer5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1) , nn.ReLU(inplace=True) )
        self.layer6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1) , nn.ReLU(inplace=True) , nn.MaxPool2d(kernel_size=2, stride=2) , nn.Dropout(p = 0.1))
                
        self.classifier = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Dropout(p = 0.1),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x, return_feature = False, level = 0, *args ,**kwargs):
        if level <= 0:
            x = self.layer1(x)
            out0 = self.layer2(x)
        else:
            out0 = x
            
        if level <= 1:            
            out1 = self.layer3(out0)
            out1 = self.layer4(out1)
        else:
            out1 = out0
            
        if level <= 2:
            out2 = self.layer5(out1)
            out2 = self.layer6(out2)
            out2 = out2.view(out2.size(0), -1)
        else:
            out2 = out1
        
        logit = self.classifier(out2)
        if return_feature==True:
            return out0,out1,out2,logit
        else:
            return logit






if __name__ == "__main__":
    print("cifar input size : 3 x 32 x 32")
    model = VGG9()
    torchinfo.summary(model, input_size=(1, 3, 32, 32), device="cpu")


