import torch
import torch.nn as nn
import torch.nn.functional as F
from models.build import ENCODER_REGISTRY

@ENCODER_REGISTRY.register()
class CNN(nn.Module):
    def __init__(self,args, num_classes, **kwargs):
        super(CNN,self).__init__()
        self.num_layers = 4
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding='same', stride=1),  # in_channels, out_channels, kernel_size
            nn.GroupNorm(2, 32),
            nn.MaxPool2d(kernel_size=2,stride=2,padding = 0),  # kernel_size, stride
            nn.ReLU()            
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding='same', stride=1),  # in_channels, out_channels, kernel_size
            nn.GroupNorm(2, 32),
            nn.MaxPool2d(kernel_size=2,stride=2,padding = 0),  # kernel_size, stride
            nn.ReLU()            
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding='same', stride=1),  # in_channels, out_channels, kernel_size
            nn.GroupNorm(2, 32),
            nn.MaxPool2d(kernel_size=2,stride=2,padding = 0),  # kernel_size, stride
            nn.ReLU()            
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding='same', stride=1),  # in_channels, out_channels, kernel_size
            nn.GroupNorm(2, 32),
            nn.MaxPool2d(kernel_size=2,stride=2,padding = 0),  # kernel_size, stride
            nn.ReLU()            
        )
        
        self.fc=nn.Sequential(
            #nn.Linear(in_features=7*7*64,out_features=2048),
            #nn.ReLU(),
            nn.Linear(in_features=5*5*32,out_features=num_classes)
        )

    def forward(self,x, no_relu=False):
        results = {}
        #print(x.shape)
        x = self.conv0(x)
        #print(x.size())
        results['layer0'] = x
        x = self.conv1(x)
        #print(x.size())
        results['layer1'] = x
        x = self.conv2(x)
        #print(x.size())
        results['layer2'] = x
        feature = (self.conv3(x))
        #print(feature.size())
        #print(feature.view(x.shape[0], -1).shape)
        #feature = F.adaptive_avg_pool2d(feature, 1)
        feature = feature.view(x.shape[0], -1)
        results['feature'] = feature
        output=self.fc(feature)
        results['logit'] = output
        results['layer3'] = output
        return results

    
@ENCODER_REGISTRY.register()
class CNNFemnist(nn.Module):
    def __init__(self,args, num_classes, **kwargs):
        super(CNNFemnist,self).__init__()
        self.num_layers = 4
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding='same', stride=1),  # in_channels, out_channels, kernel_size
            nn.GroupNorm(2, 32),
            #nn.MaxPool2d(kernel_size=2,stride=2,padding = 0),  # kernel_size, stride
            nn.ReLU()            
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding='same', stride=1),  # in_channels, out_channels, kernel_size
            nn.GroupNorm(2, 32),
            nn.MaxPool2d(kernel_size=2,stride=2,padding = 0),  # kernel_size, stride
            nn.ReLU()            
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding='same', stride=1),  # in_channels, out_channels, kernel_size
            nn.GroupNorm(2, 32),
            nn.MaxPool2d(kernel_size=2,stride=2,padding = 0),  # kernel_size, stride
            nn.ReLU()            
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding='same', stride=1),  # in_channels, out_channels, kernel_size
            nn.GroupNorm(2, 32),
            nn.MaxPool2d(kernel_size=2,stride=2,padding = 0),  # kernel_size, stride
            nn.ReLU()            
        )
        
        self.fc=nn.Sequential(
            #nn.Linear(in_features=7*7*64,out_features=2048),
            #nn.ReLU(),
            nn.Linear(in_features=3*3*32,out_features=num_classes)
        )

    def forward(self,x, no_relu=False):
        x = x.view(-1,1,28,28)
        results = {}
        #print(x.shape)
        x = self.conv0(x)
        #print(x.size())
        results['layer0'] = x
        x = self.conv1(x)
        #print(x.size())
        results['layer1'] = x
        x = self.conv2(x)
        #print(x.size())
        results['layer2'] = x
        feature = (self.conv3(x))
        results['layer3'] = feature
        #print(feature.size())
        #print(feature.view(x.shape[0], -1).shape)
        #feature = F.adaptive_avg_pool2d(feature, 1)
        feature = feature.view(x.shape[0], -1)
        results['feature'] = feature
        output=self.fc(feature)
        results['logit'] = output
        results['layer4'] = output
        return results

def leaf_celeba(num_classes=10, l2_norm=False, use_pretrained = False, transfer_learning = True, use_bn = False, use_pre_fc = False, use_bn_layer = False):
    return CNN(num_classes)