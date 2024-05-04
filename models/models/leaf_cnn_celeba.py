import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN,self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding='same'),  # in_channels, out_channels, kernel_size
            nn.GroupNorm(2, 32),
            nn.MaxPool2d(kernel_size=2,stride=2,padding = 'same'),  # kernel_size, stride
            nn.ReLU()            
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding='same'),  # in_channels, out_channels, kernel_size
            nn.GroupNorm(2, 32),
            nn.MaxPool2d(kernel_size=2,stride=2,padding = 'same'),  # kernel_size, stride
            nn.ReLU()            
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding='same'),  # in_channels, out_channels, kernel_size
            nn.GroupNorm(2, 32),
            nn.MaxPool2d(kernel_size=2,stride=2,padding = 'same'),  # kernel_size, stride
            nn.ReLU()            
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=512,kernel_size=3,padding='same'),  # in_channels, out_channels, kernel_size
            nn.GroupNorm(2, 32),
            #nn.MaxPool2d(kernel_size=2,stride=2,padding = 'same'),  # kernel_size, stride
            nn.ReLU()            
        )
        
        self.fc=nn.Sequential(
            #nn.Linear(in_features=7*7*64,out_features=2048),
            #nn.ReLU(),
            nn.Linear(in_features=225792,out_features=num_classes)
        )

    def forward(self,x):
        #print(x.shape)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        feature = (self.conv3(x))
        out = F.adaptive_avg_pool2d(out, 1)
        #print(feature.view(x.shape[0], -1).shape)
        output=self.fc(feature.view(x.shape[0], -1))
        return output

    
def leaf_celeba(num_classes=10, l2_norm=False, use_pretrained = False, transfer_learning = True, use_bn = False, use_pre_fc = False, use_bn_layer = False):
    return CNN(num_classes)