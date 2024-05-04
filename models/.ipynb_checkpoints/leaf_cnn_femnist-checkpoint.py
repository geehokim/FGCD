import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,padding=2),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),  # kernel_size, stride
            nn.Conv2d(in_channels=16,out_channels=64,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc=nn.Sequential(
            nn.Linear(in_features=7*7*64,out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048,out_features=num_classes)
        )

    def forward(self,x):
        x = x.view(-1,1,28,28)
        feature=self.conv(x)
        output=self.fc(feature.view(x.shape[0], -1))
        return output

    
def leaf_femnist(num_classes=10, l2_norm=False, use_pretrained = False, transfer_learning = True, use_bn = False, use_pre_fc = False, use_bn_layer = False):
    return CNN(num_classes)