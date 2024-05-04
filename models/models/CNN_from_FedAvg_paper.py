#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
import torch.nn.functional as F
import copy
# In[1]:
import torchinfo


class CNN_from_FedAvg_paper(nn.Module):
    # from "https://arxiv.org/pdf/1602.05629.pdf"
    def __init__(self,num_classes = 10,l2_norm = False):
        super(CNN_from_FedAvg_paper, self).__init__()
        self.flatten_size = 3136
        self.conv1 = nn.Conv2d(3, 32, 5,padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5,padding=1)
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        #self.fc3 = nn.Linear(192, num_classes)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #x = F.relu(self.fc2(x))
        #x = (self.fc3(x))

        return x

class CNN_from_FedAvg_paper_MLB(nn.Module):
    # from "https://arxiv.org/pdf/1602.05629.pdf"
    def __init__(self,num_classes = 10,l2_norm = False):
        super(CNN_from_FedAvg_paper_MLB, self).__init__()
        self.flatten_size = 3136
        self.conv1 = nn.Conv2d(3, 32, 5,padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5,padding=1)
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        #self.fc3 = nn.Linear(192, num_classes)

    def forward(self, x, return_feature=False,level = 0):
        if level <= 0:
            x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #x = F.relu(self.fc2(x))
        #x = (self.fc3(x))
        return x    
    


if __name__ == "__main__":
    
    print("cifar input size : 3 x 32 x 32")
    model = CNN_from_FedAvg_paper()
    torchinfo.summary(model, input_size=(1, 3, 32, 32), device="cpu")
      


# In[ ]:




