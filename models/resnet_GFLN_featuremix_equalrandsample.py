'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,use_bn_layer = False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes) 
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes) 

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(2, self.expansion*planes) if not use_bn_layer else nn.BatchNorm2d(self.expansion*planes) 
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(2, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(2, planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(2, self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(2, self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, l2_norm=False, use_pretrained = False, use_bn_layer = False):
        
        #use_pretrained means whether to use torch torchvision.models pretrained model, and use conv1 kernel size as 7
        
        super(ResNet, self).__init__()
        self.l2_norm = l2_norm
        self.in_planes = 64
        conv1_kernel_size = 3
        if use_pretrained:
            conv1_kernel_size = 7
            print("note that conv1_kernel_size is :",conv1_kernel_size) 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=conv1_kernel_size,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2, 64) if not use_bn_layer else nn.BatchNorm2d(64) 
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use_bn_layer =use_bn_layer)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use_bn_layer =use_bn_layer)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use_bn_layer =use_bn_layer)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, use_bn_layer =use_bn_layer)
        if l2_norm:
            self.fc = nn.Linear(512 * block.expansion, num_classes, bias=False)
        else:
            self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride,use_bn_layer = False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_bn_layer =use_bn_layer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    
    def get_global_model(self):
        
        self.global_conv1 = copy.deepcopy(self.conv1)
        
        self.global_bn1 = copy.deepcopy(self.bn1)
        
        self.global_layer1 = copy.deepcopy(self.layer1)
        
        self.global_layer2 = copy.deepcopy(self.layer2)
        
        self.global_layer3 = copy.deepcopy(self.layer3)
        
        self.global_layer4 = copy.deepcopy(self.layer4)
                
        self.global_fc = copy.deepcopy(self.fc)     
        
        for n,p in self.named_parameters():
            if 'global' in n:
                p.requires_grad = False
        
        #Wrong!!!!!!!! the parameters of the global_~~ will never freeze. 
        '''
        self.global_conv1 = copy.deepcopy(self.conv1)
        self.global_conv1.requires_grad = False
        
        self.global_bn1 = copy.deepcopy(self.bn1)
        self.global_bn1.requires_grad = False
        
        self.global_layer1 = copy.deepcopy(self.layer1)
        self.global_layer1.requires_grad = False
        
        self.global_layer2 = copy.deepcopy(self.layer2)
        self.global_layer2.requires_grad = False
        
        self.global_layer3 = copy.deepcopy(self.layer3)
        self.global_layer3.requires_grad = False
        
        self.global_layer4 = copy.deepcopy(self.layer4)
        self.global_layer4.requires_grad = False
        
        
        self.global_fc = copy.deepcopy(self.fc)
        self.global_fc.requires_grad = False
        '''

    def get_mixed_feature(self,out,out_g,rand_sample):
        return out*rand_sample + out_g*(1-rand_sample)

    def forward(self, x, return_feature=False,num_of_branch = 1):
        if return_feature:
            newsize = list(torch.ones(len(x.shape),dtype = int))
            newsize[0] = num_of_branch
            repeated_x = x.repeat(newsize)
            #print("repeated_x.shape :",repeated_x.shape)
            
            randsize = copy.deepcopy(newsize)
            randsize[0] = len(repeated_x)
            
            rand_sample = torch.rand(randsize).to(x.device)
            batch_size = len(x)
            #print("rand_sample.squeeze().unsqueeze(dim = 1) : ",rand_sample.squeeze().unsqueeze(dim = 1).shape)
            #print("rand_sample.shape : ",rand_sample.shape)
            
            out0 = F.relu(self.bn1(self.conv1(torch.cat((repeated_x,x),dim = 0))))
            out0_g = F.relu(self.global_bn1(self.global_conv1(repeated_x)))
            mixed_out0 = self.get_mixed_feature(out0[:-batch_size],out0_g,rand_sample)
            #print("out0.shape :",out0.shape)
            #print("mixed_out0.shape :",mixed_out0.shape)

            out1_g = self.global_layer1(mixed_out0)
            out1 = self.layer1(torch.cat((mixed_out0,out0[-batch_size:]),dim = 0))
            mixed_out1 = self.get_mixed_feature(out1[:-batch_size],out1_g,rand_sample)
            #print("before concat",out1.shape)
            #print("after concat",torch.cat((out1_g,out1),dim=0).shape)
            out2_g = self.global_layer2(mixed_out1)
            out2 = self.layer2(torch.cat((mixed_out1,out1[-batch_size:]),dim = 0))
            mixed_out2 = self.get_mixed_feature(out2[:-batch_size],out2_g,rand_sample)

            out3_g = self.global_layer3(mixed_out2)
            out3 = self.layer3(torch.cat((mixed_out2,out2[-batch_size:]),dim = 0))
            mixed_out3 = self.get_mixed_feature(out3[:-batch_size],out3_g,rand_sample)


            out4_g = self.global_layer4(mixed_out3)
            out4 = self.layer4(torch.cat((mixed_out3,out3[-batch_size:]),dim = 0))
            
            #print("out4_g : ",out4_g.shape)
            #print("out4 : ",out4.shape)
            
            
            mixed_out4 = self.get_mixed_feature(out4[:-batch_size],out4_g,rand_sample)
            #print("mixed_out4 : ", mixed_out4.shape)
            mixed_out4 = F.adaptive_avg_pool2d(mixed_out4, 1)
            mixed_out4 = mixed_out4.view(mixed_out4.size(0), -1)
            out4 = F.adaptive_avg_pool2d(out4, 1)
            out4 = out4.view(out4.size(0), -1)
            #mixed_out4 = self.get_mixed_feature(,out4_g,rand_sample)
            '''
            out4_g = F.adaptive_avg_pool2d(out4_g, 1)
            out4_g = out4_g.view(out4_g.size(0), -1)
            
            out4 = F.adaptive_avg_pool2d(out4, 1)
            out4 = out4.view(out4.size(0), -1)
            mixed_out4 = self.get_mixed_feature(out4,out4_g,rand_sample.squeeze().unsqueeze(dim = 1))
            '''
            #print("out4_g : ",out4_g.shape)
            #print("out4 : ",out4.shape)

            logit_g = self.global_fc(mixed_out4)
            logit = self.fc(torch.cat((mixed_out4,out4[-batch_size:]),dim = 0))
            #print("logit.shape :",logit.shape)
            mixed_logit = self.get_mixed_feature(logit[:-batch_size],logit_g,rand_sample.squeeze().unsqueeze(dim = 1))
            #print("mixed_logit.shape : ",mixed_logit.shape)

            return torch.cat((mixed_logit,logit[-batch_size:]),dim = 0)
        
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            #out = F.avg_pool2d(out, 4)
            out = F.adaptive_avg_pool2d(out, 1)
            out = out.view(out.size(0), -1)
            if self.l2_norm:
                #with torch.no_grad():
                    #w = self.linear.weight.data.clone()
                    #w = F.normalize(w, dim=1, p=2)
                    #self.linear.weight.copy_(w)
                #self.linear = F.normalize(self.linear)
                self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
                out = F.normalize(out, dim=1)
                logit = self.fc(out)
            else:
                logit = self.fc(out)


            return logit
        
        '''
        if level <= 0:
            out0 = F.relu(self.bn1(self.conv1(x)))
        else:
            out0 = x
        if level <=1:
            out1 = self.layer1(out0)
        else:
            out1 = out0
        if level <=2:
            out2 = self.layer2(out1)
        else:
            out2 = out1
        if level <=3:
            out3 = self.layer3(out2)
        else:
            out3 = out2
        if level <=4:
            out4 = self.layer4(out3)
            #out = F.avg_pool2d(out, 4)
            out4 = F.adaptive_avg_pool2d(out4, 1)
            out4 = out4.view(out4.size(0), -1)
        else:
            out4 = out3
        if self.l2_norm:
            #with torch.no_grad():
                #w = self.linear.weight.data.clone()
                #w = F.normalize(w, dim=1, p=2)
                #self.linear.weight.copy_(w)
            #self.linear = F.normalize(self.linear)
            self.fc.weight.data = F.normalize(self.linear.weight.data, p=2, dim=1)
            out4 = F.normalize(out4, dim=1)
            logit = self.fc(out4)
        else:
            logit = self.fc(out4)
            
        if return_feature==True:
            return out0,out1,out2,out3,out4,logit
        else:
            return logit
        
        '''
    def forward_classifier(self,x):
        logit = self.fc(x)
        return logit        
    
    
    def sync_online_and_global(self):
        state_dict=self.state_dict()
        for key in state_dict:
            if 'global' in key:
                x=(key.split("_global"))
                online=(x[0]+x[1])
                state_dict[key]=state_dict[online]
        self.load_state_dict(state_dict)


def ResNet18_GFLN_featuremix_equalrandsample(num_classes=10, l2_norm=False, use_pretrained = False, transfer_learning = True, use_bn = False, use_pre_fc = False, use_bn_layer = False):
    
    model =  ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, l2_norm=l2_norm, use_pretrained = use_pretrained, use_bn_layer =use_bn_layer)    

    
    if not use_pretrained:
        return model

    else:
        ImageNet_pretrained = models.resnet18(pretrained=True)
        if not transfer_learning:
            return ImageNet_pretrained
           

        my_res_dict = model.state_dict()
        res_dict = ImageNet_pretrained.state_dict()
        except_names = []
        if use_bn == False:
            except_names.extend(['bn','downsample.1'])
        if use_pre_fc == False:
            except_names.extend(['fc'])
            
         
        print("Start synking model with pretrained")
        for name in my_res_dict.keys():
            print()
            print(name)
            skip = False
            if except_names!=[]:
                for except_name in except_names:
                    if except_name in name:
                        skip = True
                        continue
            if not skip:
                try:
                    pre_par = res_dict[name]
                    if my_res_dict[name].shape == pre_par.shape:
                        my_res_dict[name] = pre_par
                        print("synk")
                    else:
                        print("Shape is not same")
                        print('my_shape:', my_res_dict[name].shape)
                        print('pretrained shape:', pre_par.shape)

                except:
                    print("Fail to synk at ", name)
                    pass            


        model.load_state_dict(my_res_dict)
        
        return model

        
            
        
        


def ResNet34(num_classes=10, l2_norm=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, l2_norm=l2_norm)


def ResNet50(num_classes=10, l2_norm=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, l2_norm=l2_norm)


def ResNet101(num_classes=10, l2_norm=False):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, l2_norm=l2_norm)


def ResNet152(num_classes=10, l2_norm=False):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, l2_norm=l2_norm)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
