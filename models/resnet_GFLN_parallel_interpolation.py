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
    def get_interpolation_model(self):
        self.inter_conv1 = copy.deepcopy(self.conv1)
        
        self.inter_bn1 = copy.deepcopy(self.bn1)
        
        self.inter_layer1 = copy.deepcopy(self.layer1)
        
        self.inter_layer2 = copy.deepcopy(self.layer2)
        
        self.inter_layer3 = copy.deepcopy(self.layer3)
        
        self.inter_layer4 = copy.deepcopy(self.layer4)
                
        self.inter_fc = copy.deepcopy(self.fc)     
        
        for n,p in self.named_parameters():
            if 'inter' in n:
                p.requires_grad = False
        self.inter = True

    def update_interpolation_model(self, weight = 0):
        this_dict = copy.deepcopy(self.state_dict())
        for key in self.state_dict().keys():
            if 'inter' in key:
                this_layer_name = key[6:]
                this_global_name = 'global_'+this_layer_name
                this_dict[key] = this_dict[this_global_name] + weight*(this_dict[this_layer_name] - this_dict[this_global_name])
        self.load_state_dict(this_dict)

        






    def forward(self, x, return_feature=False,level = 0):
        if self.inter == False:
            self.get_interpolation_model()

        self.update_interpolation_model()

        if return_feature:

            out0 = F.relu(self.bn1(self.conv1(x)))

            out1_i = self.inter_layer1(out0)
            out1_g = self.global_layer1(out0)
            out1 = self.layer1(out0)

            #print("before concat",out1.shape)
            #print("after concat",torch.cat((out1_g,out1),dim=0).shape)
            out2_g = self.global_layer2(torch.cat((out1_i,out1_g,out1),dim=0))
            out2 = self.layer2(out1)

            out3_g = self.global_layer3(torch.cat((out2_i,out2_g,out2),dim=0))
            out3 = self.layer3(out2)


            out4_g = self.global_layer4(torch.cat((out3_g,out3),dim=0))
            out4_g = F.adaptive_avg_pool2d(out4_g, 1)
            out4_g = out4_g.view(out4_g.size(0), -1)
            out4 = self.layer4(out3)
            out4 = F.adaptive_avg_pool2d(out4, 1)
            out4 = out4.view(out4.size(0), -1)

            #print("out4_g : ",out4_g.shape)
            #print("out4 : ",out4.shape)

            logit_g = self.global_fc(torch.cat((out4_g,out4),dim=0))
            logit = self.fc(out4)

            return torch.cat((logit_g,logit),dim=0)
        
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


def ResNet18_GFLN_parallel(num_classes=10, l2_norm=False, use_pretrained = False, transfer_learning = True, use_bn = False, use_pre_fc = False, use_bn_layer = False):
    
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
