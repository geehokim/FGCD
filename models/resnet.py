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
import torchvision
from models.build import ENCODER_REGISTRY


class SubspaceConv2d(nn.Conv2d):
    
    def forward(self, x):
        w = self.get_weight()
        x = F.conv2d(
            x,
            w,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return x

    def get_weight(self):
        if getattr(self, "weight1", None) is not None:
            return self.alpha * self.weight + (1-self.alpha) * self.weight1
        else:
            return self.weight
        # raise NotImplementedError


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_bn_layer=False, Conv2d=nn.Conv2d):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes) 
        self.conv2 = Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes) 

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(2, self.expansion*planes) if not use_bn_layer else nn.BatchNorm2d(self.expansion*planes) 
            )

    def forward_intermediate(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
        out_i = self.bn1(self.conv1(x))
        out = F.relu(out_i)
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        if not no_relu:
            out = F.relu(out)
        else:
            out = out
        return out, out_i

    def forward(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        if not no_relu:
            out = F.relu(out)
        else:
            out = out
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, use_bn_layer=False, Conv2d=nn.Conv2d):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(2, self.expansion*planes) if not use_bn_layer else nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(2, self.expansion*planes) if not use_bn_layer else nn.BatchNorm2d(planes)
            )

    def forward(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        if not no_relu:
            out = F.relu(out)
        else:
            out = out
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, l2_norm=False, use_pretrained=False, use_bn_layer=False,
                 last_feature_dim=512, **kwargs):
        
        #use_pretrained means whether to use torch torchvision.models pretrained model, and use conv1 kernel size as 7
        
        super(ResNet, self).__init__()
        self.l2_norm = l2_norm
        print("self.l2_norm:", self.l2_norm)
        self.in_planes = 64
        conv1_kernel_size = 3
        if use_pretrained:
            conv1_kernel_size = 7
            print("note that conv1_kernel_size is :", conv1_kernel_size) 
            # breakpoint()

        Conv2d = self.get_conv()
        Linear = self.get_linear()   
        self.conv1 = Conv2d(3, 64, kernel_size=conv1_kernel_size,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2, 64) if not use_bn_layer else nn.BatchNorm2d(64) 
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use_bn_layer=use_bn_layer)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use_bn_layer=use_bn_layer)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use_bn_layer=use_bn_layer)
        self.layer4 = self._make_layer(block, last_feature_dim, num_blocks[3], stride=2, use_bn_layer=use_bn_layer)

        self.logit_detach = False

        # self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1, use_bn_layer=use_bn_layer)
        # self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2, use_bn_layer=use_bn_layer)
        # self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=2, use_bn_layer=use_bn_layer)
        # self.layer4 = self._make_layer(block, last_feature_dim, num_blocks[3], stride=2, use_bn_layer=use_bn_layer)

        # print(self)
        

        if use_pretrained:
            resnet = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
            self.layer1.load_state_dict(resnet.layer1.state_dict(), strict=False)
            self.layer2.load_state_dict(resnet.layer2.state_dict(), strict=False)
            self.layer3.load_state_dict(resnet.layer3.state_dict(), strict=False)
            self.layer4.load_state_dict(resnet.layer4.state_dict(), strict=False)
            # self.layer2 = resnet.layer2
            # self.layer3 = resnet.layer3
            # self.layer4 = resnet.layer4
            # breakpoint()

        self.num_layers = 6 # layer0 to layer5 (fc)

        if l2_norm:
            self.fc = Linear(last_feature_dim * block.expansion, num_classes, bias=False)
        else:
            self.fc = Linear(last_feature_dim * block.expansion, num_classes)


    def get_conv(self):
        return nn.Conv2d
    
    def get_linear(self):
        return nn.Linear

    def _make_layer(self, block, planes, num_blocks, stride, use_bn_layer=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_bn_layer=use_bn_layer, Conv2d=self.get_conv()))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False):
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
            
        if return_feature==True:
            return out, logit
        else:
            return logit
        
    
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

    #
    # def ResNet18(num_classes=10, l2_norm=False, use_pretrained = False, transfer_learning = True, use_bn = False, use_pre_fc = False, use_bn_layer = False):
    #
    # model =  ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, l2_norm=l2_norm, use_pretrained = use_pretrained, use_bn_layer =use_bn_layer)
    #
    #
    # if not use_pretrained:
    #     return model
    #
    # else:
    #     ImageNet_pretrained = models.resnet18(pretrained=True)
    #     if not transfer_learning:
    #         return ImageNet_pretrained
    #
    #
    #     my_res_dict = model.state_dict()
    #     res_dict = ImageNet_pretrained.state_dict()
    #     except_names = []
    #     if use_bn == False:
    #         except_names.extend(['bn','downsample.1'])
    #     if use_pre_fc == False:
    #         except_names.extend(['fc'])
    #
    #
    #     print("Start synking model with pretrained")
    #     for name in my_res_dict.keys():
    #         print()
    #         print(name)
    #         skip = False
    #         if except_names!=[]:
    #             for except_name in except_names:
    #                 if except_name in name:
    #                     skip = True
    #                     continue
    #         if not skip:
    #             try:
    #                 pre_par = res_dict[name]
    #                 if my_res_dict[name].shape == pre_par.shape:
    #                     my_res_dict[name] = pre_par
    #                     print("synk")
    #                 else:
    #                     print("Shape is not same")
    #                     print('my_shape:', my_res_dict[name].shape)
    #                     print('pretrained shape:', pre_par.shape)
    #
    #             except:
    #                 print("Fail to synk at ", name)
    #                 pass
    #
    #
    #     model.load_state_dict(my_res_dict)
    #
    #     return model


class SubspaceResNet(ResNet):

    def get_conv(self):
        return SubspaceConv2d
    
    def get_linear(self):
        return nn.Linear

    def update_model1_weight(self, model1, alpha):
        for layer, layer1 in zip(self.named_modules(), model1.named_modules()):
            breakpoint()
            if isinstance(layer, nn.Conv2d):
                breakpoint()
                # setattr(layer, f"weight1", alpha)
                setattr(layer, f"weight1", layer1.weight)
                setattr(layer, f"alpha", alpha)
            



        # model1_state_dict = model1.state_dict()

        # self.model1 = model1




@ENCODER_REGISTRY.register()
class ResNet18(ResNet):
    def __init__(self, args, num_classes=10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs
                        #  l2_norm=args.l2_norm, use_pretrained=args.model.pretrained, use_bn_layer=args.use_bn_layer
                         )
        

@ENCODER_REGISTRY.register()
class SubspaceResNet18(SubspaceResNet):
    def __init__(self, args, num_classes=10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                         l2_norm=args.l2_norm, use_pretrained=args.use_pretrained, use_bn_layer=args.use_bn_layer
                         )



def ResNet34(num_classes=10, l2_norm=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, l2_norm=l2_norm)


def ResNet50Proc(num_classes=10, l2_norm=False, use_pretrained = False, transfer_learning = True, use_bn = False, use_pre_fc = False, use_bn_layer = False):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, l2_norm=l2_norm, use_pretrained=use_pretrained,
                   use_bn_layer=use_bn_layer)

    if not use_pretrained:
        return model


def ResNet101(num_classes=10, l2_norm=False):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, l2_norm=l2_norm)


def ResNet152(num_classes=10, l2_norm=False):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, l2_norm=l2_norm)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()