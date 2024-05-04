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
from utils import *
import numpy as np
from models.resnet import BasicBlock, Bottleneck, ResNet
from models.build import ENCODER_REGISTRY
from typing import Callable, Dict, Tuple, Union, List, Optional
from omegaconf import DictConfig

from models.resnet import SubspaceConv2d

import logging
logger = logging.getLogger(__name__)

class ResNet_base(ResNet):

    def forward_layer(self, layer, x, no_relu=True):

        if isinstance(layer, nn.Linear):
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.view(x.size(0), -1)
            out = layer(x)
        else:
            if no_relu:
                out = x
                for sublayer in layer[:-1]:
                    out = sublayer(out)
                out = layer[-1](out, no_relu=no_relu)
            else:
                out = layer(x)

        return out
    
    def forward_layer_by_name(self, layer_name, x, no_relu=True):
        layer = getattr(self, layer_name)
        return self.forward_layer(layer, x, no_relu)

    def forward_layer0(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
        out0 = self.bn1(self.conv1(x))
        if not no_relu:
            out0 = F.relu(out0)
        return out0
    

    def freeze_backbone(self):
        for n, p in self.named_parameters():
            if 'fc' not in n:
                p.requires_grad = False
        logger.warning('Freeze backbone parameters (except fc)')
        return
    
    def forward(self, x: torch.Tensor, no_relu: bool = True) -> Dict[str, torch.Tensor]:
        results = {}

        if no_relu:
            out0 = self.bn1(self.conv1(x))
            results['layer0'] = out0
            out0 = F.relu(out0)

            out = out0
            for i, sublayer in enumerate(self.layer1):
                sub_norelu = (i == len(self.layer1) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer1'] = out
            out = F.relu(out)

            for i, sublayer in enumerate(self.layer2):
                sub_norelu = (i == len(self.layer2) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer2'] = out
            out = F.relu(out)

            for i, sublayer in enumerate(self.layer3):
                sub_norelu = (i == len(self.layer3) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer3'] = out
            out = F.relu(out)

            for i, sublayer in enumerate(self.layer4):
                sub_norelu = (i == len(self.layer4) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer4'] = out
            out = F.relu(out)
            
        else:
            out0 = self.bn1(self.conv1(x))
            out0 = F.relu(out0)
            results['layer0'] = out0

            out = out0
            out = self.layer1(out)
            results['layer1'] = out
            out = self.layer2(out)
            results['layer2'] = out
            out = self.layer3(out)
            results['layer3'] = out
            out = self.layer4(out)
            results['layer4'] = out
            

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)

        # if self.l2_norm:
        #     self.fc.weight.data = F.normalize(self.fc.weight.data, p=2., dim=1)
        #     out4 = F.normalize(out4, dim=1)
        #     logit = self.fc(out4)
            
        if self.logit_detach:
            logit = self.fc(out.detach())
        else:
            logit = self.fc(out)

        results['feature'] = out
        results['logit'] = logit
        results['layer5'] = logit

        return results


class SubspaceResNet_base(ResNet_base):

    def get_conv(self):
        return SubspaceConv2d
    
    def get_linear(self):
        return nn.Linear

    

@ENCODER_REGISTRY.register()
class ResNet18_base(ResNet_base):
    
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs
                        #  l2_norm=args.model.l2_norm,
                        #  use_pretrained=args.model.pretrained, use_bn_layer=args.model.use_bn_layer
                         )
        

class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, mlp_width):
        super(MLP, self).__init__()

        self.input = nn.Linear(n_inputs, mlp_width)
        # self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(mlp_width, mlp_width)
            for _ in range(1)]
            )
        self.output = nn.Linear(mlp_width, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        # x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            # x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


@ENCODER_REGISTRY.register()
class ResNet18_FedBR(ResNet_base):
    
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs
                        #  l2_norm=args.model.l2_norm,
                        #  use_pretrained=args.model.pretrained, use_bn_layer=args.model.use_bn_layer
                         )
        feature_dim = 512
        self.discriminator = MLP(feature_dim, feature_dim, mlp_width=2*feature_dim)


    def forward(self, x: torch.Tensor, no_relu: bool = True) -> Dict[str, torch.Tensor]:
        results = {}

        if no_relu:
            out0 = self.bn1(self.conv1(x))
            results['layer0'] = out0
            out0 = F.relu(out0)

            out = out0
            for i, sublayer in enumerate(self.layer1):
                sub_norelu = (i == len(self.layer1) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer1'] = out
            out = F.relu(out)

            for i, sublayer in enumerate(self.layer2):
                sub_norelu = (i == len(self.layer2) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer2'] = out
            out = F.relu(out)

            for i, sublayer in enumerate(self.layer3):
                sub_norelu = (i == len(self.layer3) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer3'] = out
            out = F.relu(out)

            for i, sublayer in enumerate(self.layer4):
                sub_norelu = (i == len(self.layer4) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer4'] = out
            out = F.relu(out)
            
        else:
            out0 = self.bn1(self.conv1(x))
            out0 = F.relu(out0)
            results['layer0'] = out0

            out = out0
            out = self.layer1(out)
            results['layer1'] = out
            out = self.layer2(out)
            results['layer2'] = out
            out = self.layer3(out)
            results['layer3'] = out
            out = self.layer4(out)
            results['layer4'] = out
            

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)

        projected = self.discriminator(out)
            
        if self.logit_detach:
            # print("logit detach")
            logit = self.fc(out.detach())
        else:
            logit = self.fc(out)

        results['feature'] = out
        results['logit'] = logit
        results['layer5'] = logit
        results['projected'] = projected

        return results



@ENCODER_REGISTRY.register()
class ResNet18_logit_detach(ResNet_base):
    
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
        self.logit_detach = True

@ENCODER_REGISTRY.register()
class ResNet18_intermediate(ResNet_base):
    
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
        self.num_layers = 10


    def forward(self, x: torch.Tensor, no_relu: bool = True) -> Dict[str, torch.Tensor]:
        results = {}

        if no_relu:
            out0 = self.bn1(self.conv1(x))
            results['layer0'] = out0
            out0 = F.relu(out0)

            out = out0
            k = 1
            for i, sublayer in enumerate(self.layer1):
                # sub_norelu = (i == len(self.layer1) - 1)
                out = sublayer(out, no_relu=True)
                results[f'layer{k}'] = out
                out = F.relu(out)
                k += 1

            # results['layer1'] = out
            # out = F.relu(out)

            for i, sublayer in enumerate(self.layer2):
                # sub_norelu = (i == len(self.layer2) - 1)
                # out = sublayer(out, no_relu=sub_norelu)
                out = sublayer(out, no_relu=True)
                results[f'layer{k}'] = out
                out = F.relu(out)
                k += 1

            # results['layer2'] = out
            # out = F.relu(out)

            for i, sublayer in enumerate(self.layer3):
                # sub_norelu = (i == len(self.layer3) - 1)
                # out = sublayer(out, no_relu=sub_norelu)
                out = sublayer(out, no_relu=True)
                results[f'layer{k}'] = out
                out = F.relu(out)
                k += 1

            # results['layer3'] = out
            # out = F.relu(out)

            for i, sublayer in enumerate(self.layer4):
                # sub_norelu = (i == len(self.layer4) - 1)
                # out = sublayer(out, no_relu=sub_norelu)
                out = sublayer(out, no_relu=True)
                results[f'layer{k}'] = out
                out = F.relu(out)
                k += 1
            # results['layer4'] = out
            # out = F.relu(out)
            
        else:
            out0 = self.bn1(self.conv1(x))
            out0 = F.relu(out0)
            results['layer0'] = out0

            out = out0
            out = self.layer1(out)
            results['layer1'] = out
            out = self.layer2(out)
            results['layer2'] = out
            out = self.layer3(out)
            results['layer3'] = out
            out = self.layer4(out)
            results['layer4'] = out
            

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
            
        logit = self.fc(out)

        results['feature'] = out
        results['logit'] = logit
        results[f'layer{k}'] = logit

        return results



@ENCODER_REGISTRY.register()
class ResNet18_detach(ResNet_base):
    
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
        self.num_layers = 6


    def forward(self, x: torch.Tensor, no_relu: bool = True) -> Dict[str, torch.Tensor]:
        results = {}

        if no_relu:
            out0 = self.bn1(self.conv1(x))
            results['layer0'] = out0
            out0 = F.relu(out0)

            out = out0
            k = 1

            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                out_detach = out.detach()
                for i, sublayer in enumerate(layer):
                    out = sublayer(out, no_relu=True)
                    out_detach = sublayer(out_detach, no_relu=True)

                    if (i == len(layer) - 1):
                        results[f'layer{k}'] = out_detach
                        k += 1

                    out = F.relu(out)
                    out_detach = F.relu(out_detach)
                    

            # for i, sublayer in enumerate(self.layer1):
            #     # sub_norelu = (i == len(self.layer1) - 1)
            #     out = sublayer(out, no_relu=True)
            #     results[f'layer{k}'] = out
            #     out = F.relu(out)
            #     k += 1

            # # results['layer1'] = out
            # # out = F.relu(out)

            # for i, sublayer in enumerate(self.layer2):
            #     # sub_norelu = (i == len(self.layer2) - 1)
            #     # out = sublayer(out, no_relu=sub_norelu)
            #     out = sublayer(out, no_relu=True)
            #     results[f'layer{k}'] = out
            #     out = F.relu(out)
            #     k += 1

            # # results['layer2'] = out
            # # out = F.relu(out)

            # for i, sublayer in enumerate(self.layer3):
            #     # sub_norelu = (i == len(self.layer3) - 1)
            #     # out = sublayer(out, no_relu=sub_norelu)
            #     out = sublayer(out, no_relu=True)
            #     results[f'layer{k}'] = out
            #     out = F.relu(out)
            #     k += 1

            # # results['layer3'] = out
            # # out = F.relu(out)

            # for i, sublayer in enumerate(self.layer4):
            #     # sub_norelu = (i == len(self.layer4) - 1)
            #     # out = sublayer(out, no_relu=sub_norelu)
            #     out = sublayer(out, no_relu=True)
            #     results[f'layer{k}'] = out
            #     out = F.relu(out)
            #     k += 1
            # results['layer4'] = out
            # out = F.relu(out)
            
        else:
            out0 = self.bn1(self.conv1(x))
            out0 = F.relu(out0)
            results['layer0'] = out0

            out = out0
            out = self.layer1(out)
            results['layer1'] = out
            out = self.layer2(out)
            results['layer2'] = out
            out = self.layer3(out)
            results['layer3'] = out
            out = self.layer4(out)
            results['layer4'] = out
            

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
            
        logit = self.fc(out)

        results['feature'] = out
        results['logit'] = logit
        results[f'layer{k}'] = logit

        return results


@ENCODER_REGISTRY.register()
class ResNet18_dense_intermediate(ResNet_base):
    
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
        self.num_layers = 18


    def forward(self, x: torch.Tensor, no_relu: bool = True) -> Dict[str, torch.Tensor]:
        results = {}

        if no_relu:
            out0 = self.bn1(self.conv1(x))
            results['layer0'] = out0
            out0 = F.relu(out0)

            out = out0
            k = 1
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for i, sublayer in enumerate(layer):
                    # sub_norelu = (i == len(self.layer1) - 1)
                    out, out_i = sublayer.forward_intermediate(out, no_relu=True)
                    results[f'layer{k}'] = out_i
                    k += 1
                    results[f'layer{k}'] = out
                    out = F.relu(out)
                    k += 1

            
        else:
            out0 = self.bn1(self.conv1(x))
            out0 = F.relu(out0)
            results['layer0'] = out0

            out = out0
            out = self.layer1(out)
            results['layer1'] = out
            out = self.layer2(out)
            results['layer2'] = out
            out = self.layer3(out)
            results['layer3'] = out
            out = self.layer4(out)
            results['layer4'] = out
            

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
            
        logit = self.fc(out)

        results['feature'] = out
        results['logit'] = logit
        results[f'layer{k}'] = logit

        return results


@ENCODER_REGISTRY.register()
class ResNet34_base(ResNet_base):

    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs
                        #  l2_norm=args.model.l2_norm,
                        #  use_pretrained=args.model.pretrained, use_bn_layer=args.model.use_bn_layer
                         )
        




@ENCODER_REGISTRY.register()
class SubspaceResNet18_base(SubspaceResNet_base):
    def __init__(self, args, num_classes=10):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, l2_norm=args.model.l2_norm,
                         use_pretrained=args.model.pretrained, use_bn_layer=args.model.use_bn_layer)
        

# @ENCODER_REGISTRY.register()
# class ResNet18_base(ResNet):

#     def __init__(self, args, num_classes=10):
#         super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, l2_norm=args.model.l2_norm,
#                          use_pretrained=args.model.pretrained, use_bn_layer=args.model.use_bn_layer)
#         self.inter = False
#         self.fi = None
#         self.num_of_branch = 5


#     def forward_layer(self, layer, x, no_relu=True):
#         #print(x.shape)
#         #print(type(layer))
#         if isinstance(layer,nn.Linear):
#             x = F.adaptive_avg_pool2d(x, 1)
#             x = x.view(x.size(0), -1)
#             out = layer(x)
#         else:
#             if no_relu:
#                 out = x
#                 for sublayer in layer[:-1]:
#                     out = sublayer(out)
#                 out = layer[-1](out, no_relu=no_relu)
#             else:
#                 out = layer(x)

#         return out
    
#     def forward_layer_by_name(self, layer_name, x, no_relu=True):
#         layer = getattr(self,layer_name)
#         return self.forward_layer(layer, x, no_relu)
    
#     def forward_layer0(self, x, no_relu=True):
#         out0 = self.bn1(self.conv1(x))
#         if not no_relu:
#             out0 = F.relu(out0)
#         return out0
    
#     def forward(self, x, no_relu=True):
#         results = {}
#         out0 = self.bn1(self.conv1(x))
#         if no_relu:
#             results['layer0'] = out0
#             out0 = F.relu(out0)
#         else:
#             out0 = F.relu(out0)
#             results['layer0'] = out0
        
#         out1 = self.forward_layer(self.layer1, out0, no_relu=no_relu)
#         results['layer1'] = out1
#         out1 = F.relu(out1) if no_relu else out1

#         out2 = self.forward_layer(self.layer2, out1, no_relu=no_relu)
#         results['layer2'] = out2
#         out2 = F.relu(out2) if no_relu else out2

#         out3 = self.forward_layer(self.layer3, out2, no_relu=no_relu)
#         results['layer3'] = out3
#         out3 = F.relu(out3) if no_relu else out3

#         out4 = self.forward_layer(self.layer4, out3, no_relu=no_relu)
#         results['layer4'] = out4
#         out4 = F.relu(out4) if no_relu else out4

#         out4 = F.adaptive_avg_pool2d(out4, 1)
#         out4 = out4.view(out4.size(0), -1)

#         if self.l2_norm:
#             self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
#             out4 = F.normalize(out4, dim=1)
#             logit = self.fc(out4)
#         else:
#             logit = self.fc(out4)

#         results['logit'] = logit

#         return results

    