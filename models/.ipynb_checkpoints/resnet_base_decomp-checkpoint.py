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

class ResNet_base_decomp(ResNet):

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
    

    def create_backward_hook(self, name = "", one_or_two = True):
        #one_or_two : True => grad of metric feature, False => grad of CE feature



        def decompose_nonzero(grad, diff):
            #breakpoint()
            grad = grad * ((diff > 1e-10) + (diff < -1e-10))
            
            return grad

        def decompose_parallel(grad , diff, weight = 1, eliminate = 'all', clip = -1, unit = 'channel', debug = False):

            #diff is criterion
            if debug:
                breakpoint()

            
            b,c,h,w = grad.shape
            if unit=='channel':
                selected_grad = grad.view(b,c,-1)
                normalized_diff = diff.view(b,c,-1)
                normalized_diff = normalized_diff / (normalized_diff.norm(dim=2, keepdim=True) + 1e-10)
                dotproduct = (selected_grad * normalized_diff).sum(dim=2,keepdim= True)
                #dotproduct = dotproduct.view(b,c,1,1)
            elif unit== 'feature':
                selected_grad = grad.view(b,-1)
                normalized_diff = diff.view(b,-1)
                normalized_diff = normalized_diff / (normalized_diff.norm(dim=1, keepdim=True) + 1e-10)
                dotproduct = (selected_grad * normalized_diff).sum(dim=1,keepdim= True)
                #xxdotproduct = dotproduct.view(b,1,1,1)              
                
                
        
            if eliminate == 'positive':
                dotproduct = dotproduct * (dotproduct>0)
            elif eliminate == 'negative':
                dotproduct = dotproduct * (dotproduct<0)
            elif eliminate == 'all':
                pass
            else:
                raise ValueError
            # if clip > 0:
            #     dotproduct -= dotproduct.clamp(max = clip)

            parallel = dotproduct * weight * normalized_diff
            parallel = parallel.view(grad.shape)
            # print(grad[0][0])
            # print(diff[0][0])
            # print(parallel[0][0])
            # print(grad[0][0] - parallel[0][0])
            # print((diff[0][0] *grad[0][0]).sum())
            # print((diff[0][0] *(grad[0][0] - parallel[0][0])).sum())

            # breakpoint()
            return parallel

        
        def decomp_back_hook(grad):
            my_one_or_two = one_or_two
            r1, r2 = self.results1_grad, self.results2_grad
            myname = name


            #Gradient saving (instead of all retain_graph = True)
            if one_or_two:
                assert(r2[myname] is not None)
                #one_or_two : True => grad of metric feature, False => grad of CE feature


                # neg_paralle + vertical
                #grad -= decompose_parallel(grad, r2[myname],unit='channel',eliminate = 'positive')


                # save_only_zero
                #grad -= decompose_nonzero(grad, r2[myname])

                #pos_parallel only
                #grad = decompose_parallel(grad, r2[myname],unit='channel',eliminate = 'positive')

                #neg_parallel only
                #grad = decompose_parallel(grad, r2[myname],unit='channel',eliminate = 'negative')

                #all parallel
                grad = decompose_parallel(grad, r2[myname],unit='channel',eliminate = 'all')

            else:
                r2[myname] = grad

            
            return grad
        return decomp_back_hook


    def forward(self, x: torch.Tensor, no_relu: bool = True) -> Dict[str, torch.Tensor]:
        results = {}

        if no_relu:
            out0_ = self.bn1(self.conv1(x))
            results['layer0'] = out0_.clone()
            out0 = out0_.clone()
            out1_ = F.relu(out0)



            #out = out0
            for i, sublayer in enumerate(self.layer1):
                sub_norelu = (i == len(self.layer1) - 1)
                out1_ = sublayer(out1_, no_relu=sub_norelu)
            results['layer1'] = out1_.clone()
            out1 = out1_.clone()
            out2_ = F.relu(out1)


            for i, sublayer in enumerate(self.layer2):
                sub_norelu = (i == len(self.layer2) - 1)
                out2_ = sublayer(out2_, no_relu=sub_norelu)
            results['layer2'] = out2_.clone()
            out2 = out2_.clone()
            out3_ = F.relu(out2)

            for i, sublayer in enumerate(self.layer3):
                sub_norelu = (i == len(self.layer3) - 1)
                out3_ = sublayer(out3_, no_relu=sub_norelu)
            results['layer3'] = out3_.clone()
            out3 = out3_.clone()
            out4_ = F.relu(out3)

            for i, sublayer in enumerate(self.layer4):
                sub_norelu = (i == len(self.layer4) - 1)
                out4_ = sublayer(out4_, no_relu=sub_norelu)
            results['layer4'] = out4_.clone()
            out4 = out4_.clone()
            out5_ = F.relu(out4)


            #print(results['layer4'].requires_grad)

            
            ########## register hook
            if self.train and results['layer4'].requires_grad:
                self.results1_grad = {}
                self.results2_grad = {}
                self.hook1 = {}
                self.hook2 = {}
                for key, outn in zip(results.keys(), [out0, out1, out2, out3, out4]):
                    self.results1_grad[key] = None
                    self.results2_grad[key] = None
                    self.hook1[key] = results[key].register_hook(self.create_backward_hook(name= key, one_or_two=True))
                    self.hook2[key] = outn.register_hook(self.create_backward_hook(name= key, one_or_two = False))
                    
                #out3_.register_hook(self.create_backward_hook(name= 'out3_', one_or_two = False))

                # out0.register_hook(self.create_backward_hook(name= 'out0', crit_feat = results['layer0']))
                # out1.register_hook(self.create_backward_hook(name= 'out1', crit_feat = results['layer1']))
                # out2.register_hook(self.create_backward_hook(name= 'out2', crit_feat = results['layer2']))
                # out3.register_hook(self.create_backward_hook(name= 'out3', crit_feat = results['layer3']))
                # out4.register_hook(self.create_backward_hook(name= 'out4', crit_feat = results['layer4']))

    


            ###########################
            
        else:
            pass
            # out0 = self.bn1(self.conv1(x))
            # out0 = F.relu(out0)
            # results['layer0'] = out0

            # out = out0
            # out = self.layer1(out)
            # results['layer1'] = out
            # out = self.layer2(out)
            # results['layer2'] = out
            # out = self.layer3(out)
            # results['layer3'] = out
            # out = self.layer4(out)
            # results['layer4'] = out
            

        out5_ = F.adaptive_avg_pool2d(out5_, 1)
        out5_ = out5_.view(out5_.size(0), -1)

        # if self.l2_norm:
        #     self.fc.weight.data = F.normalize(self.fc.weight.data, p=2., dim=1)
        #     out4 = F.normalize(out4, dim=1)
        #     logit = self.fc(out4)
            
        logit = self.fc(out5_.clone())

        results['feature'] = out5_.clone()
        results['logit'] = logit.clone()
        results['layer5'] = logit.clone()

        # for key in results.keys():
        #     print(key, results[key].var())

        return results


# class SubspaceResNet_base(ResNet_base):

#     def get_conv(self):
#         return SubspaceConv2d
    
#     def get_linear(self):
#         return nn.Linear

    

@ENCODER_REGISTRY.register()
class ResNet18_base_decomp(ResNet_base_decomp):
    
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs
                        #  l2_norm=args.model.l2_norm,
                        #  use_pretrained=args.model.pretrained, use_bn_layer=args.model.use_bn_layer
                         )
        

# @ENCODER_REGISTRY.register()
# class ResNet34_base(ResNet_base):

#     def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
#         super().__init__(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs
#                         #  l2_norm=args.model.l2_norm,
#                         #  use_pretrained=args.model.pretrained, use_bn_layer=args.model.use_bn_layer
#                          )
        




# @ENCODER_REGISTRY.register()
# class SubspaceResNet18_base(SubspaceResNet_base):
#     def __init__(self, args, num_classes=10):
#         super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, l2_norm=args.model.l2_norm,
#                          use_pretrained=args.model.pretrained, use_bn_layer=args.model.use_bn_layer)
        

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

    