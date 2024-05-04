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

    def forward(self, x, return_feature_norelu = False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        if return_feature_norelu:
            return out
        else:
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

    def forward(self, x, return_feature_norelu = False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        if return_feature_norelu:
            return out
        else:
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

        self.inter = False
        self.fi = None
        self.num_of_branch = 5





    def _make_layer(self, block, planes, num_blocks, stride,use_bn_layer = False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_bn_layer =use_bn_layer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    

    def layer_forward_relu(self, layer, x, return_feature_norelu = False):
        out = x
        if return_feature_norelu:
            for layer_el in layer[:-1]:
                out  = layer_el(out)
            out = layer[-1](out, return_feature_norelu = return_feature_norelu)
        else:
            out = layer(out)

        return out
    
    
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

    def get_prev_model(self):

        self.prev_conv1 = copy.deepcopy(self.conv1)

        self.prev_bn1 = copy.deepcopy(self.bn1)

        self.prev_layer1 = copy.deepcopy(self.layer1)

        self.prev_layer2 = copy.deepcopy(self.layer2)

        self.prev_layer3 = copy.deepcopy(self.layer3)

        self.prev_layer4 = copy.deepcopy(self.layer4)

        self.prev_fc = copy.deepcopy(self.fc)

        for n, p in self.named_parameters():
            if 'prev' in n:
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

    def layerwise_normalize(self, fisher):
        fisher = copy.deepcopy(fisher)
        for key in fisher.keys():
            fisher[key] = (fisher[key] - fisher[key].min()) / (fisher[key].max() - fisher[key].min())
        return fisher

    def normalize_m(self, fisher):
        min_value = 100000000
        max_value = 0
        fisher = copy.deepcopy(fisher)
        for key in fisher.keys():
            mi = fisher[key].min()
            ma = fisher[key].max()
            if mi < min_value:
                min_value = mi
            if ma > max_value:
                max_value = ma
        for key in fisher.keys():
            fisher[key] = (fisher[key] - min_value) / (max_value - min_value)
        return fisher


    def update_interpolation_model(self,low = 0, high = 0, interpolation_type = None, interpolation_model='current',
                                   device='cpu', p=0.1):#, weight = 0):
        #breakpoint()
        if interpolation_type == 'deterministic':
            weight = high
        elif interpolation_type == 'stochastic':
            weight = torch.FloatTensor(1).uniform_(low, high).item()
        elif interpolation_type == 'stochastic_layerwise':
            pass
        elif interpolation_type == 'deter_fisher_select_top':
            for key in self.fi.keys():
                #import pdb; pdb.set_trace()
                shape = self.fi[key].size()
                fi = self.fi[key].view(-1)
                weight = torch.zeros_like(fi)
                num_selected = int(fi.size()[0] * p)
                vals, indices = fi.topk(num_selected)
                weight[indices] = -1
                self.fi[key] = weight.view(shape)
            pass
        elif interpolation_type == 'deter_fisher_2':
            self.fi = self.layerwise_normalize(self.fi)
            for key in self.fi.keys():
                self.fi[key] = - torch.pow(self.fi[key], 2)
        elif interpolation_type == 'deter_fisher_5':
            self.fi = self.layerwise_normalize(self.fi)
            for key in self.fi.keys():
                self.fi[key] = - torch.pow(self.fi[key], 5)
        elif interpolation_type == 'deter_fisher':
            self.fi = self.layerwise_normalize(self.fi)
            for key in self.fi.keys():
                self.fi[key] = - self.fi[key]
        elif interpolation_type == 'stoc_fisher':
            self.fi = self.layerwise_normalize(self.fi)
            for key in self.fi.keys():
                low = torch.zeros_like(self.fi[key]).detach().cpu().numpy()
                high = self.fi[key].detach().cpu().numpy()
                weight = torch.from_numpy(np.random.uniform(low, high)).to(device)
                self.fi[key] = - weight
        elif interpolation_type == 'deter_fisher_rev':
            for key in self.fi.keys():
                self.fi[key] =  - 1 + self.fi[key]
        elif interpolation_type == 'deter_fisher_norm_entire_max':
            fi = copy.deepcopy(self.fi)
            self.fi = self.normalize_m(fi)
            for key in self.fi.keys():
                self.fi[key] = - self.fi[key]
            pass
        else: 
            raise Exception("Not valid stochastic weight interpolation mode") 
        
        with torch.no_grad():
            this_dict = copy.deepcopy(self.state_dict())
            for key in self.state_dict().keys():
                if 'inter' in key:
                    if interpolation_type == "stochastic_layerwise":
                        weight = torch.FloatTensor(1).uniform_(low, high).item()
                    this_layer_name = key[6:]
                    this_global_name = 'global_' + this_layer_name
                    this_prev_name = 'prev_' + this_layer_name
                    if 'fisher' in interpolation_type:
                        if interpolation_model == 'current':
                            this_dict[key] = this_dict[this_global_name] + self.fi[this_layer_name] * (
                                        this_dict[this_layer_name] - this_dict[this_global_name])
                        elif interpolation_model == 'prev':
                            this_dict[key] = this_dict[this_global_name] + self.fi[this_prev_name] * (
                                    this_dict[this_prev_name] - this_dict[this_global_name])
                    else:
                        if interpolation_model == 'current':
                            this_dict[key] = this_dict[this_global_name] + weight*(this_dict[this_layer_name] - this_dict[this_global_name])
                        elif interpolation_model == 'prev':
                            this_dict[key] = this_dict[this_global_name] + weight * (
                                    this_dict[this_prev_name] - this_dict[this_global_name])
        self.load_state_dict(this_dict)

    def forward_layer(self, layer, layerinput, layer_name: str = None):
        out = layer(layerinput)
        # if layer4 == True:
        if layer_name == 'layer4':
            out = F.adaptive_avg_pool2d(out, 1)
            out = out.view(out.size(0), -1)
        return out
    



    def forward_stoc(self, x):
        out = x
        for layer_name in ['layer1','layer2','layer3','layer4', 'fc']:
            layers = self.get_layers(layer_name)
            branch = torch.randint(len(layers), (1,))[0]
            out = self.forward_layer(layers[branch], out, layer_name=layer_name)
        return out
    
    def forward_stoc_with_dict(self, x, layer_dict):
        # layer_dict = {layer_name: ['l', 'g'] for layer_name in ['layer1', 'layer2', 'layer3', 'layer4', 'fc']}
        out = x
        for layer_name in ['layer1','layer2','layer3','layer4', 'fc']:
            # layers = self.get_layers(layer_name)
            branches = self._get_layer_branches(layer_name, layer_dict[layer_name])
            k = torch.randint(len(branches), (1,))[0]
            out = self.forward_layer(branches[k], out, layer_name=layer_name)
        return out
    

    def forward_stoc_with_dict_includefirst(self, x, layer_dict):
        # layer_dict = {layer_name: ['l', 'g'] for layer_name in ['layer1', 'layer2', 'layer3', 'layer4', 'fc']}
        out = x
        for layer_name in ['conv1','bn1','layer1','layer2','layer3','layer4','fc']:
            # layers = self.get_layers(layer_name)
            branches = self._get_layer_branches(layer_name, layer_dict[layer_name])
            k = torch.randint(len(branches), (1,))[0]
            out = self.forward_layer(branches[k], out, layer_name=layer_name)
        return out


    def forward_stoc_inter_local_only(self, x):
        out = x
        for layer_name in ['layer1','layer2','layer3','layer4', 'fc']:
            layers = self.get_layers_inter_local_only(layer_name)
            branch = torch.randint(len(layers), (1,))[0]
            out = self.forward_layer(layers[branch], out, layer_name=layer_name)
        return out

    def forward_stoc_global_local_only(self, x):
        out = x
        for layer_name in ['layer1','layer2','layer3','layer4', 'fc']:
            layers = self.get_layers_global_local_only(layer_name)
            branch = torch.randint(len(layers), (1,))[0]
            out = self.forward_layer(layers[branch], out, layer_name=layer_name)
        return out


    def forward_stoc_globallocalinter_selectedlayer_globallocal_else(self, x, selected_level = None):
        if type(selected_level) == int:
            raise Exception("Not valid selected_level type")
        out = x
        for idx, layer_name in enumerate(['layer1','layer2','layer3','layer4', 'fc']):
            if idx == selected_level:
                layers = self.get_layers(layer_name)
            else:
                layers = self.get_layers_global_local_only(layer_name)
            branch = torch.randint(len(layers), (1,))[0]
            out = self.forward_layer(layers[branch], out, layer_name=layer_name)
        return out


    def forward_local(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = x
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        if self.l2_norm:
            self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
            out = F.normalize(out, dim=1)
            logit = self.fc(out)
        else:
            logit = self.fc(out)
        
        return logit

    def _get_layer_branches(self, layer_name, branch_types): # layer_names = ['l', 'g', 'i']
        branches = []
        if 'l' in branch_types:
            branches.append(getattr(self, layer_name))
        if 'g' in branch_types:
            branches.append(getattr(self, 'global_'+layer_name))
        if 'i' in branch_types:
            branches.append(getattr(self, 'inter_'+layer_name))
        return branches


    def get_layers(self, layername):
        local_layer = getattr(self, layername)
        global_layer = getattr(self, 'global_'+layername)
        inter_layer = getattr(self, 'inter_'+layername)
        return [local_layer, global_layer, inter_layer]

    def get_layers_inter_local_only(self, layername):
        local_layer = getattr(self, layername)
        #global_layer = getattr(self, 'global_'+layername)
        inter_layer = getattr(self, 'inter_'+layername)
        return [local_layer, inter_layer] 
    

    def get_layers_global_local_only(self, layername):
        local_layer = getattr(self, layername)
        global_layer = getattr(self, 'global_'+layername)
        #inter_layer = getattr(self, 'inter_'+layername)
        return [local_layer, global_layer] 

    def mlb_forward(self, x, return_feature=False, selected_layer='', selected_layers=[], stochastic="deterministic", num_of_stochastic_branch=1, **kwargs):
        if stochastic == "deterministic":    
            if return_feature:

                out0 = F.relu(self.bn1(self.conv1(x)))
                

                out1_i = self.inter_layer1(out0)
                out1_g = self.global_layer1(out0)
                out1 = self.layer1(out0)
                
                #print("before concat",out1.shape)
                #print("after concat",torch.cat((out1_g,out1),dim=0).shape)
                out2_i = self.inter_layer2(torch.cat((out1_i,out1_g,out1),dim=0))
                out2_g = self.global_layer2(torch.cat((out1_i,out1_g,out1),dim=0))
                out2 = self.layer2(torch.cat((out1_i,out1_g,out1),dim=0))


                out3_i = self.inter_layer3(torch.cat((out2_i,out2_g,out2),dim=0))
                out3_g = self.global_layer3(torch.cat((out2_i,out2_g,out2),dim=0))
                out3 = self.layer3(torch.cat((out2_i,out2_g,out2),dim=0))



                out4_i = self.inter_layer4(torch.cat((out3_i,out3_g,out3),dim=0))
                out4_i = F.adaptive_avg_pool2d(out4_i, 1)
                out4_i = out4_i.view(out4_i.size(0), -1)


                out4_g = self.global_layer4(torch.cat((out3_i,out3_g,out3),dim=0))
                out4_g = F.adaptive_avg_pool2d(out4_g, 1)
                out4_g = out4_g.view(out4_g.size(0), -1)

                out4 = self.layer4(torch.cat((out3_i,out3_g,out3),dim=0))
                out4 = F.adaptive_avg_pool2d(out4, 1)
                out4 = out4.view(out4.size(0), -1)

                #print("out4_g : ",out4_g.shape)
                #print("out4 : ",out4.shape)
                logit_i = self.inter_fc(torch.cat((out4_i,out4_g,out4),dim=0))
                logit_g = self.global_fc(torch.cat((out4_i,out4_g,out4),dim=0))
                logit = self.fc(torch.cat((out4_i,out4_g,out4),dim=0))

                return torch.cat((logit_i,logit_g,logit),dim=0)
            
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
                  
        elif stochastic == "stochastic":    
            if return_feature:
                #Always forward the first local layer
                out0 = F.relu(self.bn1(self.conv1(x)))

                M = num_of_stochastic_branch
                outs = []
                for m in range(M):
                    outs.append(self.forward_stoc(out0))
                
                
                outs.append(self.forward_local(out0))
                #breakpoint()
                return torch.cat(outs, dim=0)
        elif stochastic == "stochastic_inter_local_only":    
            if return_feature:
                #Always forward the first local layer
                out0 = F.relu(self.bn1(self.conv1(x)))

                M = num_of_stochastic_branch
                outs = []
                for m in range(M):
                    outs.append(self.forward_stoc_inter_local_only(out0))
                
                
                outs.append(self.forward_local(out0))
                #breakpoint()
                return torch.cat(outs, dim=0)   


        elif stochastic == "stochastic_global_local_only":    
            if return_feature:
                #Always forward the first local layer
                out0 = F.relu(self.bn1(self.conv1(x)))

                M = num_of_stochastic_branch
                outs = []
                for m in range(M):
                    outs.append(self.forward_stoc_global_local_only(out0))
                
                
                outs.append(self.forward_local(out0))
                #breakpoint()
                return torch.cat(outs, dim=0)  


        elif stochastic == "stochastic_layer_ablation":  
            #breakpoint() 
            layers_list = ['layer1', 'layer2', 'layer3', 'layer4', 'fc']
            assert selected_layer != ""
            assert selected_layer in layers_list

            if return_feature:
                #Always forward the first local layer
                out0 = F.relu(self.bn1(self.conv1(x)))

                # selected_level = 'layer2'

                layer_dict = {layer_name: ['l', 'g'] for layer_name in layers_list}#['layer1', 'layer2', 'layer3', 'layer4', 'fc']}
                layer_dict[selected_layer] = ['l', 'g', 'i']

                #breakpoint()

                M = num_of_stochastic_branch
                outs = []
                for m in range(M):
                    outs.append(self.forward_stoc_with_dict(out0, layer_dict=layer_dict))
                    # outs.append(self.forward_stoc_globallocalinter_selectedlayer_globallocal_else(out0, selected_level= level))
                
                #breakpoint()
                
                outs.append(self.forward_local(out0))
                #breakpoint()
                return torch.cat(outs, dim=0)                
        elif stochastic == "stochastic_layers_ablation":  
            #breakpoint() 
            layers_list = ['layer1', 'layer2', 'layer3', 'layer4', 'fc']
            assert selected_layers != []#""
            assert len(selected_layers) > 0
            assert type(selected_layers) == type([])
            #assert selected_layer in layers_list

            if return_feature:
                #Always forward the first local layer
                out0 = F.relu(self.bn1(self.conv1(x)))

                # selected_level = 'layer2'
  
                layer_dict = {layer_name: ['l', 'g'] for layer_name in layers_list}#['layer1', 'layer2', 'layer3', 'layer4', 'fc']}
                for sl in selected_layers:
                    layer_dict[sl] = ['l', 'g', 'i']
                #layer_dict[selected_layer] = ['l', 'g', 'i']

                #breakpoint()

                M = num_of_stochastic_branch
                outs = []
                for m in range(M):
                    outs.append(self.forward_stoc_with_dict(out0, layer_dict=layer_dict))
                    # outs.append(self.forward_stoc_globallocalinter_selectedlayer_globallocal_else(out0, selected_level= level))
                
                #breakpoint()
                
                outs.append(self.forward_local(out0))
                #breakpoint()
                return torch.cat(outs, dim=0)                  
        elif stochastic == "stochastic_ultimate":  
            #breakpoint() 
            layers_list = ['conv1','bn1','layer1', 'layer2', 'layer3', 'layer4', 'fc']
            assert selected_layers != []#""
            assert len(selected_layers) > 0
            assert type(selected_layers) == type([])
            #assert selected_layer in layers_list

            if return_feature:
                #Always forward the first local layer
                
                # selected_level = 'layer2'

                layer_dict = {layer_name: ['l', 'g'] for layer_name in layers_list}#['layer1', 'layer2', 'layer3', 'layer4', 'fc']}
                for sl in selected_layers:
                    layer_dict[sl] = ['l', 'g', 'i']
                #layer_dict[selected_layer] = ['l', 'g', 'i']

                #breakpoint()

                M = num_of_stochastic_branch
                outs = []
                for m in range(M):
                    outs.append(self.forward_stoc_with_dict_includefirst(x, layer_dict=layer_dict))
                    # outs.append(self.forward_stoc_globallocalinter_selectedlayer_globallocal_else(out0, selected_level= level))
                
                #breakpoint()
                out0 = F.relu(self.bn1(self.conv1(x)))

                outs.append(self.forward_local(out0))
                #breakpoint()
                return torch.cat(outs, dim=0)  

        elif stochastic == "interteacher":  
                    #breakpoint() 
                    layers_list = ['conv1','bn1','layer1', 'layer2', 'layer3', 'layer4', 'fc']
                    assert selected_layers != []#""
                    assert len(selected_layers) > 0
                    assert type(selected_layers) == type([])
                    #assert selected_layer in layers_list

                    if return_feature:
                        #Always forward the first local layer
                        
                        # selected_level = 'layer2'

                        layer_dict = {layer_name: ['l', 'g'] for layer_name in layers_list}#['layer1', 'layer2', 'layer3', 'layer4', 'fc']}
                        for sl in selected_layers:
                            layer_dict[sl] = ['i']
                        #layer_dict[selected_layer] = ['l', 'g', 'i']

                        #breakpoint()

                        M = num_of_stochastic_branch
                        outs = []
                        for m in range(M):
                            outs.append(self.forward_stoc_with_dict_includefirst(x, layer_dict=layer_dict))
                            # outs.append(self.forward_stoc_globallocalinter_selectedlayer_globallocal_else(out0, selected_level= level))
                        
                        #breakpoint()
                        out0 = F.relu(self.bn1(self.conv1(x)))

                        outs.append(self.forward_local(out0))
                        #breakpoint()
                        return torch.cat(outs, dim=0)  







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
    
        elif stochastic == 'None':
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            # out = F.avg_pool2d(out, 4)
            out = F.adaptive_avg_pool2d(out, 1)
            out = out.view(out.size(0), -1)
            if self.l2_norm:
                # with torch.no_grad():
                # w = self.linear.weight.data.clone()
                # w = F.normalize(w, dim=1, p=2)
                # self.linear.weight.copy_(w)
                # self.linear = F.normalize(self.linear)
                self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
                out = F.normalize(out, dim=1)
                logit = self.fc(out)
            else:
                logit = self.fc(out)

            return logit

        else:
            raise Exception("Not valid stochastic mode")
        


    def inter_forward(self, x, return_feature=False,level = 0):
        if level <= 0:
            out0 = F.relu(self.inter_bn1(self.inter_conv1(x)))
        else:
            out0 = x
        if level <=1:
            out1 = self.inter_layer1(out0)
        else:
            out1 = out0
        if level <=2:
            out2 = self.inter_layer2(out1)
        else:
            out2 = out1
        if level <=3:
            out3 = self.inter_layer3(out2)
        else:
            out3 = out2
        if level <=4:
            out4 = self.inter_layer4(out3)
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
            self.inter_fc.weight.data = F.normalize(self.inter_fc.weight.data, p=2, dim=1)
            out4 = F.normalize(out4, dim=1)
            logit = self.inter_fc(out4)
        else:
            logit = self.inter_fc(out4)
            
        if return_feature==True:
            return out0,out1,out2,out3,out4,logit
        else:
            return logit  
        

    def metric_forward_feature_reluX(self, x, return_feature=False,level = 0, return_feature_norelu = False):

        if level <= 0:
            if return_feature_norelu:
                out0 = (self.bn1(self.conv1(x)))
            else:
                out0 = F.relu(self.bn1(self.conv1(x)))
        else:
            out0 = x

        #breakpoint()
        if level <=1:
            out1 = self.layer_forward_relu(self.layer1, F.relu(out0) if return_feature_norelu else out0, return_feature_norelu =return_feature_norelu)
        else:
            out1 = out0
        if level <=2:
            out2 = self.layer_forward_relu(self.layer2, F.relu(out1) if return_feature_norelu else out1, return_feature_norelu =return_feature_norelu)
        else:
            out2 = out1
        if level <=3:
            out3 = self.layer_forward_relu(self.layer3, F.relu(out2) if return_feature_norelu else out2, return_feature_norelu =return_feature_norelu)
        else:
            out3 = out2

        if level <=4:
            out4 = self.layer_forward_relu(self.layer4, F.relu(out3) if return_feature_norelu else out3, return_feature_norelu =return_feature_norelu)
            #breakpoint()
            if return_feature_norelu:
                relu_out4 = F.relu(out4)
                relu_out4 = F.adaptive_avg_pool2d(relu_out4, 1)
                relu_out4 = relu_out4.view(relu_out4.size(0), -1) 


                out4 = F.adaptive_avg_pool2d(out4, 1)
                out4 = out4.view(out4.size(0), -1)

                fc_input = relu_out4
            else:
                out4 = F.adaptive_avg_pool2d(out4, 1)
                out4 = out4.view(out4.size(0), -1)
                fc_input = out4
        else:
            out4 = out3
            fc_input = out4

        
        if self.l2_norm:

            self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
            fc_input = F.normalize(fc_input, dim=1)
            logit = self.fc(fc_input)
        else:
            logit = self.fc(fc_input)
            
        if return_feature==True:
            return out0,out1,out2,out3,out4,logit
        else:
            return logit        

    def metric_forward(self, x, return_feature=False,level = 0):
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
            self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
            out4 = F.normalize(out4, dim=1)
            logit = self.fc(out4)
        else:
            logit = self.fc(out4)
            
        if return_feature==True:
            return out0,out1,out2,out3,out4,logit
        else:
            return logit  

    def basic_forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        if self.l2_norm:
            # with torch.no_grad():
            # w = self.linear.weight.data.clone()
            # w = F.normalize(w, dim=1, p=2)
            # self.linear.weight.copy_(w)
            # self.linear = F.normalize(self.linear)
            self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
            out = F.normalize(out, dim=1)
            logit = self.fc(out)
        else:
            logit = self.fc(out)

        return logit


    def forward(self, x, return_feature=False, level = 0, stochastic = "deterministic", num_of_stochastic_branch = 1, do_mlb = False, return_feature_norelu = False , selected_layer = "" , selected_layers = []):
        #breakpoint()
        #(self, x, return_feature=False, selected_layer='', stochastic="deterministic", num_of_stochastic_branch=1, **kwargs)
        if do_mlb:
            result = self.mlb_forward(x = x, return_feature = return_feature, stochastic = stochastic , num_of_stochastic_branch = num_of_stochastic_branch, selected_layer = selected_layer, selected_layers = selected_layers)
        else:
            if return_feature_norelu == True:
                result = self.metric_forward_feature_reluX(x, return_feature, level = level, return_feature_norelu = return_feature_norelu)
            else:
                result = self.metric_forward(x, return_feature, level = level)
            
        return result
            
            
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


def ResNet18_GFLN(num_classes=10, l2_norm=False, use_pretrained = False, transfer_learning = True, use_bn = False, use_pre_fc = False, use_bn_layer = False):
    #
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

        
            
        
        


# def ResNet34(num_classes=10, l2_norm=False):
#     return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, l2_norm=l2_norm)
#
#
# def ResNet50(num_classes=10, l2_norm=False):
#     return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, l2_norm=l2_norm)
#
#
# def ResNet101(num_classes=10, l2_norm=False):
#     return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, l2_norm=l2_norm)
#
#
# def ResNet152(num_classes=10, l2_norm=False):
#     return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, l2_norm=l2_norm)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
