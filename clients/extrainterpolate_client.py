#!/usr/bin/env python
# coding: utf-8
import copy
import time

import matplotlib.pyplot as plt
import torch.multiprocessing as mp

from utils import *
from utils.metrics import evaluate
from utils.visualize import __log_test_metric__, umap_allmodels, cka_allmodels, log_fisher_diag
from models import build_encoder, get_model
from typing import Callable, Dict, Tuple, Union, List
from utils.logging_utils import AverageMeter
from torch.cuda.amp import autocast, GradScaler

import logging
logger = logging.getLogger(__name__)

from clients.build import CLIENT_REGISTRY
from clients import Client

from utils import LossManager
from utils.train_utils import apply_label_noise
from omegaconf import DictConfig

from clients.interpolate_client import Interpolater, InterpolateClient


class ExtraInterpolater(Interpolater):

    def __init__(self,
                 local_model: nn.Module,
                 global_model: nn.Module,
                #  interpolation_type: str,
                 args: DictConfig):
        
        self.args = args
        self.local_model = local_model
        self.global_model = global_model
        self.inter_model = copy.deepcopy(global_model)
        self.inter_args = self.args.client.interpolation

        # for model in [self.global_model, self.inter_model]:
        for model in [self.global_model]:
            for n, p in model.named_parameters():
                p.requires_grad = False
        
        return

    def to(self, device):
        self.local_model.to(device)
        self.global_model.to(device)
        self.inter_model.to(device)
    
    def get_models(self, branch) -> List:
        models = []
        if 'l' in branch:
            models.append(self.local_model)
        if 'g' in branch:
            models.append(self.global_model)
        if 'i' in branch:
            models.append(self.inter_model)

        return models


    def get_interpolate_model(self, factor, stochastic=False):
        # Note: it will change self.inter_model
        inter_state_dict = self.inter_model.state_dict()
        global_state_dict = self.global_model.state_dict()
        local_state_dict = self.local_model.state_dict()

        factors = torch.Tensor([1, 0, factor])
        branch_index = torch.randint(3, (5,))
        branch_factors = factors[branch_index]

        for key in inter_state_dict.keys():
            if stochastic:
                inter_factor = branch_factors[0] if 'layer1' in key else branch_factors[1] if 'layer2' in key else branch_factors[2] if 'layer3' in key else branch_factors[3] if 'layer4' in key else branch_factors[4] if 'fc' in key else 1
            else:
                inter_factor = factor 

            # breakpoint()
            # inter_factor.to(self.device) 
            inter_state_dict[key] = global_state_dict[key] + inter_factor * (local_state_dict[key] - global_state_dict[key])
                        
        self.inter_model.load_state_dict(inter_state_dict)
        # stoc_inter_results = self.trainer.evaler.eval(model=inter_model, epoch=global_epoch)
        return self.inter_model
    


    def update(self):
        # How to construct the interpolated model

        with torch.no_grad():
            inter_state_dict = self.inter_model.state_dict()
            global_state_dict = self.global_model.state_dict()
            local_state_dict = self.local_model.state_dict()

            if self.inter_args.type == "noise":
                factor = torch.FloatTensor(1).uniform_(self.inter_args.low, self.inter_args.high).item()

                for key in inter_state_dict.keys():
                    diff = (local_state_dict[key] - global_state_dict[key]).detach()
                    norm = torch.norm(diff)
                    noise = torch.rand(diff.size()).to(diff.device)
                    noise -= noise.mean()
                    noise *= norm/torch.norm(noise)
                    
                    # print(norm)
                    inter_state_dict[key] = global_state_dict[key] + factor * noise

            elif self.inter_args.type == "noise_negative":
                # factor = torch.FloatTensor(1).uniform_(self.inter_args.low, self.inter_args.high).item()

                for key in inter_state_dict.keys():
                    diff = (local_state_dict[key] - global_state_dict[key]).detach()

                    norm = torch.norm(diff)
                    while True:
                        noise = torch.rand(diff.size()).to(diff.device)
                        noise -= noise.mean()
                        cosine = torch.dot(diff.reshape(-1), noise.reshape(-1))
                        # breakpoint()
                        if cosine <= self.inter_args.high:
                            break

                    # breakpoint()
                    # print(cosine)
                    noise *= norm/torch.norm(noise)
                    # print(norm)
                    inter_state_dict[key] = global_state_dict[key] + noise

            elif self.inter_args.type == "stochastic_noise":
                factor = torch.FloatTensor(1).uniform_(self.inter_args.low, self.inter_args.high).item()
                # noise = torch.rand(global_state_dict.size()).to(diff.device)
                for key in inter_state_dict.keys():
                    noise = torch.rand(global_state_dict[key].size()).to(global_state_dict[key].device)
                    noise -= noise.mean()
                    diff = (local_state_dict[key] - global_state_dict[key]).detach()
                    norm = torch.norm(diff)
                    noise *= norm/torch.norm(noise)
                    # breakpoint()
                    inter_state_dict[key] = global_state_dict[key] + factor * (local_state_dict[key] - global_state_dict[key]) + noise

            elif self.inter_args.type == "stochastic_param":
                for key in inter_state_dict.keys():
                    factor = torch.FloatTensor(1).uniform_(self.inter_args.low, self.inter_args.high).item()    
                    inter_state_dict[key] = global_state_dict[key] + factor * (local_state_dict[key] - global_state_dict[key])

            else:
                if self.inter_args.type == "stochastic":
                    factor = torch.FloatTensor(1).uniform_(self.inter_args.low, self.inter_args.high).item()
                elif self.inter_args.type == "fixed":
                    factor = self.inter_args.factor
                elif self.inter_args.type == "distribution":
                    # factor = -0.1 * self.class_counts[0] / self.class_counts.mean()
                    factor = -0.2 * self.class_counts[0] / self.class_counts.mean()
                
                # print("factor : ", factor)
                for key in inter_state_dict.keys():
                    inter_state_dict[key] = global_state_dict[key] + factor * (local_state_dict[key] - global_state_dict[key])
                            
            self.inter_model.load_state_dict(inter_state_dict)

        return



    def forward_stoc_layerwise(self, x: torch.Tensor, repeat: int, no_relu: bool, ):
        
        initial_out = self.local_model.forward_layer0(x, no_relu=False)

        outs = []
        for m in range(repeat):
            out = initial_out

            # indices = ''
            for layer_name in ['layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                if self.inter_args.get('branches'):
                    branch = self.inter_args.branches.get(layer_name)
                else:
                    branch = self.inter_args.branch

                branch_probs = torch.ones(len(branch))
                if self.inter_args.get('branch_probs') and len(self.inter_args.get('branch_probs')) == len(branch):
                    branch_probs = torch.Tensor(self.inter_args.get('branch_probs'))

                branch_probs /= branch_probs.sum()

                models = self.get_models(branch=branch)

                branch_dist = torch.distributions.categorical.Categorical(probs=branch_probs)

                branch_index = branch_dist.sample()
                # indices += str(branch_index)
                out = models[branch_index].forward_layer_by_name(layer_name, x=out, no_relu=no_relu)

            # print(indices)
            outs.append(out)

        return outs
    

    def forward_stoc(self, x: torch.Tensor, repeat: int, no_relu: bool, ):
        
        initial_out = self.local_model.forward_layer0(x, no_relu=False)

        outs = []
        for m in range(repeat):
            out = initial_out

            # indices = ''
            results = {}
            results["layer0"] = out
            for layer_name in ['layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                if self.inter_args.get('branches'):
                    branch = self.inter_args.branches.get(layer_name)
                else:
                    branch = self.inter_args.branch

                branch_probs = torch.ones(len(branch))
                if self.inter_args.get('branch_probs') and len(self.inter_args.get('branch_probs')) == len(branch):
                    branch_probs = torch.Tensor(self.inter_args.get('branch_probs'))

                branch_probs /= branch_probs.sum()

                models = self.get_models(branch=branch)

                branch_dist = torch.distributions.categorical.Categorical(probs=branch_probs)

                branch_index = branch_dist.sample()
                # indices += str(branch_index)
                out = models[branch_index].forward_layer_by_name(layer_name, x=out, no_relu=no_relu)
                results[layer_name] = out
            results['logit'] = results['fc']

            # print(indices)
            outs.append(results)

        return outs
    

    def forward_stoc_backbonefc(self, x: torch.Tensor, repeat: int, no_relu: bool, ):
        
        initial_out = self.local_model.forward_layer0(x, no_relu=False)

        outs = []
        for m in range(repeat):
            out = initial_out
            branch = self.inter_args.branch
            models = self.get_models(branch=branch)
            branch_indices = torch.randint(len(models), (2,))

            for layer_name in ['layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                models = self.get_models(branch=branch)
                branch_index = branch_indices[1] if layer_name == 'fc' else branch_indices[0]
                out = models[branch_index].forward_layer_by_name(layer_name, x=out, no_relu=no_relu)
            outs.append(out)

        return outs
    

    def forward_stoc_whole(self, x: torch.Tensor, repeat: int, no_relu: bool, ):

        initial_out = self.local_model.forward_layer0(x, no_relu=False)

        outs = []
        for m in range(repeat):
            out = initial_out
            if self.inter_args.get('branches'):
                branch = self.inter_args.branches.get(layer_name)
            else:
                branch = self.inter_args.branch
            models = self.get_models(branch=branch)
            branch_index = torch.randint(len(models), (1,))[0]

            for layer_name in ['layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                out = models[branch_index].forward_layer_by_name(layer_name, x=out, no_relu=no_relu)
            outs.append(out)

        return outs
    
    def forward_local(self, x: torch.Tensor, no_relu: bool):
        return self.local_model(x, no_relu=no_relu)

    def forward_inter(self, x: torch.Tensor, no_relu: bool):
        return self.inter_model(x, no_relu=no_relu)

    def forward(self, x: torch.Tensor,  repeat: int = 1, no_relu: bool = False) -> Dict:
        if self.inter_args.forward_type == "layer":
            stoc_outs = self.forward_stoc_layerwise(x, repeat=repeat, no_relu=no_relu)
            local_out = self.forward_local(x, no_relu=no_relu)
            results = {
                "logit_stoc": stoc_outs,
                "logit_local": local_out["logit"]
            }

        elif self.inter_args.forward_type == "whole":
            stoc_outs = self.forward_stoc_whole(x, repeat=repeat, no_relu=no_relu)
            local_out = self.forward_local(x, no_relu=no_relu)
            results = {
                "logit_stoc": stoc_outs,
                "logit_local": local_out["logit"]
            }

        elif self.inter_args.forward_type == "backbonefc":
            stoc_outs = self.forward_stoc_backbonefc(x, repeat=repeat, no_relu=no_relu)
            local_out = self.forward_local(x, no_relu=no_relu)
            results = {
                "logit_stoc": stoc_outs,
                "logit_local": local_out["logit"]
            }

        elif self.inter_args.forward_type == "mlb":
            stoc_outs = self.forward_inter(x, no_relu=no_relu)
            local_out = self.forward_local(x, no_relu=no_relu)
            results = {
                "logit_stoc": [stoc_outs["logit"]],
                "logit_local": local_out["logit"]
            }
        return results

    




# @CLIENT_REGISTRY.register()
# class InterpolateClient(Client):

#     def __init__(self, args, client_index, loader=None):
#         self.args = args
#         self.client_index = client_index
#         # self.loader = loader
#         self.criterion = nn.CrossEntropyLoss()
#         return

#     def setup(self, model, device, local_dataset, local_lr, global_epoch, trainer, **kwargs):
#         self.model = model
#         self.global_model = copy.deepcopy(model)
#         self.interpolater = Interpolater(local_model=self.model, global_model=self.global_model, args=self.args)

#         self.device = device
#         self.loader = DataLoader(local_dataset, batch_size=self.args.batch_size, shuffle=True)
        
#         self.optimizer = optim.SGD(self.interpolater.local_model.parameters(), lr=local_lr, momentum=self.args.optimizer.momentum,
#                                    weight_decay=self.args.optimizer.wd)

#         self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, 
#                                                      lr_lambda=lambda epoch: self.args.trainer.local_lr_decay ** epoch)
        
        
#         class_counts = np.sort([*local_dataset.class_dict.values()])[::-1]
#         self.interpolater.class_counts = class_counts

#         self.trainer = trainer
#         # self.loss_manager = LossManager(self.args, {'cls': self.args.lambda1}, epoch=global_epoch)

#     def __repr__(self):
#         print(f'{self.__class__} {self.client_index}, {"data : "+len(self.loader.dataset) if self.loader else ""}')
    

#     def local_train(self, global_epoch, **kwargs):
#         # self.model.to(self.device)
#         # self.global_model.to(self.device)
#         self.interpolater.to(self.device)
#         scaler = GradScaler()
#         start = time.time()
#         loss_meter = AverageMeter('Loss', ':.2f')
#         inter_ce_losses_meter = AverageMeter('CELoss', ':.2f')
#         inter_kl_losses_meter = AverageMeter('KLLoss', ':.2f')
#         time_meter = AverageMeter('BatchTime', ':3.1f')


#         loss_dict = {}
#         # logger.info(f"[Client {self.client_index}] Local training start")
#         client_args = self.args.client

#         for local_epoch in range(self.args.trainer.local_epochs):
#             end = time.time()

#             self.interpolater.update()

#             for i, (images, labels) in enumerate(self.loader):
                
#                 if client_args.get('label_noise'):
#                     noise_ratio = client_args.label_noise.ratio
#                     # breakpoint()
#                     labels = apply_label_noise(labels, 100, noise_ratio) #TODO: num_classes

#                 images, labels = images.to(self.device), labels.to(self.device)


#                 self.interpolater.local_model.zero_grad()
#                 no_relu = not client_args.interpolation.feature_relu
#                 with autocast(enabled=self.args.use_amp):
                    
#                     self.interpolater.update()
#                     # results = self.model(images)
#                     results = self.interpolater.forward(images, repeat=client_args.interpolation.repeat, no_relu=no_relu)
#                     logit_local = results["logit_local"]
#                     main_celoss = self.criterion(logit_local, labels) # if errors occur, use ResNet18_base instead of ResNet18_GFLN

#                     ce_losses, kl_losses = [], []
#                     #for m in range(self.interpolater.inter_args.repeat):

#                     for m in range(len(results["logit_stoc"])):
#                         logit_m = results["logit_stoc"][m]
#                         ce_losses.append(self.criterion(logit_m, labels))
#                         kl_losses.append(KLD(logit_m, logit_local, T=client_args.interpolation.temp))
                    
#                     #ce_losses, kl_losses = torch.cat(ce_losses), torch.cat(kl_losses)
#                     inter_ce_loss, inter_kl_loss = sum(ce_losses)/len(ce_losses), sum(kl_losses)/len(kl_losses)
#                     loss = client_args.ce_loss.weight * main_celoss + \
#                         client_args.interpolation.ce_weight * inter_ce_loss + \
#                         client_args.interpolation.kl_weight * inter_kl_loss

#                 scaler.scale(loss).backward()
#                 scaler.unscale_(self.optimizer)
#                 torch.nn.utils.clip_grad_norm_(self.interpolater.local_model.parameters(), 10)
#                 scaler.step(self.optimizer)
#                 scaler.update()

#                 loss_meter.update(main_celoss.item(), images.size(0))
#                 inter_ce_losses_meter.update(inter_ce_loss.item(), images.size(0))
#                 inter_kl_losses_meter.update(inter_kl_loss.item(), images.size(0))
#                 time_meter.update(time.time() - end)
#                 end = time.time()
#             self.scheduler.step()

#             if global_epoch > 0 and global_epoch % 200 == 0:
#             # if global_epoch % 5 == 0:
#                 local_loss_dict = self.local_evaluate(global_epoch=global_epoch, local_epoch=local_epoch)
#                 # print(local_loss_dict)
#                 loss_dict.update(local_loss_dict)
        
#         logger.info(f"[C{self.client_index}] End. Time: {end-start:.2f}s, CE: {loss_meter.avg:.3f}, InterCE: {inter_ce_losses_meter.avg:.3f}, InterKL: {inter_kl_losses_meter.avg:.3f}")
        
#         #self.model.to('cpu')
#         self.interpolater.to('cpu')

#         loss_dict.update({
#             f'loss/{self.args.dataset.name}/cls': loss_meter.avg,
#             f'loss/{self.args.dataset.name}/inter_cls': inter_ce_losses_meter.avg,
#             f'loss/{self.args.dataset.name}/inter_kl': inter_kl_losses_meter.avg,
#         })

#         if global_epoch > 0 and global_epoch % 20 == 0:
#             local_loss_dict = self.local_evaluate(global_epoch=global_epoch)
#             loss_dict.update(local_loss_dict)
 
#         return self.interpolater.local_model, loss_dict
    

    
#     def local_evaluate(self, global_epoch, local_epoch='', num_major_class=20, factors=[-1, -0.5, 0, 0.5], **kwargs):

#         N = len(self.loader.dataset)
#         D = num_major_class
#         # C = len(self.loader.dataset.classes) # error

#         class_ids = np.array([int(key) for key in [*self.loader.dataset.class_dict.keys()]])
#         class_counts_id = np.argsort([*self.loader.dataset.class_dict.values()])[::-1]
#         sorted_class_ids = class_ids[class_counts_id]

#         loss_dict = {}

#         local_results = self.trainer.evaler.eval(model=self.interpolater.local_model, epoch=global_epoch)

#         desc = '' if len(str(local_epoch)) == 0 else f'_l{local_epoch}'

#         for factor in factors:
#             inter_model = self.interpolater.get_interpolate_model(factor=factor)
#             inter_results = self.trainer.evaler.eval(model=inter_model, epoch=global_epoch)

#             loss_dict.update({
#                 f'acc/{self.args.dataset.name}/inter{factor}{desc}': inter_results["acc"],
#                 f'class_acc/{self.args.dataset.name}/inter{factor}/top{D}{desc}': inter_results["class_acc"][sorted_class_ids[:D]].mean(),
#                 f'class_acc/{self.args.dataset.name}/inter{factor}/else{D}{desc}': inter_results["class_acc"][sorted_class_ids[D:]].mean(),
#             })

#             stoc_inter_model = self.interpolater.get_interpolate_model(factor=factor, stochastic=True)
#             stoc_inter_results = self.trainer.evaler.eval(model=stoc_inter_model, epoch=global_epoch)

#             loss_dict.update({
#                 f'acc/{self.args.dataset.name}/stoc_inter{factor}{desc}': stoc_inter_results["acc"],
#                 f'class_acc/{self.args.dataset.name}/stoc_inter{factor}/top{D}{desc}': stoc_inter_results["class_acc"][sorted_class_ids[:D]].mean(),
#                 f'class_acc/{self.args.dataset.name}/stoc_inter{factor}/else{D}{desc}': stoc_inter_results["class_acc"][sorted_class_ids[D:]].mean(),
#             })

#         loss_dict.update({
#             f'acc/{self.args.dataset.name}/local{desc}': local_results["acc"],
#             f'class_acc/{self.args.dataset.name}/local/top{D}{desc}': local_results["class_acc"][sorted_class_ids[:D]].mean(),
#             f'class_acc/{self.args.dataset.name}/local/else{D}{desc}': local_results["class_acc"][sorted_class_ids[D:]].mean(),
#         })

#         logger.warning(f'[C{self.client_index}, E{global_epoch}-{local_epoch}] Local Model: {local_results["acc"]:.2f}%')

#         return loss_dict





@CLIENT_REGISTRY.register()
class ExtraInterpolateClient(InterpolateClient):

    def setup(self, model, device, local_dataset, local_lr, global_epoch, trainer, **kwargs):
        self.model = model
        self.global_model = copy.deepcopy(model)
        self.interpolater = Interpolater(local_model=self.model, global_model=self.global_model, args=self.args)

        self.device = device
        self.loader = DataLoader(local_dataset, batch_size=self.args.batch_size, shuffle=True)
        
        self.optimizer = optim.SGD(self.interpolater.local_model.parameters(), lr=local_lr, momentum=self.args.optimizer.momentum,
                                   weight_decay=self.args.optimizer.wd)

        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, 
                                                     lr_lambda=lambda epoch: self.args.trainer.local_lr_decay ** epoch)
        
        
        class_counts = np.sort([*local_dataset.class_dict.values()])[::-1]
        self.interpolater.class_counts = class_counts
        # breakpoint()

        self.trainer = trainer