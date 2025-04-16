#!/usr/bin/env python
# coding: utf-8
import copy
import time

import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import gc

from utils import *
from utils.metrics import evaluate
from utils.visualize import __log_test_metric__, umap_allmodels, cka_allmodels, log_fisher_diag
from models import build_encoder, get_model
from typing import Callable, Dict, Tuple, Union, List
from utils.logging_utils import AverageMeter
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim

import logging
logger = logging.getLogger(__name__)

from clients.build import CLIENT_REGISTRY

from utils import LossManager

@CLIENT_REGISTRY.register()
class Client():

    def __init__(self, args, client_index, model=None, loader=None):
        self.args = args
        self.client_index = client_index
        # self.loader = loader  
        self.model = model
        self.global_model = copy.deepcopy(model)
        self.criterion = nn.CrossEntropyLoss()
        return

    def setup(self, model, device, local_dataset, global_epoch, local_lr, trainer, **kwargs):
        # if self.model is None:
        #     self.model = model
        # else:
        #     self.model.load_state_dict(model.state_dict())
        self._update_model(model)

        # if self.global_model is None:
        #     self.global_model = copy.deepcopy(self.model)
        # else:
        #     self.global_model.load_state_dict(model.state_dict())
        self._update_global_model(model)

        for fixed_model in [self.global_model]:
            for n, p in fixed_model.named_parameters():
                p.requires_grad = False

        self.device = device
        # self.num_layers = self.model.num_layers
        # self.num_layers = 6

        # self.loader = DataLoader(local_dataset, batch_size=self.args.batch_size, shuffle=True)
        # train_sampler = None
        # if self.args.dataset.num_instances > 0:
        #     train_sampler = RandomClasswiseSampler(local_dataset, num_instances=self.args.dataset.num_instances)
        # self.loader =  DataLoader(local_dataset, batch_size=self.args.batch_size, sampler=train_sampler, shuffle=train_sampler is None,
        #                            num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, drop_last=False)

        # --------------------
        # SAMPLER
        # Sampler which balances labelled and unlabelled examples in each batch
        # --------------------
        label_len = len(local_dataset.labelled_dataset)
        unlabelled_len = len(local_dataset.unlabelled_dataset)
        sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(local_dataset))]
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(local_dataset))
        self.loader = DataLoader(local_dataset, batch_size=self.args.batch_size, sampler=sampler, shuffle=False,
                                 num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, drop_last=False)
        self.optimizer = optim.SGD(self.model.parameters(), lr=local_lr, momentum=self.args.optimizer.momentum,
                                   weight_decay=self.args.optimizer.wd)
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                     lr_lambda=lambda epoch: self.args.trainer.local_lr_decay ** epoch)
        self.class_counts = np.sort([*local_dataset.class_dict.values()])[::-1]
        #self.num_classes = len(self.loader.dataset.dataset.classes)
        self.num_classes = len(self.args.dataset.seen_classes) + len(self.args.dataset.unseen_classes)

        # self.class_stats['ratio'] = torch.zeros(self.num_classes)
        # for class_key in local_dataset.class_dict:
        #     self.class_stats['ratio'][int(class_key)] = local_dataset.class_dict[class_key]
            # self.class_ratios[int(class_key)] = local_dataset.class_dict[class_key]

        

        sorted_key = np.sort([*local_dataset.class_dict.keys()])
        sorted_class_dict = {} 
        for key in sorted_key:  
            sorted_class_dict[key] = local_dataset.class_dict[key]
        
        if self.args.client.get('LC'):
            self.label_distrib = torch.zeros(len(local_dataset.dataset.classes), device=self.device)
            for key in sorted_class_dict:
                self.label_distrib[int(key)] = sorted_class_dict[key]
            
        if global_epoch == 0:
            logger.warning(f"Class counts : {self.class_counts}")
            logger.info(f"Sorted class dict : {sorted_class_dict}")

        self.sorted_class_dict = sorted_class_dict

        self.trainer = trainer

    # def setup(self, model, device, local_dataset, local_lr, global_epoch, trainer, **kwargs):
    #     # if self.model is None:
    #     # self.model = model
    #     self._update_model(model)
    #     # else:
    #     #     self.model.load_state_dict(model.state_dict())

    #     self.device = device
    #     # self.loader = DataLoader(local_dataset, batch_size=self.args.batch_size, shuffle=True)
    #     train_sampler = None
    #     if self.args.dataset.num_instances > 0:
    #         train_sampler = RandomClasswiseSampler(local_dataset, num_instances=self.args.dataset.num_instances)   
    #     self.loader =  DataLoader(local_dataset, batch_size=self.args.batch_size, sampler=train_sampler, shuffle=train_sampler is None,
    #                                num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)
        
    #     self.optimizer = optim.SGD(model.parameters(), lr=local_lr, momentum=self.args.optimizer.momentum,
    #                                weight_decay=self.args.optimizer.wd)
    #     self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, 
    #                                                  lr_lambda=lambda epoch: self.args.trainer.local_lr_decay ** epoch)
    
    #     self.trainer = trainer
    #     self.class_counts = np.sort([*local_dataset.class_dict.values()])[::-1]
    #     if global_epoch == 0:
    #         logger.info(f"Class counts : {self.class_counts}")
        
    #     # self.loss_manager = LossManager(self.args, {'cls': self.args.lambda1}, epoch=global_epoch)

    def __repr__(self):
        print(f'{self.__class__} {self.client_index}, {"data : "+len(self.loader.dataset) if self.loader else ""}')
    
    def _update_model(self, state_dict):
        self.model.load_state_dict(state_dict)
    
    def _update_global_model(self, state_dict):
        self.global_model.load_state_dict(state_dict)
    

    def get_weights(self, epoch=None):

        weights = {
            "cls": 1
        }
        
        return weights

    def local_train(self, global_epoch, **kwargs):
        self.global_epoch = global_epoch

        self.model.to(self.device)
        scaler = GradScaler()
        start = time.time()
        loss_meter = AverageMeter('Loss', ':.2f')
        time_meter = AverageMeter('BatchTime', ':3.1f')

        # logger.info(f"[Client {self.client_index}] Local training start")

        self.weights = self.get_weights(epoch=global_epoch)

        if global_epoch % 50 == 0:
            print(self.weights)

        for local_epoch in range(self.args.trainer.local_epochs):
            end = time.time()
            for i, (images, labels, uq_idxs, mask_lab) in enumerate(self.loader):
                
                if len(images.size()) == 3:
                    images = images.unsqueeze(0)
                
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()

                with autocast(enabled=self.args.use_amp):
                    losses = self._algorithm(images, labels)
                    loss = sum([self.weights[loss_key]*losses[loss_key] for loss_key in losses])
                    # results = self.model(images)
                    # loss = self.criterion(results["logit"], labels) # if errors occur, use ResNet18_base instead of ResNet18_GFLN

                # if self.args.get('debugs'):
                #     breakpoint()
                try:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                    scaler.step(self.optimizer)
                    scaler.update()
                except Exception as e:
                    print(e)
                    # breakpoint()
                #loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gr_clipping_max_norm)
                #self.optimizer.step()

                loss_meter.update(loss.item(), images.size(0))
                time_meter.update(time.time() - end)
                end = time.time()
            self.scheduler.step()
        
        logger.info(f"[C{self.client_index}] End. Time: {end-start:.2f}s, Loss: {loss_meter.avg:.3f}")

        self.model.to('cpu')

        loss_dict = {
            f'loss/{self.args.dataset.name}/cls': loss_meter.avg,
        }
        # del results
        gc.collect()

        return self.model.state_dict(), loss_dict
    

    def _algorithm(self, images, labels, ) -> Dict:
        losses = defaultdict(float)

        # results = self.model(images)
        # no_relu = not self.args.client.metric_loss.feature_relu
        results = self.model(images)
        # loss = self.criterion(results["logit"], labels) # if errors occur, use ResNet18_base instead of ResNet18_GFLN
        cls_loss = self.criterion(results["logit"], labels)
        losses["cls"] = cls_loss

        del results
        return losses


