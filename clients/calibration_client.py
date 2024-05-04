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

import logging
logger = logging.getLogger(__name__)

from clients.build import CLIENT_REGISTRY

from utils import LossManager

@CLIENT_REGISTRY.register()
class CalibrationClient(Client):

    # def __init__(self, args, client_index, loader=None):
    #     self.args = args
    #     self.client_index = client_index
    #     # self.loader = loader  
    #     self.model = None
    #     self.criterion = nn.CrossEntropyLoss()
    #     return

    # def setup(self, model, device, local_dataset, local_lr, global_epoch, trainer, **kwargs):
    #     # if self.model is None:
    #     self.model = model
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
        
        # self.loss_manager = LossManager(self.args, {'cls': self.args.lambda1}, epoch=global_epoch)

    # def __repr__(self):
    #     print(f'{self.__class__} {self.client_index}, {"data : "+len(self.loader.dataset) if self.loader else ""}')
    

    def get_weights(self, epoch=None):

        weights = {
            "cls": 1,
            "calibration": 1,
        }
        
        return weights
    

    def _algorithm(self, images, labels, ) -> Dict:
        losses = defaultdict(float)

        # results = self.model(images)
        # no_relu = not self.args.client.metric_loss.feature_relu
        results = self.model(images)
        # loss = self.criterion(results["logit"], labels) # if errors occur, use ResNet18_base instead of ResNet18_GFLN
        cls_loss = self.criterion(results["logit"], labels)

        breakpoint()
        losses["cls"] = cls_loss

        del results
        return losses


