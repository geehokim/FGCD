#!/usr/bin/env python
# coding: utf-8
import copy
import time
import gc

import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler


from utils import *
from utils.metrics import evaluate
from utils.visualize import __log_test_metric__, umap_allmodels, cka_allmodels, log_fisher_diag
from models import build_encoder, get_model
from typing import Callable, Dict, Tuple, Union, List
from utils.logging_utils import AverageMeter

import logging
logger = logging.getLogger(__name__)
# coloredlogs.install(level='INFO', fmt='%(asctime)s %(name)s[%(process)d] %(message)s', datefmt='%m-%d %H:%M:%S')

from clients.build import CLIENT_REGISTRY
from clients import Client
from clients.interpolate_client import Interpolater
from clients.metric_client import MetricClient
from clients.extrainterpolate_client import ExtraInterpolateClient


@CLIENT_REGISTRY.register()
class MetricExtraClient(MetricClient,ExtraInterpolateClient):

    def __init__(self, args, client_index):
        self.args = args
        self.client_index = client_index
        self.loader = None
        self.interpolater = None

        self.model = None
        self.global_model = None

        self.metric_criterions = {'metric': None, 'metric2': None, 'metric3': None, 'metric4': None, }
        # self.metric_criterions = defaultdict()
        args_metric = args.client.metric_loss
        self.global_epoch = 0

        self.pairs = {}
        for pair in args_metric.pairs:
            self.pairs[pair.name] = pair
            self.metric_criterions[pair.name] = MetricLoss(pair=pair, **args_metric)
        
        self.criterion = nn.CrossEntropyLoss()




        return

    def setup(self, model, device, local_dataset, global_epoch, local_lr, trainer, **kwargs):
        if self.model is None:
            self.model = model
        else:
            self.model.load_state_dict(model.state_dict())

        if self.global_model is None:
            self.global_model = copy.deepcopy(self.model)
        else:
            self.global_model.load_state_dict(model.state_dict())

        for fixed_model in [self.global_model]:
            for n, p in fixed_model.named_parameters():
                p.requires_grad = False

        self.device = device
        self.num_layers = model.num_layers #TODO: self.model.num_layers
        # self.num_layers = 6

        # self.loader = DataLoader(local_dataset, batch_size=self.args.batch_size, shuffle=True)
        train_sampler = None
        if self.args.dataset.num_instances > 0:
            train_sampler = RandomClasswiseSampler(local_dataset, num_instances=self.args.dataset.num_instances)   
        self.loader =  DataLoader(local_dataset, batch_size=self.args.batch_size, sampler=train_sampler, shuffle=train_sampler is None,
                                   num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)

        self.optimizer = optim.SGD(self.model.parameters(), lr=local_lr, momentum=self.args.optimizer.momentum, weight_decay=self.args.optimizer.wd)
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, 
                                                     lr_lambda=lambda epoch: self.args.trainer.local_lr_decay ** epoch)
        
        
        self.class_counts = np.sort([*local_dataset.class_dict.values()])[::-1]
        

        sorted_key = np.sort([*local_dataset.class_dict.keys()])
        sorted_class_dict = {} 
        for key in sorted_key:  
            sorted_class_dict[key] = local_dataset.class_dict[key]

        if global_epoch == 0:
            logger.info(f"Class counts : {self.class_counts}")
            logger.info(f"Sorted class dict : {sorted_class_dict}")

        self.trainer = trainer

        #extrapolate part
        self.interpolater = Interpolater(local_model=self.model, global_model=self.global_model, args=self.args)


    def _algorithm_metric(self, local_results, global_results, labels):

        losses = {
            'cossim': [],
        }

        metric_args = self.args.client.metric_loss

        for l in range(self.num_layers):

            train_layer = False
            if l in metric_args.branch_level:
                train_layer = True
                
            local_feature_l = local_results[f"layer{l}"]
            global_feature_l = global_results[f"layer{l}"]

            if len(local_feature_l.shape) == 4:
                local_feature_l = F.adaptive_avg_pool2d(local_feature_l, 1)
                global_feature_l = F.adaptive_avg_pool2d(global_feature_l, 1)

            # Feature Cossim Loss
            if self.args.client.feature_align_loss.align_type == 'l2':
                loss_cossim = F.mse_loss(local_feature_l.squeeze(-1).squeeze(-1), global_feature_l.squeeze(-1).squeeze(-1))
            else:
                loss_cossim = F.cosine_embedding_loss(local_feature_l.squeeze(-1).squeeze(-1), global_feature_l.squeeze(-1).squeeze(-1), torch.ones_like(labels))
            losses['cossim'].append(loss_cossim)

            # Metric Loss
            if train_layer:
                for metric_name in self.metric_criterions:
                    metric_criterion = self.metric_criterions[metric_name]

                    if metric_criterion is not None:
                        if metric_criterion.pair.get('branch_level'):
                            train_layer = l in metric_criterion.pair.branch_level
                        

                        topk_neg = metric_args.topk_neg
                        if metric_args.get('topk_neg_end'):
                            neg_start = metric_args.topk_neg
                            neg_end, end_epoch = metric_args.topk_neg_end
                            progress = min(1, self.global_epoch/end_epoch)
                            topk_neg = int(neg_start + (neg_end - neg_start) * progress)

                        if train_layer:
                            
                            if metric_criterion.adapt_sample:
                                local_dataset = self.loader.dataset
                                minor_classes = [int(key) for key in local_dataset.class_dict if local_dataset.class_dict[key] < len(local_dataset)/local_dataset.num_classes]

                                if metric_criterion.adapt_sample == 'minor_class':
                                    weight_mask = [label in minor_classes for label in labels]
                                elif metric_criterion.adapt_sample == 'major_class':
                                    weight_mask = [label not in minor_classes for label in labels]
                                elif metric_criterion.adapt_sample == 'class_balance':
                                    class_counts = torch.zeros(len(local_dataset.dataset.classes))
                                    for key in local_dataset.class_dict:
                                        class_counts[int(key)] = local_dataset.class_dict[key]

                                    class_weights = 1/class_counts
                                    weight_mask = class_weights[labels]
                                    weight_mask /= weight_mask.mean()
                                elif metric_criterion.adapt_sample == 'class_balance_square':
                                    class_counts = torch.zeros(len(local_dataset.dataset.classes))
                                    for key in local_dataset.class_dict:
                                        class_counts[int(key)] = local_dataset.class_dict[key]

                                    class_weights = (1/class_counts)**2
                                    weight_mask = class_weights[labels]
                                    weight_mask /= weight_mask.mean()
                                elif metric_criterion.adapt_sample == 'class_balance_sqrt':
                                    class_counts = torch.zeros(len(local_dataset.dataset.classes))
                                    for key in local_dataset.class_dict:
                                        class_counts[int(key)] = local_dataset.class_dict[key]

                                    class_weights = torch.sqrt(1/class_counts)
                                    weight_mask = class_weights[labels]
                                    weight_mask /= weight_mask.mean()
                                else:
                                    raise ValueError

                                weight_mask = torch.FloatTensor(weight_mask).to(self.device)
                                loss_metric = weight_mask * metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, target=labels, reduction=False, topk_neg=topk_neg)
                            else:
                                loss_metric = metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, target=labels, reduction=False, topk_neg=topk_neg)

                            if metric_name not in losses:
                                losses[metric_name] = []
                            losses[metric_name].append(loss_metric.mean())
            
        for loss_name in losses:
            try:
                losses[loss_name] = torch.mean(torch.stack(losses[loss_name])) if len(losses[loss_name]) > 0 else 0
            except:
                breakpoint()

        return losses


    def _algorithm(self, images, labels,) -> Dict:

        losses = defaultdict(float)
        no_relu = not self.args.client.metric_loss.feature_relu
        results = self.model(images, no_relu=no_relu)

        cls_loss = self.criterion(results["logit"], labels)
        losses["cls"] = cls_loss

        with torch.no_grad():
            global_results = self.global_model(images, no_relu=no_relu)

        losses.update(self._algorithm_metric(local_results=results, global_results=global_results, labels=labels))               

        return losses


    # @property
    def get_weights(self, epoch=None):
        args_metric = self.args.client.metric_loss

        weights = {
            "cls": self.args.client.ce_loss.weight,
            "cossim": self.args.client.feature_align_loss.weight,
        }
        
        for pair in args_metric.pairs:
            weights[pair.name] = pair.weight
        return weights

    @property
    def current_progress(self):
        return self.global_epoch / self.args.trainer.global_rounds

    def local_train(self, global_epoch, **kwargs):

        self.global_epoch = global_epoch

        self.model.to(self.device)
        self.global_model.to(self.device)
        if self.interpolater:
            self.interpolater.to(self.device)


        #extra part
        client_args = self.args.client
        inter_ce_losses_meter = AverageMeter('CELoss', ':.2f')
        inter_kl_losses_meter = AverageMeter('KLLoss', ':.2f')
        #

        scaler = GradScaler()
        start = time.time()
        loss_meter = AverageMeter('Loss', ':.2f')
        time_meter = AverageMeter('BatchTime', ':3.1f')

        # logger.info(f"[Client {self.client_index}] Local training start")
        # self.global_model = copy.deepcopy(self.model)
        self.weights = self.get_weights(epoch=global_epoch)

        if global_epoch % 50 == 0:
            print(self.weights)
            print(self.pairs)
            
        for local_epoch in range(self.args.trainer.local_epochs):
            end = time.time()

            for i, (images, labels) in enumerate(self.loader):

                images, labels = images.to(self.device), labels.to(self.device)
                no_relu = not client_args.interpolation.feature_relu
                self.interpolater.update()
                self.model.zero_grad()

                #with autocast(enabled=self.args.use_amp):
                losses = self._algorithm(images, labels)            
                loss_metric = sum([self.weights[loss_key]*losses[loss_key] for loss_key in losses])
                    

                #scaler.scale(loss_metric).backward()
                # scaler.unscale_(self.optimizer)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)


                #extra part begin
                #self.interpolater.local_model.zero_grad()

                #with autocast(enabled=self.args.use_amp):
                # results = self.model(images)
                results = self.interpolater.forward(images, repeat=client_args.interpolation.repeat, no_relu=no_relu)
                logit_local = results["logit_local"]
                main_celoss = self.criterion(logit_local, labels) # if errors occur, use ResNet18_base instead of ResNet18_GFLN

                ce_losses, kl_losses = [], []
                #for m in range(self.interpolater.inter_args.repeat):

                for m in range(len(results["logit_stoc"])):
                    logit_m = results["logit_stoc"][m]
                    ce_losses.append(self.criterion(logit_m, labels))
                    kl_losses.append(KLD(logit_m, logit_local, T=client_args.interpolation.temp))
                
                #ce_losses, kl_losses = torch.cat(ce_losses), torch.cat(kl_losses)
                inter_ce_loss, inter_kl_loss = sum(ce_losses)/len(ce_losses), sum(kl_losses)/len(kl_losses)
                # loss_extra = client_args.ce_loss.weight * main_celoss + \
                #     client_args.interpolation.ce_weight * inter_ce_loss + \
                #     client_args.interpolation.kl_weight * inter_kl_loss


                #After 10.7 pm 7:34, delete main_celoss in extra because it already exist in the loss_metric
                loss_extra =  client_args.interpolation.ce_weight * inter_ce_loss +   client_args.interpolation.kl_weight * inter_kl_loss       

                #extra part end
                loss =  loss_metric + loss_extra
                
                loss.backward()
                # scaler.scale(loss).backward()
                # scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()
                # scaler.step(self.optimizer)
                # scaler.update()
                #print("loss, loss_metric, loss_extra: ",loss, loss_metric, loss_extra)
                #breakpoint()
                loss_meter.update(loss.item(), images.size(0))
                #extra
                inter_ce_losses_meter.update(inter_ce_loss.item(), images.size(0))
                inter_kl_losses_meter.update(inter_kl_loss.item(), images.size(0))
                #
                time_meter.update(time.time() - end)
                end = time.time()
            self.scheduler.step()
        
        logger.info(f"[C{self.client_index}] End. Time: {end-start:.2f}s, Loss: {loss_meter.avg:.3f}, InterCE: {inter_ce_losses_meter.avg:.3f}, InterKL: {inter_kl_losses_meter.avg:.3f}")

        self.model.to('cpu')
        if self.interpolater:
            self.interpolater.to('cpu')

        loss_dict = {f'loss/{self.args.dataset.name}/{loss_key}': float(losses[loss_key]) for loss_key in losses}
        #extra
        loss_dict.update({
            f'loss/{self.args.dataset.name}/inter_cls': inter_ce_losses_meter.avg,
            f'loss/{self.args.dataset.name}/inter_kl': inter_kl_losses_meter.avg,
        })

        #


        # if global_epoch > 0 and self.args.eval.local_freq > 0 and global_epoch % self.args.eval.local_freq == 0:
        #     local_loss_dict = self.local_evaluate(global_epoch=global_epoch)
        #     loss_dict.update(local_loss_dict)



        #Delete some property
        if self.args.get('debugj'):
            breakpoint()
        gc.collect()

        return self.model, loss_dict


    
    def local_evaluate(self, global_epoch, local_epoch='', num_major_class=20, factors=[-1, 0], **kwargs):
        logger.info("Do not use local evaluate in client (due to memory leak)")
        return

        N = len(self.loader.dataset)
        D = num_major_class
        # C = len(self.loader.dataset.classes) # error

        class_ids = np.array([int(key) for key in [*self.loader.dataset.class_dict.keys()]])
        class_counts_id = np.argsort([*self.loader.dataset.class_dict.values()])[::-1]
        sorted_class_ids = class_ids[class_counts_id]

        loss_dict = {}

        local_results = self.trainer.evaler.eval(model=self.model, epoch=global_epoch, device=self.device)
        # return loss_dict

        desc = '' if len(str(local_epoch)) == 0 else f'_l{local_epoch}'


        if self.interpolater is not None:
            for factor in factors:
                inter_model = self.interpolater.get_interpolate_model(factor=factor)
                inter_results = self.trainer.evaler.eval(model=inter_model, epoch=global_epoch, device=self.device)

                loss_dict.update({
                    f'acc/{self.args.dataset.name}/inter{factor}{desc}': inter_results["acc"],
                    f'class_acc/{self.args.dataset.name}/inter{factor}/top{D}{desc}': inter_results["class_acc"][sorted_class_ids[:D]].mean(),
                    f'class_acc/{self.args.dataset.name}/inter{factor}/else{D}{desc}': inter_results["class_acc"][sorted_class_ids[D:]].mean(),
                })

                # stoc_inter_model = self.interpolater.get_interpolate_model(factor=factor, stochastic=True)
                # stoc_inter_results = self.trainer.evaler.eval(model=stoc_inter_model, epoch=global_epoch, device=self.device)

                # loss_dict.update({
                #     f'acc/{self.args.dataset.name}/stoc_inter{factor}{desc}': stoc_inter_results["acc"],
                #     f'class_acc/{self.args.dataset.name}/stoc_inter{factor}/top{D}{desc}': stoc_inter_results["class_acc"][sorted_class_ids[:D]].mean(),
                #     f'class_acc/{self.args.dataset.name}/stoc_inter{factor}/else{D}{desc}': stoc_inter_results["class_acc"][sorted_class_ids[D:]].mean(),
                # })

        loss_dict.update({
            f'acc/{self.args.dataset.name}/local{desc}': local_results["acc"],
            f'class_acc/{self.args.dataset.name}/local/top{D}{desc}': local_results["class_acc"][sorted_class_ids[:D]].mean(),
            f'class_acc/{self.args.dataset.name}/local/else{D}{desc}': local_results["class_acc"][sorted_class_ids[D:]].mean(),
        })

        logger.warning(f'[C{self.client_index}, E{global_epoch}-{local_epoch}] Local Model: {local_results["acc"]:.2f}%')

        return loss_dict
    













