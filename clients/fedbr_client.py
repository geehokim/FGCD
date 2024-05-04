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

from utils.train_utils import apply_label_noise

import logging
logger = logging.getLogger(__name__)
# coloredlogs.install(level='INFO', fmt='%(asctime)s %(name)s[%(process)d] %(message)s', datefmt='%m-%d %H:%M:%S')

from clients.build import CLIENT_REGISTRY
from clients import Client
from clients.interpolate_client import Interpolater



@CLIENT_REGISTRY.register()
class FedBRClient(Client):

    def __init__(self, args, client_index, model):
        self.args = args
        self.client_index = client_index
        self.loader = None
        self.interpolater = None

        self.model = model
        self.global_model = copy.deepcopy(model)

        self.metric_criterions = {'metric': None, 'metric2': None, 'metric3': None, 'metric4': None, }
        # self.metric_criterions = defaultdict()
        args_metric = args.client.metric_loss
        self.global_epoch = 0

        self.pairs = {}
        for pair in args_metric.pairs:
            self.pairs[pair.name] = pair
            self.metric_criterions[pair.name] = MetricLoss(pair=pair, **args_metric)
        
        self.criterion = nn.CrossEntropyLoss()
        if self.args.client.get('LC'):
            self.FedLC_criterion = FedLC

        self.decorr_criterion = FedDecorrLoss()

        self.class_stats = {
            'ratio': None,
            'cov': defaultdict(),
            'cov_global': defaultdict(),
        }

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
        self.num_layers = self.model.num_layers #TODO: self.model.num_layers
        # self.num_layers = 6

        # self.loader = DataLoader(local_dataset, batch_size=self.args.batch_size, shuffle=True)
        train_sampler = None
        if self.args.dataset.num_instances > 0:
            train_sampler = RandomClasswiseSampler(local_dataset, num_instances=self.args.dataset.num_instances)   
        self.loader =  DataLoader(local_dataset, batch_size=self.args.batch_size, sampler=train_sampler, shuffle=train_sampler is None,
                                   num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)

        self.optimizer = optim.SGD(self.model.parameters(), lr=local_lr, momentum=self.args.optimizer.momentum, weight_decay=self.args.optimizer.wd)

        self.adv_optimizer = optim.SGD(self.model.discriminator.parameters(), lr=local_lr, momentum=self.args.optimizer.momentum, weight_decay=self.args.optimizer.wd)
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, 
                                                     lr_lambda=lambda epoch: self.args.trainer.local_lr_decay ** epoch)
        
        
        self.class_counts = np.sort([*local_dataset.class_dict.values()])[::-1]

        self.num_classes = len(self.loader.dataset.dataset.classes)

        self.class_stats['ratio'] = torch.zeros(self.num_classes)
        for class_key in local_dataset.class_dict:
            self.class_stats['ratio'][int(class_key)] = local_dataset.class_dict[class_key]

        

        sorted_key = np.sort([*local_dataset.class_dict.keys()])
        sorted_class_dict = {} 
        for key in sorted_key:  
            sorted_class_dict[key] = local_dataset.class_dict[key]
        
            
        if global_epoch == 0:
            logger.warning(f"Class counts : {self.class_counts}")
            logger.info(f"Sorted class dict : {sorted_class_dict}")

        self.sorted_class_dict = sorted_class_dict

        self.trainer = trainer


    def _algorithm_metric(self, local_results, global_results, labels,):

        losses = {
            'cossim': [],
        }


        metric_args = self.args.client.metric_loss
        
        
        unique_labels = labels.unique()
        # if metric_args.get('sampling') == 'balance_progress':
        #     S = labels.size(0)//len(unique_labels) + 1
        S = labels.size(0)//len(unique_labels) + 1
        # S = 2
        bal_indices = []
        clamp_indices = []
        new_label_i = self.num_classes + 1
        splited_labels = labels.clone()

        for label in unique_labels:
            assign = (labels==label).nonzero().squeeze(1)
            label_indices = torch.randperm(assign.size(0))[:S]
            bal_indices.append(assign[label_indices])


        bal_indices = torch.cat(bal_indices)


        for l in range(self.num_layers):

            train_layer = False
            if metric_args.branch_level is False or l in metric_args.branch_level:
                train_layer = True
            # print(l, train_layer)
                
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

                            if True:
                                
                                class_ratio = self.class_stats['ratio'][labels]
                                # class_ratio = None
                                # uncertainty = self.get_entropy(local_results['logit'])
                                uncertainty = F.cross_entropy(local_results["logit"], labels, reduction='none').detach()
                                # uncertainty /= uncertainty.max()

                                loss_metric = metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, target=labels, reduction=False, topk_neg=topk_neg,
                                                            uncertainty=uncertainty, class_ratio=class_ratio, level=l, progress=self.current_progress)
            
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

        if self.args.client.get('label_noise'):
            noise_ratio = self.args.client.label_noise.ratio
            labels = apply_label_noise(labels, noise_ratio)

        cls_loss = self.criterion(results["logit"], labels)
        losses["cls"] = cls_loss

        
        if self.args.client.get('LC'):
            LC_loss = self.FedLC_criterion(self.label_distrib, results["logit"], labels, self.args.client.LC.tau)
            losses["LC"] = LC_loss

        if self.args.client.get('decorr_loss'):
            decorr_loss = self.decorr_criterion(results["feature"])
            losses["decorr"] = decorr_loss

        from utils.loss import KL_u_p_loss
        uniform_loss = KL_u_p_loss(results["logit"]).mean()
        losses["uniform"] = uniform_loss

        with torch.no_grad():
            global_results = self.global_model(images, no_relu=no_relu)

        breakpoint()

        losses.update(self._algorithm_metric(local_results=results, global_results=global_results, labels=labels,))               

        features = {
            "local": results,
            "global": global_results
        }

        return losses, features


    # @property
    def get_weights(self, epoch=None):
        args_metric = self.args.client.metric_loss

        weights = {
            "cls": self.args.client.ce_loss.weight,
            "cossim": self.args.client.feature_align_loss.weight,
            "uniform": self.args.client.ce_loss.get('uniform_weight') or 0,
            # 'decorr': self.args.client.get('decorr_loss').get('w') or 0,
        }
        
        # for pair in args_metric.pairs:
        #     weights[pair.name] = pair.weight

        for pair in args_metric.pairs:

            if pair.get('weights'):
                for weight_epoch in pair.weights:
                    if epoch >= weight_epoch:
                        weight = pair.weights[weight_epoch]
                weights[pair.name] = weight
            else:
                weights[pair.name] = pair.weight
                    

        if self.args.client.get('LC'):
            weights['LC'] = self.args.client.LC.weight
        if self.args.client.get('decorr_loss'):
            weights['decorr'] = self.args.client.decorr_loss.weight
            
        return weights

    @property
    def current_progress(self):
        return self.global_epoch / self.args.trainer.global_rounds


    def get_entropy(self, logit, relative=True):
        local_score = F.softmax(logit, 1).detach().double()
        entropy = torch.distributions.Categorical(local_score).entropy()
        uniform_entropy = torch.distributions.Categorical(torch.ones_like(local_score)).entropy()
        entropy_ = entropy / uniform_entropy
        # entropy_meter.update(entropy_.item())  
        return entropy_


    def local_train(self, global_epoch, **kwargs):

        self.global_epoch = global_epoch

        self.model.to(self.device)
        self.global_model.to(self.device)
        if self.interpolater:
            self.interpolater.to(self.device)

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
            
        entropy_meter = AverageMeter('Entropy', ':.2f')
        global_entropy_meter = AverageMeter('Entropy', ':.2f')
        var_meter = None

        all_features_local = defaultdict(list)
        all_features_global = defaultdict(list)
        all_labels = []
        
        for local_epoch in range(self.args.trainer.local_epochs):
            all_features_local = defaultdict(list)
            all_features_global = defaultdict(list)
            all_labels = []



            end = time.time()
            for i, (images, labels) in enumerate(self.loader):

                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()

                with autocast(enabled=self.args.use_amp):
                    losses, features = self._algorithm(images, labels)


                    #For only visualizing loss(not use for training)
                    for loss_key in losses:
                        if loss_key not in self.weights.keys():
                            self.weights[loss_key] = 0

                    loss = sum([self.weights[loss_key]*losses[loss_key] for loss_key in losses])


                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                scaler.step(self.optimizer)
                scaler.update()


                # Entropy
                entropy = self.get_entropy(features['local']['logit']).mean()
                global_entropy = self.get_entropy(features['global']['logit']).mean()

                entropy_meter.update(entropy.item())                
                global_entropy_meter.update(global_entropy.item())                


                loss_meter.update(loss.item(), images.size(0))
                time_meter.update(time.time() - end)

                # feature_var
                l_feat = features['local']
                if var_meter is None:
                    var_meter = {}
                    for key in l_feat.keys():
                        var_meter[key] = AverageMeter('var', ':.2f')
                for key in l_feat.keys():
                    this_feat = l_feat[key].view(len(l_feat[key]), -1)
                    this_feat_var = (this_feat.var(dim=1)).mean()
                    var_meter[key].update(this_feat_var.detach().clone().to('cpu'))
                    

                end = time.time()
            self.scheduler.step()
        
        logger.info(f"[C{self.client_index}] End. Time: {end-start:.2f}s, Loss: {loss_meter.avg:.3f}, Rel Entropy: {entropy_meter.avg:.3f}/{global_entropy_meter.avg:.3f}")

        self.model.to('cpu')
        self.global_model.to('cpu')
        if self.interpolater:
            self.interpolater.to('cpu')

        loss_dict = {f'loss/{self.args.dataset.name}/{loss_key}': float(losses[loss_key]) for loss_key in losses}
        loss_dict.update({
            f'entropy/{self.args.dataset.name}/train/local': entropy_meter.avg,
            f'entropy/{self.args.dataset.name}/train/global': global_entropy_meter.avg,
        })

        for key in var_meter.keys():
            loss_dict.update({
                f'feature_var/{self.args.dataset.name}/train/local/{key}': var_meter[key].avg,
            })

        # if global_epoch > 0 and self.args.eval.local_freq > 0 and global_epoch % self.args.eval.local_freq == 0:
        #     local_loss_dict = self.local_evaluate(global_epoch=global_epoch)
        #     loss_dict.update(local_loss_dict)
        
        #Delete some property
        for name in ['results1_grad', 'results2_grad', 'hook1', 'hook2']:
            if hasattr(self.model, name):
                if 'hook' in name:
                    for val in getattr(self.model, name).values():
                        val.remove()
                delattr(self.model, name)



        gc.collect()

        return self.model.state_dict(), loss_dict


    def update_cov_results(self, all_features_local, all_features_global, all_labels):

        all_labels = torch.cat(all_labels)

        # for key in all_features_local:
        for key in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5']:
            all_features_local_l = torch.cat(all_features_local[key])
            cov_results = _get_covariance_results(all_features_local_l.squeeze(), all_labels, self.num_classes)
            self.class_stats['cov'][key] = cov_results
        # for key in all_features_global:
            all_features_global_l = torch.cat(all_features_global[key])
            cov_global_results = _get_covariance_results(all_features_global_l.squeeze(), all_labels, self.num_classes)
            self.class_stats['cov_global'][key] = cov_global_results


    
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
    





