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
class MetricExtraClient_whichlearnfaster(MetricClient,ExtraInterpolateClient):

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
        #Before 10/11 02:35, all experiments using metric_extra_client_whichlearnfaster_exp is useless if weight of loc_var_sample_prototype,inverse_var_of_prototypes,diff_loc_glob_prototypes_mean > 0 (eventhough weight is all zero, ~~prototype loss is useless)
        # since self.num_classes for local client means only num_classes which client has class label data
        # for example, CIFAR 100 dataset, client has 22 classes data, so self.num_classes is 22, not 100
        #self.num_classes = self.loader.dataset.num_classes
        self.num_classes = len(self.loader.dataset.dataset.classes)


    def _get_prototypes(self, features, labels):
        #Before 10/11 02:35, all experiments using metric_extra_client_whichlearnfaster_exp is useless if weight of loc_var_sample_prototype,inverse_var_of_prototypes,diff_loc_glob_prototypes_mean > 0 (eventhough weight is all zero, ~~prototype loss is useless)
        # since self.num_classes for local client means only num_classes which client has class label data
        # for example, CIFAR 100 dataset, client has 22 classes data, so self.num_classes is 22, not 100
        num_classes = self.num_classes






        #num_classes = len(self.loader.dataset.dataset.classes)
        prototypes = torch.zeros((num_classes, features.size(1))).to(self.device)
        #breakpoint()


        

        for i in range(num_classes):
            if (labels==i).sum() > 0:
                this_proto = features[labels==i].mean(0)
                if len(this_proto.shape) > 1:
                    prototypes[i] = this_proto.squeeze()


        valid_num_classes = labels.bincount().nonzero().size(0)
        prototypes_ = prototypes[~prototypes[:, 0].isnan(), :]
        #valid_num_classes = prototypes_.size(0)
        valid_class_counts = labels.bincount()[labels.bincount().nonzero()]
        
        prototypes_mean =  prototypes_.sum(0, keepdim=True) / valid_num_classes

        var_sample_prototype = torch.zeros((num_classes))
        for i in range(num_classes):
            if (labels==i).sum() > 0:
                #Before 10/12 02:35, All experiments measure the thing about local_var_sample_prototype wrong... 
                #var_sample_prototype[i] = torch.norm((features[labels==i] - prototypes[i]), dim =1).mean()
                #breakpoint()
                this_features = features[labels==i]
                if len(this_features.shape) > 1:
                    this_features = this_features.view(len(this_features) , -1)
                var_sample_prototype[i] = torch.norm(( this_features - prototypes[i]), dim =1).mean()
                #breakpoint()
                

        var_sample_prototype = var_sample_prototype.sum() / valid_num_classes
        var_of_prototypes = torch.norm((prototypes_ - prototypes_mean), dim = 1).sum() / valid_num_classes
        return prototypes_, prototypes_mean, var_sample_prototype, var_of_prototypes


    def _algorithm_metric(self, local_results, global_results, labels):

        losses = {
            'cossim': [],
            'loc_var_sample_prototype':[],
            'inverse_var_of_prototypes':[],
            'diff_loc_glob_prototypes_mean':[]

        }

        # if self.args.client.get('label_noise'):
        #     noise_ratio = self.args.client.label_noise.ratio
        #     labels_ = apply_label_noise(labels, noise_ratio)
        #     labels = labels_

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
                loc_prototypes_, loc_prototypes_mean, loc_var_sample_prototype, loc_var_of_prototypes = self._get_prototypes(local_feature_l, labels)
                with torch.no_grad():
                    glob_prototypes_, glob_prototypes_mean, glob_var_sample_prototype, glob_var_of_prototypes = self._get_prototypes(global_feature_l, labels)
                losses['loc_var_sample_prototype'].append(loc_var_sample_prototype)
                losses['inverse_var_of_prototypes'].append(1/(0.1+loc_var_of_prototypes))
                losses['diff_loc_glob_prototypes_mean'].append(torch.norm((glob_prototypes_mean -  loc_prototypes_mean),dim=1))
                #print("Layer ",l,", loc_var_of_prototypes :",loc_var_of_prototypes)

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
                                    weight_mask = torch.FloatTensor(weight_mask).to(self.device)
                                elif metric_criterion.adapt_sample == 'major_class':
                                    weight_mask = [label not in minor_classes for label in labels]
                                    weight_mask = torch.FloatTensor(weight_mask).to(self.device)
                                elif metric_criterion.adapt_sample == 'class_balance':
                                    class_counts = torch.zeros(len(local_dataset.dataset.classes))
                                    for key in local_dataset.class_dict:
                                        class_counts[int(key)] = local_dataset.class_dict[key]

                                    class_weights = 1/class_counts
                                    weight_mask = class_weights[labels]
                                    weight_mask /= weight_mask.mean()
                                    weight_mask = torch.FloatTensor(weight_mask).to(self.device)

                                elif metric_criterion.adapt_sample == 'uncertainty':

                                    # local_entropy = self.get_entropy(local_results['logit'])
                                    global_entropy = self.get_entropy(global_results['logit'])
                                    # unc = torch.log(global_entropy / local_entropy)
                                    global_entropy /= global_entropy.mean()
                                    weight_mask = global_entropy.detach()

                                elif metric_criterion.adapt_sample == 'rel_uncertainty':

                                    local_entropy = self.get_entropy(local_results['logit'])
                                    global_entropy = self.get_entropy(global_results['logit'])
                                    # unc = torch.log(global_entropy / local_entropy)
                                    unc = torch.log(local_entropy / global_entropy) # penalizing overconfident samples
                                    weight_mask = unc.detach()

                                else:
                                    raise ValueError

                                
                                loss_metric = weight_mask * metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, target=labels, reduction=False, topk_neg=topk_neg)
                            else:
                                uncertainty = self.get_entropy(global_results['logit'])
                                loss_metric = metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, target=labels, reduction=False, topk_neg=topk_neg,
                                                               uncertainty=uncertainty)

                            if metric_name not in losses:
                                losses[metric_name] = []
                            losses[metric_name].append(loss_metric.mean())
            
        for loss_name in losses:
            try:
                losses[loss_name] = torch.mean(torch.stack(losses[loss_name])) if len(losses[loss_name]) > 0 else 0
            except:
                breakpoint()

        #breakpoint()

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

        # from utils.loss import KL_u_p_loss
        # uniform_loss = KL_u_p_loss(results["logit"]).mean()
        # losses["uniform"] = uniform_loss

        with torch.no_grad():
            global_results = self.global_model(images, no_relu=no_relu)

        losses.update(self._algorithm_metric(local_results=results, global_results=global_results, labels=labels))               

        features = {
            "local": results,
            "global": global_results
        }
        #breakpoint()
        return losses, features


    # @property
    def get_weights(self, epoch=None):
        args_metric = self.args.client.metric_loss

        weights = {
            "cls": self.args.client.ce_loss.weight,
            "cossim": self.args.client.feature_align_loss.weight,
            'loc_var_sample_prototype': self.args.client.loc_var_sample_prototype.weight,
            'inverse_var_of_prototypes': self.args.client.inverse_var_of_prototypes.weight,
            'diff_loc_glob_prototypes_mean':self.args.client.diff_loc_glob_prototypes_mean.weight
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

        # save for which learn faster
        models_dict = {}
        initial_local_model = copy.deepcopy(self.model)
        models_dict['saved_initial_model'] = initial_local_model


        for local_epoch in range(self.args.trainer.local_epochs):
            end = time.time()

            for i, (images, labels) in enumerate(self.loader):

                images, labels = images.to(self.device), labels.to(self.device)
                no_relu = not client_args.interpolation.feature_relu
                self.interpolater.update()
                self.model.zero_grad()

                #with autocast(enabled=self.args.use_amp):
                losses, features = self._algorithm(images, labels)            
                #breakpoint()
                #print(losses)
                loss_metric = sum([self.weights[loss_key]*losses[loss_key] for loss_key in losses])
                    

                #scaler.scale(loss_metric).backward()
                # scaler.unscale_(self.optimizer)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)


                #extra part begin
                #self.interpolater.local_model.zero_grad()

                #with autocast(enabled=self.args.use_amp):
                # results = self.model(images)
                if client_args.interpolation.ce_weight+client_args.interpolation.kl_weight  >0:
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

                    loss_extra =  client_args.interpolation.ce_weight * inter_ce_loss +   client_args.interpolation.kl_weight * inter_kl_loss    
                    loss =  loss_metric + loss_extra
                    #After 10.7 pm 7:34, delete main_celoss in extra because it already exist in the loss_metric
                else:
                    inter_ce_loss, inter_kl_loss = torch.tensor([0]),torch.tensor([0])
                    loss = loss_metric
                   

                #extra part end
                
                
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
            if local_epoch % 1 == 0:
                this_str = 'saved_model_at_' + str(local_epoch)
                models_dict[this_str] = {}
                models_dict[this_str]['epoch'] = local_epoch
                models_dict[this_str]['model'] = copy.deepcopy(self.model)
        
        logger.info(f"[C{self.client_index}] End. Time: {end-start:.2f}s, Loss: {loss_meter.avg:.3f}, InterCE: {inter_ce_losses_meter.avg:.3f}, InterKL: {inter_kl_losses_meter.avg:.3f}")

        models_dict['saved_last_model'] = copy.deepcopy(self.model)
        distance_dict = {}
        for k in list(models_dict.keys()):
            if 'saved_model_at_' in k:
                models_dict['model_now'] = models_dict[k]['model']
                models_dict = cal_distances_between_models(models_dict)
                for name in ['distance_current_last','ratio_distance_current_last']:
                    for key in models_dict[name].keys():
                        this_str = str(name) + "/" + str(key) + "/local_epoch" + str( models_dict[k]['epoch']) 
                        distance_dict[this_str] = models_dict[name][key]




        self.model.to('cpu')
        self.global_model.to('cpu')
        if self.interpolater:
            self.interpolater.to('cpu')

        loss_dict = {f'loss/{self.args.dataset.name}/{loss_key}': float(losses[loss_key]) for loss_key in losses}
        #extra
        loss_dict.update({
            f'loss/{self.args.dataset.name}/inter_cls': inter_ce_losses_meter.avg,
            f'loss/{self.args.dataset.name}/inter_kl': inter_kl_losses_meter.avg,
        })

        #
        loss_dict.update(distance_dict)


        # if global_epoch > 0 and self.args.eval.local_freq > 0 and global_epoch % self.args.eval.local_freq == 0:
        #     local_loss_dict = self.local_evaluate(global_epoch=global_epoch)
        #     loss_dict.update(local_loss_dict)
        
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
    













