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
class MetricClient(Client):

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
            if self.args.get('ml2'):
                self.metric_criterions[pair.name] = MetricLoss2(pair=pair, **args_metric)
            elif 'triplet' in pair.name:
                self.metric_criterions[pair.name] = TripletLoss(pair=pair, **args_metric)
            elif self.args.get('rel_mode'):
                self.metric_criterions[pair.name] = MetricLoss_rel(pair=pair, **args_metric)
            elif args_metric.get('criterion_type'):
                if args_metric.criterion_type == 'unsupervised':
                    self.metric_criterions[pair.name] = UnsupMetricLoss(pair=pair, **args_metric)
                elif args_metric.criterion_type == 'subset':
                    self.metric_criterions[pair.name] = MetricLossSubset(pair=pair, **args_metric)
                else:
                    raise ValueError
            else:
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
                                   num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, drop_last=False)

        self.optimizer = optim.SGD(self.model.parameters(), lr=local_lr, momentum=self.args.optimizer.momentum, weight_decay=self.args.optimizer.wd)
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, 
                                                     lr_lambda=lambda epoch: self.args.trainer.local_lr_decay ** epoch)
        
        
        self.class_counts = np.sort([*local_dataset.class_dict.values()])[::-1]

        self.num_classes = len(self.loader.dataset.dataset.classes)

        self.class_stats['ratio'] = torch.zeros(self.num_classes)
        for class_key in local_dataset.class_dict:
            self.class_stats['ratio'][int(class_key)] = local_dataset.class_dict[class_key]
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


    def _algorithm_metric(self, local_results, global_results, labels,):

        losses = {
            'cossim': [],
        }

        # if self.args.client.get('label_noise'):
        #     noise_ratio = self.args.client.label_noise.ratio
        #     labels_ = apply_label_noise(labels, noise_ratio)
        #     labels = labels_

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

            # if self.args.get('debuga'):
                # breakpoint()
            # if len(assign) < S-1:
            #     pass
            #     # clamp_indices.append(assign[:1])
            # else:
            #     clamp_indices.append(assign[label_indices])

            # while len(assign) > S:
            #     assign = assign[S:]
            #     splited_labels[assign] = new_label_i
            #     new_label_i += 1

            # if len(assign) > S:
            #     splited_labels[assign[len(assign)//2:]] = new_label_i
            #     new_label_i += 1
        
        bal_indices = torch.cat(bal_indices)

        # if len(clamp_indices) > 0:
        # try:
            # clamp_indices = torch.cat(clamp_indices)
        # except:
        #     breakpoint()


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

            if f"layer{l}_aug" in local_results:
                local_feature_l_aug = local_results[f"layer{l}_aug"]
                global_feature_l_aug = global_results[f"layer{l}_aug"]

                if len(local_feature_l_aug.shape) == 4:
                    local_feature_l_aug = F.adaptive_avg_pool2d(local_feature_l_aug, 1)
                    global_feature_l_aug = F.adaptive_avg_pool2d(global_feature_l_aug, 1)

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

                            if metric_criterion.sampling:
                                if metric_criterion.sampling == 'balance':

                                    if self.args.get('debuga'):
                                        breakpoint()

                                    loss_metric = metric_criterion(old_feat=global_feature_l[bal_indices], 
                                                                   new_feat=local_feature_l[bal_indices],
                                                                   target=labels[bal_indices],
                                                                   reduction=False, topk_neg=topk_neg)
                                    
                                elif metric_criterion.sampling == 'clamp':

                                    if len(clamp_indices) == 0:
                                        loss_metric = torch.zeros(1)
                                    else:
                                        loss_metric = metric_criterion(old_feat=global_feature_l[clamp_indices], 
                                                                    new_feat=local_feature_l[clamp_indices],
                                                                    target=labels[clamp_indices],
                                                                    reduction=False, topk_neg=topk_neg)

                                elif metric_criterion.sampling == 'split':
                                    loss_metric = metric_criterion(old_feat=global_feature_l, 
                                                                   new_feat=local_feature_l,
                                                                   target=splited_labels,
                                                                   reduction=False, topk_neg=topk_neg)

                                elif metric_criterion.sampling == 'half':
                                    L = global_feature_l.size(0)
                                    loss_metric1 = metric_criterion(old_feat=global_feature_l[:L//2], 
                                                                   new_feat=local_feature_l[:L//2],
                                                                   target=labels[:L//2],
                                                                   reduction=False, topk_neg=topk_neg)
                                    loss_metric2 = metric_criterion(old_feat=global_feature_l[L//2:], 
                                                                    new_feat=local_feature_l[L//2:],
                                                                    target=labels[L//2:],
                                                                    reduction=False, topk_neg=topk_neg)
                                    
                                    loss_metric = torch.cat((loss_metric1, loss_metric2))

                                elif metric_criterion.sampling == 'half1':
                                    L = global_feature_l.size(0)
                                    loss_metric1 = metric_criterion(old_feat=global_feature_l[:L//2], 
                                                                   new_feat=local_feature_l[:L//2],
                                                                   target=labels[:L//2],
                                                                   reduction=False, topk_neg=topk_neg)
                                    loss_metric = loss_metric1

                                elif metric_criterion.sampling == 'progress2':

                                    if metric_args.get('sampling_ranges'):
                                        sampling_ranges = metric_args.sampling_ranges
                                        for range_epoch in sampling_ranges:
                                            if self.global_epoch >= range_epoch:
                                                sampling_range = sampling_ranges[range_epoch]

                                        # weights[pair.name] = weight

                                    L = global_feature_l.size(0)
                                    # split = int(1/sampling_range)

                                    loss_metric = metric_criterion(old_feat=global_feature_l, 
                                                                   new_feat=local_feature_l,
                                                                   target=labels,
                                                                   sampling_range=sampling_range,
                                                                   reduction=False, topk_neg=topk_neg)
                                    
                                elif metric_criterion.sampling == 'progress':

                                    if metric_args.get('sampling_ranges'):
                                        sampling_ranges = metric_args.sampling_ranges
                                        for range_epoch in sampling_ranges:
                                            if self.global_epoch >= range_epoch:
                                                sampling_range = sampling_ranges[range_epoch]

                                    L = global_feature_l.size(0)
                                    split = int(1/sampling_range)

                                    loss_metric = []
                                    for i in range(split):
                                        # loss_metric_i = metric_criterion(old_feat=global_feature_l[int(i*(L//split)):int((i+1)*L//split)], 
                                        #                            new_feat=local_feature_l[int(i*(L//split)):int((i+1)*L//split)],
                                        #                            target=labels[int(i*(L//split)):int((i+1)*L//split)],
                                        #                            reduction=False, topk_neg=topk_neg)
                                        range_i = torch.range(int(i*(L*sampling_range)), int((i+1)*L*sampling_range)).long()
                                        loss_metric_i = metric_criterion(old_feat=global_feature_l[range_i], 
                                                                   new_feat=local_feature_l[range_i],
                                                                   target=labels[range_i],
                                                                   reduction=False, topk_neg=topk_neg)
                                        loss_metric.append(loss_metric_i)
                                        
                                    loss_metric = torch.cat(loss_metric)
                                    
                                else:
                                    raise ValueError


                            
                            elif metric_criterion.adapt_sample:
                                overall_weight = 1.

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

                                elif metric_criterion.adapt_sample == 'class_balance_overall':
                                    class_counts = torch.zeros(len(local_dataset.dataset.classes))
                                    for key in local_dataset.class_dict:
                                        class_counts[int(key)] = local_dataset.class_dict[key]

                                    class_weights = 1/class_counts
                                    mean_class_weights = (class_weights * class_counts).nansum() / len(local_dataset)
                                    weight_mask = class_weights[labels]
                                    weight_mask /= mean_class_weights
                                    weight_mask = torch.FloatTensor(weight_mask).to(self.device)

                                elif metric_criterion.adapt_sample == 'batch_class_balance':
                                    # class_counts = torch.zeros(len(local_dataset.dataset.classes))
                                    # for key in local_dataset.class_dict:
                                    #     class_counts[int(key)] = local_dataset.class_dict[key]
                                    batch_class_counts = labels.bincount()

                                    class_weights = 1/batch_class_counts
                                    weight_mask = class_weights[labels]
                                    weight_mask /= weight_mask.mean()

                                elif metric_criterion.adapt_sample == 'class_balance_sq':
                                    class_counts = torch.zeros(len(local_dataset.dataset.classes))
                                    for key in local_dataset.class_dict:
                                        class_counts[int(key)] = local_dataset.class_dict[key]

                                    class_weights = (1/class_counts)**2
                                    weight_mask = class_weights[labels]
                                    weight_mask /= weight_mask.mean()
                                    weight_mask = torch.FloatTensor(weight_mask).to(self.device)

                                elif metric_criterion.adapt_sample == 'class_balance_sqrt':
                                    class_counts = torch.zeros(len(local_dataset.dataset.classes))
                                    for key in local_dataset.class_dict:
                                        class_counts[int(key)] = local_dataset.class_dict[key]

                                    class_weights = (1/class_counts)**0.5
                                    weight_mask = class_weights[labels]
                                    weight_mask /= weight_mask.mean()
                                    weight_mask = torch.FloatTensor(weight_mask).to(self.device)

                                elif metric_criterion.adapt_sample == 'rev_class_balance':
                                    class_counts = torch.zeros(len(local_dataset.dataset.classes))
                                    for key in local_dataset.class_dict:
                                        class_counts[int(key)] = local_dataset.class_dict[key]

                                    class_weights = class_counts
                                    weight_mask = class_weights[labels]
                                    weight_mask /= weight_mask.mean()
                                    weight_mask = torch.FloatTensor(weight_mask).to(self.device)

                                elif metric_criterion.adapt_sample == 'rev_class_balance_sq':
                                    class_counts = torch.zeros(len(local_dataset.dataset.classes))
                                    for key in local_dataset.class_dict:
                                        class_counts[int(key)] = local_dataset.class_dict[key]

                                    class_weights = class_counts ** 2
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
                                    # unc = (global_entropy / local_entropy)
                                    unc = torch.log(global_entropy / local_entropy) # penalizing overconfident samples
                                    weight_mask = unc.detach()
                                    weight_mask = torch.clamp(weight_mask, 0)

                                elif metric_criterion.adapt_sample in ['within_cov', 'within_cov_rel', 'within_cov_rel2', 'class_sep', 'within+ratio', 'within_all']:

                                    class_counts = torch.zeros(len(local_dataset.dataset.classes))
                                    for key in local_dataset.class_dict:
                                        class_counts[int(key)] = local_dataset.class_dict[key]

                                    class_counts = torch.FloatTensor(class_counts).to(self.device)

                                    layer_name = f"layer{l}"

                                    if self.class_stats['cov'].get(layer_name):
                                        layer_stat = self.class_stats['cov'][layer_name]
                                        total_cov = layer_stat["total_cov"]
                                        within_cov_class = layer_stat["within_cov_class"]
                                        rel_cov_class = within_cov_class / total_cov

                                        # self.class_stats['cov'][layer_name]["within_cov_class"] / self.class_stats['cov_global'][layer_name]["within_cov_class"]
                                        # self.class_stats['cov'][layer_name]["within_cov"] / self.class_stats['cov_global'][layer_name]["within_cov"]
                                        # self.class_stats['cov'][layer_name]["within_cov"] / self.class_stats['cov'][layer_name]["total_cov"]
                                        # self.class_stats['cov_global'][layer_name]["within_cov"] / self.class_stats['cov_global'][layer_name]["total_cov"]

                                        layer_stat_global = self.class_stats['cov_global'][layer_name]
                                        within_cov_class_global = layer_stat_global["within_cov_class"]
                                        total_cov_global = layer_stat_global["total_cov"]
                                        rel_cov_class_global = within_cov_class_global / total_cov_global

                                        rel_local_global = rel_cov_class / rel_cov_class_global
                                        
                                        if metric_criterion.adapt_sample == 'within_cov_rel':
                                            class_weights = rel_local_global.detach()
                                        elif metric_criterion.adapt_sample == 'within_cov_rel2':
                                            class_weights = (within_cov_class / within_cov_class_global).detach()
                                        elif metric_criterion.adapt_sample == 'within_cov':
                                            class_weights = within_cov_class.detach() 
                                        elif metric_criterion.adapt_sample == 'within_all':
                                            overall_weight = self.class_stats['cov'][layer_name]['within_cov']/self.class_stats['cov_global'][layer_name]['within_cov']
                                            # print(overall_weight, l)
                                            overall_weight = 1
                                            class_weights = torch.ones_like(class_counts)
                                        elif metric_criterion.adapt_sample == 'class_sep':
                                            class_weights = layer_stat['within_dist_class']/layer_stat['total_dist_class']

                                        elif metric_criterion.adapt_sample == 'within+ratio':
                                            class_weight1 = (1/class_counts)**0.5
                                            # weight_mask = class_weights[labels]

                                            class_weights2 = (within_cov_class / within_cov_class_global).detach()

                                            class_weights = (class_weight1 * class_weights2)**0.5

                                            # local_w = layer_stat["within_cov"] / layer_stat["total_cov"]
                                            # global_w = layer_stat_global["within_cov"] / layer_stat_global["total_cov"]

                                            # overall_weight = local_w / global_w
                                            # if torch.rand(1).item() < 0.001:
                                            #     print(overall_weight, l)
                                            # class_weights = within_cov_class.detach()/class_counts
                                        else:
                                            raise ValueError

                                        weight_mask = class_weights[labels]
                                        weight_mask /= weight_mask.nanmean()
                                        weight_mask[weight_mask.isnan()] = 1
                                        # print(weight_mask.mean(), weight_mask.min(), weight_mask.max(), l)
                                    else:
                                        if metric_criterion.adapt_sample == 'within+ratio':
                                            class_weights = (1/class_counts)**0.5
                                            weight_mask = class_weights[labels]
                                            weight_mask /= weight_mask.nanmean()
                                            weight_mask[weight_mask.isnan()] = 1
                                        else:
                                            weight_mask = 1

                                else:
                                    raise ValueError

                                
                                loss_metric = overall_weight * weight_mask * metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, target=labels, reduction=False, topk_neg=topk_neg)

                            else:
                                # overconfidence
                                # class_counts = torch.Tensor(list(self.class_counts))
                                # # class_counts = labels.bincount()
                                # # class_counts = class_counts[class_counts.nonzero()].squeeze()
                                # class_entropy = torch.distributions.Categorical(class_counts).entropy()
                                # uniform_entropy = torch.distributions.Categorical(torch.ones_like(class_counts)).entropy()


                                # uncertainty = 1 - class_entropy / uniform_entropy
                                class_ratio = self.class_stats['ratio'][labels.to('cpu')]
                                # uncertainty = self.get_entropy(local_results['logit'])
                                uncertainty = F.cross_entropy(local_results["logit"], labels, reduction='none').detach()

                                # if self.args.get('debugs'):
                                #     print("unc : ", uncertainty)
                                #     breakpoint()
                                # breakpoint()
                                # uncertainty = (1 - self.get_entropy(local_results['logit']) / self.get_entropy(global_results['logit'])).clamp(0)

                                if self.args.get('ml2'):
                                    #print(self.args.get('ml2'))
                                    loss_metric, loss_dist_opt_sim, pos_mean_sim, neg_mean_sim = metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, target=labels, reduction=False, topk_neg=topk_neg,
                                                               uncertainty=uncertainty, class_ratio=class_ratio, level=l, progress=self.current_progress, mode = str(self.args.get('ml2')))
                                elif self.args.get('rel_mode'):
                                    loss_metric = metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, target=labels, reduction=False, topk_neg=topk_neg,
                                                               uncertainty=uncertainty, class_ratio=class_ratio, level=l, progress=self.current_progress, mode =self.args.get('rel_mode'))

                                
                                elif 'triplet' in metric_name:
                                    loss_metric, pair_poss, pair_negs = metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, target=labels, reduction=False, topk_neg=topk_neg,
                                                               uncertainty=uncertainty, class_ratio=class_ratio, level=l, progress=self.current_progress, pos_weight = self.weights[metric_name + "/pos"], neg_weight = self.weights[metric_name + "/neg"], threshold = self.weights[metric_name + "/threshold"])

                                elif self.args.client.metric_loss.get('criterion_type') == 'unsupervised':
                                    loss_metric = metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, new_feat_aug=local_feature_l_aug, target=labels, reduction=False, topk_neg=topk_neg,
                                                               uncertainty=uncertainty, class_ratio=class_ratio, level=l, progress=self.current_progress)

                                else:
                                    loss_metric = metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, target=labels, reduction=False, topk_neg=topk_neg,
                                                               uncertainty=uncertainty, class_ratio=class_ratio, level=l, progress=self.current_progress)

            
                            if metric_name not in losses:
                                losses[metric_name] = []
                                if 'triplet' in metric_name:
                                    #metric_name is pair name
                                    for name_pos in pair_poss.keys():
                                        losses[metric_name + "/" + name_pos +"/avgpos"] = [] #pair_poss[name_pos]
                                    for name_neg in pair_negs.keys():
                                        losses[metric_name + "/" + name_neg +"/avgneg"] = [] #pair_negs[name_neg]
                                else:
                                    if self.args.get('ml2'):
                                        losses[metric_name + "_dist_opt_sim"] = []
                                        losses[metric_name + "_pos_mean_sim"] = []
                                        losses[metric_name + "_neg_mean_sim"] = []
                                



                            if 'triplet' in metric_name:
                                losses[metric_name].append(loss_metric.mean())
                                
                                for name_pos in pair_poss.keys():
                                    losses[metric_name + "/" + name_pos +"/avgpos"].append(pair_poss[name_pos]) 
                                for name_neg in pair_negs.keys():
                                    losses[metric_name + "/" + name_neg +"/avgneg"].append(pair_negs[name_neg])


                            else:
                                losses[metric_name].append(loss_metric.mean())
                                if self.args.get('ml2'):
                                    losses[metric_name + "_dist_opt_sim"].append(loss_dist_opt_sim.mean())
                                    losses[metric_name + "_pos_mean_sim"].append(pos_mean_sim.mean())
                                    losses[metric_name + "_neg_mean_sim"].append(neg_mean_sim.mean())

        for loss_name in losses:
            try:
                losses[loss_name] = torch.mean(torch.stack(losses[loss_name])) if len(losses[loss_name]) > 0 else 0
            except:
                breakpoint()

        return losses


    def _algorithm(self, images, labels, images_aug=None) -> Dict:


        losses = defaultdict(float)
        no_relu = not self.args.client.metric_loss.feature_relu
        results = self.model(images, no_relu=no_relu)
        with torch.no_grad():
            global_results = self.global_model(images, no_relu=no_relu)

        results_aug, global_results_aug = None, None
        if images_aug is not None:
            results_aug = self.model(images_aug, no_relu=no_relu)
            with torch.no_grad():
                global_results_aug = self.global_model(images_aug, no_relu=no_relu)

            for key in results_aug:
                results[f"{key}_aug"] = results_aug[key]
                global_results[f"{key}_aug"] = global_results_aug[key]

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

        ## Weight L2 loss
        prox_loss = 0
        fixed_params = {n:p for n,p in self.global_model.named_parameters()}
        for n, p in self.model.named_parameters():
            prox_loss += ((p-fixed_params[n].detach())**2).sum()  

        losses["prox"] = prox_loss

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
            if 'triplet' in pair.name:
                #Triplet weights are given to forwarding metric_criterion, unlike MetricLoss
                weights[pair.name] = pair.weight
                weights[pair.name + "/pos"] = pair.pos_weight
                weights[pair.name + "/neg"] = pair.pos_weight
                weights[pair.name + "/threshold"] = pair.threshold
            else:
                if pair.get('weights'):
                    for weight_epoch in pair.weights:
                        if epoch >= weight_epoch:
                            weight = pair.weights[weight_epoch]
                    weights[pair.name] = weight
                    if self.args.get('ml2'): 
                        weights[pair.name + "_dist_opt_sim"] = weight
                        weights[pair.name + "_pos_mean_sim"] = weight*0
                        weights[pair.name + "_neg_mean_sim"] = weight*0

                else:
                    weights[pair.name] = pair.weight
                    if self.args.get('ml2'): 
                        weights[pair.name + "_dist_opt_sim"] = pair.weight
                        weights[pair.name + "_pos_mean_sim"] = pair.weight*0
                        weights[pair.name + "_neg_mean_sim"] = pair.weight*0

        if self.args.client.get('LC'):
            weights['LC'] = self.args.client.LC.weight
        if self.args.client.get('decorr_loss'):
            weights['decorr'] = self.args.client.decorr_loss.weight
        if self.args.client.get('prox_loss'):
            weights['prox'] = self.args.client.prox_loss.weight
            
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

            # if self.args.get('debugs'):
            #     with torch.no_grad():
            #         for i, (images, labels) in enumerate(self.loader):
            #             images, labels = images.to(self.device), labels.to(self.device)
            #             self.model.zero_grad()

            #             with autocast(enabled=self.args.use_amp):
            #                 losses, features = self._algorithm(images, labels,)

                            
            #                 # for key in features['local']:
            #                 for key in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5']:
            #                     feat_local = features['local'][key].detach()
            #                     if len(feat_local.shape) == 4:
            #                         feat_local = F.adaptive_avg_pool2d(feat_local, 1)
            #                     all_features_local[key].append(feat_local)
            #                 # for key in features['global']:
            #                     feat_global = features['global'][key].detach()
            #                     if len(feat_global.shape) == 4:
            #                         feat_global = F.adaptive_avg_pool2d(feat_global, 1)
            #                     all_features_global[key].append(feat_global)

            #                 all_labels.append(labels)

            #         self.update_cov_results(all_features_local, all_features_global, all_labels)

                # if local_epoch > 1:
                #     breakpoint()

            end = time.time()
            for i, (images, labels) in enumerate(self.loader):

                images_aug = None
                # if len(images) == 2:
                #     print("augmenting?")
                #     print("image_size:", images.size())
                #     images, images_aug = images
                #     images_aug = images_aug.to(self.device)
                # elif len(images.size()) == 3:
                #     print("dimension expended!")
                #     images = images.unsqueeze(0)

                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()

                with autocast(enabled=self.args.use_amp):
                    losses, features = self._algorithm(images, labels, images_aug=images_aug)

                    # if self.args.get('debugs'):
                    #     # for key in features['local']:
                    #     for key in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5']:
                    #         feat_local = features['local'][key].detach()
                    #         if len(feat_local.shape) == 4:
                    #             feat_local = F.adaptive_avg_pool2d(feat_local, 1)
                    #         all_features_local[key].append(feat_local)
                    #     # for key in features['global']:
                    #         feat_global = features['global'][key].detach()
                    #         if len(feat_global.shape) == 4:
                    #             feat_global = F.adaptive_avg_pool2d(feat_global, 1)
                    #         all_features_global[key].append(feat_global)

                    #     all_labels.append(labels)

                        # if local_epoch == 0 and (i+1) % 2 == 0 and (i+1) < len(self.loader):
                        #     self.update_cov_results(all_features_local, all_features_global, all_labels)
                        # all_features_local.append(features['local'])
                        # all_features_global.append(features['global'])
                        # breakpoint()

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

                # if self.args.get('debugs'):
                #     breakpoint()

                # Entropy
                entropy = self.get_entropy(features['local']['logit']).mean()
                global_entropy = self.get_entropy(features['global']['logit']).mean()

                # if self.args.get('debugs'):
                #     breakpoint()

                # local_score = F.softmax(features['local']['logit'], 1).detach().double()
                # entropy = torch.distributions.Categorical(local_score).entropy().mean()
                # uniform_entropy = torch.distributions.Categorical(torch.ones_like(local_score)).entropy().mean()
                # entropy_ = entropy / uniform_entropy
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

            ### calculate covs
            # if self.args.get('debugs'):
                # print("update whole", i)
                # self.update_cov_results(all_features_local, all_features_global, all_labels)
                # all_labels = torch.cat(all_labels)
                # # for key in all_features_local:
                # for key in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5']:
                #     all_features_local[key] = torch.cat(all_features_local[key])
                #     cov_results = _get_covariance_results(all_features_local[key].squeeze(), all_labels, self.num_classes)
                #     self.class_stats['cov'][key] = cov_results
                # # for key in all_features_global:
                #     all_features_global[key] = torch.cat(all_features_global[key])
                #     cov_global_results = _get_covariance_results(all_features_global[key].squeeze(), all_labels, self.num_classes)
                #     self.class_stats['cov_global'][key] = cov_global_results
                # breakpoint()

        
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
    













############### OUTDATED #####################



@CLIENT_REGISTRY.register()
class MetricInterpolateClient(MetricClient):

    # pass

    def setup(self, model, device, local_dataset, local_lr, global_epoch, trainer, **kwargs):
        
        # if self.model is None:
        #     self.model = model
        # else:
        #     self.model.load_state_dict(model.state_dict())

        # if self.global_model is None:
        #     self.global_model = copy.deepcopy(self.model)
        # else:
        #     self.global_model.load_state_dict(model.state_dict())
        self._update_model(model)
        self._update_global_model(model)

        for fixed_model in [self.global_model]:
            for n, p in fixed_model.named_parameters():
                p.requires_grad = False

        self.interpolater = Interpolater(local_model=self.model, global_model=self.global_model, args=self.args)

        self.device = device
        self.num_layers = model.num_layers
        #print("model.num_layers:", model.num_layers)
        #self.num_layers = 6

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
        self.trainer = trainer

        sorted_key = np.sort([*local_dataset.class_dict.keys()])
        sorted_class_dict = {} 
        for key in sorted_key:  
            sorted_class_dict[key] = local_dataset.class_dict[key]

        # self.class_ratios = torch.zeros(200)

        # for class_key in local_dataset.class_dict:
        #     self.class_ratios[int(class_key)] = local_dataset.class_dict[class_key]
        for class_key in local_dataset.class_dict:
            self.class_stats['ratio'][int(class_key)] = local_dataset.class_dict[class_key]

        if global_epoch == 0:
            logger.info(f"Class counts : {self.class_counts}")
            logger.info(f"Sorted class dict : {sorted_class_dict}")
            

    # def get_weights(self, epoch=None):
    #     args_metric = self.args.client.metric_loss
    #     weights = {
    #         "cls": self.args.client.ce_loss.weight,
    #         "cossim": self.args.client.feature_align_loss.weight,}
        
    #     for pair in args_metric.pairs:
    #         if pair.get('weights'):
    #             for weight_epoch in pair.weights:
    #                 if epoch >= weight_epoch:
    #                     weight = pair.weights[weight_epoch]
    #             weights[pair.name] = weight
    #         else:
    #             weights[pair.name] = pair.weight
    #     return weights
    


    def _algorithm(self, images, labels, ) -> Dict:

        args_metric = self.args.client.metric_loss

        losses = defaultdict(float)
        no_relu = not args_metric.feature_relu

        results = self.model(images, no_relu=no_relu)
        cls_loss = self.criterion(results["logit"], labels)
        losses["cls"] = cls_loss

        with torch.no_grad():
            global_results = self.global_model(images, no_relu=no_relu)

        if args_metric.interpolation_type == 'stochastic':
            inter_results = self.interpolater.forward_stoc(images, repeat=1, no_relu=no_relu)[0]
        elif args_metric.interpolation_type == 'full':
            inter_results = self.interpolater.forward_inter(images, no_relu=no_relu)
        elif args_metric.interpolation_type == 'ensemble':
            inter_results = {key: (results[key]+global_results[key])/2 for key in results}

        losses.update(self._algorithm_metric(local_results=results, global_results=inter_results, labels=labels))   

        features = {
            "local": results,
            "global": global_results
        }
        return losses, features


    # def _algorithm(self, images, labels,) -> Dict:

    #     losses = defaultdict(float)
    #     no_relu = not self.args.client.metric_loss.feature_relu
    #     results = self.model(images, no_relu=no_relu)

    #     if self.args.client.get('label_noise'):
    #         noise_ratio = self.args.client.label_noise.ratio
    #         labels = apply_label_noise(labels, noise_ratio)

    #     cls_loss = self.criterion(results["logit"], labels)
    #     losses["cls"] = cls_loss

    #     from utils.loss import KL_u_p_loss
    #     uniform_loss = KL_u_p_loss(results["logit"]).mean()
    #     losses["uniform"] = uniform_loss

    #     # if self.args.get('debugs'):
    #     #     breakpoint()

    #     with torch.no_grad():
    #         global_results = self.global_model(images, no_relu=no_relu)

    #     losses.update(self._algorithm_metric(local_results=results, global_results=global_results, labels=labels))               

    #     features = {
    #         "local": results,
    #         "global": global_results
    #     }

    #     return losses, features








@CLIENT_REGISTRY.register()
class MetricInterpolateUpdateClient(MetricClient):

    # pass

    def setup(self, model, device, local_dataset, local_lr, global_epoch, trainer, **kwargs):
        
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

        self.interpolater = Interpolater(local_model=self.model, global_model=self.global_model, args=self.args)

        self.device = device
        self.num_layers = model.num_layers
        #print("model.num_layers:", model.num_layers)
        #self.num_layers = 6

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

        self.num_classes = len(self.loader.dataset.dataset.classes)
        self.class_stats['ratio'] = torch.zeros(self.num_classes)
        for class_key in local_dataset.class_dict:
            self.class_stats['ratio'][int(class_key)] = local_dataset.class_dict[class_key]

        self.trainer = trainer

        sorted_key = np.sort([*local_dataset.class_dict.keys()])
        sorted_class_dict = {} 
        for key in sorted_key:  
            sorted_class_dict[key] = local_dataset.class_dict[key]

        # self.class_ratios = torch.zeros(200)

        # for class_key in local_dataset.class_dict:
        #     self.class_ratios[int(class_key)] = local_dataset.class_dict[class_key]
        for class_key in local_dataset.class_dict:
            self.class_stats['ratio'][int(class_key)] = local_dataset.class_dict[class_key]

        if global_epoch == 0:
            logger.info(f"Class counts : {self.class_counts}")
            logger.info(f"Sorted class dict : {sorted_class_dict}")
            


    def _algorithm(self, images, labels, ) -> Dict:

        args_metric = self.args.client.metric_loss

        losses = defaultdict(float)
        no_relu = not args_metric.feature_relu

        results = self.model(images, no_relu=no_relu)
        cls_loss = self.criterion(results["logit"], labels)
        losses["cls"] = cls_loss

        with torch.no_grad():
            global_results = self.global_model(images, no_relu=no_relu)

        if args_metric.interpolation_type == 'stochastic':
            inter_results = self.interpolater.forward_stoc(images, repeat=1, no_relu=no_relu)[0]
        elif args_metric.interpolation_type == 'full':
            inter_results = self.interpolater.forward_inter(images, no_relu=no_relu)
        elif args_metric.interpolation_type == 'ensemble':
            inter_results = {key: (results[key]+global_results[key])/2 for key in results}

        losses.update(self._algorithm_metric(local_results=results, global_results=inter_results, labels=labels))   

        features = {
            "local": results,
            "global": global_results
        }
        return losses, features


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

        
        for local_epoch in range(self.args.trainer.local_epochs):

            # breakpoint()
            interpolate_state_dict = self.interpolater.get_interpolate_state_dict(factor=0.5)
            self.model.load_state_dict(interpolate_state_dict)

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

                # if self.args.get('debugs'):
                #     breakpoint()

                # Entropy
                entropy = self.get_entropy(features['local']['logit']).mean()
                global_entropy = self.get_entropy(features['global']['logit']).mean()

                # if self.args.get('debugs'):
                #     breakpoint()

                # local_score = F.softmax(features['local']['logit'], 1).detach().double()
                # entropy = torch.distributions.Categorical(local_score).entropy().mean()
                # uniform_entropy = torch.distributions.Categorical(torch.ones_like(local_score)).entropy().mean()
                # entropy_ = entropy / uniform_entropy
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

        return self.model, loss_dict





# concatenating layer0~layer4 features as a single feature (not work well)
@CLIENT_REGISTRY.register()
class MetricConcatenateClient(MetricInterpolateClient):


    def _algorithm_metric(self, local_results, global_results, labels):

        losses = {
            'cossim': [],
        }

        local_feature, global_feature = [], []

        for l in range(self.num_layers):

            train_layer = False
            if l in self.args.client.metric_loss.branch_level:
                train_layer = True
                
            local_feature_l = local_results[f"layer{l}"]
            global_feature_l = global_results[f"layer{l}"]

            if len(local_feature_l.shape) == 4:
                local_feature_l = F.adaptive_avg_pool2d(local_feature_l, 1)
                global_feature_l = F.adaptive_avg_pool2d(global_feature_l, 1)

            # Metric Loss
            if train_layer:
                local_feature.append(local_feature_l)
                global_feature.append(global_feature_l)

        local_feature = torch.cat(local_feature, 1)
        global_feature = torch.cat(global_feature, 1)

        for metric_name in self.metric_criterions:
            metric_criterion = self.metric_criterions[metric_name]
            if metric_criterion is not None:
                loss_metric = metric_criterion(old_feat=global_feature, new_feat=local_feature, target=labels, reduction=False)

                if metric_name not in losses:
                    losses[metric_name] = []

                losses[metric_name].append(loss_metric.mean())
            
        for loss_name in losses:
            try:
                losses[loss_name] = torch.mean(torch.stack(losses[loss_name])) if len(losses[loss_name]) > 0 else 0
            except:
                breakpoint()

        return losses










#======# Use Extrapolation for negative augmentation

@CLIENT_REGISTRY.register()
class MetricExtrapolateAugmentClient(MetricInterpolateClient):

    def _algorithm(self, images, labels, ) -> Dict:

        args_metric = self.args.client.metric_loss

        losses = defaultdict(float)
        no_relu = not self.args.client.metric_loss.feature_relu

        results = self.model(images, no_relu=no_relu)
        cls_loss = self.criterion(results["logit"], labels)
        losses["cls"] = cls_loss

        # with torch.no_grad():
            # global_results = self.global_model(images, no_relu=no_relu)

        if args_metric.interpolation_type == 'stochastic':
        # inter_results = self.interpolater.forward_inter(images, no_relu=no_relu)
            inter_results = self.interpolater.forward_stoc(images, repeat=1, no_relu=no_relu)[0]
        elif args_metric.interpolation_type == 'full':
            inter_results = self.interpolater.forward_inter(images, no_relu=no_relu)

        # breakpoint()
        # inter_results = {"logit": inter_results[0]}
        # B = len(labels)
        # for i in range(B-1):
            # if labels[i-1] != labels[i]:
        not_same_label = labels[1:]!=labels[:-1]
        new_labels = torch.ones_like(labels)[1:][not_same_label] * 1001
        breakpoint()
        for key in inter_results:
            feature_l = inter_results[key]
            new_feature_l = (feature_l[:-1] + feature_l[1:])/2

            inter_results[key] = torch.cat((feature_l, new_feature_l[not_same_label]), 0)

        for key in results:
            feature_l = results[key]
            new_feature_l = (feature_l[:-1] + feature_l[1:])/2
            results[key] = torch.cat((feature_l, new_feature_l[not_same_label]), 0)

        labels = torch.cat((labels, new_labels))

        losses.update(self._algorithm_metric(local_results=results, global_results=inter_results, labels=labels))   
        return losses



    def _algorithm_metric(self, local_results, global_results, labels):

        losses = {
            'cossim': [],
        }

        for l in range(self.num_layers):

            train_layer = False
            if l in self.args.client.metric_loss.branch_level:
                train_layer = True
                
            local_feature_l = local_results[f"layer{l}"]
            global_feature_l = global_results[f"layer{l}"]

            if len(local_feature_l.shape) == 4:
                local_feature_l = F.adaptive_avg_pool2d(local_feature_l, 1)
                global_feature_l = F.adaptive_avg_pool2d(global_feature_l, 1)

            # Feature Cossim Loss
            if self.args.client.feature_align_loss.align_type == 'l2':
                loss_cossim = F.mse_loss(local_feature_l.squeeze(), global_feature_l.squeeze())
            else:
                loss_cossim = F.cosine_embedding_loss(local_feature_l.squeeze(), global_feature_l.squeeze(), torch.ones_like(labels))
            losses['cossim'].append(loss_cossim)

            # Metric Loss
            if train_layer:
                for metric_name in self.metric_criterions:
                    metric_criterion = self.metric_criterions[metric_name]
                    if metric_criterion is not None:
                        loss_metric = metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, target=labels, reduction=False)

                        if metric_name not in losses:
                            losses[metric_name] = []

                        losses[metric_name].append(loss_metric[:self.args.batch_size].mean())
            
        for loss_name in losses:
            try:
                losses[loss_name] = torch.mean(torch.stack(losses[loss_name])) if len(losses[loss_name]) > 0 else 0
            except:
                breakpoint()

        return losses






















@CLIENT_REGISTRY.register()
class MetricExtrapolateUniformClient(MetricInterpolateClient):


    def _algorithm_metric(self, local_results, global_results, labels):

        losses = {
            'cossim': [],
            # 'metric': [],
            # 'metric2': [],
            # 'metric3': [],
            # 'metric4': [],
        }

        for l in range(self.num_layers):

            train_layer = False
            if l in self.args.client.metric_loss.branch_level:
                train_layer = True

            local_feature_l = local_results[f"layer{l}"]
            global_feature_l = global_results[f"layer{l}"]
            if len(local_feature_l.shape) == 4:
                local_feature_l = F.adaptive_avg_pool2d(local_feature_l, 1)
                global_feature_l = F.adaptive_avg_pool2d(global_feature_l, 1)

            # Feature Cossim Loss
            if self.args.client.feature_align_loss.align_type == 'l2':
                loss_cossim = F.mse_loss(local_feature_l.squeeze(), global_feature_l.squeeze())
            else:
                loss_cossim = F.cosine_embedding_loss(local_feature_l.squeeze(), global_feature_l.squeeze(), torch.ones_like(labels))
            losses['cossim'].append(loss_cossim)

            # Metric Loss
            if train_layer:
                for metric_name in self.metric_criterions:
                    metric_criterion = self.metric_criterions[metric_name]
                    # breakpoint()

                    if metric_criterion is not None:
                        if metric_criterion.pairs.get('label') == 'random':
                            loss_metric = metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, target=labels, reduction=False)
                        else:
                            loss_metric = metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, target=labels, reduction=False)
                        # breakpoint()
                        if metric_name not in losses:
                            losses[metric_name] = []

                        losses[metric_name].append(loss_metric.mean())
            
        for loss_name in losses:
            try:
                losses[loss_name] = torch.mean(torch.stack(losses[loss_name])) if len(losses[loss_name]) > 0 else 0
            except:
                breakpoint()

        return losses
    

    def _algorithm_uniform(self, inter_results, labels):
        logits = inter_results['logit']
        uniform_tensors = torch.ones(logits.size())
        uniform_dists = torch.autograd.Variable(uniform_tensors / logits.size(1)).to(self.device)
        T = 1
        instance_losses = F.kl_div(F.log_softmax(logits/T, dim=1), uniform_dists, reduce=False).sum(dim=1)
        losses = {
            'uniform': instance_losses.mean()
        }
        return losses

    def local_train(self, global_epoch, **kwargs):
        self.interpolater.to(self.device)
        self.model.to(self.device)
        self.model.train()
        scaler = GradScaler()

        start = time.time()
        loss_meter = AverageMeter('Loss', ':.2f')
        # uniform_loss_meter = AverageMeter('UniformLoss', ':.2f')
        time_meter = AverageMeter('BatchTime', ':3.1f')

        losses = defaultdict(float)
        args_metric = self.args.client.metric_loss
        weights = {
            "cls": self.args.client.ce_loss.weight,
            "cossim": self.args.client.feature_align_loss.weight,
            "uniform": self.args.client.interpolation.uniform_weight,
            }
        for pair in args_metric.pairs:
            weights[pair.name] = pair.weight

        if global_epoch % 50 == 0:
            print(weights)
            print(self.pairs)
            
        for local_epoch in range(self.args.trainer.local_epochs):
            end = time.time()
            self.interpolater.update(progress=global_epoch/self.args.trainer.global_rounds)                    

            for i, (images, labels) in enumerate(self.loader):    
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()

                no_relu = not self.args.client.metric_loss.feature_relu

                with autocast(enabled=self.args.use_amp):                    
                    results = self.model(images, no_relu=no_relu)
                    cls_loss = self.criterion(results["logit"], labels)
                    losses["cls"] = cls_loss

                    with torch.no_grad():
                        global_results = self.global_model(images, no_relu=no_relu)
                        # inter_results = self.interpolater.forward_inter(images, no_relu=no_relu)

                    # inter_results = self.interpolater.forward_stoc_layerwise(images, repeat=1, no_relu=no_relu)
                    # inter_results = {"logit": inter_results[0]}
                    inter_results = self.interpolater.forward_stoc(images, repeat=1, no_relu=no_relu)[0]

                    losses.update(self._algorithm_metric(local_results=results, global_results=global_results, labels=labels))               
                    losses.update(self._algorithm_uniform(inter_results=inter_results, labels=labels))
                
                    loss = sum([weights[loss_key]*losses[loss_key] for loss_key in losses])

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                scaler.step(self.optimizer)
                scaler.update()

                loss_meter.update(loss.item(), images.size(0))
                time_meter.update(time.time() - end)
                end = time.time()

            self.scheduler.step()
        
        logger.info(f"[C{self.client_index}] End. Time: {end-start:.2f}s, Loss: {loss_meter.avg:.3f}")

        self.model.to('cpu')
        self.interpolater.to('cpu')

        loss_dict = {f'loss/{self.args.dataset.name}/{loss_key}': float(losses[loss_key]) for loss_key in losses}


        if global_epoch > 0 and self.args.eval.local_freq > 0 and global_epoch % self.args.eval.local_freq == 0:
            local_loss_dict = self.local_evaluate(global_epoch=global_epoch)
            loss_dict.update(local_loss_dict)

        return self.model, loss_dict



@CLIENT_REGISTRY.register()
class MetricExtrapolateContrastiveClient(MetricInterpolateClient):

    def local_train(self, global_epoch, **kwargs):
        self.interpolater.to(self.device)
        self.model.to(self.device)
        scaler = GradScaler()

        start = time.time()
        loss_meter = AverageMeter('Loss', ':.2f')
        time_meter = AverageMeter('BatchTime', ':3.1f')

        losses = defaultdict(float)
        args_metric = self.args.client.metric_loss
        weights = {
            "cls": self.args.client.ce_loss.weight,
            "cossim": self.args.client.feature_align_loss.weight,
            }
        for pair in args_metric.pairs:
            weights[pair.name] = pair.weight

        if global_epoch % 50 == 0:
            print(weights)
            print(self.pairs)
            
        for local_epoch in range(self.args.trainer.local_epochs):
            end = time.time()
            self.interpolater.update()                    

            for i, (images, labels) in enumerate(self.loader):    
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()

                no_relu = not self.args.client.metric_loss.feature_relu

                with autocast(enabled=self.args.use_amp):                    
                    results = self.model(images, no_relu=no_relu)
                    cls_loss = self.criterion(results["logit"], labels)
                    losses["cls"] = cls_loss

                    # with torch.no_grad():
                        # global_results = self.global_model(images, no_relu=no_relu)

                    if args_metric.interpolation_type == 'stochastic':
                    # inter_results = self.interpolater.forward_inter(images, no_relu=no_relu)
                        inter_results = self.interpolater.forward_stoc(images, repeat=1, no_relu=no_relu)[0]
                    elif args_metric.interpolation_type == 'full':
                        inter_results = self.interpolater.forward_inter(images, no_relu=no_relu)
                    # inter_results = {"logit": inter_results[0]}

                    # breakpoint()

                    losses.update(self._algorithm_metric(local_results=results, global_results=inter_results, labels=labels))               
                
                    loss = sum([weights[loss_key]*losses[loss_key] for loss_key in losses])

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                scaler.step(self.optimizer)
                scaler.update()

                loss_meter.update(loss.item(), images.size(0))
                time_meter.update(time.time() - end)
                end = time.time()

            self.scheduler.step()
        
        logger.info(f"[C{self.client_index}] End. Time: {end-start:.2f}s, Loss: {loss_meter.avg:.3f}")

        self.model.to('cpu')
        self.interpolater.to('cpu')

        loss_dict = {f'loss/{self.args.dataset.name}/{loss_key}': float(losses[loss_key]) for loss_key in losses}


        if global_epoch > 0 and global_epoch % 20 == 0:
            local_loss_dict = self.local_evaluate(global_epoch=global_epoch)
            loss_dict.update(local_loss_dict)

        return self.model, loss_dict
    

    def _algorithm_metric(self, local_results, global_results, labels):

        losses = {
            'cossim': [],
            # 'metric': [],
            # 'metric2': [],
            # 'metric3': [],
            # 'metric4': [],
        }

        for l in range(self.num_layers):

            train_layer = False
            if l in self.args.client.metric_loss.branch_level:
                train_layer = True

            local_feature_l = local_results[f"layer{l}"]
            global_feature_l = global_results[f"layer{l}"]
            if len(local_feature_l.shape) == 4:
                local_feature_l = F.adaptive_avg_pool2d(local_feature_l, 1)
                global_feature_l = F.adaptive_avg_pool2d(global_feature_l, 1)

            # Feature Cossim Loss
            if self.args.client.feature_align_loss.align_type == 'l2':
                loss_cossim = F.mse_loss(local_feature_l.squeeze(), global_feature_l.squeeze())
            else:
                loss_cossim = F.cosine_embedding_loss(local_feature_l.squeeze(), global_feature_l.squeeze(), torch.ones_like(labels))
            losses['cossim'].append(loss_cossim)

            # Metric Loss
            if train_layer:
                for metric_name in self.metric_criterions:
                    metric_criterion = self.metric_criterions[metric_name]
                    # breakpoint()

                    if metric_criterion is not None:
                        # breakpoint()
                        # B = global_feature_l.size(0)
                        # targets = torch.arange(B).repeat(2).to(labels.device)
                        targets = torch.cat((labels, labels), 0)
                        whole_feature_l = torch.cat((local_feature_l, global_feature_l), 0)
                        loss_metric = metric_criterion(old_feat=whole_feature_l, new_feat=whole_feature_l, target=targets, reduction=False)

                        # loss_metric = metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, target=labels, reduction=False)
                        # breakpoint()
                        if metric_name not in losses:
                            losses[metric_name] = []

                        losses[metric_name].append(loss_metric.mean())
            
        for loss_name in losses:
            try:
                losses[loss_name] = torch.mean(torch.stack(losses[loss_name])) if len(losses[loss_name]) > 0 else 0
            except:
                breakpoint()

        return losses
    

    '''
    def local_train(self, global_epoch, **kwargs):
        self.interpolater.to(self.device)
        self.model.to(self.device)
        scaler = GradScaler()

        start = time.time()
        loss_meter = AverageMeter('Loss', ':.2f')
        time_meter = AverageMeter('BatchTime', ':3.1f')

        # losses = defaultdict(float)
        # weights = {
        #     "cls": self.args.client.ce_loss.weight,
        #     "cossim": self.args.client.feature_align_loss.weight,}
        
        # for pair in args_metric.pairs:
        #     if pair.get('weights'):
        #         for weight_epoch in pair.weights:
        #             if global_epoch >= weight_epoch:
        #                 weight = pair.weights[weight_epoch]
        #         weights[pair.name] = weight
        #     else:
        #         weights[pair.name] = pair.weight
        self.weights = self.get_weights(epoch=global_epoch)

        if global_epoch % 50 == 0:
            print(self.weights)
            print(self.pairs)
            
        for local_epoch in range(self.args.trainer.local_epochs):
            end = time.time()
            self.interpolater.update()
                    

            for i, (images, labels) in enumerate(self.loader):    
                images, labels = images.to(self.device), labels.to(self.device)

                self.model.zero_grad()

                with autocast(enabled=self.args.use_amp):                    

                    losses = defaultdict(float)
                    results = self.model(images, no_relu=no_relu)
                    cls_loss = self.criterion(results["logit"], labels)
                    losses["cls"] = cls_loss

                    # with torch.no_grad():
                        # global_results = self.global_model(images, no_relu=no_relu)

                    if args_metric.interpolation_type == 'stochastic':
                    # inter_results = self.interpolater.forward_inter(images, no_relu=no_relu)
                        inter_results = self.interpolater.forward_stoc(images, repeat=1, no_relu=no_relu)[0]
                    elif args_metric.interpolation_type == 'full':
                        inter_results = self.interpolater.forward_inter(images, no_relu=no_relu)
                    # inter_results = {"logit": inter_results[0]}

                    # breakpoint()

                    losses.update(self._algorithm_metric(local_results=results, global_results=inter_results, labels=labels))               
                
                    loss = sum([weights[loss_key]*losses[loss_key] for loss_key in losses])

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                scaler.step(self.optimizer)
                scaler.update()

                loss_meter.update(loss.item(), images.size(0))
                time_meter.update(time.time() - end)
                end = time.time()

            self.scheduler.step()
        
        logger.info(f"[C{self.client_index}] End. Time: {end-start:.2f}s, Loss: {loss_meter.avg:.3f}")

        self.model.to('cpu')
        self.interpolater.to('cpu')

        loss_dict = {f'loss/{self.args.dataset.name}/{loss_key}': float(losses[loss_key]) for loss_key in losses}


        if global_epoch > 0 and global_epoch % 20 == 0:
        # if True:
            local_loss_dict = self.local_evaluate(global_epoch=global_epoch)
            loss_dict.update(local_loss_dict)

        return self.model, loss_dict
    '''


@torch.no_grad()
def _get_covariance_results(features, labels, num_classes):
    # features = features.double()
    features = torch.nn.functional.normalize(features, p=2, dim=1).detach()

    total_cov = torch.cov(features.T)
    # feat_ = features.reshape(num_classes, -1, features.size(1))

    prototypes = torch.zeros((num_classes, features.size(1))).to(features)
    for i in range(num_classes):
        prototypes[i] = features[labels==i].mean(0)

    # within_diff = (feat_ - feat_.mean(1, keepdim=True)).reshape(-1, features.size(1))
    # within_cov = torch.mm(within_diff.T, within_diff) / (features.size(0)-1)
    within_diff = features - prototypes[labels]
    within_cov = torch.mm(within_diff.T, within_diff) / (features.size(0)-1)

    within_cov_classes = [0] * num_classes
    for i in range(num_classes):
        within_diff_i = within_diff[labels==i]
        within_cov_i = torch.mm(within_diff_i.T, within_diff_i) / (within_diff_i.size(0)-1)
        # within_cov_i = torch.mm(within_diff_i.T, within_diff_i) / (within_diff_i.size(0))
        within_cov_classes[i] = torch.trace(within_cov_i)

    within_cov_classes = torch.stack(within_cov_classes)

    # prototypes = feat_.mean(1) # C * dim
    # btn_diff = prototypes - prototypes.mean(0, keepdim=True)
    # btn_cov = torch.mm(btn_diff.T, btn_diff) / feat_.size(0)
    
    '''
    btn_diff = prototypes_ - prototypes_.mean(0, keepdim=True)
    btn_cov = torch.mm(btn_diff.T, btn_diff) / feat_.size(0)

    p1 = prototypes_.unsqueeze(1).repeat(1, num_classes, 1)
    p2 = prototypes_.unsqueeze(0).repeat(num_classes, 1, 1)
    proto_cosines = F.cosine_similarity(p1, p2, dim=2)

    proto_cosines_ = torch.masked_select(proto_cosines, (1-torch.eye(num_classes)).bool())
    mean_cosines = proto_cosines_.nanmean()
    var_cosines =  proto_cosines_.var()
    collapse_error = F.mse_loss(proto_cosines_, torch.ones(num_classes*(num_classes-1))*(-1/(num_classes-1)))
    '''

    # try:
    valid_num_classes = labels.bincount().nonzero().size(0)
    prototypes_ = prototypes[~prototypes[:, 0].isnan(), :]
    valid_num_classes = prototypes_.size(0)
    valid_class_counts = labels.bincount()[labels.bincount().nonzero()]

    btn_diff = prototypes_ - prototypes_.mean(0, keepdim=True)
    # btn_cov = torch.mm(btn_diff.T, btn_diff) / valid_num_classes
    btn_cov = torch.cov(btn_diff.T, fweights=valid_class_counts.squeeze())
    # torch.cov(btn_diff.T, fweights=vc.squeeze())

    # (within_cov_classes * labels.bincount()).nansum() / labels.size(0)
    # breakpoint()

    p1 = prototypes_.unsqueeze(1).repeat(1, valid_num_classes, 1)
    p2 = prototypes_.unsqueeze(0).repeat(valid_num_classes, 1, 1)
    proto_cosines = F.cosine_similarity(p1, p2, dim=2)


    proto_cosines_ = torch.masked_select(proto_cosines, (1-torch.eye(valid_num_classes).to(proto_cosines)).bool())
    mean_cosines = proto_cosines_.mean()
    var_cosines =  proto_cosines_.var()



    # Class separation metric (from VCI paper)
    sim = torch.mm(features, features.t())
    dist = 1 - sim
    B = labels.size(0)
    classwise_mask = labels.expand(B, B).eq(labels.expand(B, B).T)
    pos_mask = classwise_mask.float()
    pos_mask[pos_mask==0] = torch.nan
    pos_dist = (dist*pos_mask).nanmean(1)
    neg_mask = classwise_mask.float()
    neg_mask[neg_mask==1] = torch.nan
    neg_mask[neg_mask==0] = 1
    neg_dist = (dist*neg_mask).nanmean(1)
    total_dist = dist.mean(1)
    d_within = [0] * num_classes
    d_total = [0] * num_classes
    for i in range(num_classes):
        d_within[i] = pos_dist[labels==i].mean()
        d_total[i] = total_dist[labels==i].mean()
    d_within = torch.stack(d_within)
    d_total = torch.stack(d_total)

    # if valid_num_classes > 1:
    # try:
    #     collapse_error = F.mse_loss(proto_cosines_, torch.ones(valid_num_classes*(valid_num_classes-1))*(-1/(valid_num_classes-1)))
    # except:
    #     collapse_error = 0

    # vci = 1 - torch.trace(torch.mm(torch.linalg.pinv(total_cov), btn_cov))/torch.linalg.matrix_rank(btn_cov)

    results = {
        "total_cov": torch.trace(total_cov),
        "within_cov": torch.trace(within_cov),
        "between_cov": torch.trace(btn_cov),

        "within_cov_class": within_cov_classes,

        "within_dist_class": d_within,
        "total_dist_class": d_total,

        # "vci": vci,
    }
    return results