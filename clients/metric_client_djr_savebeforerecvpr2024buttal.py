#!/usr/bin/env python
# coding: utf-8
import copy
import time
import gc

import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from collections import OrderedDict


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
class MetricClient_djr(Client):

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


        self.djr_update = False
        if args_metric.get('djr'):
            if args_metric.djr.use ==True:
                self.djr_update = True
                assert(self.args.client.metric_loss.feature_relu == True)
        

        self.pairs = {}
        for pair in args_metric.pairs:
            self.pairs[pair.name] = pair
            if self.djr_update:
                self.metric_criterions[pair.name] = MetricLoss_djr(pair=pair, **args_metric)
            elif self.args.get('ml2'):
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

        self.softmax = nn.Softmax(dim=1)

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
        
        self.label_distrib = torch.zeros(len(local_dataset.dataset.classes), device=self.device)
        for key in sorted_class_dict:
            self.label_distrib[int(key)] = sorted_class_dict[key]
            
        if global_epoch == 0:
            logger.warning(f"Class counts : {self.class_counts}")
            logger.info(f"Sorted class dict : {sorted_class_dict}")

        self.sorted_class_dict = sorted_class_dict

        self.trainer = trainer


        #### For logging D_jr
        self.log_djr = False
        if self.args.get('Djr_log_freq'):
            if global_epoch % self.args.get('Djr_log_freq') == 0:
                self.log_djr = True
        self.djr_dict = {}



        if self.log_djr:
            self.djrloader =  DataLoader(local_dataset, batch_size=len(local_dataset), sampler=train_sampler, shuffle=train_sampler is None,
                            num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)
        major_crit = len(self.loader.dataset) /len(local_dataset.dataset.classes)
        dist_dict = defaultdict(OrderedDict)
        dist_dict['is_major'] = torch.zeros(len(local_dataset.dataset.classes), device=self.device, dtype = torch.bool)
        dist_dict['is_minor'] = torch.zeros(len(local_dataset.dataset.classes), device=self.device, dtype = torch.bool)
        # for cl in range(len(local_dataset.dataset.classes)):
        #     if str(cl) not in sorted_class_dict:
        #         sorted_class_dict[str(cl)] = 0
        sorted_sample_classes = sorted(sorted_class_dict.items(), key = lambda a:a[1], reverse = True)
        most_sample = sorted_sample_classes[0][-1]
        for key, val in sorted_sample_classes:
            if val >= major_crit:
                dist_dict['major_dict'][key] = val
                dist_dict['is_major'][int(key)] = True
            else:
                dist_dict['minor_dict'][key] = val
                dist_dict['is_minor'][int(key)] = True

            if val == sorted_sample_classes[0][-1]:
                dist_dict['most_major'][key] = val

            if val == sorted_sample_classes[-1][-1]:
                dist_dict['most_minor'][key] = val

        self.dist_dict = dist_dict

        self.num_samples_vector = torch.zeros((self.num_classes, 1),device = self.device)
        for key, val in sorted_class_dict.items():
            self.num_samples_vector[int(key)] = val

        self.nj_div_nr = cal_att_j_div_r(self.num_samples_vector)
        self.djr_dict['num_samples'] = self.num_samples_vector.t()
        #breakpoint()








    def _algorithm_metric(self, local_results, global_results, labels):

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
        
        # if self.args.get('debuga'):
        #     breakpoint()

        bal_indices = torch.cat(bal_indices)

        # if len(clamp_indices) > 0:
        # try:
            # clamp_indices = torch.cat(clamp_indices)
        # except:
        #     breakpoint()
        index_of_train_layer = []
        
        for l in range(self.num_layers):

            train_layer = False
            if metric_args.branch_level is False or l in metric_args.branch_level:
                train_layer = True
                index_of_train_layer.append(l)
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
                for metric_idx, metric_name in enumerate(self.metric_criterions):
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
                                    # loss_metric2 = metric_criterion(old_feat=global_feature_l[L//2:], 
                                    #                                 new_feat=local_feature_l[L//2:],
                                    #                                 target=labels[L//2:],
                                    #                                 reduction=False, topk_neg=topk_neg)
                                    
                                    # loss_metric = torch.cat((loss_metric1, loss_metric2))
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

                                        # weights[pair.name] = weight

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

                                    # breakpoint()

                                    class_weights = 1/class_counts
                                    mean_class_weights = (class_weights * class_counts).nansum() / len(local_dataset)
                                    weight_mask = class_weights[labels]
                                    weight_mask /= mean_class_weights
                                    # weight_mask /= weight_mask.mean()
                                    # print(weight_mask.mean(), l)
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
                                    # if weight_mask.mean() > 1000:
                                    #     breakpoint()
                                    #     print(weight_mask.mean(), l)

                                elif metric_criterion.adapt_sample in ['within_cov', 'within_cov_rel', 'within_cov_rel2', 'class_sep', 'within+ratio', 'within_all']:

                                    class_counts = torch.zeros(len(local_dataset.dataset.classes))
                                    for key in local_dataset.class_dict:
                                        class_counts[int(key)] = local_dataset.class_dict[key]

                                    class_counts = torch.FloatTensor(class_counts).to(self.device)

                                    # class_cov = self.class_stats['cov']
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

                                        # breakpoint()
                                        
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
                                            # breakpoint()
                                            # print("!")
                                            # print("ha  ", weight_mask.mean(), weight_mask.min(), weight_mask.max(), l)
                                        else:
                                            weight_mask = 1
                                    # weight_mask = covs_global["vci"] / covs_local["vci"]
                                    # weight_mask = 1
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

                                # if self.args.get('debugs'):
                                #     breakpoint()

                                class_ratio = self.class_stats['ratio'][labels.cpu()]
                                # class_ratio = self.class_stats['ratio'].to(labels.device)[labels]
                                # class_ratio = None
                                # uncertainty = self.get_entropy(local_results['logit'])
                                uncertainty = F.cross_entropy(local_results["logit"], labels, reduction='none').detach()
                                # uncertainty /= uncertainty.max()

                                # if self.args.get('debugs'):
                                #     print("unc : ", uncertainty)
                                #     breakpoint()
                                # breakpoint()
                                # uncertainty = (1 - self.get_entropy(local_results['logit']) / self.get_entropy(global_results['logit'])).clamp(0)
                                if self.djr_update and self.log_djr and metric_idx==0:
                                    metric_criterion.__set_num_classes__(self.num_classes)
                                    loss_metric, Dx_dict = metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, target=labels, reduction=False, topk_neg=topk_neg,
                                                               uncertainty=uncertainty, class_ratio=class_ratio, level=l, progress=self.current_progress, djr =self.log_djr, class_major = self.batch_major_stat['this_batch_major'], class_minor = self.batch_minor_stat['this_batch_minor'], djr_dict = (self.djr_dict))
                                elif self.args.get('ml2'):
                                    #print(self.args.get('ml2'))
                                    loss_metric, loss_dist_opt_sim, pos_mean_sim, neg_mean_sim = metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, target=labels, reduction=False, topk_neg=topk_neg,
                                                               uncertainty=uncertainty, class_ratio=class_ratio, level=l, progress=self.current_progress, mode = str(self.args.get('ml2')))
                                elif self.args.get('rel_mode'):
                                    loss_metric = metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, target=labels, reduction=False, topk_neg=topk_neg,
                                                               uncertainty=uncertainty, class_ratio=class_ratio, level=l, progress=self.current_progress, mode =self.args.get('rel_mode'))

                                # elif self.log_djr and metric_idx==0:
                                #     metric_criterion.__set_num_classes__(self.num_classes)
                                #     loss_metric, self.feat_norm_stat, self.feat_norm_ratio_stat, self.feat_cos_stat = metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, target=labels, reduction=False, topk_neg=topk_neg,
                                #                                uncertainty=uncertainty, class_ratio=class_ratio, level=l, progress=self.current_progress, djr =self.log_djr, class_major = self.batch_major_stat['this_batch_major'], class_minor = self.batch_minor_stat['this_batch_minor'])
                                
                                elif 'triplet' in metric_name:
                                    loss_metric, pair_poss, pair_negs = metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, target=labels, reduction=False, topk_neg=topk_neg,
                                                               uncertainty=uncertainty, class_ratio=class_ratio, level=l, progress=self.current_progress, pos_weight = self.weights[metric_name + "/pos"], neg_weight = self.weights[metric_name + "/neg"], threshold = self.weights[metric_name + "/threshold"])



                                else:
                                    if self.args.get('debuga'):
                                        breakpoint()
                                        
                                    loss_metric = metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, target=labels, reduction=False, topk_neg=topk_neg,
                                                               uncertainty=uncertainty, class_ratio=class_ratio, level=l, progress=self.current_progress)


                                # loss_metric = metric_criterion(old_feat=global_feature_l, new_feat=local_feature_l, target=labels, reduction=False, topk_neg=topk_neg,
                                #                                uncertainty=uncertainty, class_ratio=class_ratio, level=l, progress=self.current_progress)

            
                            if metric_name not in losses:
                                #breakpoint()
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


                                    if self.log_djr and metric_idx==0:
                                        for key in Dx_dict.keys():
                                            losses[key] = []
                                        # mj = ['major', 'minor']
                                        # mj_comb = ['major_major','major_minor','minor_major','minor_minor']
                                        # for n in mj:
                                        #     losses['djr_prob_1_minus_prr_'+n]= []
                                        #     losses['djr_phi_'+n] = []
                                        #     losses['djr_cos_same_' + n] = []

                                        # for nc in mj_comb:
                                        #     losses['djr_prob_prj_'+ nc] = []
                                        #     #1_minus_prr / prj
                                        #     losses['djr_prob_ratio_' + nc] = []
                                        #     losses['djr_phi_ratio_' + nc] = []
                                        #     losses['djr_cos_diff_' + nc] = []
                                        #     losses['djr_cos_ratio_' + nc] = []
                                        #     losses['djr_Djr_'+nc] = []

                                        
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

                                if self.log_djr and metric_idx==0:
                                    for key in Dx_dict.keys():
                                        losses[key].append(Dx_dict[key].mean())
                                    # mj = ['major', 'minor']
                                    # mj_comb = ['major_major','major_minor','minor_major','minor_minor']

                                    # #Calculate metrics

                                    # def cal_ratio(mj, stat, name, one_minus = False):    
                                    #     for n1 in mj:
                                    #         for n2 in mj:
                                    #             for cl in stat[n1+'_self'].keys():
                                    #                 val = stat[n1+'_self'][cl]
                                    #                 stat[name + n1+'_'+n2][cl] = {}
                                    #                 for cl2 in stat[n1+'_'+n2][cl].keys():
                                    #                     val2 = stat[n1+'_'+n2][cl][cl2]
                                    #                     if one_minus:
                                    #                         stat[name + n1+'_'+n2][cl][cl2] = (1-val)/val2
                                    #                     else:
                                    #                         stat[name + n1+'_'+n2][cl][cl2] = val/val2
                                    #                 stat[name + n1+'_'+n2 +'_minclass'][cl] = min_dict(stat[name + n1+'_'+n2][cl])
                                    #             stat[name + n1+'_'+n2 +'_minall'] = min_dict(stat[name + n1+'_'+n2 +'_minclass'])


                                    # cal_ratio(mj,self.prob_stat,'djr_prob_ratio_', one_minus = True)
                                    # cal_ratio(mj,self.feat_cos_stat,'djr_cos_ratio_', one_minus = False)
                                    
                                    # Djr = defaultdict(type({}))
                                    # #self.feat_cos_stat['djr_cos_ratio_major_major']
                                    # #self.prob_stat['djr_cos_ratio_major_major']
                                    # #self.feat_norm_ratio_stat['major_major']

                                    # for n1 in mj:
                                    #     for n2 in mj:
                                    #         D_pr = self.prob_stat['djr_prob_ratio_' + n1+'_'+n2]
                                    #         D_no = self.feat_norm_ratio_stat[n1+'_'+n2]
                                    #         D_co = self.feat_cos_stat['djr_cos_ratio_' + n1+'_'+n2]
                                    #         for cl in D_pr.keys():
                                    #             #breakpoint()
                                    #             D_pr_val = D_pr[cl]
                                    #             D_no_val = D_no[cl]
                                    #             D_co_val = D_co[cl]
                                    #             Djr[n1+'_'+n2 + '_raw_val'][cl] = {}

                                    #             for cl2 in D_pr_val.keys():
                                    #                 val = D_pr_val[cl2] * D_no_val[cl2] * D_co_val[cl2]
                                    #                 Djr[n1+'_'+n2 +'_raw_val'][cl][cl2] = val

                                    #             Djr[n1+'_'+n2 +'_minclass'][cl] = min_dict(Djr[n1+'_'+n2 +'_raw_val'][cl])
                                    #         Djr[n1+'_'+n2 +'_minall'] = min_dict(Djr[n1+'_'+n2 +'_minclass'])
                                    #         #breakpoint()
                                            
                                    # #breakpoint()


                                    # #Logging metrics
                                    # for n in mj:
                                    #     append_or_not(losses['djr_prob_1_minus_prr_'+n], (self.prob_stat[n + '_self_minall']), one_minus = True)
                                    #     append_or_not(losses['djr_phi_'+n],self.feat_norm_stat[n + '_self_minall'])
                                    #     append_or_not(losses['djr_cos_same_' + n], self.feat_cos_stat[n + '_self_minall'])

                                    # for nc in mj_comb:
                                    #     append_or_not(losses['djr_prob_prj_'+ nc] ,(self.prob_stat[nc + '_minall']))
                                    #     #1_minus_prr / prj
                                    #     append_or_not(losses['djr_prob_ratio_' + nc], self.prob_stat['djr_prob_ratio_' + nc +'_minall'])

                                    #     append_or_not(losses['djr_phi_ratio_' + nc] , self.feat_norm_ratio_stat[nc + '_minall'])

                                    #     append_or_not(losses['djr_cos_diff_' + nc] , self.feat_cos_stat[nc + '_minall'])
                                    #     append_or_not(losses['djr_cos_ratio_' + nc], self.feat_cos_stat['djr_cos_ratio_' + nc +'_minall'])
                                                      
                                    #     append_or_not(losses['djr_Djr_'+nc],Djr[nc + '_minall'])
                                    #breakpoint()

                                    #self.feat_norm_stat, self.feat_norm_ratio_stat, self.feat_cos_stat



        for loss_name in list(losses):
            try:
                for idx, l in zip(index_of_train_layer ,losses[loss_name]):
                    losses[loss_name + "_layer" + str(idx)] = l
                losses[loss_name] = torch.mean(torch.stack(losses[loss_name])) if len(losses[loss_name]) > 0 else None

            except:
                breakpoint()
        return losses


    def _algorithm(self, images, labels) -> Dict:


        losses = defaultdict(float)
        no_relu = not self.args.client.metric_loss.feature_relu
        if self.log_djr:
            no_relu = False
        results = self.model(images, no_relu=no_relu)

        if self.args.client.get('label_noise'):
            noise_ratio = self.args.client.label_noise.ratio
            labels = apply_label_noise(labels, noise_ratio)
        probability_vector  = self.softmax(results["logit"])
        self.unique_labels = torch.unique(labels)
        averaged_probabilities = get_avg_data_per_class(probability_vector, labels, self.num_classes, unique_labels = self.unique_labels)
        self.averaged_probabilities = averaged_probabilities
        #self.djr_dict[averaged_probabilities] = averaged_probabilities
        oneminus_truelabel_probabilities = (1 - torch.diag(averaged_probabilities)).unsqueeze(dim = 1)
        self.Ompjj_div_Omprr = cal_att_j_div_r(oneminus_truelabel_probabilities)
        #breakpoint()
        cls_loss = self.criterion(results["logit"], labels)



        #For D_jr
        if self.log_djr:
            probability_vector  = self.softmax(results["logit"])
            self.unique_labels = torch.unique(labels)
            averaged_probabilities = get_avg_data_per_class(probability_vector, labels, self.num_classes, unique_labels = self.unique_labels)
            self.djr_dict['averaged_probabilities'] = averaged_probabilities.detach().clone().t()
            #self.djr_dict['']
            batch_major_stat = defaultdict()
            batch_minor_stat = defaultdict()
            this_batch_minor = []
            this_batch_major = []
            for ul in self.unique_labels:
                ul = ul.item()
                if str(ul) in self.dist_dict['major_dict'].keys():
                    this_batch_major.append(ul)
                else:
                    this_batch_minor.append(ul)

            batch_major_stat['this_batch_major'] = this_batch_major
            batch_minor_stat['this_batch_minor'] = this_batch_minor
            
            self.batch_major_stat = batch_major_stat
            self.batch_minor_stat = batch_minor_stat

            #breakpoint()
            self.prob_stat = get_major_minor_stat(self.batch_major_stat['this_batch_major'], self.batch_minor_stat['this_batch_minor'], averaged_probabilities)



            
        #
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

        # if self.args.get('debugs'):
        #     breakpoint()

        with torch.no_grad():
            global_results = self.global_model(images, no_relu=no_relu)

        losses.update(self._algorithm_metric(local_results=results, global_results=global_results, labels=labels))               
        #breakpoint()

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
        losses_meter_dict = {}

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

            if self.args.get('debugs'):
                with torch.no_grad():
                    for i, (images, labels) in enumerate(self.loader):
                        images, labels = images.to(self.device), labels.to(self.device)
                        self.model.zero_grad()

                        with autocast(enabled=self.args.use_amp):
                            losses, features = self._algorithm(images, labels,)

                            
                            # for key in features['local']:
                            for key in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5']:
                                feat_local = features['local'][key].detach()
                                if len(feat_local.shape) == 4:
                                    feat_local = F.adaptive_avg_pool2d(feat_local, 1)
                                all_features_local[key].append(feat_local)
                            # for key in features['global']:
                                feat_global = features['global'][key].detach()
                                if len(feat_global.shape) == 4:
                                    feat_global = F.adaptive_avg_pool2d(feat_global, 1)
                                all_features_global[key].append(feat_global)

                            all_labels.append(labels)

                    self.update_cov_results(all_features_local, all_features_global, all_labels)

                # if local_epoch > 1:
                #     breakpoint()

            end = time.time()
            for i, (images, labels) in enumerate(self.loader):

                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()

                with autocast(enabled=self.args.use_amp):
                    losses, features = self._algorithm(images, labels)

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

                    
                    loss_sum = []
                    for loss_key in losses:
                        #print(loss_key, losses[loss_key])
                        if losses[loss_key]!=None and self.weights[loss_key]>0:
                            loss_sum.append(self.weights[loss_key]*losses[loss_key])

                    loss = sum(loss_sum)#sum([self.weights[loss_key]*losses[loss_key] for loss_key in losses])
                    # breakpoint()

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
                for loss_key in losses:
                    if losses[loss_key]!=None:
                        if loss_key not in losses_meter_dict:
                            losses_meter_dict[loss_key] = AverageMeter('loss_key', ':.2f')
                        losses_meter_dict[loss_key].update(losses[loss_key].item(), images.size(0))

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


        #Calculate Djr
        #breakpoint()
        # if self.log_djr:
        #     with torch.no_grad():
        #         for i, (images, labels) in enumerate(self.djrloader):
        #             images, labels = images.to(self.device), labels.to(self.device)
        #             self.model.zero_grad()
        #             with autocast(enabled=self.args.use_amp):
        #                 losses, features = self._algorithm(images, labels)


        #                 #For only visualizing loss(not use for training)
        #                 for loss_key in losses:
        #                     if loss_key not in self.weights.keys():
        #                         self.weights[loss_key] = 0

                        
        #                 loss_sum = []
        #                 for loss_key in losses:
        #                     #print(loss_key, losses[loss_key])
        #                     if losses[loss_key]!=None and self.weights[loss_key]>0:
        #                         loss_sum.append(self.weights[loss_key]*losses[loss_key])
        #                 loss = sum(loss_sum)#sum([self.weights[loss_key]*losses[loss_key] for loss_key in losses])
        #             loss_meter.update(loss.item(), images.size(0))
        #             for loss_key in losses:
        #                 if losses[loss_key]!=None:
        #                     if loss_key not in losses_meter_dict:
        #                         losses_meter_dict[loss_key] = AverageMeter('loss_key', ':.2f')
        #                     losses_meter_dict[loss_key].update(losses[loss_key].item(), images.size(0))

        #             time_meter.update(time.time() - end)
        #             end = time.time()
            
        #
        
        logger.info(f"[C{self.client_index}] End. Time: {end-start:.2f}s, Loss: {loss_meter.avg:.3f}, Rel Entropy: {entropy_meter.avg:.3f}/{global_entropy_meter.avg:.3f}")

        self.model.to('cpu')
        self.global_model.to('cpu')
        if self.interpolater:
            self.interpolater.to('cpu')

        loss_dict = {f'loss/{self.args.dataset.name}/{loss_key}':(losses_meter_dict[loss_key].avg) for loss_key in losses_meter_dict}

        #loss_dict = {f'loss/{self.args.dataset.name}/{loss_key}': float(losses[loss_key]) for loss_key in losses}
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
    













