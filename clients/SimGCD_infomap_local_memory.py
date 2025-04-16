#!/usr/bin/env python
# coding: utf-8
import copy
import time

import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import gc

import torch.nn.functional

from utils import *
from utils.metrics import evaluate
from utils.visualize import __log_test_metric__, umap_allmodels, cka_allmodels, log_fisher_diag
from models import build_encoder, get_model
from typing import Callable, Dict, Tuple, Union, List
from utils.logging_utils import AverageMeter
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)

from clients.build import CLIENT_REGISTRY
import wandb
from tqdm import tqdm
import random
import numpy as np
from utils.train_utils import update_prior, update_prior_threshold
from finch import FINCH

from utils import LossManager, extract_local_features_unlabelled
from utils.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
from torch.optim import lr_scheduler
from utils.sampler import RandomMultipleGallerySamplerNoCam, IterLoader


class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs,
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """


        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))


        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

@CLIENT_REGISTRY.register()
class SimGCD_Infomap_Client():

    def __init__(self, args, client_index, loader=None, evaler=None):
        self.args = args
        self.client_index = client_index
        # self.loader = loader  
        # self.model = model
        # self.global_model = copy.deepcopy(model)
        self.criterion = nn.CrossEntropyLoss()
        self.sup_con_crit = SupConLoss(temperature=self.args.client.sup_temperature,
                                       base_temperature=self.args.client.sup_temperature)
        
        self.g_clipping = args.client.g_clipping
        self.evaler = evaler
        self.cluster_criterion = DistillLoss(
            args.client.warmup_teacher_temp_epochs,
            # args.trainer.global_rounds,
            args.trainer.local_epochs,
            args.client.n_views,
            args.client.warmup_teacher_temp,
            args.client.teacher_temp,
        )
        self.num_classes = len(self.args.dataset.seen_classes) + len(self.args.dataset.unseen_classes)
        self.prior_dist = torch.ones(self.num_classes) / self.num_classes
        self.gt_prior = None
        self.aligned_preds = None
        self.num_clusters = 10
        self.ind_map_pred_to_gt = None
        self.bfloat16_support = check_bfloat16_support()
        self.pk_sampler = None
        return

    def wandb_log(self, log: Dict, step: int = None):
        if self.args.wandb:
            wandb.log(log, step=step)

    def evaluate(self, epoch: int, local_datasets: List[torch.utils.data.Dataset] = None) -> Dict:

        results = self.evaler.eval(model=self.model, epoch=epoch)
        all_acc = results["all_acc"]
        new_acc = results["new_acc"]
        old_acc = results["old_acc"]

        wandb_dict = {
            f"all_acc/{self.args.dataset.name}": all_acc,
            f"old_acc/{self.args.dataset.name}": old_acc,
            f"new_acc/{self.args.dataset.name}": new_acc,
        }

        logger.warning(
            f'[Epoch {epoch}] Test ALL Acc: {all_acc:.2f}%, OLD Acc: {old_acc:.2f}%, NEW Acc: {new_acc:.2f}%')

        plt.close()

        self.wandb_log(wandb_dict, step=epoch)
        return {
            "all_acc": all_acc,
            "new_acc": new_acc,
            "old_acc": old_acc
        }

    def evaluate_local_trainset(self, epoch: int, global_epoch=0, local_datasets: List[torch.utils.data.Dataset] = None) -> Dict:

        # results_before = self.evaler.local_trainset_eval(model=self.global_model, epoch=epoch,
        #                                           local_train_loader=copy.deepcopy(self.loader))
        # conf_plot_before = self.evaler.plot_confusion_matrix(results_before["conf_matrix"])

        results = self.evaler.local_trainset_eval(model=self.model, epoch=epoch, local_train_loader=copy.deepcopy(self.loader))
        conf_plot = self.evaler.plot_confusion_matrix(results["conf_matrix"])

        all_acc = results["all_acc"]
        new_acc = results["new_acc"]
        old_acc = results["old_acc"]
        # all_acc_before = results_before["all_acc"]
        # new_acc_before = results_before["new_acc"]
        # old_acc_before = results_before["old_acc"]

        wandb_dict = {
            f"{self.args.dataset.name}/client{self.client_index}/local_all_acc": all_acc,
            f"{self.args.dataset.name}/client{self.client_index}/local_old_acc": old_acc,
            f"{self.args.dataset.name}/client{self.client_index}/local_new_acc": new_acc,
            # f"local_all_acc_{self.client_index}_before/{self.args.dataset.name}": all_acc_before,
            # f"local_old_acc_{self.client_index}_before/{self.args.dataset.name}": old_acc_before,
            # f"local_new_acc_{self.client_index}_before/{self.args.dataset.name}": new_acc_before,
            f"{self.args.dataset.name}/client{self.client_index}/conf_matrix": conf_plot["confusion_matrix"],
            # f"client_{self.client_index}_conf_matrix_before": conf_plot_before["confusion_matrix"],
        }

        logger.warning(
            f'[Epoch {epoch}] Local Trainset ALL Acc: {all_acc:.2f}%, OLD Acc: {old_acc:.2f}%, NEW Acc: {new_acc:.2f}%')


        self.wandb_log(wandb_dict, step=global_epoch)
        # return {
        #     "all_acc": all_acc,
        #     "new_acc": new_acc,
        #     "old_acc": old_acc
        # }
        ind_map = results['ind_map']
        sorted_ind_map = dict(sorted(ind_map.items()))

        return sorted_ind_map
        

    def evaluate_local_trainset_unlabelled(self, epoch: int, global_epoch=0,
                                local_datasets: List[torch.utils.data.Dataset] = None) -> Dict:


        results = self.evaler.local_trainset_eval_unlabelled_logit(model=self.model, epoch=epoch,
                                                  local_train_loader=copy.deepcopy(self.loader))
        
        all_acc = results["all_acc"]
        new_acc = results["new_acc"]
        old_acc = results["old_acc"]

        wandb_dict = {
            f"{self.args.dataset.name}/client{self.client_index}/local_all_acc": all_acc,
            f"{self.args.dataset.name}/client{self.client_index}/local_old_acc": old_acc,
            f"{self.args.dataset.name}/client{self.client_index}/local_new_acc": new_acc,
            f"{self.args.dataset.name}/client{self.client_index}/local_cluster_acc": results["cluster_acc"],
            f"{self.args.dataset.name}/client{self.client_index}/local_cluster_old_acc": results["cluster_old_acc"],
            f"{self.args.dataset.name}/client{self.client_index}/local_cluster_new_acc": results["cluster_new_acc"],
        }

        logger.warning(
            f'[Epoch {epoch}] Local Trainset Unlabelled ALL Acc: {all_acc:.2f}%, OLD Acc: {old_acc:.2f}%, NEW Acc: {new_acc:.2f}%')
        logger.warning(
            f'[Epoch {epoch}] Local Trainset Unlabelled Cluster Acc: {results["cluster_acc"]:.2f}%, Cluster Old Acc: {results["cluster_old_acc"]:.2f}%, Cluster New Acc: {results["cluster_new_acc"]:.2f}%')

        plt.close()

        self.wandb_log(wandb_dict, step=epoch)
        return {
            "local_cluster_acc": results["cluster_acc"],
            "local_cluster_old_acc": results["cluster_old_acc"],
            "local_cluster_new_acc": results["cluster_new_acc"]
        }
    
    def get_params_groups(self, model):
        regularized = []
        not_regularized = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

    def setup(self, model, device, local_dataset, global_epoch, local_lr, trainer, **kwargs):
       
        if self.args.client.ce_type == 'ce':
            self.ce = nn.CrossEntropyLoss()
        elif self.args.client.ce_type == 'margin':
            self.ce = MarginLoss(m=-1*self.args.client.margin_m, s=self.args.client.margin_s)
        else:
            raise ValueError(f"Invalid ce_type: {self.args.client.ce_type}")
        

        self.optimizer_state_dict = kwargs['optimizer_state_dict']
        self.model = model
        self.model.to(device)
        self.global_model = copy.deepcopy(self.model)
        self.device = device

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
                                 num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, drop_last=False,
                                 )
        
        if self.args.client.data_range == 'unlabelled':
            if self.pk_sampler is None:
                PK_sampler = RandomMultipleGallerySamplerNoCam(local_dataset.unlabelled_dataset, self.args.num_instances)
                self.pk_sampler = PK_sampler
            self.train_cluster_loader = IterLoader(DataLoader(local_dataset.unlabelled_dataset, batch_size=self.args.batch_size, sampler=PK_sampler, shuffle=False,
                                    num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, drop_last=False,
                                    ))
        elif self.args.client.data_range == 'all':
            if self.pk_sampler is None:
                PK_sampler = RandomMultipleGallerySamplerNoCam(local_dataset, self.args.num_instances)
                self.pk_sampler = PK_sampler
            self.train_cluster_loader = IterLoader(DataLoader(local_dataset, batch_size=self.args.batch_size, sampler=PK_sampler, shuffle=False,
                                    num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, drop_last=False,
                                    ))
        else:
            raise ValueError(f"Invalid cluster_type: {self.args.client.cluster_type}")
    
        self.labelled_class_set = set(local_dataset.labelled_dataset.targets)
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=local_lr, momentum=self.args.optimizer.momentum, weight_decay=self.args.optimizer.wd)
        
        if self.optimizer_state_dict is not None:
            new_state_dict = {
            'param_groups' : self.optimizer.state_dict()['param_groups'],
            'state' : copy.deepcopy(self.optimizer_state_dict)
            }
            self.optimizer.load_state_dict(new_state_dict)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = local_lr
            self.optimizer.zero_grad()
        
        self.exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
                                    self.optimizer,
                                    T_max=self.args.trainer.local_epochs,
                                    eta_min=local_lr * 1e-3,
                                    )

        self.class_counts = np.sort([*local_dataset.class_dict.values()])[::-1]


        sorted_key = np.sort([*local_dataset.class_dict.keys()])
        sorted_class_dict = {} 
        for key in sorted_key:  
            sorted_class_dict[key] = local_dataset.class_dict[key]
            
        if global_epoch == 0:
            logger.warning(f"Class counts : {self.class_counts}")
            logger.info(f"Sorted class dict : {sorted_class_dict}")
            
            ## init classifier weights with cluster centroids
            if self.args.client.re_init_prototypes:
                new_prototypes = self.classifiier_init_clustering(copy.deepcopy(self.model), copy.deepcopy(self.loader))
                self.model.proj_layer.last_layer.parametrizations.weight.original1.data[:].copy_(new_prototypes[:])
        
        self.sorted_class_dict = sorted_class_dict

    def __repr__(self):
        print(f'{self.__class__} {self.client_index}, {"data : "+len(self.loader.dataset) if self.loader else ""}')

    def get_weights(self, epoch=None):

        weights = {
            "self_cls": self.args.client.unsup_cls_weight,
            "sup_cls": self.args.client.sup_cls_weight,
            "sup_con": self.args.client.sup_con_weight,
            "con": self.args.client.unsup_con_weight,
            "memax": self.args.client.memax_weight * (self.args.client.unsup_cls_weight),
        }
        
        return weights

    def local_train(self, global_epoch, **kwargs):
        self.global_epoch = global_epoch
        self.model.train()
        self.model.to(self.device)
        scaler = GradScaler()
        start = time.time()
        loss_meter = AverageMeter('Loss', ':.2f')
        memax_loss_meter = AverageMeter('Me_Max_Loss', ':.4f')
        time_meter = AverageMeter('BatchTime', ':3.1f')

        self.weights = self.get_weights(epoch=global_epoch)

        self.prior_bank = None
        
        for local_epoch in range(self.args.trainer.local_epochs):
            end = time.time()
            self.model.train()
            self.model.to(self.device)

            for i, (images, labels, uq_idxs, mask_lab) in enumerate(self.loader):
                

                if self.args.client.n_views > 1:
                    images = torch.cat(images, dim=0)
                
                images, labels = images.to(self.device), labels.to(self.device)
                # Create mask tensors where all elements are 1
                # For labeled data
                mask_lab = mask_lab[:, 0]
                mask_lab = mask_lab.to(self.device).bool()
                # mask_lab_u = mask_lab_u[:, 0]
                # mask_lab_u = mask_lab_u.to(self.device).bool()

                losses = self._algorithm(images, labels, uq_idxs, mask_lab, global_epoch, local_epoch)
                loss = sum([self.weights[loss_key]*losses[loss_key] for loss_key in losses])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                loss_meter.update(loss.item(), images.size(0))
                memax_loss_meter.update(losses['memax'].item(), images.size(0))
                time_meter.update(time.time() - end)
                end = time.time()

            self.exp_lr_scheduler.step()
            print(f'Learning rate: {self.exp_lr_scheduler.get_last_lr()}')

            if self.args.trainer.local_eval and (local_epoch+ 1) % self.args.trainer.local_eval_freq == 0:
                ind_map = self.evaluate_local_trainset_unlabelled(epoch=local_epoch, global_epoch=global_epoch, local_datasets=None)


        # Local clustering and get local centroids
        if self.args.client.get_local_centroid and global_epoch >= self.args.client.start_update:
            cluster_means, cluster_targets = self.get_centroids(copy.deepcopy(self.model), copy.deepcopy(self.loader),  self.evaler, self.num_clusters, self.device)
        else:
            cluster_means = None
            cluster_targets = None
        
        logger.info(f"[C{self.client_index}] End. Time: {end-start:.2f}s, Loss: {loss_meter.avg:.3f}")
        self.model.to('cpu')

        loss_dict = {
            f'loss/{self.args.dataset.name}/cls': loss_meter.avg,
            f'loss/{self.args.dataset.name}/memax': memax_loss_meter.avg,
        }
        state_dict = self.model.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()

        results = {
            'cluster_means': cluster_means,
            'cluster_targets': cluster_targets,
            'optimizer_state_dict': optimizer_state_dict['state'],
        }

        # Flush model memories
        self._flush_memory()


        return state_dict, loss_dict, results

    def classifiier_init_clustering(self, model, loader, num_clusters=200):
        model.eval()
        model.to(self.device)
        feats_labelled, feats_proj_labelled, logits_labelled, targets_labelled, feats_unlabelled, feats_proj_unlabelled, logits_unlabelled, targets_unlabelled, mask = extract_local_features(self.args, model, loader, self.evaler, self.device)

        targets_unlabelled = targets_unlabelled.astype(int)
        targets_labelled = targets_labelled.astype(int)
        feats_labelled = F.normalize(feats_labelled, dim=1)
        feats_unlabelled = F.normalize(feats_unlabelled, dim=1)
        all_feats = torch.cat([feats_labelled, feats_unlabelled], dim=0)
        all_feats_proj = torch.cat([feats_proj_labelled, feats_proj_unlabelled], dim=0)
        all_targets = np.concatenate([targets_labelled, targets_unlabelled])

        ## Semi-supervised K-Means
        ## No balanced setting
        cluster_size = None
        kmeanssem = SemiSupKMeans(k=num_clusters, tolerance=1e-4,
                                max_iterations=10, init='k-means++',
                                n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                                mode=None, protos=None, cluster_size=cluster_size)
        kmeanssem.fit_mix(feats_unlabelled, feats_labelled, torch.from_numpy(targets_labelled).long())
        req_c = kmeanssem.labels_.cpu().numpy()

        ## Eval Clustering results
        all_acc, old_acc, new_acc, w, ind_map_gt = self.evaler.split_cluster_acc_v2(all_targets.astype(int), req_c, mask)
        cluster_prototypes = {}
        for cluster_idx in np.unique(req_c):
            cluster_indices = np.where(req_c == cluster_idx)[0]
            cluster_prototype = all_feats[cluster_indices].mean(dim=0)
            cluster_prototypes[cluster_idx] = cluster_prototype

        # Sort ind_map_gt by keys
        # ind_map_gt = dict(sorted(ind_map_gt.items()))
        # Convert cluster_prototypes values to list and reorder based on ind_map_gt values
        cluster_prototypes_list = list(cluster_prototypes.values())
        # aligned_prototypes = torch.stack([cluster_prototypes_list[i] for i in ind_map_gt.values()])
        cluster_prototypes = torch.stack(cluster_prototypes_list)
        # aligned_prototypes = cluster_prototypes.values()[list(ind_map_gt.values())]
        return cluster_prototypes

    
    def estimate_prior(self, model):
        start_time = time.time()
        feats_labelled, feats_proj_labelled, logits_labelled, targets_labelled, feats_unlabelled, feats_proj_unlabelled, logits_unlabelled, targets_unlabelled, mask = extract_local_features(self.args, model, self.loader, self.evaler, self.device)
        end_time = time.time()
        print(f'Time taken to extract local features: {end_time - start_time:.4f} seconds')

        
    def get_centroids(self, model, loader, evaler, K, device):

        # get features from model
        feats_unlabelled, feats_proj_unlabelled, logits_unlabelled, targets_unlabelled, mask = extract_local_features_unlabelled(self.args, model, loader, evaler, device)
        
        targets_unlabelled = torch.from_numpy(targets_unlabelled).long()


        if self.args.client.centroid_type == 'finch':
            ## Finch clustering
            c, num_clust, req_c = FINCH(feats_unlabelled.cpu().numpy(), req_clust=self.num_clusters, distance='cosine', verbose=False)
            
            unique_targets, counts = np.unique(targets_unlabelled.cpu().numpy(), return_counts=True)
            print(dict(zip(unique_targets, counts)))
            D = 10
            w = np.zeros((self.num_clusters, D), dtype=int)
            for i in range(req_c.size):
                w[req_c[i], targets_unlabelled[i]] += 1

            ind = linear_assignment(w.max() - w)
            ind = np.vstack(ind).T


            ind_map = {j: i for i, j in ind}
            print(ind_map)
            cluster_means = []
            cluster_targets = []
            
            unseen_classes = torch.unique(targets_unlabelled[targets_unlabelled >= len(self.args.dataset.seen_classes)]).tolist()
            unseen_ind_map = {k: v for k, v in ind_map.items() if k in unseen_classes}
            for gt_cls, cluster_id in unseen_ind_map.items():
                cluster_indices = np.where(req_c == cluster_id)[0]
                if len(cluster_indices) > 0:
                    cluster_mean = (feats_unlabelled[cluster_indices]).mean(dim=0)
                    cluster_means.append(cluster_mean)
                    cluster_targets.append(gt_cls)
            
            cluster_means = torch.stack(cluster_means)
            cluster_targets = torch.LongTensor(cluster_targets)
        elif self.args.client.centroid_type == 'preds':
            cluster_means = []
            cluster_targets = []
            # Get predictions from logits
            preds = logits_unlabelled.argmax(dim=1)
            
            # Find predictions for unseen classes (>= num_seen_classes)
            unseen_preds = torch.unique(preds[preds >= len(self.args.dataset.seen_classes)])
            
            # Calculate centroids for each unseen class prediction
            for pred in unseen_preds:
                # Get indices where prediction matches current class
                pred_indices = (preds == pred).nonzero().squeeze()
                
                if len(pred_indices.shape) == 0:
                    pred_indices = pred_indices.unsqueeze(0)
                    
                if len(pred_indices) > 0:
                    # Calculate mean of features for this prediction
                    pred_centroid = feats_unlabelled[pred_indices.cpu()].mean(dim=0)
                    cluster_means.append(pred_centroid)
                    cluster_targets.append(pred.item())
            
            if len(cluster_means) > 0:
                cluster_means = torch.stack(cluster_means)
                cluster_targets = torch.LongTensor(cluster_targets)
            else:
                cluster_means = None
                cluster_targets = None
            

        elif self.args.client.centroid_type == 'ind_map':
            ind_map_pred_to_gt = self.ind_map_pred_to_gt
            req_c_unlabelled = self.req_c_unlabelled
            cluster_means = []
            cluster_targets = []

            unseen_classes = [v for v in ind_map_pred_to_gt.values() if v >= len(self.args.dataset.seen_classes)]
            unseen_clusters = [k for k, v in ind_map_pred_to_gt.items() if v in unseen_classes]

            for cluster_idx, class_idx in ind_map_pred_to_gt.items():
                if class_idx >= len(self.args.dataset.seen_classes):
                    cluster_indices = np.where(req_c_unlabelled == cluster_idx)[0]
                    if len(cluster_indices) > 0:
                        cluster_mean = feats_unlabelled[cluster_indices].mean(dim=0)
                        cluster_means.append(cluster_mean)
                        cluster_targets.append(class_idx)

            cluster_means = torch.stack(cluster_means)
            cluster_targets = torch.LongTensor(cluster_targets)

        return cluster_means, cluster_targets

    def info_nce_logits(self, features):

        b_ = 0.5 * int(features.size(0))

        labels = torch.cat([torch.arange(b_) for i in range(self.args.client.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.args.client.unsup_temperature
        return logits, labels

    def info_nce_logits_inter(self, features):

        b_ = 0.5 * int(features.size(0))

        labels = torch.cat([torch.arange(b_) for i in range(self.args.client.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # Find the most similar sample (excluding self)
        sim_matrix_no_diag = similarity_matrix.clone()
        sim_matrix_no_diag.fill_diagonal_(-float('inf'))
        most_similar = sim_matrix_no_diag.max(dim=1).indices

        # Add the most similar sample as a positive
        for i in range(labels.size(0)):
            labels[i, most_similar[i]] = 1

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.args.client.unsup_temperature
        return logits, labels
    

    def _algorithm(self, images, labels, uq_idxs, mask_lab, global_epoch=0, local_epoch=0) -> Dict:

        losses = defaultdict(float)

        student_proj, student_out = self.model(images)
        teacher_out = student_out.detach()

        # clustering, sup
        sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
        sup_labels = torch.cat([labels[mask_lab] for _ in range(2)], dim=0)
        cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

        # clustering, unsup
        # cluster_loss = self.cluster_criterion(student_out, teacher_out, global_epoch)
        # for local training, reproduce
        images_c, labels_c, uq_idxs_c, mask_lab_c = self.train_cluster_loader.next()
        if self.args.client.n_views > 1:
            images_c = torch.cat(images_c, dim=0)
        images_c, labels_c = images_c.to(self.device), labels_c.to(self.device)
        student_proj_c, student_out_c = self.model(images_c)
        teacher_out_c = student_out_c.detach()

        cluster_loss = self.cluster_criterion(student_out_c, teacher_out_c, local_epoch)
        avg_probs = (student_out_c / 0.1).softmax(dim=1).mean(dim=0)
        me_max_loss = - torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
        #cluster_loss += self.args.client.memax_weight * me_max_loss

        # represent learning, unsup
        contrastive_logits, contrastive_labels = self.info_nce_logits(features=student_proj)
        contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

        # representation learning, sup
        student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
        student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
        sup_con_labels = labels[mask_lab]
        sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

        
        losses["self_cls"] =  cluster_loss
        losses["sup_cls"] = cls_loss
        losses["sup_con"] =sup_con_loss
        losses["con"] = contrastive_loss
        losses["memax"] = me_max_loss

        # del results
        return losses
        
        # losses = defaultdict(float)

        # # Concatenate images and do single forward pass
        # all_images = torch.cat([images_l, images_u], dim=0)
        # all_feat, all_proj, all_out = self.model(all_images, return_all=True)
        # student_feat_l, student_proj_l, student_out_l = all_feat[:len(images_l)], all_proj[:len(images_l)], all_out[:len(images_l)]
        # student_feat_u, student_proj_u, student_out_u = all_feat[len(images_l):], all_proj[len(images_l):], all_out[len(images_l):]
        # teacher_out_u = student_out_u.detach()

        # # clustering, sup
        # sup_logits = torch.cat([f[mask_lab_l] for f in (student_out_l / 0.1).chunk(2)], dim=0)
        # sup_labels = torch.cat([labels_l[mask_lab_l] for _ in range(2)], dim=0)
        # cls_loss = self.ce(sup_logits, sup_labels)

        # # clustering, unsup
        # # cluster_loss = self.cluster_criterion(student_out_u, teacher_out_u, global_epoch)
        # cluster_loss = self.cluster_criterion(student_out_u, teacher_out_u, local_epoch)
        # avg_probs = (student_out_u / 0.1).softmax(dim=1).mean(dim=0)

        # ## Minimize point-wise entropy L2 distance
        # if self.args.client.distance_type == 'jsd':
        #     avg_probs = torch.clamp(avg_probs, 1e-8, 1)
        #     prior_dist = torch.clamp(self.prior_dist, 1e-8, 1)
        #     intermediate_dist = (avg_probs + prior_dist) / 2

        #     entropy_distance = 0.5 * torch.sum( avg_probs * torch.log(avg_probs /intermediate_dist)) + 0.5 * torch.sum( prior_dist * torch.log(prior_dist /intermediate_dist))

        # elif self.args.client.distance_type == 'jsd_noclamp':
        #     avg_probs += 1e-8
        #     prior_dist = self.prior_dist + 1e-8
        #     intermediate_dist = (avg_probs + prior_dist) / 2

        #     entropy_distance = 0.5 * torch.sum( avg_probs * torch.log(avg_probs /intermediate_dist)) + 0.5 * torch.sum( prior_dist * torch.log(prior_dist /intermediate_dist))
        # elif self.args.client.distance_type == 'jsd_noclamp_binary':
        #     avg_probs += 1e-8
        #     prior_dist = self.prior_dist + 1e-8
        #     intermediate_dist = (avg_probs + prior_dist) / 2

        #     entropy_distance = 0.5 * torch.sum( avg_probs * torch.log(avg_probs /intermediate_dist)) + 0.5 * torch.sum( prior_dist * torch.log(prior_dist /intermediate_dist))

        #     binary_avg_probs = torch.cat([avg_probs[:len(self.args.dataset.seen_classes)], avg_probs[len(self.args.dataset.seen_classes):]])
        #     binary_prior_dist = torch.cat([prior_dist[:len(self.args.dataset.seen_classes)], prior_dist[len(self.args.dataset.seen_classes):]])
        #     binary_intermediate_dist = (binary_avg_probs + binary_prior_dist) / 2

        #     binary_entropy_distance = 0.5 * torch.sum( binary_avg_probs * torch.log(binary_avg_probs /binary_intermediate_dist)) + 0.5 * torch.sum( binary_prior_dist * torch.log(binary_prior_dist /binary_intermediate_dist))
        #     entropy_distance += binary_entropy_distance
            
        # elif self.args.client.distance_type == 'simgcd':
        #     entropy_distance = - torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
        # else:
        #     raise ValueError(f"Distance type {self.args.client.distance_type} not supported")
        # me_max_loss = entropy_distance

        # # represent learning, unsup
        # contrastive_logits_l, contrastive_labels_l = self.info_nce_logits(features=student_proj_l)
        # contrastive_logits_u, contrastive_labels_u = self.info_nce_logits(features=student_proj_u)
        # contrastive_loss_l = torch.nn.CrossEntropyLoss()(contrastive_logits_l, contrastive_labels_l)
        # contrastive_loss_u = torch.nn.CrossEntropyLoss()(contrastive_logits_u, contrastive_labels_u)
        # contrastive_loss = (contrastive_loss_l + contrastive_loss_u) / 2

        # # representation learning, sup
        # student_proj_l = torch.cat([f[mask_lab_l].unsqueeze(1) for f in student_proj_l.chunk(2)], dim=1)
        # student_proj_l = torch.nn.functional.normalize(student_proj_l, dim=-1)
        # sup_con_labels_l = labels_l[mask_lab_l]
        # sup_con_loss = SupConLoss()(student_proj_l, labels=sup_con_labels_l)

        # # Losses
        # losses["self_cls"] =  cluster_loss
        # losses["sup_cls"] = cls_loss
        # losses["sup_con"] =sup_con_loss
        # losses["con"] = contrastive_loss
        # losses["memax"] = me_max_loss

        # return losses


    def _flush_memory(self):
        del self.model
        #del self.global_model
        del self.optimizer
        del self.loader
        del self.class_counts
        del self.sorted_class_dict
        torch.cuda.empty_cache()
        gc.collect()

