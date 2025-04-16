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
from datasets.base import MergedDatasetCluster
from sklearn import metrics
from torch.nn import functional as F
from models.vision_transformer_simgcd import ClientGMM

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
class GCLClient():

    def __init__(self, args, client_index, loader=None, evaler=None):
        self.args = args
        self.client_index = client_index
        self.criterion = nn.CrossEntropyLoss()
        self.sup_con_crit = SupConLoss(temperature=self.args.client.sup_temperature,
                                       base_temperature=self.args.client.sup_temperature)
        
        self.g_clipping = args.client.g_clipping
        self.evaler = evaler
        self.cluster_criterion = DistillLoss(
            args.client.warmup_teacher_temp_epochs,
            args.trainer.global_rounds,
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

        results = self.evaler.local_trainset_eval(model=self.model, epoch=epoch, local_train_loader=copy.deepcopy(self.loader))
        conf_plot = self.evaler.plot_confusion_matrix(results["conf_matrix"])

        all_acc = results["all_acc"]
        new_acc = results["new_acc"]
        old_acc = results["old_acc"]

        wandb_dict = {
            f"{self.args.dataset.name}/client{self.client_index}/local_all_acc": all_acc,
            f"{self.args.dataset.name}/client{self.client_index}/local_old_acc": old_acc,
            f"{self.args.dataset.name}/client{self.client_index}/local_new_acc": new_acc,
            f"{self.args.dataset.name}/client{self.client_index}/conf_matrix": conf_plot["confusion_matrix"],
        }

        logger.warning(
            f'[Epoch {epoch}] Local Trainset ALL Acc: {all_acc:.2f}%, OLD Acc: {old_acc:.2f}%, NEW Acc: {new_acc:.2f}%')


        self.wandb_log(wandb_dict, step=global_epoch)
        ind_map = results['ind_map']
        sorted_ind_map = dict(sorted(ind_map.items()))

        return sorted_ind_map
        

    def get_params_groups(self, model, local_lr):
        params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "proj_layer" in name:
                params.append({"params": param, "lr": local_lr * 0.1})
            else:
                params.append({"params": param})

        # Add cluster means and covariances to parameter groups
        # if hasattr(self, 'client_gmm'):
        if self.args.client.train_gmm:
            params.append({"params": self.client_gmm.cluster_means, "lr": local_lr * 0.1})
            params.append({"params": self.client_gmm.cluster_log_covariances, "lr": local_lr * 0.1})
        return params

    

    def setup(self, model, device, local_dataset, global_epoch, local_lr, trainer, **kwargs):
        
        self.ce = nn.CrossEntropyLoss()
        self.model = model
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
        
        
        self.labelled_class_set = set(local_dataset.labelled_dataset.targets)

        cluster_labels = self.init_gmm_using_semi_finch(self.args, copy.deepcopy(self.model), copy.deepcopy(self.loader), self.evaler, self.num_classes, global_epoch, self.device)
        self.cluster_labels = cluster_labels
        updated_local_dataset = MergedDatasetCluster(copy.deepcopy(local_dataset.labelled_dataset), copy.deepcopy(local_dataset.unlabelled_dataset), class_dict=local_dataset.class_dict, cluster_labels=cluster_labels)

        self.updated_loader = DataLoader(updated_local_dataset, batch_size=self.args.batch_size, sampler=sampler, shuffle=False,
                                 num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, drop_last=False,
                                 )
        
        params_groups = self.get_params_groups(self.model, local_lr)
        self.optimizer = optim.SGD(
            params_groups, lr=local_lr, momentum=self.args.optimizer.momentum, weight_decay=self.args.optimizer.wd)

        self.class_counts = np.sort([*local_dataset.class_dict.values()])[::-1]


        sorted_key = np.sort([*local_dataset.class_dict.keys()])
        sorted_class_dict = {} 
        for key in sorted_key:  
            sorted_class_dict[key] = local_dataset.class_dict[key]
            
        if global_epoch == 0:
            logger.warning(f"Class counts : {self.class_counts}")
            logger.info(f"Sorted class dict : {sorted_class_dict}")

        self.sorted_class_dict = sorted_class_dict

        
        self.prior_dist = self.prior_dist.to(self.device)

    def __repr__(self):
        print(f'{self.__class__} {self.client_index}, {"data : "+len(self.loader.dataset) if self.loader else ""}')

    
    def init_gmm_using_semi_finch(self, args, model, loader, evaler, num_classes, global_epoch, device):
        feats_labelled, feats_proj_labelled, logits_labelled, targets_labelled, feats_unlabelled, feats_proj_unlabelled, logits_unlabelled, targets_unlabelled, mask = extract_local_features(args, model, loader, evaler, device)

        targets_unlabelled = targets_unlabelled.astype(int)
        targets_labelled = targets_labelled.astype(int)
        all_feats = torch.cat([feats_labelled, feats_unlabelled], dim=0)
        all_feats_proj = torch.cat([feats_proj_labelled, feats_proj_unlabelled], dim=0)
        all_targets = np.concatenate([targets_labelled, targets_unlabelled])
        #all_logits = torch.cat([logits_labelled, logits_unlabelled], dim=0)

        all_feats_proj = F.normalize(all_feats_proj, dim=-1)
        all_feats = F.normalize(all_feats, dim=-1)

        orig_dist = metrics.pairwise.pairwise_distances(all_feats_proj.numpy(), all_feats_proj.numpy(), metric='cosine')
        orig_dist_copy = copy.deepcopy(orig_dist)
        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)


        orig_dist_labelled = orig_dist_copy[:len(targets_labelled)]
        for cls in np.unique(targets_labelled):
            indices = np.where(targets_labelled == cls)[0]
            cls_dist = orig_dist_labelled[indices]
            cls_rank = np.argmax(cls_dist, axis=1)
            initial_rank[indices] = cls_rank


        #c, num_clust, _ = FINCH(all_feats_proj.numpy(), initial_rank=initial_rank, distance='cosine', verbose=True)
        c, num_clust, _ = FINCH(all_feats_proj.numpy(),  distance='cosine', verbose=True)

        class_set_labelled = np.unique(targets_labelled)

        # estimate num_clusters
        best_acc = -1
        best_k = 0
        best_idx = -1
        for i, k in enumerate(num_clust):
            
            preds = c[:, i]
            # Calculate accuracies for this client
            all_acc, old_acc, new_acc, w, _ = evaler.log_accs_from_preds(
                y_true=all_targets,
                y_pred=preds,
                mask=mask,
                T=0,
                eval_funcs=['v2'],
                save_name=f'Local Clustering Client {0}'
            )
            print(f'k: {k}, old_acc: {old_acc}, new_acc: {new_acc} | all_acc: {all_acc}')
            if old_acc > best_acc:
                best_acc = old_acc
                best_k = k
                best_idx = i
        
        self.num_clusters = best_k
        #self.num_clusters = len(np.unique(all_targets))
        print(f'Best k: {best_k}')
        preds = c[:, best_idx]

        ## calculate cluster means and covariances
        cluster_means = []
        cluster_covariances = []
        for cluster_id in range(self.num_clusters):
            cluster_indices = np.where(preds == cluster_id)[0]
            #cluster_indices = np.where(all_targets == cluster_id)[0]
            cluster_feats = all_feats_proj[cluster_indices]
            
            # Calculate mean
            mean = cluster_feats.mean(dim=0)
            cluster_means.append(mean)
            
            # Calculate diagonal covariance
            # Calculate covariance using torch.cov()
            cov = torch.cov(cluster_feats.T)  # Transpose to get features in columns
            # Get diagonal elements of covariance matrix

            diag_cov = torch.diag(cov)
            diag_cov = torch.clamp(diag_cov, min=1e-6)
            diag_cov = torch.sqrt(diag_cov)
            diag_cov = torch.log(diag_cov + 1e-6)
            # if torch.any(diag_cov < 0):
            #     raise ValueError("Negative values detected in diagonal covariance matrix")
            cluster_covariances.append(diag_cov)
        
        # Convert lists to tensors
        cluster_means = torch.stack(cluster_means)
        cluster_covariances = torch.stack(cluster_covariances)

        self.client_gmm = ClientGMM(args, cluster_means.size(-1), self.num_clusters)
        self.client_gmm.cluster_means.data.copy_(cluster_means)
        self.client_gmm.cluster_log_covariances.data.copy_(cluster_covariances)

        return preds
        #return all_targets


        
    
    def get_weights(self, epoch=None):

        weights = {
            "sup_con": self.args.client.sup_con_weight,
            "con": self.args.client.unsup_con_weight,
            "local_gcl": self.args.client.local_gcl_weight,
        }
        
        return weights

    def local_train(self, global_epoch, **kwargs):
        self.global_epoch = global_epoch
        self.model.train()
        self.model.to(self.device)
        scaler = GradScaler()
        start = time.time()
        loss_meter = AverageMeter('Loss', ':.2f')
        local_gcl_meter = AverageMeter('Local GCL', ':.2f')
        time_meter = AverageMeter('BatchTime', ':3.1f')

        self.weights = self.get_weights(epoch=global_epoch)
            
        if global_epoch % 50 == 0:
            print(self.weights)

        for local_epoch in range(self.args.trainer.local_epochs):
            end = time.time()

            self.model.train()
            self.model.to(self.device)
            self.client_gmm.to(self.device)
            self.client_gmm.train()
            for i, (images, labels, uq_idxs, mask_lab, cluster_label) in enumerate(self.updated_loader):

                if self.args.client.n_views > 1:
                   images = torch.cat(images, dim=0)
                
                images, labels, cluster_label = images.to(self.device), labels.to(self.device), cluster_label.to(self.device)
                mask_lab = mask_lab[:, 0]
                mask_lab = mask_lab.to(self.device).bool()

                with autocast(enabled=self.args.use_amp, dtype=torch.bfloat16 if self.bfloat16_support else torch.float16):
                    losses = self._algorithm(images, labels, uq_idxs, mask_lab, global_epoch, cluster_label)
                    loss = sum([self.weights[loss_key]*losses[loss_key] for loss_key in losses])
                    
                self.optimizer.zero_grad()

                try:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    # if self.g_clipping:
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                    scaler.step(self.optimizer)
                    scaler.update()
                except Exception as e:
                    print(e)

                loss_meter.update(loss.item(), images.size(0))
                local_gcl_meter.update(losses["local_gcl"].item(), images.size(0))
                time_meter.update(time.time() - end)
                end = time.time()

            if self.args.trainer.local_eval and (local_epoch+ 1) % self.args.trainer.local_eval_freq == 0:
                ind_map = self.evaluate_local_trainset(epoch=local_epoch, global_epoch=global_epoch, local_datasets=None)


        cluster_means = None
        cluster_targets = None
        
        
        logger.info(f"[C{self.client_index}] End. Time: {end-start:.2f}s, Loss: {loss_meter.avg:.3f}")
        self.model.to('cpu')
        self.client_gmm.to('cpu')
        self.client_gmm.eval()

        loss_dict = {
            f'loss/{self.args.dataset.name}/cls': loss_meter.avg,
            f'loss/{self.args.dataset.name}/local_gcl': local_gcl_meter.avg,
        }
        state_dict = self.model.state_dict()

        results = {
            'cluster_means': cluster_means,
            'cluster_targets': cluster_targets,
        }

        # Flush model memories
        self._flush_memory()


        return state_dict, loss_dict, results

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
    

    def _algorithm(self, images, labels, uq_idxs, mask_lab, global_epoch=0, cluster_label=None) -> Dict:

        losses = defaultdict(float)

        batch_size = images.size(0)
        num_clusters = self.num_clusters

        student_feat, student_proj, _ = self.model(images, return_all=True)

        student_proj_normalized = F.normalize(student_proj, dim=-1)
        student_proj = F.normalize(student_proj, dim=-1)

        # represent learning, unsup
        contrastive_logits, contrastive_labels = self.info_nce_logits(features=student_proj_normalized)
        contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

        # representation learning, sup
        student_proj_masked = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj_normalized.chunk(2)], dim=1)
        student_proj_masked = torch.nn.functional.normalize(student_proj_masked, dim=-1)
        sup_con_labels = labels[mask_lab]
        sup_con_loss = self.sup_con_crit(student_proj_masked, labels=sup_con_labels)

        # student_feats = [batch_size, dim], cluster_means = [num_clusters, dim], cluster_covariances = [num_clusters, dim]

        student_proj = student_proj.unsqueeze(1)
        mask_cluster = torch.ones(batch_size, num_clusters)
        mask_cluster = mask_cluster.to(self.device)
        cluster_label = torch.cat([cluster_label for _ in range(self.args.client.n_views)])
        mask_cluster[torch.arange(batch_size), cluster_label] = 0
        mask_cluster = mask_cluster.bool()

        # c_mean = self.cluster_means.unsqueeze(0).repeat(batch_size, 1, 1)
        # c_var = self.cluster_covariances.unsqueeze(0).repeat(batch_size, 1, 1)
        # repeated_cluster_means = self.client_gmm.cluster_means.unsqueeze(0).repeat(batch_size, 1, 1)
        # non_target_means = repeated_cluster_means.masked_select(mask_cluster.unsqueeze(-1)).view(batch_size, -1, self.client_gmm.cluster_means.size(-1))
        # target_means = self.client_gmm.cluster_means[cluster_label]

        # repeated_cluster_stds = self.client_gmm.cluster_covariances.unsqueeze(0).repeat(batch_size, 1, 1)
        # non_target_stds = repeated_cluster_stds.masked_select(mask_cluster.unsqueeze(-1)).view(batch_size, -1, self.client_gmm.cluster_covariances.size(-1))
        # target_stds = self.client_gmm.cluster_covariances[cluster_label]
        # denom = torch.sqrtnon_target_stds
        
        # Add epsilon to avoid division by zero
        epsilon = 1e-6
        #cov_l2_norm = torch.abs(self.client_gmm.cluster_covariances).sum(-1)
        ## compute log(|sigma_y| ^ (-1/2)) term
        cluster_covariances = torch.exp(self.client_gmm.cluster_log_covariances)
        # log_sigma = -0.5 * torch.sum(torch.log(cluster_covariances + epsilon), dim=-1)
        log_sigma = - 0.5 * torch.log(cluster_covariances + epsilon)

        diff = student_proj - self.client_gmm.cluster_means.unsqueeze(0)
        norm_diff = diff / (cluster_covariances.unsqueeze(0) + epsilon)
        norm_sq = (norm_diff ** 2)
        norm_sq =  -0.5 * norm_sq * 1.3

        # diff = student_proj - self.client_gmm.cluster_means.unsqueeze(0)
        # norm_diff  = diff / (cluster_covariances.unsqueeze(0) + epsilon)
        # norm_diff = torch.clamp(norm_diff, min=-1e4, max=1e4)
        # norm_sq = (norm_diff ** 2).sum(-1)
        # norm_sq =  -0.5 * norm_sq * 1.3

        non_target_norm_sq = norm_sq.masked_select(mask_cluster.unsqueeze(-1)).view(batch_size, -1, cluster_covariances.size(-1))
        non_target_log_sigma = log_sigma.masked_select(mask_cluster.unsqueeze(-1)).view(batch_size, -1, cluster_covariances.size(-1))

        # non_target_norm_sq = norm_sq.masked_select(mask_cluster).view(batch_size, -1)
        # non_target_log_sigma = log_sigma.unsqueeze(0).repeat(batch_size, 1).masked_select(mask_cluster).view(batch_size, -1)

        denominator = non_target_norm_sq + non_target_log_sigma
        # denominator = torch.logsumexp(denominator, dim=-1)
        
        denominator = torch.logsumexp(denominator, dim=1)

        if torch.isnan(denominator).any():
            print("Nan is detected!! at denominator")

        # target_norm_sq = norm_sq.masked_select(~mask_cluster).view(batch_size, -1)
        # target_log_sigma = log_sigma.unsqueeze(0).repeat(batch_size, 1).masked_select(~mask_cluster).view(batch_size, -1)
        target_norm_sq = norm_sq.masked_select(~mask_cluster.unsqueeze(-1)).view(batch_size, -1, cluster_covariances.size(-1))
        target_log_sigma = log_sigma.unsqueeze(0).repeat(batch_size, 1, 1).masked_select(~mask_cluster.unsqueeze(-1)).view(batch_size, -1, cluster_covariances.size(-1))
        
        numer = target_norm_sq.squeeze() + target_log_sigma.squeeze()
        if torch.isnan(numer).any():
            print("Nan is detected!! at numerator")

        log_likelihood = numer - denominator
        gmm_loss = -log_likelihood.mean()

        
        
        # cov_det = torch.prod(self.client_gmm.cluster_covariances, dim=-1)
        # #cov_l1_norm = torch.norm(self.client_gmm.cluster_covariances, dim=-1, p=1)
        # log_cov_det = torch.log(cov_det + epsilon)
        # #s_gmm = (student_proj - self.cluster_means.unsqueeze(0)) / (self.cluster_covariances.unsqueeze(0) + epsilon)
        # log_s_gmm = (student_proj - self.client_gmm.cluster_means.unsqueeze(0)) / (self.client_gmm.cluster_covariances.unsqueeze(0) + epsilon)
        # log_s_gmm = - 0.5 * torch.norm(log_s_gmm, dim=-1, p=2) * 1.3
        # #s_gmm = torch.logsumexp(s_gmm, dim=-1)

        # # target_s_gmm = s_gmm.masked_select(~mask_cluster.unsqueeze(-1)).view(batch_size, -1, self.cluster_covariances.size(-1)) # [batch_size, 1, dim]
        # non_target_s_gmm = log_s_gmm.masked_select(mask_cluster).view(batch_size, -1) # [batch_size, num_clusters-1, dim]
        # non_target_cov_det = log_cov_det.unsqueeze(0).repeat(batch_size, 1).masked_select(mask_cluster).view(batch_size, -1) # [batch_size, num_clusters-1, dim]

        # if torch.isnan(non_target_s_gmm).any():
        #     print("Nan is detected!! at non_target_s_gmm")

        # if torch.isnan(non_target_cov_det).any():
        #     print("Nan is detected!! at non_target_cov")
        # # denom = torch.sqrt((1 / non_target_stds + epsilon)) * non_target_s_gmm
        # denom = -0.5 * non_target_cov_det + non_target_s_gmm
        # denom = torch.logsumexp(denom, dim=-1)
        # if torch.isnan(denom).any():
        #     print("Nan is detected!! at denominator")

        # # numer = 0.5 * torch.log(target_stds + epsilon) + 0.5 * 1.3 * ((student_proj.squeeze() - target_means) / (target_stds + epsilon))**2
        # numer = 0.5 * torch.log(torch.prod(target_stds, dim=-1) + epsilon) + 0.5 * 1.3 * torch.norm((student_proj.squeeze() - target_means) / (target_stds + epsilon), dim=-1, p=2)
        # # numer = 0.5 * torch.log(target_stds.sum(-1) + epsilon)
        # # numer = 0.5 * 1.3 * (((student_proj.squeeze() - target_means) / (target_stds + epsilon))**2).sum(-1)
        # if torch.isnan(numer).any():
        #     print("Nan is detected!! at numerator")
        
        # tmp = 0.5 * 1.3 * (((student_proj.squeeze() - target_means) / (target_stds + epsilon))**2).sum(-1)
        # if torch.isnan(tmp).any():
        #     print("Nan is detected!! at numer s_gmm")
        # gmm_loss = numer + denom
        # gmm_loss = gmm_loss.mean()


        #gmm_nll = torch.nn.GaussianNLLLoss()
        #gmm_loss1 = gmm_nll(student_feat, c_mean, c_var**2)
        #gmm_loss = F.gaussian_nll_loss(student_feat, target_means, target_stds**2)

        # gmm_loss = 0.5 * torch.log(target_stds) + 0.5 * (((student_feat - target_means) / target_stds)**2) * 1.3
        # reg_loss = 0.5 * torch.log(target_stds) - (student_feat.squeeze() * target_means).sum(-1)

        alpha = 0.00
        local_gcl_loss = gmm_loss
        # Debugging: Print intermediate values
        # print(f"norm_diff: {norm_diff.mean().item()}")
        # print(f"norm_sq: {norm_sq.mean().item()}")
        # print(f"log_sigma: {log_sigma.mean().item()}")
        # print(f"numer: {numer.mean().item()}")
        # print(f"denominator: {denominator.mean().item()}")
        print(f'gmm_loss: {gmm_loss.item()}')    



        

        # gmm_loss2 = 0.5 * torch.log(c_var) + 0.5 * (((student_feat - c_mean) / c_var)**2) * 1
        # #gmm_loss = 0.5 * torch.log(self.cluster_covariances.unsqueeze(0)) + 0.5 * (((student_feat - self.cluster_means.unsqueeze(0)) / self.cluster_covariances.unsqueeze(0))**2) * 1.3
        
        
        # mask_cluster = torch.ones(batch_size, num_clusters)
        # mask_cluster = mask_cluster.to(self.device)
        # cluster_label = torch.cat([cluster_label for _ in range(self.args.client.n_views)])
        # mask_cluster[torch.arange(batch_size), cluster_label] = 0
        # mask_cluster = mask_cluster.bool()
        
        # gmm_cal = (s_gmm * (torch.norm(self.cluster_covariances, dim=-1).unsqueeze(0) ** 0.5))
        # denom = gmm_cal.masked_fill(~mask_cluster, 0)  # Using masked_fill to zero out False values
        # denom = gmm_cal.sum(-1)

        # numer = gmm_cal.masked_fill(mask_cluster, 0) 
        # numer = numer.sum(-1)

        # gmm_loss = -torch.log(numer + 1e-8) + torch.log(denom + 1e-8)

        
        # Losses
        losses["sup_con"] =sup_con_loss
        losses["con"] = contrastive_loss
        losses["local_gcl"] = local_gcl_loss

        return losses


    def _flush_memory(self):
        del self.model
        #del self.global_model
        del self.optimizer
        del self.loader
        del self.class_counts
        del self.sorted_class_dict
        torch.cuda.empty_cache()
        gc.collect()

