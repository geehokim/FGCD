#!/usr/bin/env python
# coding: utf-8
import copy
import time

import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import gc

import torch.nn.functional

from utils import IterLoader
from utils import *
from utils.metrics import evaluate
from utils.visualize import __log_test_metric__, umap_allmodels, cka_allmodels, log_fisher_diag
from models import build_encoder, get_model
from typing import Callable, Dict, Tuple, Union, List
from utils.logging_utils import AverageMeter
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from collections import defaultdict
from datasets.base import MergedDatasetCluster

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
from sklearn import metrics
import contextlib
from utils import get_dist_nbr, cluster_by_semi_infomap
from sklearn.mixture import GaussianMixture
from utils.faster_mix_k_means_pytorch import K_Means_MADD

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


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
        dist = -torch.cdist(anchor_feature, contrast_feature)
        anchor_dot_contrast = torch.div(dist, self.temperature)
        
        # anchor_dot_contrast = torch.div(
        #     torch.matmul(anchor_feature, contrast_feature.T),
        #     self.temperature)

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


class SupConLoss2(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss2, self).__init__()
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
        # dist = -torch.cdist(anchor_feature, contrast_feature)
        # anchor_dot_contrast = torch.div(dist, self.temperature)
        
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
class GCDClient_InfoMapCon_SeenHard():

    def __init__(self, args, client_index, loader=None, evaler=None):
        self.args = args
        self.client_index = client_index
        self.criterion = nn.CrossEntropyLoss()
        self.sup_con_crit = SupConLoss2(temperature=self.args.client.sup_temperature,
                                       base_temperature=self.args.client.sup_temperature)
        # self.sup_con_crit_orig = SupConLoss2(temperature=self.args.client.sup_temperature,
        #                                base_temperature=self.args.client.sup_temperature)
        self.cluster_con_crit = SupConLoss2(temperature=self.args.client.cluster_temperature,
                                       base_temperature=self.args.client.cluster_temperature)
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
        self.contrastive_cluster_weight_schedule = np.concatenate((
        np.linspace(args.client.cluster_con_weight * 0.1,
                    args.client.cluster_con_weight, args.client.contrastive_cluster_warmup_epochs),
        np.ones(args.trainer.global_rounds - args.client.contrastive_cluster_warmup_epochs) * args.client.cluster_con_weight
        ))
        
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
        

    def get_params_groups(self, model, cluster_loss, local_lr):
        backbones = []
        cluster_prototypes = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            else:
                backbones.append(param)
        for name, param in cluster_loss.named_parameters():
            if not param.requires_grad:
                continue
            else:
                cluster_prototypes.append(param)
        return [{'params': backbones, 'lr': local_lr}, {'params': cluster_prototypes, 'lr': local_lr * self.args.client.cluster_lr_scale}]
    

    def setup(self, model, device, local_dataset, global_epoch, local_lr, trainer, **kwargs):
        
        self.ce = nn.CrossEntropyLoss()
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.optimizer_state_dict = kwargs['optimizer_state_dict']

        # --------------------
        # SAMPLER
        # Sampler which balances labelled and unlabelled examples in each batch
        # --------------------
        label_len = len(local_dataset.labelled_dataset)
        unlabelled_len = len(local_dataset.unlabelled_dataset)
        sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(local_dataset))]
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(local_dataset))
        self.loader = DataLoader(local_dataset, batch_size=self.args.batch_size, shuffle=False, sampler=sampler,
                                 num_workers=self.args.num_workers,  drop_last=False)
        # self.loader = DataLoader(local_dataset, batch_size=self.args.batch_size, shuffle=False,
        #                         num_workers=self.args.num_workers,  drop_last=False)
        
        self.labelled_class_set = set(local_dataset.labelled_dataset.targets)
        
        cluster_labels, cluster_feature = self.semi_infomap(self.args, self.model, copy.deepcopy(self.loader), self.evaler, self.num_classes, global_epoch, self.device)
        self.cluster_labels = cluster_labels
        # self.cluster_labels = np.concatenate([local_dataset.labelled_dataset.targets, local_dataset.unlabelled_dataset.targets])
        
        
        updated_local_dataset = MergedDatasetCluster(copy.deepcopy(local_dataset.labelled_dataset), copy.deepcopy(local_dataset.unlabelled_dataset), class_dict=local_dataset.class_dict, cluster_labels=self.cluster_labels)
        # RC_sampler = RandomClasswiseSampler(updated_local_dataset, num_instances=16)
        # logger('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

        contrastive_cluster_train_loader = IterLoader(
                    DataLoader(updated_local_dataset,
                               batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                               shuffle=True, pin_memory=True, drop_last=False))
        contrastive_cluster_train_loader.new_epoch()
        self.contrastive_cluster_train_loader_predefine = contrastive_cluster_train_loader

        # self.updated_loader = DataLoader(updated_local_dataset, batch_size=self.args.batch_size, shuffle=False, sampler=sampler, 
        #                          num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, drop_last=False,
        #                          )
        # self.updated_loader = DataLoader(updated_local_dataset, batch_size=self.args.batch_size, shuffle=False,
        #                          num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, drop_last=False,
        #                          )

        cluster_prototypes = generate_cluster_features(self.cluster_labels, cluster_feature)
        self.cluster_loss = ClusterLoss(num_features=cluster_prototypes.size(1), num_samples=cluster_prototypes.size(0), temperature=self.args.client.cluster_temperature, momentum=self.args.client.memory_momentum, device=self.device)
        self.cluster_loss.prototypes = F.normalize(cluster_prototypes, dim=1)
        # self.cluster_loss.update_prototypes(cluster_prototypes.size(1), cluster_prototypes.size(0), device=self.device)
        # self.cluster_loss.to(self.device)
        
        if global_epoch < 5:
            local_lr = local_lr * global_epoch / 5
        
        # params_groups = self.get_params_groups(self.model, self.cluster_loss, local_lr)
        print(f'local_lr: {local_lr}')
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=local_lr, momentum=self.args.optimizer.momentum, weight_decay=self.args.optimizer.wd)

        # self.cluster_optimizer = optim.SGD(
        #     self.cluster_loss.parameters(), lr=local_lr * self.args.client.cluster_lr_scale, momentum=self.args.optimizer.momentum, weight_decay=self.args.optimizer.wd)    
        # self.cluster_optimizer = optim.SGD(
        #     self.cluster_loss.parameters(), lr=0, momentum=self.args.optimizer.momentum, weight_decay=self.args.optimizer.wd)    

        # if self.optimizer_state_dict is not None:
        #     new_state_dict = {
        #     'param_groups' : self.optimizer.state_dict()['param_groups'],
        #     'state' : copy.deepcopy(self.optimizer_state_dict)
        #     }
        #     self.optimizer.load_state_dict(new_state_dict)
            
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = local_lr
        #     # self.optimizer.param_groups[-1]['lr'] = local_lr * self.args.client.cluster_lr_scale
        #     self.optimizer.zero_grad()
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

    def semi_infomap(self, args, model, loader, evaler, num_classes, global_epoch, device):

        model.eval()
        feats_labelled, feats_proj_labelled, logits_labelled, targets_labelled, feats_unlabelled, feats_proj_unlabelled, logits_unlabelled, targets_unlabelled, mask = extract_local_features(args, model, loader, evaler, device, labelled_test_transform=False)
        targets_unlabelled = targets_unlabelled.astype(int)
        targets_labelled = targets_labelled.astype(int)

        feats_labelled = F.normalize(feats_labelled, dim=-1)
        feats_unlabelled = F.normalize(feats_unlabelled, dim=-1)
        
        all_feats = torch.cat([feats_labelled, feats_unlabelled], dim=0)
        all_feats_proj = torch.cat([feats_proj_labelled, feats_proj_unlabelled], dim=0)
        all_targets = np.concatenate([targets_labelled, targets_unlabelled])
        #all_logits = torch.cat([logits_labelled, logits_unlabelled], dim=0)
        if_labeled = torch.zeros(len(all_targets))
        if_labeled[:len(targets_labelled)] = 1
        if_labeled = if_labeled.bool()
        client_mask = all_targets < len(self.args.dataset.seen_classes)

        # all_feats_proj = F.normalize(all_feats_proj, dim=-1)
        # all_feats = F.normalize(all_feats, dim=-1)

        if args.client.cluster_feature == 'feats':
            cluster_feature = all_feats.clone()
        elif args.client.cluster_feature == 'feats_proj':
            cluster_feature = all_feats_proj.clone()
        else:
            raise ValueError(f"Invalid cluster feature: {args.client.cluster_feature}")

        # infomap clustering, k1 = num_of neighbours, k2 = min num_of clusters, eps = min similarity
        # cluster_feature = cluster_feature.to(torch.bfloat16)
        kmeanssem_madd = K_Means_MADD(k=len(np.unique(all_targets)), tolerance=1e-4, max_iterations=10, init='k-means++',
                 n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=None, mode=None, protos=None,cluster_size=None,is_code=False)
        
        kmeanssem_madd.fit_mix(feats_unlabelled, feats_labelled, torch.from_numpy(targets_labelled))

        cluster_feature = F.normalize(cluster_feature, dim=-1).cpu().numpy()
        feat_dists, feat_nbrs = get_dist_nbr(features=cluster_feature, k=args.client.k1, knn_method='faiss_cpu', device=0)
        pseudo_labels = cluster_by_semi_infomap(feat_nbrs, feat_dists, min_sim=args.client.eps, cluster_num=args.client.k2, label_mark=all_targets, if_labeled=if_labeled, args=args)
        
        # note that pseudo_labels contains -1 for unassigned examples
        pseudo_labels = pseudo_labels.astype(np.intp)

        # Calculate accuracies for this client
        unique_pseudo_labels, pseudo_label_counts = np.unique(pseudo_labels, return_counts=True)
        pseudo_label_set = dict(zip(unique_pseudo_labels, pseudo_label_counts))
        logger.info(f"Pseudo label set: {pseudo_label_set}")

        all_acc, old_acc, new_acc, _, _ = evaler.log_accs_from_preds(
            y_true=all_targets,
            y_pred=pseudo_labels,
            mask=client_mask,
            T=0,
            eval_funcs=['v2'],
            save_name=f'Local Infomap accuracy'
        )

        print(f'Local Infomap accuracy - All Acc: {all_acc:.4f}, Seen Acc: {old_acc:.4f}, Unseen Acc: {new_acc:.4f}')
        if args.wandb:
            wandb.log({
                    f"{args.dataset.name}/{self.client_index}/infomap_all_acc": all_acc * 100,
                f"{args.dataset.name}/{self.client_index}/infomap_old_acc": old_acc * 100,
                f"{args.dataset.name}/{self.client_index}/infomap_new_acc": new_acc * 100,
            }, step=global_epoch)

        ## Filter out -1 values from pseudo_labels
        cluster_set = set(pseudo_labels)
        cluster_prototypes = generate_cluster_features(pseudo_labels, all_feats.clone())

        ## Get labelled centroids
        unique_labels_labelled = np.unique(targets_labelled).tolist()
        class_centroids_labelled = [feats_labelled[targets_labelled == label].mean(dim=0) for label in unique_labels_labelled]
        class_centroids_labelled = torch.stack(class_centroids_labelled)

        # Initialize the mapping dictionary
        ind_map_gt_to_pred = {}
        assigned_clusters = set()
        remaining_clusters = cluster_set

        seen_class_similarities = F.cosine_similarity(class_centroids_labelled.unsqueeze(1), cluster_prototypes.unsqueeze(0), dim=2)

        ## Filter out the classes that have wasserstein distance less than threshold
        threshold = args.client.align_threshold
        for i, label in enumerate(unique_labels_labelled):
            class_centroid = class_centroids_labelled[i]
            simiilarities_between_clusters = seen_class_similarities[i]
            # Get indices sorted by similarity in descending order
            sorted_indices = torch.argsort(simiilarities_between_clusters, descending=True)
            # Find first unassigned cluster index
            for idx in sorted_indices:
                cluster_ind = idx.item()
                if cluster_ind not in assigned_clusters:
                    break
            print(f'class {label} - selected cluster index: {cluster_ind}')
            ## Estimate distribution of the cluster
            if args.client.clust_feats == 'feats':
                cluster_feats = all_feats[pseudo_labels == cluster_ind]
            elif args.client.clust_feats == 'feats_proj':
                cluster_feats = all_feats_proj[pseudo_labels == cluster_ind]
            gmm_cluster = GaussianMixture(n_components=1).fit(cluster_feats.cpu().clone().numpy())
            cluster_mean = gmm_cluster.means_[0]
            cluster_var = np.diag(gmm_cluster.covariances_[0])
            diff = class_centroid.cpu().clone().numpy() - cluster_mean
            inv_var = 1 / cluster_var
            mean_var = np.sqrt(np.sum(cluster_var))
            wasserstein_dist = np.sqrt(np.sum((diff ** 2) * inv_var))
            print(f'class {label} - wasserstein distance: {wasserstein_dist} - mean_var: {mean_var} - threshold: {args.client.align_threshold * mean_var}')
            if wasserstein_dist <= threshold * mean_var:
                ind_map_gt_to_pred[label] = cluster_ind
                assigned_clusters.add(cluster_ind)
                remaining_clusters.remove(cluster_ind)
        
        print(f'ind_map_gt_to_pred: {ind_map_gt_to_pred}')
        num_remaining_clusters = len(remaining_clusters)
        num_of_total_clusters = len(unique_labels_labelled) + num_remaining_clusters
        
        #new semi-supervised pseudo_label_mapping
        pseudo_labels_unlabelled = pseudo_labels[len(targets_labelled):]
        result_pseudo_labels = np.zeros_like(pseudo_labels) - 2
        for idx, label in enumerate(unique_labels_labelled):
            if label in ind_map_gt_to_pred.keys():
                cluster_idx = ind_map_gt_to_pred[label]
                result_pseudo_labels[:len(targets_labelled)][label == targets_labelled] = idx
                result_pseudo_labels[len(targets_labelled):][pseudo_labels_unlabelled == cluster_idx] = idx
            else:
                result_pseudo_labels[:len(targets_labelled)][label == targets_labelled] = idx
        cnt = len(unique_labels_labelled)
        for idx, cluster_idx in enumerate(remaining_clusters):
            if cluster_idx != -1:
                if cluster_idx in pseudo_labels_unlabelled:
                    result_pseudo_labels[len(targets_labelled):][pseudo_labels_unlabelled == cluster_idx] = cnt
                    
                    cnt += 1
            else:
                result_pseudo_labels[len(targets_labelled):][pseudo_labels_unlabelled == cluster_idx] = -1
            # print(f'result_pseudo_labels_set: {set(result_pseudo_labels)}')
        assert np.all(result_pseudo_labels != -2)

        print(f'num_result_pseudo_labels: {len(np.unique(result_pseudo_labels))}')
        print(f'result_pseudo_labels_set: {set(result_pseudo_labels)}')
        print(f'result_pseudo_labels: {result_pseudo_labels}')

        if args.client.cluster_feature == 'feats':
            return result_pseudo_labels, all_feats
        elif args.client.cluster_feature == 'feats_proj':
            return result_pseudo_labels, all_feats_proj
        else:
            raise ValueError(f"Invalid cluster feature: {args.client.cluster_feature}")
    
    def get_weights(self, epoch=None):

        weights = {
            "sup_con": self.args.client.sup_con_weight,
            "con": self.args.client.unsup_con_weight,
            "cluster_con":  self.contrastive_cluster_weight_schedule[epoch],
        }
        
        return weights

    def local_train(self, global_epoch, **kwargs):
        self.global_epoch = global_epoch
        self.model.train()
        self.model.to(self.device)
        scaler = GradScaler()
        start = time.time()
        loss_meter = AverageMeter('Loss', ':.2f')
        cluster_con_loss_meter = AverageMeter('Cluster_Con_Loss', ':.2f')
        time_meter = AverageMeter('BatchTime', ':3.1f')

        self.weights = self.get_weights(epoch=global_epoch)
            
        if global_epoch % 50 == 0:
            print(self.weights)

        for local_epoch in range(self.args.trainer.local_epochs):
            end = time.time()

            self.model.train()
            self.model.to(self.device)
            self.cluster_loss.train()
            self.cluster_loss.to(self.device)
            for i, (images, labels, uq_idxs, mask_lab) in enumerate(self.loader):

                # old_prototypes = self.cluster_loss.prototypes.data
            #for i, (images, labels, uq_idxs, mask_lab) in enumerate(self.updated_loader):

                if self.args.client.n_views > 1:
                    images = torch.cat(images, dim=0)
                
                images, labels = images.to(self.device), labels.to(self.device)
                mask_lab = mask_lab[:, 0]
                mask_lab = mask_lab.to(self.device).bool()
                
                if True not in mask_lab:
                    continue


                # with autocast(enabled=self.args.use_amp, dtype=torch.bfloat16 if self.bfloat16_support else torch.float16):
                losses = self._algorithm(i, images, labels, uq_idxs, mask_lab, global_epoch)
                loss = sum([self.weights[loss_key]*losses[loss_key] for loss_key in losses])
                    
                self.optimizer.zero_grad()
                # self.cluster_optimizer.zero_grad()

                # try:
                #     scaler.scale(loss).backward()
                #     # scaler.unscale_(self.optimizer)
                #     # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                #     scaler.step(self.optimizer)
                #     scaler.update()
                # except Exception as e:
                #     print(e)

                loss.backward()
                self.optimizer.step()
                # self.cluster_optimizer.step()

                ## EMA prototypes update
                # self.cluster_loss.prototypes.data = old_prototypes * 0.9 + self.cluster_loss.prototypes.data * 0.1
                
                loss_meter.update(loss.item(), images.size(0))
                cluster_con_loss_meter.update(losses['cluster_con'].item(), images.size(0))
                time_meter.update(time.time() - end)
                end = time.time()

            if self.args.trainer.local_eval and (local_epoch+ 1) % self.args.trainer.local_eval_freq == 0:
                ind_map = self.evaluate_local_trainset(epoch=local_epoch, global_epoch=global_epoch, local_datasets=None)

            if self.args.local_test_eval:
                results = self.evaler.eval(model=self.model, epoch=local_epoch)
                all_acc = results["all_acc"]
                new_acc = results["new_acc"]
                old_acc = results["old_acc"]
                wandb_dict = {
                    f"{self.args.dataset.name}/global/all_acc": all_acc,
                    f"{self.args.dataset.name}/global/old_acc": old_acc,
                    f"{self.args.dataset.name}/global/new_acc": new_acc
                    }
                print(f"all_acc: {all_acc:.2f}%, new_acc: {new_acc:.2f}%, old_acc: {old_acc:.2f}%")
                wandb.log(wandb_dict, step=local_epoch)


        cluster_means = None
        cluster_targets = None
        
        logger.info(f"[C{self.client_index}] End. Time: {end-start:.2f}s, Loss: {loss_meter.avg:.3f}")
        self.model.to('cpu')
        self.cluster_loss.to('cpu')

        loss_dict = {
            f'loss/{self.args.dataset.name}/cls': loss_meter.avg,
            f'loss/{self.args.dataset.name}/cluster_con': cluster_con_loss_meter.avg,
        }
        state_dict = self.model.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        # # Map the momentum buffer in the optimizer state_dict to CPU
        # if 'state' in optimizer_state_dict:
        #     for state in optimizer_state_dict['state'].values():
        #         if 'momentum_buffer' in state:
        #             state['momentum_buffer'] = state['momentum_buffer'].cpu()

        results = {
            'cluster_means': cluster_means,
            'cluster_targets': cluster_targets,
            'optimizer_state_dict': optimizer_state_dict['state'],
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
    

    def _algorithm(self, batch_idx, images, labels, uq_idxs, mask_lab, global_epoch=0) -> Dict:

        losses = defaultdict(float)
        

        student_feat, student_proj, _ = self.model(images, return_all=True)

        student_proj = F.normalize(student_proj, dim=-1)
        student_feat = F.normalize(student_feat, dim=-1)

        # represent learning, unsup
        contrastive_logits, contrastive_labels = self.info_nce_logits(features=student_proj)
        contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

        # representation learning, sup
        student_proj_labelled = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
        # student_proj_labelled = torch.nn.functional.normalize(student_proj_labelled, dim=-1)
        sup_con_labels = labels[mask_lab]
        sup_con_loss = self.sup_con_crit(student_proj_labelled, labels=sup_con_labels)
        
        cluster_images, cluster_labels, cluster_uq_idxs, cluster_mask_lab, cluster_pseudo_labels = self.contrastive_cluster_train_loader_predefine.next()

        if batch_idx == 0:
            print(f'cluster_pseudo_labels: {cluster_pseudo_labels}')

        if self.args.client.n_views > 1:
            cluster_images = torch.cat(cluster_images, dim=0)
                
        cluster_images, cluster_labels = cluster_images.to(self.device), cluster_labels.to(self.device)
        cluster_mask_lab = cluster_mask_lab[:, 0]
        cluster_mask_lab = cluster_mask_lab.to(self.device).bool()
        cluster_pseudo_labels = cluster_pseudo_labels.to(self.device)

        cluster_pseudo_labels2 = cluster_pseudo_labels.detach().clone()
        cluster_pseudo_labels = torch.cat([cluster_pseudo_labels, cluster_pseudo_labels2], dim=0).to(self.device)

        valid_mask = cluster_pseudo_labels != -1
        cluster_images = cluster_images[valid_mask]
        cluster_pseudo_labels = cluster_pseudo_labels[valid_mask]

        cluster_feat, cluster_proj, _ = self.model(cluster_images, return_all=True)
        cluster_proj = F.normalize(cluster_proj, dim=-1)
        cluster_feat = F.normalize(cluster_feat, dim=-1)

        cluster_con_loss = self.cluster_loss(cluster_feat, target=cluster_pseudo_labels)



        
        
        # cluster_labels_unlabelled = cluster_label[~mask_lab]
        
        # cluster_feats = student_feat
        # # Filter out -1 values from cluster_labels_unlabelled
        # cluster_label2 = cluster_label.detach().clone()
        # cluster_label = torch.cat([cluster_label, cluster_label2], dim=0).to(self.device)
        # valid_mask = cluster_label != -1
        # valid_contrast_feats = cluster_feats[valid_mask]
        # valid_cluster_labels = cluster_label[valid_mask]
        # # cluster_con_loss = self.cluster_con_crit(contrast_feats_unlabelled, labels=cluster_labels_unlabelled)
        # cluster_con_loss = self.cluster_loss(valid_contrast_feats, target=valid_cluster_labels)
        # # cluster_con_loss = 0
        # # for i in range(0, min(self.args.client.cluster_con_levels, cluster_label.size(1))):
        # #     cluster_con_loss_i = self.cluster_con_crit(student_proj_unlabelled, labels=cluster_labels_unlabelled[:, i])
        # #     cluster_con_loss += cluster_con_loss_i / (2 ** (i+1))
            
        # Losses
        losses["sup_con"] =sup_con_loss
        losses["con"] = contrastive_loss
        # losses["con"] = torch.tensor(0.0)
        # losses["cluster_con"] = cluster_con_loss.to(self.device)
        losses["cluster_con"] = cluster_con_loss

        return losses


    def _flush_memory(self):
        del self.model
        #del self.global_model
        del self.optimizer
        del self.loader
        # del self.updated_loader
        del self.class_counts
        del self.sorted_class_dict
        torch.cuda.empty_cache()
        gc.collect()

