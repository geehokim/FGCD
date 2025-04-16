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

import logging
logger = logging.getLogger(__name__)

from clients.build import CLIENT_REGISTRY
import wandb
from tqdm import tqdm
import random
import numpy as np

from utils import LossManager
from torch.optim import lr_scheduler

from utils.misc import extract_local_features_only
from datasets.base import MergedDatasetClusterSemi
from utils.infomap_cluster_utils import cluster_by_semi_infomap, get_dist_nbr
import collections
from utils import cluster_acc, np, linear_assignment




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
class SimGCDOnlinePriorMatchingSemiClients():

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
            args.trainer.global_rounds,
            # args.trainer.local_epochs,
            args.client.n_views,
            args.client.warmup_teacher_temp,
            args.client.teacher_temp,
        )
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
            f"{self.args.dataset.name}/client{self.client_index}/local_all_acc": all_acc,
            f"{self.args.dataset.name}/client{self.client_index}/local_old_acc": old_acc,
            f"{self.args.dataset.name}/client{self.client_index}/local_new_acc": new_acc,
            f"{self.args.dataset.name}/client{self.client_index}/local_cluster_acc": results["cluster_acc"],
            f"{self.args.dataset.name}/client{self.client_index}/local_cluster_old_acc": results["cluster_old_acc"],
            f"{self.args.dataset.name}/client{self.client_index}/local_cluster_new_acc": results["cluster_new_acc"],
        }

        logger.warning(
            f'[Epoch {epoch}] Test ALL Acc: {all_acc:.2f}%, OLD Acc: {old_acc:.2f}%, NEW Acc: {new_acc:.2f}%')
        logger.warning(
            f'[Epoch {epoch}] Test Cluster Acc: {results["cluster_acc"]:.2f}%, Cluster Old Acc: {results["cluster_old_acc"]:.2f}%, Cluster New Acc: {results["cluster_new_acc"]:.2f}%')

        plt.close()

        self.wandb_log(wandb_dict, step=epoch)
        return {
            "all_acc": all_acc,
            "new_acc": new_acc,
            "old_acc": old_acc
        }
    
    
    def evaluate_local_trainset_unlabelled(self, epoch: int, global_epoch=0,
                                local_datasets: List[torch.utils.data.Dataset] = None) -> Dict:


        results = self.evaler.local_trainset_eval_unlabelled(model=self.model, epoch=epoch,
                                                  local_train_loader=copy.deepcopy(self.loader_local_eval))
        
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

        self.wandb_log(wandb_dict, step=global_epoch)
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

    def setup(self, model, device, local_dataset, global_epoch, local_lr, **kwargs):
        self.model = model
        self.global_model = copy.deepcopy(self.model)
        self.device = device
        self.local_dataset = copy.deepcopy(local_dataset)
        self.local_lr = local_lr
        

        self.loader_for_clustering = DataLoader(local_dataset, batch_size=4096, shuffle=False,
                                 num_workers=self.args.num_workers, pin_memory=False, drop_last=False,
                                 )
        cluster_labels, act_protos, local_prototypes, labeled_prototypes = self.local_clustering_semi(self.args, self.model, self.loader_for_clustering, self.evaler, global_epoch, self.device)
        self.cluster_labels = cluster_labels
        
        local_dataset_training = MergedDatasetClusterSemi(copy.deepcopy(local_dataset.labelled_dataset), copy.deepcopy(local_dataset.unlabelled_dataset), cluster_labels=self.cluster_labels)
        # local_dataset_training = copy.deepcopy(local_dataset)
        # local_dataset_training.labelled_dataset.targets = copy.deepcopy(self.cluster_labels)
        # local_dataset_unlabelled_clustered = copy.deepcopy(local_dataset.unlabelled_dataset)
        # local_dataset_unlabelled_clustered.targets = self.cluster_labels

        len_num_seen_classes = len(self.args.dataset.seen_classes)
        self.model.proj_layer.act_protos = act_protos
        self.model.proj_layer.local_prototypes.data[len_num_seen_classes:act_protos] = local_prototypes[len_num_seen_classes:]
        if global_epoch == 0:
            self.model.proj_layer.local_prototypes.data[:len_num_seen_classes] = labeled_prototypes[:]

        # if self.args.client.reiniit_labeled_prototypes:
        #     self.model.proj_layer.last_layer.parametrizations.weight.original1.data[:len(labeled_prototypes)] = labeled_prototypes
        
        # --------------------
        # SAMPLER
        # Sampler which balances labelled and unlabelled examples in each batch
        # --------------------
        label_len = len(local_dataset.labelled_dataset)
        unlabelled_len = len(local_dataset.unlabelled_dataset)
        sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(local_dataset))]
        sample_weights = torch.DoubleTensor(sample_weights)
        if self.args.client.sampler == 'weighted':
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(local_dataset))
        elif self.args.client.sampler == 'class_balanced':
            sampler = RandomMultipleGallerySamplerNoCam(local_dataset_training, self.args.num_instances)
        self.loader = DataLoader(local_dataset_training, batch_size=self.args.batch_size, sampler=sampler, shuffle=False,
                                 num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, drop_last=False,
                                 )
        self.loader_local_eval = DataLoader(self.local_dataset, batch_size=1024, shuffle=False,
                                 num_workers=self.args.num_workers, pin_memory=False, drop_last=False,
                                 )
        
        
        # cluster_train_loader = IterLoader(
        #             DataLoader(local_dataset_unlabelled_clustered,
        #                        batch_size=self.args.batch_size, num_workers=self.args.num_workers,
        #                        shuffle=True, pin_memory=True, drop_last=False))
        # cluster_train_loader.new_epoch()

        # model init
        
        
        # initialize local memeory using cluster centroids
        print(f'Global Epoch {global_epoch+1} Learning Rate: {self.local_lr}')
        params_groups = self.get_params_groups(self.model)
        self.optimizer = optim.SGD(
            params_groups, lr=local_lr, momentum=self.args.optimizer.momentum, weight_decay=self.args.optimizer.wd)
        # self.exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        #                             self.optimizer,
        #                             T_max=self.args.trainer.local_epochs,
        #                             eta_min=local_lr * 1e-3,
        #                             )

        self.class_counts = np.sort([*local_dataset.class_dict.values()])[::-1]
        self.num_classes = len(self.args.dataset.seen_classes) + len(self.args.dataset.unseen_classes)

        sorted_key = np.sort([*local_dataset.class_dict.keys()])
        sorted_class_dict = {} 
        for key in sorted_key:  
            sorted_class_dict[key] = local_dataset.class_dict[key]
            
        # if global_epoch == 0:
        #     logger.warning(f"Class counts : {self.class_counts}")
        #     logger.info(f"Sorted class dict : {sorted_class_dict}")

        self.sorted_class_dict = sorted_class_dict


    def __repr__(self):
        print(f'{self.__class__} {self.client_index}, {"data : "+len(self.loader.dataset) if self.loader else ""}')

    def get_weights(self, epoch=None):

        weights = {
            "self_cls": self.args.client.unsup_cls_weight,
            "sup_cls": self.args.client.sup_cls_weight,
            "sup_con": self.args.client.sup_con_weight,
            "con": self.args.client.unsup_con_weight,
            "memax": self.args.client.memax_weight * self.args.client.unsup_cls_weight,
        }
        
        return weights

    
    def local_clustering_semi(self, args, model, loader, evaler, global_epoch, device, local_epoch=0):

        model.eval()
        feats_labelled, targets_labelled, feats_unlabelled, targets_unlabelled, mask = extract_local_features_only(args, model, loader, evaler, device, labelled_test_transform=False)
        targets_unlabelled = targets_unlabelled.astype(int)
        targets_labelled = targets_labelled.astype(int)
        all_feats = torch.cat([feats_labelled, feats_unlabelled], dim=0)
        all_targets = np.concatenate([targets_labelled, targets_unlabelled])
        if_labeled = torch.zeros(len(all_targets))
        if_labeled[:len(targets_labelled)] = 1
        if_labeled = if_labeled.bool()
        client_mask = all_targets < len(self.args.dataset.seen_classes)
        client_mask_unlabelled = client_mask[len(targets_labelled):]


        cluster_feature = all_feats.clone()
        cluster_feature = F.normalize(cluster_feature, dim=-1).cpu().numpy()
        all_targets = torch.from_numpy(all_targets)
        
        # infomap clustering, k1 = num_of neighbours, k2 = min num_of clusters, eps = min similarity
        # cluster_feature = cluster_feature.to(torch.bfloat16)
        feat_dists, feat_nbrs = get_dist_nbr(features=cluster_feature, k=args.client.top_k, knn_method='faiss-gpu', device=device)
        pseudo_labels = cluster_by_semi_infomap(feat_nbrs, feat_dists, min_sim=args.client.tao_f, cluster_num=args.client.k2, label_mark=all_targets, if_labeled=if_labeled, args=args)
        pseudo_labels = pseudo_labels.astype(int)
        # idx2label, idx_single, dists = cluster_by_infomap(cluster_feature, k=args.client.top_k, tao_f=args.client.tao_f, dataset_name=args.dataset.name)


        ## merged labelled examples to pseudo_labels
        ## first get the mapping from pred to gt
        pseudo_labels_labelled = pseudo_labels[:len(targets_labelled)].astype(int)
        D = max(pseudo_labels_labelled.max(), targets_labelled.max()) + 1
        w = np.zeros((D, D), dtype=int)
        for i in range(pseudo_labels_labelled.size):
            w[pseudo_labels_labelled[i], targets_labelled[i]] += 1

        ind = linear_assignment(w.max() - w)
        ind = np.vstack(ind).T

        ind_map_pred_to_gt = {j: i for i, j in ind}
        ind_map_gt_to_pred = {i: j for i, j in ind if i in np.unique(targets_labelled)}
        ind_map_pred_to_gt_labelled = {j + len(self.args.dataset.seen_classes): i for i, j in ind_map_gt_to_pred.items()}
        total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / targets_labelled.size
        print(f"Total clustering accuracy for labelled data: {total_acc}")

        # Add len(self.args.dataset.seen_classes) + 1 to all non-negative labels
        pseudo_labels_unlabelled = np.array([pl + len(self.args.dataset.seen_classes) if pl != -1 else -1 for pl in pseudo_labels[len(targets_labelled):]])
        updated_pseudo_labels_unlabelled = []
        for i in pseudo_labels_unlabelled:
            if i in ind_map_pred_to_gt_labelled:
                updated_pseudo_labels_unlabelled.append(ind_map_pred_to_gt_labelled[i])
            else:
                updated_pseudo_labels_unlabelled.append(i)

        # Remap pseudo_labels_unlabelled to have consecutive class labels starting from len(self.args.dataset.seen_classes)+1
        # Find unique labels that are greater than or equal to len(self.args.dataset.seen_classes)+1
        unique_unseen_labels = sorted(set([label for label in updated_pseudo_labels_unlabelled if (label >= len(self.args.dataset.seen_classes) and label != -1)]))
        
        # Create mapping from old labels to new consecutive labels
        unseen_label_map = {old_label: len(self.args.dataset.seen_classes)+idx for idx, old_label in enumerate(unique_unseen_labels)}
        
        # Apply the mapping to pseudo_labels_unlabelled
        for i in range(len(updated_pseudo_labels_unlabelled)):
            if updated_pseudo_labels_unlabelled[i] > len(self.args.dataset.seen_classes):
                updated_pseudo_labels_unlabelled[i] = unseen_label_map[updated_pseudo_labels_unlabelled[i]]

        pseudo_labels[:len(targets_labelled)] = targets_labelled
        pseudo_labels[len(targets_labelled):] = updated_pseudo_labels_unlabelled
        
        logger.info(f"Remapped cluster labels to consecutive integers: {len(unique_unseen_labels)} unique clusters")

        centers = collections.defaultdict(list)

        for i, label in enumerate(pseudo_labels):
            if label == -1:
                continue
            centers[label].append(all_feats[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0)
        
        # estimate_k = centers.size(0)
        estimate_k = max(np.unique(pseudo_labels)) + 1
        self.estimate_k = estimate_k
        print(f"Estimated number of clusters: {estimate_k}")
        print(f"gt_to_pred: {ind_map_gt_to_pred}")
        act_protos = estimate_k
        local_prototypes = centers.to(device)

        targets_labelled_unique = np.unique(targets_labelled)
        centroids_labelled = []
        for cls_id in targets_labelled_unique:
            data_idx = np.where(targets_labelled == cls_id)[0]
            data = feats_labelled[data_idx]
            prototype = data.mean(dim=0, keepdims=True)
            centroids_labelled.append(prototype)

        labeled_prototypes = torch.cat(centroids_labelled, dim=0).to(device)

        # # note that pseudo_labels contains -1 for unassigned examples
        # # pseudo_labels = preds_unsingle.astype(np.intp)
        # # 방법 1: -1로 초기화하여 클러스터링되지 않은 샘플 처리
        # sorted_cluster_labels = [-1] * len(cluster_feature)
        # for idx in idx2label:
        #     sorted_cluster_labels[idx] = idx2label[idx]
        # sorted_cluster_labels = np.array(sorted_cluster_labels)

        # Calculate accuracies for this client
        unique_pseudo_labels, pseudo_label_counts = np.unique(pseudo_labels, return_counts=True)
        pseudo_label_set = dict(zip(unique_pseudo_labels, pseudo_label_counts))
        logger.info(f"Pseudo label set: {pseudo_label_set}")

        all_acc, old_acc, new_acc, _, _ = evaler.log_accs_from_preds(
            y_true=all_targets.numpy(),
            y_pred=pseudo_labels,
            mask=client_mask,
            T=0,
            eval_funcs=['v2'],
            save_name=f'Local Infomap accuracy'
        )
        print(f'Local Infomap accuracy Unlabelled - All Acc: {all_acc:.4f}, Seen Acc: {old_acc:.4f}, Unseen Acc: {new_acc:.4f}')
        if args.wandb:
            wandb.log({
                    f"{args.dataset.name}/client{self.client_index}/infomap_all_acc": all_acc * 100,
                f"{args.dataset.name}/client{self.client_index}/infomap_old_acc": old_acc * 100,
                f"{args.dataset.name}/client{self.client_index}/infomap_new_acc": new_acc * 100,
            }, step=local_epoch)
        
        
        return pseudo_labels, act_protos, local_prototypes, labeled_prototypes
    
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

        if global_epoch % 50 == 0:
            print(self.weights)

        # self.evaluate_local_trainset_unlabelled(epoch=0, local_datasets=None)

        for local_epoch in range(self.args.trainer.local_epochs):
            end = time.time()
            self.model.train()
            self.model.to(self.device)
            for i, (images, labels, uq_idxs, mask_lab) in tqdm(enumerate(self.loader), total=len(self.loader)):

                # if len(images.size()) == 3:
                #     images = images.unsqueeze(0)
                if self.args.client.n_views > 1:
                   images = torch.cat(images, dim=0)
                
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                mask_lab = mask_lab[:, 0]
                mask_lab = mask_lab.to(self.device, non_blocking=True).bool()

                online_prior = torch.zeros(self.estimate_k)
                epsilon = torch.ones(self.estimate_k) * 1e-7
                online_prior += epsilon
                online_prior = online_prior.to(self.device)

                valid_mask = ~mask_lab & (labels != -1).bool()
                labels_unlabelled = labels[valid_mask]
                class_weights = torch.ones_like(labels_unlabelled, dtype=torch.float).to(self.device)
                online_prior.index_add_(0, labels_unlabelled, class_weights)
                online_prior = online_prior / len(labels_unlabelled)  # Normalize to get probability distribution

                # with autocast(enabled=self.args.use_amp):
                    #losses = self._algorithm(images, labels, uq_idxs, mask_lab, global_epoch)
                losses = self._algorithm(images, labels, uq_idxs, mask_lab, global_epoch, local_epoch, online_prior)
                loss = sum([self.weights[loss_key]*losses[loss_key] for loss_key in losses])
                    # results = self.model(images)
                    # loss = self.criterion(results["logit"], labels) # if errors occur, use ResNet18_base instead of ResNet18_GFLN

                # if self.args.get('debugs'):
                #     breakpoint()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_meter.update(loss.item(), images.size(0))
                memax_loss_meter.update(losses['memax'].item(), images.size(0))
                time_meter.update(time.time() - end)
                end = time.time()

            if self.args.trainer.local_eval and (local_epoch+ 1) % self.args.trainer.local_eval_freq == 0:
            # if self.args.trainer.local_eval:
                #self.evaluate(epoch=local_epoch, local_datasets=None)
                self.evaluate_local_trainset_unlabelled(epoch=local_epoch, local_datasets=None, global_epoch=global_epoch)
            # self.scheduler.step()
        
        logger.info(f"[C{self.client_index}] End. Time: {end-start:.2f}s, Loss: {loss_meter.avg:.3f}")
        self.model.to('cpu')

        loss_dict = {
            f'loss/{self.args.dataset.name}/cls': loss_meter.avg,
            f'loss/{self.args.dataset.name}/memax': memax_loss_meter.avg,
        }


        state_dict = self.model.state_dict()

        results = {
            'cluster_means': None,
            'cluster_targets': None,
            'optimizer_state_dict': self.optimizer.state_dict(),
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
    

    def _algorithm(self, images, labels, uq_idxs, mask_lab, global_epoch=0, local_epoch=0, online_prior=None) -> Dict:

        losses = defaultdict(float)

        student_proj, student_out, student_out_clustered = self.model(images)
        teacher_out = student_out.detach()
        teacher_out_clustered = student_out_clustered.detach()

        # clustering, sup
        sup_logits = torch.cat([f[mask_lab] for f in (student_out_clustered / 0.1).chunk(2)], dim=0)
        sup_labels = torch.cat([labels[mask_lab] for _ in range(2)], dim=0)
        cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

        # clustering, unsup
        # cluster_loss = self.cluster_criterion(student_out, teacher_out, global_epoch)
        # for local training, reproduce
        ### for labelled data
        valid_mask = (labels != -1)
        teacher_out = torch.cat([f[valid_mask] for f in teacher_out_clustered.chunk(2)], dim=0)
        student_out = torch.cat([f[valid_mask] for f in student_out_clustered.chunk(2)], dim=0)
        cluster_loss = self.cluster_criterion(student_out, teacher_out, global_epoch)

        # ### for unlabelled_data
        # valid_mask = ~mask_lab & (labels != -1)
        # teacher_out_clustered_unlabelled = torch.cat([f[valid_mask] for f in teacher_out_clustered.chunk(2)], dim=0)
        # student_out_clustered_unlabelled = torch.cat([f[valid_mask] for f in student_out_clustered.chunk(2)], dim=0)

        # cluster_loss_unlabelled = self.cluster_criterion(student_out_clustered_unlabelled, teacher_out_clustered_unlabelled, global_epoch)

        # num_labelled = mask_lab.sum().item()
        # num_unlabelled = len(mask_lab) - num_labelled
        # cluster_loss = cluster_loss_labelled * num_labelled + cluster_loss_unlabelled * num_unlabelled
        # cluster_loss = cluster_loss / (num_labelled + num_unlabelled)

        valid_mask = ~mask_lab & (labels != -1)
        student_out_unlabelled = torch.cat([f[valid_mask] for f in student_out_clustered.chunk(2)], dim=0)
        avg_probs = (student_out_unlabelled / 0.1).softmax(dim=1).mean(dim=0)
        
        if self.args.client.distance_type == 'jsd':
            avg_probs = torch.clamp(avg_probs, 1e-8, 1)
            online_prior = torch.clamp(online_prior, 1e-8, 1)
            intermediate_dist = (avg_probs + online_prior) / 2

            entropy_distance = 0.5 * torch.sum( avg_probs * torch.log(avg_probs /intermediate_dist)) + 0.5 * torch.sum( online_prior * torch.log(online_prior /intermediate_dist))
        elif self.args.client.distance_type == 'jsd_noclamp':
            avg_probs += 1e-8
            online_prior = online_prior + 1e-8
            intermediate_dist = (avg_probs + online_prior) / 2

            entropy_distance = 0.5 * torch.sum( avg_probs * torch.log(avg_probs /intermediate_dist)) + 0.5 * torch.sum( online_prior * torch.log(online_prior /intermediate_dist))
        elif self.args.client.distance_type == 'simgcd':
            entropy_distance = - torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
        
        me_max_loss = entropy_distance

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

    def _flush_memory(self):
        del self.model
        #del self.global_model
        del self.optimizer
        del self.loader
        del self.class_counts
        del self.sorted_class_dict
        torch.cuda.empty_cache()
        gc.collect()

