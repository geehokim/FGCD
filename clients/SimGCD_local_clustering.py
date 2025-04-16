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
from utils import extract_feature
from finch import FINCH
from utils import LossManager
from collections import OrderedDict



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
class SimGCD_Local_Clustering_Client():

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
        self.clustering_result = {'best_val_acc': 0}
        self.decorr_criterion = FedDecorrLoss()

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

        results_before = self.evaler.local_trainset_eval(model=self.global_model, epoch=epoch,
                                                  local_train_loader=copy.deepcopy(self.loader))
        conf_plot_before = self.evaler.plot_confusion_matrix(results_before["conf_matrix"])

        results = self.evaler.local_trainset_eval(model=self.model, epoch=epoch, local_train_loader=copy.deepcopy(self.loader))
        conf_plot = self.evaler.plot_confusion_matrix(results["conf_matrix"])
        umap_plot = self.evaler.visualize_local_umap(model=copy.deepcopy(self.model), client_idx=self.client_index,
                                                     all_feats=results['all_feats'],
                                                     targets=results['targets'], epoch=epoch)

        all_acc = results["all_acc"]
        new_acc = results["new_acc"]
        old_acc = results["old_acc"]
        all_acc_before = results_before["all_acc"]
        new_acc_before = results_before["new_acc"]
        old_acc_before = results_before["old_acc"]

        wandb_dict = {
            f"local_all_acc_{self.client_index}/{self.args.dataset.name}": all_acc,
            f"local_old_acc_{self.client_index}/{self.args.dataset.name}": old_acc,
            f"local_new_acc_{self.client_index}/{self.args.dataset.name}": new_acc,
            f"local_all_acc_{self.client_index}_before/{self.args.dataset.name}": all_acc_before,
            f"local_old_acc_{self.client_index}_before/{self.args.dataset.name}": old_acc_before,
            f"local_new_acc_{self.client_index}_before/{self.args.dataset.name}": new_acc_before,
            f"client_{self.client_index}_conf_matrix_after": conf_plot["confusion_matrix"],
            f"client_{self.client_index}_conf_matrix_before": conf_plot_before["confusion_matrix"],
            f"client{self.client_index}_umap": umap_plot["umap"],
        }

        logger.warning(
            f'[Epoch {epoch}] Local Trainset ALL Acc: {all_acc:.2f}%, OLD Acc: {old_acc:.2f}%, NEW Acc: {new_acc:.2f}%')


        self.wandb_log(wandb_dict, step=global_epoch)
        # return {
        #     "all_acc": all_acc,
        #     "new_acc": new_acc,
        #     "old_acc": old_acc
        # }

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
        # if self.model is None:
        #     self.model = model
        # else:
        #     self.model.load_state_dict(model.state_dict())
        # self._update_model(model)
        self.model = model
        self.global_model = copy.deepcopy(self.model)
        self.clustering_result = {'best_val_acc': 0}


        # if self.global_model is None:
        #     self.global_model = copy.deepcopy(self.model)
        # else:
        #     self.global_model.load_state_dict(model.state_dict())
        #self._update_global_model(model)

        # self.global_model = copy.deepcopy(model)
        #
        # for fixed_model in [self.global_model]:
        #     for n, p in fixed_model.named_parameters():
        #         p.requires_grad = False

        self.device = device
        # self.num_layers = self.model.num_layers
        # self.num_layers = 6

        # self.loader = DataLoader(local_dataset, batch_size=self.args.batch_size, shuffle=True)
        # train_sampler = None
        # if self.args.dataset.num_instances > 0:
        #     train_sampler = RandomClasswiseSampler(local_dataset, num_instances=self.args.dataset.num_instances)
        # self.loader =  DataLoader(local_dataset, batch_size=self.args.batch_size, sampler=train_sampler, shuffle=train_sampler is None,
        #                            num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, drop_last=False)

        # --------------------
        # SAMPLER
        # Sampler which balances labelled and unlabelled examples in each batch
        # --------------------




        self.class_counts = np.sort([*local_dataset.class_dict.values()])[::-1]


        sorted_key = np.sort([*local_dataset.class_dict.keys()])
        sorted_class_dict = {} 
        for key in sorted_key:  
            sorted_class_dict[key] = local_dataset.class_dict[key]
            
        if global_epoch == 0:
            logger.warning(f"Class counts : {self.class_counts}")
            logger.info(f"Sorted class dict : {sorted_class_dict}")

        self.sorted_class_dict = sorted_class_dict

        ### Get validation set (labelled data)

        if global_epoch >= self.args.client.start_update:
            ### Cluster all training data and estimate k based on the accuracy on the validation set.
            estimation_result = self._estimate_k(copy.deepcopy(local_dataset))

            ### update prior
            current_prior_est = self.prior_estimation(estimation_result)
            uniform_prior = torch.ones(self.num_classes) / self.num_classes

            alpha = self.args.client.update_lambda
            #self.prior_dist =  alpha * current_prior_est + (1 - alpha) * uniform_prior
            self.prior_dist = current_prior_est

        self.prior_dist = self.prior_dist.to(self.device)
        print(self.prior_dist)



        # Init dataloader for the local dataset
        label_len = len(local_dataset.labelled_dataset)
        unlabelled_len = len(local_dataset.unlabelled_dataset)
        sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(local_dataset))]
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(local_dataset))
        self.loader = DataLoader(local_dataset, batch_size=self.args.batch_size, sampler=sampler, shuffle=False,
                                 num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, drop_last=False,
                                 )

        # ### Update prior based on the clustering result
        # self.update_prior(estimation_result['y_pred'], estimation_result['num_clust'])
        # self.prior_dist = self.prior_dist.to(self.device)

        params_groups = self.get_params_groups(self.model)
        self.optimizer = optim.SGD(
            params_groups, lr=local_lr, momentum=self.args.optimizer.momentum, weight_decay=self.args.optimizer.wd)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=200,
        #                                                       eta_min=self.args.trainer.local_lr * 1e-3)

    def __repr__(self):
        print(f'{self.__class__} {self.client_index}, {"data : "+len(self.loader.dataset) if self.loader else ""}')


    def _estimate_k(self, local_dataset):
        test_transform = copy.deepcopy(self.evaler.test_loader.dataset.transform)
        labelled_set = local_dataset.labelled_dataset
        unlabelled_set = local_dataset.unlabelled_dataset
        labelled_set.transform = test_transform
        unlabelled_set.transform = test_transform

        result = {}

        feats_labelled, targets_labelled, _ = extract_feature(copy.deepcopy(self.model), dataset=labelled_set,
                                                           device=self.device)
        num_labeled = len(targets_labelled)
        feats_unlabelled, targets_unlabelled, _ = extract_feature(copy.deepcopy(self.model), dataset=unlabelled_set,
                                                           device=self.device)
        all_feats = torch.cat([feats_labelled.cpu(), feats_unlabelled.cpu()], dim=0)
        all_targets = torch.cat([targets_labelled.cpu(), targets_unlabelled.cpu()])

        labelled_class_set = set(targets_labelled.cpu().numpy().tolist())

        ## Use Finch Algorithm
        acc_results = {}
        best_val_acc = 0
        k = 0
        result = {}
        total_classes = list(range(len(self.args.dataset.seen_classes) + len(self.args.dataset.unseen_classes)))
        old_classes = total_classes[:len(self.args.dataset.seen_classes)]
        new_classes = total_classes[len(self.args.dataset.seen_classes):]
        for num_cluster in range(len(labelled_class_set)+1, len(labelled_class_set) * 3):
            c, num_clust, req_c = FINCH(all_feats.numpy(), req_clust=num_cluster, distance='cosine', verbose=False)
            y_pred = req_c[:len(targets_labelled)]
            # y_true = targets.astype(int)
            y_true = targets_labelled.cpu().numpy().astype(int)

            D = max(y_pred.max(), y_true.max()) + 1
            w = np.zeros((D, D), dtype=int)
            for i in range(y_pred.size):
                w[y_pred[i], y_true[i]] += 1

            ind = linear_assignment(w.max() - w)
            ind = np.vstack(ind).T
            ind_map = {j: i for i, j in ind}

            class_set_labeled = set(y_true)
            ind_map_labeled = {i: ind_map[i] for i in y_true}

            total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

            acc_results[num_cluster] = total_acc
            #if total_acc > best_val_acc:
            if total_acc > self.clustering_result['best_val_acc']:
                self.clustering_result['best_val_acc'] = total_acc
                k = num_cluster
                self.clustering_result.update({
                    'num_clust': k,
                    'all_feats': all_feats,
                    'all_preds': req_c,
                    'num_labeled': num_labeled,
                    'y_true_labeled': y_true,
                    'seen_class_map': ind_map_labeled,
                    'unlabeled_class_set': torch.unique(targets_unlabelled).long(),
                    'clustering_map_for_test': ind,
                })
        self.clustering_result.update({
                'local_seen_classes': set(y_true)
        })
        print(acc_results)
        best_val_acc = self.clustering_result['best_val_acc']
        k = self.clustering_result['num_clust']
        print(f'best k: {k}, best_val_acc: {best_val_acc}')

        ## Get preds on entire dataset and check accuracy on entire dataset
        c, num_clust, req_c = FINCH(all_feats.numpy(), req_clust=k, distance='cosine', verbose=False)
        y_pred = req_c
        # y_true = targets.astype(int)
        y_true = all_targets.numpy().astype(int)

        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=int)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1

        ind = linear_assignment(w.max() - w)
        ind = np.vstack(ind).T

        ind_map = {j: i for i, j in ind}
        total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
        old_acc = 0
        total_old_instances = 0
        old_classes_local = []
        new_classes_local = []
        for i in ind_map:
            if i in old_classes:
                old_classes_local.append(i)
            else:
                new_classes_local.append(i)
        for i in old_classes_local:
            old_acc += w[ind_map[i], i]
            total_old_instances += sum(w[:, i])
        old_acc /= total_old_instances

        new_acc = 0
        total_new_instances = 0
        for i in new_classes_local:
            new_acc += w[ind_map[i], i]
            total_new_instances += sum(w[:, i])
        new_acc /= total_new_instances

        print(f'train_cluster_acc using best k, all: {total_acc}, old: {old_acc}, new: {new_acc}')
        return self.clustering_result


    def prior_estimation(self, estimation_result):
        all_feats = estimation_result['all_feats']
        all_preds = estimation_result['all_preds']
        num_labeled = estimation_result['num_labeled']
        feats_unlabeled = all_feats[num_labeled:]
        y_true_labeled = estimation_result['y_true_labeled']
        seen_class_map = estimation_result['seen_class_map']
        clustering_map = estimation_result['clustering_map_for_test']
        print(clustering_map)

        ## init prior
        prior = torch.zeros(self.num_classes)

        ## update prior for labeled dataset
        for cls in y_true_labeled:
            smooth_max = self.args.client.smooth_max
            smooth_onehot = torch.ones(self.num_classes) * (1 - smooth_max) / (self.num_classes - 1)
            smooth_onehot[int(cls)] = smooth_max
            prior += smooth_onehot

        ## update prior for unlabeled seen

        # Compute prototypes using all feats
        class_features = {}
        class_counts = {}
        for pred, feat in zip(all_preds, all_feats):

            if pred not in class_features:
                class_features[pred] = []
                class_counts[pred] = 0
            class_features[pred].append(feat)
            class_counts[pred] += 1

        prototypes = []
        for cls, feats in class_features.items():
            prototype = torch.stack(feats).mean(dim=0)
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes)
        prototypes = F.normalize(prototypes, dim=-1)

        # Compute cosine similarity between prototypes and unmatched classifiers
        classifier_weights = self.model.proj_layer.last_layer.parametrizations.weight.original1.data
        classifier_weights = F.normalize(classifier_weights, dim=-1)

        # # Invert the classifier indices
        # inverted_indices = torch.arange(classifier_weights.size(0) - 1, -1, -1)
        # inverted_classifier_weights = classifier_weights[inverted_indices]

        similarity_matrix = torch.matmul(prototypes, classifier_weights.T)
        ind = linear_assignment(similarity_matrix.max() - similarity_matrix)
        ind = np.vstack(ind).T

        # Map the inverted indices back to the original indices
        #ind[:, 1] = inverted_indices[ind[:, 1]].cpu().numpy()

        ind_map_pred_to_true = {i: j for i, j in ind}


        # Update prior for unlabeled unseen classes
        preds_unlabeled = all_preds[num_labeled:]
        for pred in preds_unlabeled:
            smooth_max = self.args.client.smooth_max
            smooth_onehot = torch.ones(self.num_classes) * (1 - smooth_max) / (self.num_classes - 1)
            smooth_onehot[ind_map_pred_to_true[pred]] = smooth_max
            prior += smooth_onehot


        # for prototype_idx, classifier_idx in zip(row_ind, col_ind):
        #     true_class = classifier_idx
        #     smooth_max = self.args.client.smooth_max
        #     smooth_onehot = torch.ones(self.num_classes) * (1 - smooth_max) / (self.num_classes - 1)
        #     smooth_onehot[true_class] = smooth_max
        #     prior += smooth_onehot * class_counts[prototype_idx]
        


        # # inv -> {pred: true}
        # inv_seen_class_map = {j: i for i, j in seen_class_map.items()}
        # mask = np.array([True if pred in inv_seen_class_map.keys() else False for pred in y_pred_unlabeled])
        #
        # y_pred_unlabeled_seen = y_pred_unlabeled[mask]
        #
        # transformed_y_pred = [inv_seen_class_map[pred] for pred in y_pred_unlabeled_seen]
        #
        # for cls in transformed_y_pred:
        #     smooth_max = self.args.client.smooth_max
        #     smooth_onehot = torch.ones(self.num_classes) * (1 - smooth_max) / (self.num_classes - 1)
        #     smooth_onehot[int(cls)] = smooth_max
        #     prior += smooth_onehot
        #
        # ## Matching between prototypes of unlabeled unseen and classifiers
        # y_pred_unlabeled_unseen = y_pred_unlabeled[~mask]
        # feats_unlabeled_unseen = feats_unlabeled[~mask]
        #
        # # Compute prototypes for unlabeled unseen classes more efficiently
        # class_features = {}
        # class_counts = {}
        # for pred, feat in zip(y_pred_unlabeled_unseen, feats_unlabeled_unseen):
        #     if pred not in class_features:
        #         class_features[pred] = []
        #         class_counts[pred] = 0
        #     class_features[pred].append(feat)
        #     class_counts[pred] += 1
        #
        # prototypes_unseen = []
        # for cls, feats in class_features.items():
        #     prototype = torch.stack(feats).mean(dim=0)
        #     prototypes_unseen.append(prototype)
        # prototypes_unseen = torch.stack(prototypes_unseen)
        # prototypes_unseen = F.normalize(prototypes_unseen, dim=-1)
        # unique_unseen_classes = np.array(list(class_features.keys()))
        #
        # # Get unmatched classifiers
        # matched_classes = set(seen_class_map.keys())
        # unmatched_classifiers = [i for i in range(self.num_classes) if i not in matched_classes]
        #
        # # Compute cosine similarity between prototypes and unmatched classifiers
        # classifier_weights = self.model.proj_layer.last_layer.parametrizations.weight.original1.data
        # unmatched_weights = classifier_weights[unmatched_classifiers]
        # unmatched_weights = F.normalize(unmatched_weights, dim=-1)
        #
        # similarity_matrix = torch.matmul(prototypes_unseen, unmatched_weights.T)
        # row_ind, col_ind = linear_assignment(similarity_matrix.max() - similarity_matrix)
        #
        # # Update prior for unlabeled unseen classes
        # for prototype_idx, classifier_idx in zip(row_ind, col_ind):
        #     true_class = unmatched_classifiers[classifier_idx]
        #     smooth_max = self.args.client.smooth_max
        #     smooth_onehot = torch.ones(self.num_classes) * (1 - smooth_max) / (self.num_classes - 1)
        #     smooth_onehot[true_class] = smooth_max
        #     prior += smooth_onehot * class_counts[unique_unseen_classes[prototype_idx]]

        prior = prior / len(all_preds)
        return prior

    def _update_classifier_head(self, estimation_result):
        feats = estimation_result['all_feats']
        preds = estimation_result['y_pred']
        preds = torch.tensor(preds).long()
        k = estimation_result['num_clust']

        # 2. Calculate class prototypes
        prototypes = []
        for i in range(k):
            # Find indices of samples belonging to class i
            class_indices = (preds == i).nonzero().squeeze()

            if class_indices.numel() > 0:
                # Calculate mean of features for class i
                class_feats = feats[class_indices]
                prototype = F.normalize(torch.mean(class_feats, dim=0), dim=0)
            else:
                raise False

            prototypes.append(prototype)

        # Stack prototypes into a single tensor
        prototypes = torch.stack(prototypes)
        in_dim = self.model.proj_layer.in_dim
        out_dim = k
        last_layer = nn.Linear(in_dim, k, bias=False)
        last_layer.weight.data = prototypes
        self.model.proj_layer._init_param_normalized_last_layer(last_layer, freeze=self.args.client.freeze_proto)

    def _transform_targets(self, dataset, seen_class_map, seen_classes):
        ind_map = {j : i for i, j in seen_class_map}
        aligned_targets = np.array([ind_map[i] if i in seen_classes else i for i in dataset.targets])
        dataset.targets = aligned_targets

    def get_weights(self, epoch=None):

        weights = {
            "self_cls": self.args.client.unsup_cls_weight,
            "sup_cls": self.args.client.sup_cls_weight,
            "sup_con": self.args.client.sup_con_weight,
            "con": self.args.client.unsup_con_weight,
            "memax": self.args.client.memax_weight * (self.args.client.unsup_cls_weight),
            'decorr': self.args.client.decorr_loss_weight,
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

        if global_epoch % 50 == 0:
            print(self.weights)

        for local_epoch in range(self.args.trainer.local_epochs):
            end = time.time()


            self.model.to(self.device)
            for i, (images, labels, uq_idxs, mask_lab) in tqdm(enumerate(self.loader), total=len(self.loader)):

                # if len(images.size()) == 3:
                #     images = images.unsqueeze(0)
                if self.args.client.n_views > 1:
                   images = torch.cat(images, dim=0)
                
                images, labels = images.to(self.device), labels.to(self.device)
                mask_lab = mask_lab[:, 0]
                mask_lab = mask_lab.to(self.device).bool()

                with autocast(enabled=self.args.use_amp):
                    #losses = self._algorithm(images, labels, uq_idxs, mask_lab, global_epoch)
                    losses = self._algorithm(images, labels, uq_idxs, mask_lab, global_epoch)
                    loss = sum([self.weights[loss_key]*losses[loss_key] for loss_key in losses])
                    # results = self.model(images)
                    # loss = self.criterion(results["logit"], labels) # if errors occur, use ResNet18_base instead of ResNet18_GFLN

                # if self.args.get('debugs'):
                #     breakpoint()
                self.optimizer.zero_grad()

                try:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    if self.g_clipping:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                    scaler.step(self.optimizer)
                    scaler.update()
                except Exception as e:
                    print(e)
                # loss.backward()
                # self.optimizer.step()

                    # breakpoint()
                #loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gr_clipping_max_norm)
                #self.optimizer.step()

                loss_meter.update(loss.item(), images.size(0))
                memax_loss_meter.update(losses['memax'].item(), images.size(0))
                time_meter.update(time.time() - end)
                end = time.time()
            #self.scheduler.step()
            #print(self.scheduler.get_last_lr())

            if self.args.trainer.local_eval and (local_epoch+ 1) % 1 == 0:
            # if self.args.trainer.local_eval:
                # self.evaluate(epoch=local_epoch, local_datasets=None)
                self.evaluate_local_trainset(epoch=local_epoch, global_epoch=global_epoch, local_datasets=None)

            # self.scheduler.step()
            # if self.args.client.update_prior and (local_epoch+ 1) % self.args.client.update_prior_freq == 0 and global_epoch >= self.args.client.start_update:
            #     self.update_prior(epoch=local_epoch)

        logger.info(f"[C{self.client_index}] End. Time: {end-start:.2f}s, Loss: {loss_meter.avg:.3f}")
        self.model.to('cpu')

        loss_dict = {
            f'loss/{self.args.dataset.name}/cls': loss_meter.avg,
            f'loss/{self.args.dataset.name}/memax': memax_loss_meter.avg,
        }
        state_dict = self.model.state_dict()
        # filtered_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     if 'last_layer' not in k:
        #         filtered_state_dict[k] = v


        local_labelled_centroids, local_centroids = self.get_centroids()

        centroids = {
            'local_labelled_centroids': local_labelled_centroids,
            'local_centroids': local_centroids,
        }


        # Flush model memories
        self._flush_memory()


        return  state_dict, loss_dict, centroids

    def get_centroids(self):

        test_transform = copy.deepcopy(self.evaler.test_loader.dataset.transform)
        local_dataset = copy.deepcopy(self.loader.dataset)
        labelled_dataset = local_dataset.labelled_dataset
        labelled_dataset.transform = test_transform
        unlabelled_dataset = local_dataset.unlabelled_dataset
        unlabelled_dataset.transform = test_transform
        
        feats_labelled, targets_labelled, logits_labelled = extract_feature(copy.deepcopy(self.model),
                                                                            dataset=labelled_dataset,
                                                                            device=self.device)

        feats_unlabelled, targets_unlabelled, logits_unlabelled = extract_feature(copy.deepcopy(self.model),
                                                                            dataset=unlabelled_dataset,
                                                                            device=self.device)
        
        num_labelled = len(targets_labelled)

        all_feats = torch.cat([feats_labelled.cpu(), feats_unlabelled.cpu()], dim=0)
        all_feats = F.normalize(all_feats, dim=-1)
        all_targets = torch.cat([targets_labelled.cpu(), targets_unlabelled.cpu()])
        all_logits = torch.cat([logits_labelled.cpu(), logits_unlabelled.cpu()])

        local_centroids_dict = {}
        for feat, target in zip(feats_labelled, targets_labelled):
            if local_centroids_dict.get(target.item()) is None:
                local_centroids_dict[target.item()] = [feat, ]
            else:
                local_centroids_dict[target.item()].append(feat)

        # Get centroids via feature mean and save # of samples
        for j in local_centroids_dict:
            local_centroids_dict[j] = torch.stack(local_centroids_dict[j])
            local_centroids_dict[j] = (local_centroids_dict[j].size(0), local_centroids_dict[j].mean(dim=0))

        all_preds = all_logits.argmax(1)

        cents = {}
        for feat, pred in zip(all_feats, all_preds):
            if cents.get(pred.item()) is None:
                cents[pred.item()] = [feat, ]
            else:
                cents[pred.item()].append(feat)
        for i in cents:
            feat_mean = torch.stack(cents[i]).mean(0)
            #feat_mean = F.normalize(feat_mean, dim=-1)
            cents[i] = feat_mean

        return local_centroids_dict, cents


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
    

    def _algorithm(self, images, labels, uq_idxs, mask_lab, global_epoch=0) -> Dict:

        losses = defaultdict(float)

        student_feats, student_proj, student_out = self.model(images, return_all=True)
        teacher_out = student_out.detach()

        # clustering, sup
        sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
        sup_labels = torch.cat([labels[mask_lab] for _ in range(2)], dim=0)
        cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

        # clustering, unsup
        cluster_loss = self.cluster_criterion(student_out, teacher_out, global_epoch)
        avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)

        ## Entropy regularizarion
        if global_epoch < self.args.client.start_update:
            me_max_loss = - torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
        else:
            ## Minimize point-wise entropy L2 distance
            if self.args.client.distance_type == 'l2':
                preds_entropy = torch.log(avg_probs ** (-avg_probs)) + math.log(float(len(avg_probs)))
                target_entropy = torch.log(self.prior_dist ** (-self.prior_dist)) + math.log(float(len(avg_probs)))
                entropy_distance = torch.sum((target_entropy - preds_entropy) ** 2)
            elif self.args.client.distance_type == 'l2_no_norm':
                preds_entropy = torch.log(avg_probs ** (-avg_probs))
                target_entropy = torch.log(self.prior_dist ** (-self.prior_dist))
                entropy_distance = torch.sum((target_entropy - preds_entropy) ** 2)
            elif self.args.client.distance_type == 'abs':
                preds_entropy = torch.log(avg_probs ** (-avg_probs)) + math.log(float(len(avg_probs)))
                target_entropy = torch.log(self.prior_dist ** (-self.prior_dist)) + math.log(float(len(avg_probs)))
                entropy_distance = torch.sum(torch.abs(target_entropy - preds_entropy))
            elif self.args.client.distance_type == 'sum':
                preds_entropy = torch.sum(torch.log(avg_probs ** (-avg_probs)))
                target_entropy = torch.sum(torch.log(self.prior_dist ** (-self.prior_dist)))
                entropy_distance = (target_entropy - preds_entropy) ** 2
            elif self.args.client.distance_type == 'seen_only':
                preds_entropy = torch.log(avg_probs ** (-avg_probs)) + math.log(float(len(avg_probs)))
                target_entropy = torch.log(self.prior_dist ** (-self.prior_dist)) + math.log(float(len(avg_probs)))
                entropy_distance = torch.sum(((target_entropy - preds_entropy) ** 2)[:len(self.args.dataset.seen_classes)])
            elif self.args.client.distance_type == 'unseen_only':
                preds_entropy = torch.log(avg_probs ** (-avg_probs)) + math.log(float(len(avg_probs)))
                target_entropy = torch.log(self.prior_dist ** (-self.prior_dist)) + math.log(float(len(avg_probs)))
                entropy_distance = torch.sum(((target_entropy - preds_entropy) ** 2)[len(self.args.dataset.seen_classes):])
            elif self.args.client.distance_type == 'arb_classes':
                preds_entropy = torch.log(avg_probs ** (-avg_probs)) + math.log(float(len(avg_probs)))
                target_entropy = torch.log(self.prior_dist ** (-self.prior_dist)) + math.log(float(len(avg_probs)))
                entropy_distance = torch.sum(((target_entropy - preds_entropy) ** 2)[:self.args.client.reg_classes])
            me_max_loss = entropy_distance
        #me_max_loss = - torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
        #cluster_loss += self.args.client.memax_weight * me_max_loss

        # represent learning, unsup
        contrastive_logits, contrastive_labels = self.info_nce_logits(features=student_proj)
        contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

        # representation learning, sup
        student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
        student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
        sup_con_labels = labels[mask_lab]
        sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

        # decorr loss
        decorr_loss = self.decorr_criterion(student_feats)


        
        losses["self_cls"] =  cluster_loss
        losses["sup_cls"] = cls_loss
        losses["sup_con"] =sup_con_loss
        losses["con"] = contrastive_loss
        losses["memax"] = me_max_loss
        losses["decorr"] = decorr_loss

        # del results
        return losses


    def update_prior(self, preds, k):

        ## Make Calss dict
        class_dict = defaultdict(int)
        for pred in preds:
            class_dict[str(pred)] += 1

        ## Update prior preds
        prior = torch.zeros(k)
        total = 0
        for cls in class_dict:
            if not self.args.client.label_smoothing:
                prior[int(cls)] += class_dict[cls]
            else:
                smooth_max = self.args.client.smooth_max
                smooth_values = torch.ones(k) * (1 -  smooth_max) / (k - 1)
                smooth_values[int(cls)] = smooth_max
                prior += smooth_values * class_dict[cls]

            total += class_dict[cls]
        prior = prior.float() / total

        # prior = prior.to(self.device)
        # self.prior_dist = self.args.client.update_lambda * self.prior_dist + (1 - self.args.client.update_lambda) * prior
        self.prior_dist = prior

    def _flush_memory(self):
        del self.model
        #del self.global_model
        del self.optimizer
        del self.loader
        del self.class_counts
        del self.sorted_class_dict
        torch.cuda.empty_cache()
        gc.collect()

