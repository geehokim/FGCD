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
from utils.train_utils import update_prior, update_prior_ovc
from finch import FINCH

from utils import LossManager, extract_local_features_unlabelled



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
class SimGCD_Update_Prior_Global_OV_Client():

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
        if self.args.client.ce_type == 'ce':
            self.ce = nn.CrossEntropyLoss()
        elif self.args.client.ce_type == 'margin':
            self.ce = MarginLoss(m=-1*self.args.client.margin_m, s=self.args.client.margin_s)
        else:
            raise ValueError(f"Invalid ce_type: {self.args.client.ce_type}")


        self.model = model
        self.global_model = copy.deepcopy(self.model)

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
        label_len = len(local_dataset.labelled_dataset)
        unlabelled_len = len(local_dataset.unlabelled_dataset)
        sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(local_dataset))]
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(local_dataset))
        self.loader = DataLoader(local_dataset, batch_size=self.args.batch_size, sampler=sampler, shuffle=False,
                                 num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, drop_last=False,
                                 )
        
        self.labelled_class_set = set(local_dataset.labelled_dataset.targets)

        
        self.class_counts = np.sort([*local_dataset.class_dict.values()])[::-1]


        sorted_key = np.sort([*local_dataset.class_dict.keys()])
        sorted_class_dict = {} 
        for key in sorted_key:  
            sorted_class_dict[key] = local_dataset.class_dict[key]
            
        if global_epoch == 0:
            logger.warning(f"Class counts : {self.class_counts}")
            logger.info(f"Sorted class dict : {sorted_class_dict}")

        self.sorted_class_dict = sorted_class_dict

        # ## Calculate GT Prior Distribution
        # prior = torch.zeros(self.num_classes)
        # total = 0
        # for cls in self.sorted_class_dict:
        #     if not self.args.client.label_smoothing:
        #         prior[int(cls)] += self.sorted_class_dict[cls]
        #     else:
        #         smooth_max = self.args.client.smooth_max
        #         smooth_values = torch.ones(self.num_classes) * (1 - smooth_max) / (self.num_classes - 1)
        #         smooth_values[int(cls)] = smooth_max
        #         prior += smooth_values * self.sorted_class_dict[cls]
        #
        #     total += self.sorted_class_dict[cls]
        # prior = prior.float() / total
        # if self.args.client.softmax_prior:
        #     self.prior_dist = (prior / self.args.client.prior_temp).softmax(dim=0)
        # else:
        #     self.prior_dist = prior
        #
        # ## shuffle unseen prior dist every time
        # if self.args.client.shuffle_unseen_prior:
        #     unseen_prior_values = self.prior_dist[self.args.dataset.unseen_classes]
        #     # Shuffle the unseen prior values
        #     unseen_prior_values = unseen_prior_values[torch.randperm(len(unseen_prior_values))]
        #     # Replace the values in prior_dist with shuffled unseen values
        #     for i, cls in enumerate(self.args.dataset.unseen_classes):
        #         self.prior_dist[cls] = unseen_prior_values[i]

        if self.gt_prior is None:
            prior = torch.zeros(self.num_classes)
            epsilon = torch.ones(self.num_classes) * 1e-7
            prior += epsilon
            total = 0
            for cls in self.sorted_class_dict:
                if not self.args.client.label_smoothing:
                    prior[int(cls)] += self.sorted_class_dict[cls]
                else:
                    smooth_max = self.args.client.smooth_max
                    smooth_values = torch.ones(self.num_classes) * (1 - smooth_max) / (self.num_classes - 1)
                    smooth_values[int(cls)] = smooth_max
                    prior += smooth_values * self.sorted_class_dict[cls]

                total += self.sorted_class_dict[cls]
            prior = prior.float() / total
            self.gt_prior = prior

        if global_epoch >= self.args.client.start_update and global_epoch >= self.args.trainer.warmup:
            num_clusters, est_prior_current, all_acc, old_acc, new_acc, aligned_w, aligned_preds, ind_map_pred_to_gt, req_c, cluster_prototypes = update_prior_ovc(self.args, self.prior_dist, copy.deepcopy(self.model),
                                            copy.deepcopy(self.loader), self.evaler,
                                            self.num_classes, global_epoch, self.device, K=len(list(sorted_class_dict.keys())))
            self.req_c_unlabelled = req_c
            self.ind_map_pred_to_gt = ind_map_pred_to_gt
            self.num_clusters = num_clusters
            self.prior_dist = self.args.client.update_lambda * est_prior_current + (
                    1 - self.args.client.update_lambda) * self.prior_dist.cpu()
            logger.info(f'Client {self.client_index} - Epoch {global_epoch} - All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}')

            aligned_conf_plot = self.evaler.plot_confusion_matrix(aligned_w)
            
            wandb_dict = {
                f"{self.args.dataset.name}/client{self.client_index}/cluster_acc_all": all_acc,
                f"{self.args.dataset.name}/client{self.client_index}/cluster_acc_old": old_acc,
                f"{self.args.dataset.name}/client{self.client_index}/cluster_acc_new": new_acc,
                f"{self.args.dataset.name}/client{self.client_index}/cluster_confusion": aligned_conf_plot,
            }
            self.wandb_log(wandb_dict, step=global_epoch)

            # Update local clustering results
            self.aligned_preds = aligned_preds

        if self.args.client.use_gt_prior:
            print(f"Using GT prior: {self.gt_prior}")
            self.prior_dist = self.gt_prior
        self.prior_dist = self.prior_dist.to(self.device)

        if self.args.client.over_clustering:
            self.cluster_prototypes = nn.utils.parametrizations.weight_norm(nn.Linear(cluster_prototypes.shape[1], cluster_prototypes.shape[0], bias=False)).to(self.device)
            self.cluster_prototypes.parametrizations.weight.original0.data.fill_(1)
            self.cluster_prototypes.parametrizations.weight.original0.requires_grad = False
            self.cluster_prototypes.parametrizations.weight.original1.data.copy_(cluster_prototypes)

        
        params_groups = self.get_params_groups(self.model)
        if self.args.client.over_clustering:
            params_groups.append({"params": self.cluster_prototypes.parameters()})
        self.optimizer = optim.SGD(
            params_groups, lr=local_lr, momentum=self.args.optimizer.momentum, weight_decay=self.args.optimizer.wd)


    def __repr__(self):
        print(f'{self.__class__} {self.client_index}, {"data : "+len(self.loader.dataset) if self.loader else ""}')

    def get_weights(self, epoch=None):

        weights = {
            "self_cls": self.args.client.unsup_cls_weight,
            "sup_cls": self.args.client.sup_cls_weight,
            "sup_con": self.args.client.sup_con_weight,
            "con": self.args.client.unsup_con_weight,
            "memax": self.args.client.memax_weight * (self.args.client.unsup_cls_weight),
            "centroids_cls": self.args.client.centroids_cls_weight,
        }
        
        return weights

    def local_train(self, global_epoch, **kwargs):
        self.global_epoch = global_epoch

        # if global_epoch < self.args.trainer.warmup:
        #     self.model.freeze_extractor()
        # else:
        #     self.model.unfreeze_extractor()

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

            # if (local_epoch % 2) == 0:
            #     for name, param in self.model.named_parameters():
            #         if 'original1' in name:
            #             param.requires_grad = True
            #         else:
            #             param.requires_grad = False
            #
            #     print("Classifiers Learning Phase")
            # elif (local_epoch % 2) == 1:
            #
            #     for m in self.model.base_vit.parameters():
            #         m.requires_grad = False
            #
            #     for m in self.model.proj_layer.parameters():
            #         m.requires_grad = True
            #     self.model.proj_layer.last_layer.parametrizations.weight.original0.requires_grad = False
            #     self.model.proj_layer.last_layer.parametrizations.weight.original1.requires_grad = False
            #
            #     # Only finetune layers from block 'args.grad_from_block' onwards
            #     for name, m in self.model.base_vit.named_parameters():
            #         if 'block' in name:
            #             block_num = int(name.split('.')[1])
            #             if block_num >= self.args.model.grad_from_block:
            #                 m.requires_grad = True
            #     print("Extractor Learning Phase")

            self.model.train()
            self.model.to(self.device)
            for i, (images, labels, uq_idxs, mask_lab) in enumerate(self.loader):

                # if len(images.size()) == 3:
                #     images = images.unsqueeze(0)
                if self.args.client.n_views > 1:
                   images = torch.cat(images, dim=0)
                
                images, labels = images.to(self.device), labels.to(self.device)
                mask_lab = mask_lab[:, 0]
                mask_lab = mask_lab.to(self.device).bool()

                with autocast(enabled=self.args.use_amp, dtype=torch.bfloat16 if self.bfloat16_support else torch.float16):
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

            if self.args.trainer.local_eval and (local_epoch+ 1) % self.args.trainer.local_eval_freq == 0:
            # if self.args.trainer.local_eval:
                # self.evaluate(epoch=local_epoch, local_datasets=None)
                ind_map = self.evaluate_local_trainset(epoch=local_epoch, global_epoch=global_epoch, local_datasets=None)

            # self.scheduler.step()


        # Local clustering and get local centroids
        if self.args.client.get_local_centroid and global_epoch >= self.args.client.start_update:
            cluster_means, cluster_targets = self.get_centroids(copy.deepcopy(self.model), copy.deepcopy(self.loader),  self.evaler, self.num_clusters, self.device)
        else:
            cluster_means = None
            cluster_targets = None
            # centroids_targets = None
            # local_labelled_centroids_dict = None
        
        logger.info(f"[C{self.client_index}] End. Time: {end-start:.2f}s, Loss: {loss_meter.avg:.3f}")
        self.model.to('cpu')

        loss_dict = {
            f'loss/{self.args.dataset.name}/cls': loss_meter.avg,
            f'loss/{self.args.dataset.name}/memax': memax_loss_meter.avg,
        }
        state_dict = self.model.state_dict()

        # local_labelled_centroids, local_centroids = self.get_centroids()

        # centroids = {
        #     'local_labelled_centroids': local_labelled_centroids,
        #     'local_centroids': local_centroids,
        # }

        results = {
            # 'local_centroids': local_centroids_dict,
            # 'local_labelled_centroids': local_labelled_centroids_dict,
            'cluster_means': cluster_means,
            'cluster_targets': cluster_targets,
        }

        # Flush model memories
        self._flush_memory()


        return state_dict, loss_dict, results

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
        # # Get local centroids for labelled
        # local_labelled_centroids_dict = defaultdict(list)
        # for feat, target in zip(feats_labelled, targets_labelled):
        #     local_labelled_centroids_dict[target].append(feat)
        
        # for key in local_labelled_centroids_dict:
        #     local_labelled_centroids_dict[key] = torch.stack(local_labelled_centroids_dict[key]).mean(dim=0)

        # #local_centroids_dict = defaultdict(list)
        # local_centroids_dict = {}
        # if self.args.client.local_centroid_type == 'finch':
        #     all_feats_np = all_feats.numpy()
        #     c, num_clust, req_c = FINCH(all_feats_np, req_clust=K, distance='cosine', verbose=False)
        #     req_c = req_c.astype(int)


        #     D = max(req_c.max(), all_targets.max()) + 1
        #     w = np.zeros((D, D), dtype=int)
        #     for i in range(len(req_c)):
        #         w[req_c[i], all_targets[i]] += 1

        #     ind = linear_assignment(w.max() - w)
        #     ind = np.vstack(ind).T
        #     ind_map_pred_to_gt = {i: j for i, j in ind}
            
        #     # Calculate centroids for each cluster
        #     for cluster_ind in range(K):
        #         cluster_feats = all_feats[req_c == cluster_ind]
        #         if len(cluster_feats) > 0:
        #             local_centroids_dict[cluster_ind] = (cluster_feats.mean(dim=0), ind_map_pred_to_gt[cluster_ind])
        
        # elif self.args.client.local_centroid_type == 'preds':
        #     # updates feats_labelled
        #     for feat, target in zip(feats_labelled, targets_labelled):
        #         local_centroids_dict[target].append(feat)
        #     # update for unlabelled
        #     for feat, logit in zip(feats_unlabelled, logits_unlabelled):
        #         local_centroids_dict[int(logit.argmax().item())].append(feat)
        # elif self.args.client.local_centroid_type == 'classifier_weights':
        #     print(f'classifier ind_map: {ind_map}')
        #     classifier_weights = model.proj_layer.last_layer.parametrizations.weight.original1.data.clone().cpu()
        #     # Compare classifier weights with self.global_model's classifier weights
        #     global_classifier_weights = self.global_model.proj_layer.last_layer.parametrizations.weight.original1.data.clone().cpu()

        #     # local_centroids = []
        #     # targets = []
        #     for i, weight in enumerate(classifier_weights):
        #         if not torch.equal(weight, global_classifier_weights[i]):
        #             local_centroids_dict[i] = (weight, ind_map[i])
        #             # local_centroids.append(weight)
        #             # targets.append(ind_map[i])
        # else:
        #     assert False

        # # Sort local_centroids_dict by the keys
        # local_centroids_dict = dict(sorted(local_centroids_dict.items()))
        # local_centroids = []
        # for i in local_centroids_dict:
        #     local_centroids.append(torch.stack(local_centroids_dict[i]).mean(dim=0))
        # local_centroids = torch.stack(local_centroids)


        # local_centroids = torch.stack(local_centroids)
        # centroids_targets = torch.LongTensor(targets)
        
        # return local_centroids_dict, local_labelled_centroids_dict

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
    

    def _algorithm(self, images, labels, uq_idxs, mask_lab, global_epoch=0) -> Dict:

        losses = defaultdict(float)

        student_feat, student_proj, student_out = self.model(images, return_all=True)
        teacher_out = student_out.detach()

        # clustering, sup
        sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
        sup_labels = torch.cat([labels[mask_lab] for _ in range(2)], dim=0)
        cls_loss = self.ce(sup_logits, sup_labels)

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
            elif self.args.client.distance_type == 'kl':
                avg_probs = torch.clamp(avg_probs, 1e-7, 1)
                prior_dist = torch.clamp(self.prior_dist, 1e-7, 1)
                entropy_distance = torch.sum(prior_dist * torch.log(prior_dist / avg_probs))

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

        # Class association loss using global centroids
        if self.args.client.over_clustering:

            centroids_logits = self.cluster_prototypes(student_feat)
            centroids_labels = centroids_logits.clone().detach().to(self.device)
            centroids_ce_loss = F.cross_entropy(centroids_logits / 0.1, centroids_labels)
            avg_probs_centroids = (centroids_logits / 0.1).softmax(dim=1).mean(dim=0)
            me_max_loss_centroids = - torch.sum(torch.log(avg_probs_centroids ** (-avg_probs_centroids))) + math.log(float(len(avg_probs_centroids)))
            centroids_ce_loss += self.args.client.memax_weight * me_max_loss_centroids
        else:
            centroids_ce_loss = 0

        # Losses
        losses["self_cls"] =  cluster_loss
        losses["sup_cls"] = cls_loss
        losses["sup_con"] =sup_con_loss
        losses["con"] = contrastive_loss
        losses["memax"] = me_max_loss
        losses["centroids_cls"] = centroids_ce_loss

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

