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
class GCDClient_New():

    def __init__(self, args, client_index, loader=None, evaler=None):
        self.args = args
        self.client_index = client_index
        self.criterion = nn.CrossEntropyLoss()
        self.sup_con_crit = SupConLoss(temperature=self.args.client.sup_temperature,
                                       base_temperature=self.args.client.sup_temperature)
        
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
                params.append({"params": param, "lr": local_lr})
            else:
                params.append({"params": param})
        return params
        # regularized = []
        # not_regularized = []
        # for name, param in model.named_parameters():
        #     if not param.requires_grad:
        #         continue
        #     # we do not regularize biases nor Norm parameters
        #     if name.endswith(".bias") or len(param.shape) == 1:
        #         not_regularized.append(param)
        #     else:
        #         regularized.append(param)
        # return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

    

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

    def get_weights(self, epoch=None):

        weights = {
            "sup_con": self.args.client.sup_con_weight,
            "con": self.args.client.unsup_con_weight,
        }
        
        return weights

    def local_train(self, global_epoch, **kwargs):
        self.global_epoch = global_epoch
        self.model.train()
        self.model.to(self.device)
        scaler = GradScaler()
        start = time.time()
        loss_meter = AverageMeter('Loss', ':.2f')
        time_meter = AverageMeter('BatchTime', ':3.1f')

        self.weights = self.get_weights(epoch=global_epoch)
            
        if global_epoch % 50 == 0:
            print(self.weights)

        for local_epoch in range(self.args.trainer.local_epochs):
            end = time.time()


            self.model.train()
            self.model.to(self.device)
            for i, (images, labels, uq_idxs, mask_lab) in enumerate(self.loader):

                if self.args.client.n_views > 1:
                   images = torch.cat(images, dim=0)
                
                images, labels = images.to(self.device), labels.to(self.device)
                mask_lab = mask_lab[:, 0]
                mask_lab = mask_lab.to(self.device).bool()

                with autocast(enabled=self.args.use_amp, dtype=torch.bfloat16 if self.bfloat16_support else torch.float16):
                    losses = self._algorithm(images, labels, uq_idxs, mask_lab, global_epoch)
                    loss = sum([self.weights[loss_key]*losses[loss_key] for loss_key in losses])
                    
                self.optimizer.zero_grad()

                try:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    scaler.step(self.optimizer)
                    scaler.update()
                except Exception as e:
                    print(e)

                loss_meter.update(loss.item(), images.size(0))
                time_meter.update(time.time() - end)
                end = time.time()

            if self.args.trainer.local_eval and (local_epoch+ 1) % self.args.trainer.local_eval_freq == 0:
                ind_map = self.evaluate_local_trainset(epoch=local_epoch, global_epoch=global_epoch, local_datasets=None)


        cluster_means = None
        cluster_targets = None
        
        logger.info(f"[C{self.client_index}] End. Time: {end-start:.2f}s, Loss: {loss_meter.avg:.3f}")
        self.model.to('cpu')

        loss_dict = {
            f'loss/{self.args.dataset.name}/cls': loss_meter.avg,
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
    

    def _algorithm(self, images, labels, uq_idxs, mask_lab, global_epoch=0) -> Dict:

        losses = defaultdict(float)

        student_feat, student_proj, _ = self.model(images, return_all=True)

        student_proj = F.normalize(student_proj, dim=-1)

        # represent learning, unsup
        contrastive_logits, contrastive_labels = self.info_nce_logits(features=student_proj)
        contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

        # representation learning, sup
        student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
        student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
        sup_con_labels = labels[mask_lab]
        sup_con_loss = self.sup_con_crit(student_proj, labels=sup_con_labels)

        
        # Losses
        losses["sup_con"] =sup_con_loss
        losses["con"] = contrastive_loss

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

