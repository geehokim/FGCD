from pathlib import Path
from typing import Callable, Dict, Tuple, Union, List, Type, Any
from argparse import Namespace
from collections import defaultdict

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import tqdm
import wandb
import gc
import math

import pickle, os
import numpy as np


import logging
logger = logging.getLogger(__name__)
# coloredlogs.install(level='INFO', fmt='%(asctime)s %(name)s[%(process)d] %(message)s', datefmt='%m-%d %H:%M:%S')



import time, io, copy

from trainers.build import TRAINER_REGISTRY

from servers import Server
from clients import Client

from datasets.data_utils import DatasetSplit, DatasetSplitMultiViews
from datasets.data_utils import get_local_datasets

from utils.logging_utils import AverageMeter

from torch.utils.data import DataLoader

from utils import terminate_processes, initalize_random_seed, save_checkpoint
from omegaconf import DictConfig,OmegaConf

from loss_landscape import net_plotter, plot_2D, plot_surface
from loss_landscape.projection import setup_PCA_directions_fed, project_fed
import loss_landscape.projection as proj
from loss_landscape.plot_surface import crunch_fed

from netcal.metrics import ECE
import matplotlib.pyplot as plt
from trainers.nomp_base_trainer import BaseTrainer
import torch.nn.functional as F
from utils import linear_assignment
from utils.infomap_cluster_utils2 import cluster_by_infomap


@TRAINER_REGISTRY.register()
class PriorUpdateTrainer(BaseTrainer):

    def __init__(self,
                 model: nn.Module,
                 client_type: Type,
                 server: Server,
                 evaler_type: Type,
                 datasets: Dict,
                 device: torch.device,
                 args: DictConfig,
                 multiprocessing: Dict = None,
                 **kwargs) -> None:

        super(PriorUpdateTrainer, self).__init__(model, client_type, server, evaler_type, datasets, device, args,
                 multiprocessing, **kwargs)

    def local_update(self, device, task_queue):

        task = task_queue
        client = self.clients[task['client_idx']]
        local_dataset = self.local_dataset_split_ids[task['client_idx']]
        setup_inputs = {
            # 'model': copy.deepcopy(task['model']) if self.args.multiprocessing else copy.deepcopy(self.model),
            'model': task['model'],
            'device': device,
            'local_dataset': copy.deepcopy(local_dataset),
            'local_lr': task['local_lr'],
            'global_epoch': task['global_epoch'],
            'trainer': self,
            'optimizer_state_dict': task['optimizer_state_dict'],
        }
        client.setup(**setup_inputs)
        local_model_state_dict, local_loss_dict, results = client.local_train(global_epoch=task['global_epoch'])
        
        # local_centroids = results['local_centroids']
        # local_labelled_centroids = results['local_labelled_centroids']
        cluster_means = results['cluster_means']
        cluster_targets = results['cluster_targets']
        cluster_mask = results.get('novel_class_mask', None)
        local_labelled_class_set = None
        optimizer_state_dict = results['optimizer_state_dict']

        ind_map_pred_to_gt = None
        return local_model_state_dict, local_loss_dict, None, None, cluster_means, cluster_targets, local_labelled_class_set, ind_map_pred_to_gt, cluster_mask, optimizer_state_dict


    def train(self) -> Dict:

        M = max(int(self.participation_rate * self.num_clients), 1)
        optimizer_state_dict = None

        if self.args.eval_first:
            self.evaluate(epoch=0, local_datasets=None)

        for epoch in range(self.start_round, self.global_rounds):
            self.lr_update(epoch=epoch)

            global_state_dict = copy.deepcopy(self.model.state_dict())
            prev_model_weight = copy.deepcopy(self.model.state_dict())
            
            # Select clients
            if self.args.trainer.get('client_selection'):
                selection = self.args.trainer.client_selection
                if selection.mode == 'fix': # Always select the first M clients (fixed)
                    selected_client_ids = range(M)
                elif selection.mode == 'sequential': # Sequentially select the clients (R rounds per each client)
                    round = selection.rounds_per_client
                    selected_client_ids = [(epoch // round) % self.num_clients]
            else:
                if self.participation_rate < 1.:
                    selected_client_ids = np.random.choice(range(self.num_clients), M, replace=False)
                else:
                    selected_client_ids = range(len(self.clients))
            logger.info(f"Global epoch {epoch}, Selected client : {selected_client_ids}")

            current_lr = self.lr
            local_weights = defaultdict(list)
            local_loss_dicts = defaultdict(list)
            local_deltas = defaultdict(list)
            local_optimizer_state_dicts = defaultdict(list)

            local_models = []
            local_est_priors = []
            local_gt_priors = []
            local_centroids_list = []
            local_labelled_centroids_list = []
            local_labelled_class_set_list = []
            local_ind_map_pred_to_gt_list = []
            local_novel_cluster_means_list = []
            local_novel_cluster_targets_list = []
            local_novel_class_mask_list = []

            # Client-side
            start = time.time()
            for i, client_idx in enumerate(selected_client_ids):
                task_queue_input = {
                    # 'model': self.model if self.args.multiprocessing else None,
                    'model': copy.deepcopy(self.model),
                    'client_idx': client_idx,
                    #'lr': current_lr,
                    'local_lr': current_lr,
                    'global_epoch': epoch,
                    'optimizer_state_dict': optimizer_state_dict,
                }

                local_state_dict, local_loss_dict, local_est_prior, local_gt_prior, cluster_means, cluster_targets, local_labelled_class_set, ind_map_pred_to_gt, cluster_mask, local_optimizer_state_dict = self.local_update(self.device, task_queue_input)
                for loss_key in local_loss_dict:
                    local_loss_dicts[loss_key].append(local_loss_dict[loss_key])

                # local_state_dict = local_model.state_dict()
                local_est_priors.append(local_est_prior)
                local_gt_priors.append(local_gt_prior)
                local_models.append(local_state_dict)
                local_novel_cluster_means_list.append(cluster_means)
                local_novel_cluster_targets_list.append(cluster_targets)
                local_labelled_class_set_list.append(local_labelled_class_set)
                local_ind_map_pred_to_gt_list.append(ind_map_pred_to_gt)
                local_novel_class_mask_list.append(cluster_mask)

                for param_key in local_state_dict:
                    local_weights[param_key].append(local_state_dict[param_key])
                    #local_deltas[param_key].append(local_state_dict[param_key] - global_state_dict[param_key])

                for param_key in local_optimizer_state_dict:
                    local_optimizer_state_dicts[param_key].append(local_optimizer_state_dict[param_key])

            logger.info(f"Global epoch {epoch}, Train End. Total Time: {time.time() - start:.2f}s")

            # Server-side
            updated_global_state_dict, gathered_local_prototypes, _ = self.server.aggregate(local_weights, local_deltas, local_optimizer_state_dicts,
                                                              selected_client_ids, copy.deepcopy(self.model), local_novel_cluster_means_list, local_novel_cluster_targets_list, local_labelled_class_set_list, current_lr,
                                                              epoch, local_novel_class_mask_list)
            
            
            
            # if self.args.trainer.ema:
            #     # Perform EMA (Exponential Moving Average) update
            #     for param_name, param_tensor in self.model.named_parameters():
            #         if param_name in updated_global_state_dict:
            #             updated_global_state_dict[param_name] = self.args.trainer.ema_lambda * param_tensor + (1 - self.args.trainer.ema_lambda) * updated_global_state_dict[param_name]
            print(self.model.load_state_dict(updated_global_state_dict, strict=False))


            if self.args.client.global_clustering:
                cluster_labels, act_protos_global, global_prototypes = self.global_clustering(gathered_local_prototypes)
                self.model.proj_layer.act_protos_global.data = torch.tensor(act_protos_global).long()
                self.model.proj_layer.global_prototypes.data[:act_protos_global] = global_prototypes
                

            # optimizer_state_dict = updated_optimizer_state_dict
            

            # if global_clustering_result_w is not None:
            #     conf_results = self.evaler.plot_confusion_matrix(global_clustering_result_w.astype(int))
            #     self.wandb_log({
            #         f"{self.args.dataset.name}/global/global_centroids_clustering_conf": conf_results['confusion_matrix']
            #     }, step=epoch)

            # if  self.args.server.type == 'ServerNovelClustering':
            #     if epoch >= self.args.client.start_update:
            #         self.model.update_novel_classifier_weights(updated_global_state_dict['global_centroids.weight'])
            #     # if epoch >= self.args.client.start_update:
            #     #     self.model.update_novel_classifier_weights(updated_global_state_dict['global_centroids.weight'])
            #     # else:
            #     #     self.model.update_global_centroids_as_classifier_weights()
            #     pass
            #     # it is equal to fedavg
            # elif self.args.server.type == 'ServerNovelClusteringClsWeights':
            #     self.model.update_novel_classifier_weights(updated_global_state_dict['global_centroids.weight'])
            # elif self.args.server.type == 'ServerClusteringClsWeights':
            #     self.model.update_classifier_weights(updated_global_state_dict['global_centroids_all.weight'])
            # elif self.args.server.type == 'ServerClusteringClassifierWeights':
            #     self.model.update_classifier_weights(updated_global_state_dict['global_centroids_all.weight'])
                    

            # if  epoch >= self.args.client.start_update:
            #     if 'ServerNovelClustering' in self.args.server.type:
            #         # global_centroids = updated_global_state_dict['global_centroids.weight']
            #         # prev_classifier_weight = prev_model_weight['proj_layer.last_layer.parametrizations.weight.original1']

            #         # ## Align global centroids using linear assignment
            #         # # compute pairwise cosine similarity
            #         # cosine_similarity = torch.matmul(F.normalize(global_centroids, dim=1), F.normalize(prev_classifier_weight, dim=1).T)
            #         # cosine_similarity = cosine_similarity.cpu().numpy()
            #         # ind = linear_assignment(cosine_similarity.max() - cosine_similarity)
            #         # ind = np.vstack(ind).T
            #         # ind_map_proto_pred_to_gt = {i: j for i, j in ind}
            #         # print(f'ind_map_proto_pred_to_gt: {ind_map_proto_pred_to_gt}')
            #         # aligned_global_centroids = global_centroids[list(ind_map_proto_pred_to_gt.values())]
            #         # updated_global_state_dict['global_centroids.weight'] = torch.tensor(aligned_global_centroids)

            #         if self.args.server.use_global_centroids_as_classifier:
            #             ## Update classifier weight using global centroids
            #             self.model.update_novel_classifier_weights(updated_global_state_dict['global_centroids.weight'])
            #         else:
            #             print('Not updating classifier weight using global centroids')

            local_datasets = [self.local_dataset_split_ids[client_id] for client_id in selected_client_ids]

            # Logging
            wandb_dict = {loss_key: np.mean(local_loss_dicts[loss_key]) for loss_key in local_loss_dicts}
            wandb_dict['lr'] = self.lr
            
            # try:
            model_device = next(self.model.parameters()).device
            
            if self.args.eval.freq > 0 and (epoch+1) % self.args.eval.freq == 0:
                self.evaluate(epoch=epoch, local_models=local_models, local_datasets=local_datasets)


            if (self.args.save_freq > 0 and (epoch + 1) % self.args.save_freq == 0) or (epoch + 1 == self.args.trainer.global_rounds):
                self.save_model(local_models, prev_global_model_state=prev_model_weight, epoch=epoch)

            self.wandb_log(wandb_dict, step=epoch)
            gc.collect()

        return


    def global_clustering(self, gathered_local_prototypes):
        
        cluster_feature = torch.cat(gathered_local_prototypes, dim=0)
        cluster_feature = F.normalize(cluster_feature, dim=-1).cpu().numpy()
        
        # infomap clustering, k1 = num_of neighbours, k2 = min num_of clusters, eps = min similarity
        # cluster_feature = cluster_feature.to(torch.bfloat16)
        idx2label, idx_single, dists = cluster_by_infomap(cluster_feature, k=self.args.client.global_top_k, tao_f=self.args.client.global_tao_f, dataset_name=self.args.dataset.name)

        # Remap cluster labels to be consecutive integers starting from 0
        unique_labels = sorted(set(idx2label.values()))
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        
        # Update idx2label with the new consecutive labels
        for idx in idx2label:
            idx2label[idx] = label_map[idx2label[idx]]
            
        logger.info(f"Remapped cluster labels to consecutive integers: {len(unique_labels)} unique clusters")

        # get centroid using the idx2label
        preds_unsingle          = []  
        idxs_unsingle           = [] 
        for i in idx2label.keys():
            preds_unsingle.append(idx2label[i])
            idxs_unsingle.append(i)
        preds_unsingle      = np.array(preds_unsingle)
        preds_unique_label  = np.unique(preds_unsingle)
        feats_unsingle      = cluster_feature[idxs_unsingle]
        centroid            = []
        for cls_id in preds_unique_label:
            data_idx    = np.where(preds_unsingle==cls_id)[0]
            data        = feats_unsingle[data_idx] # [n, d]
            prototype   = data.mean(axis=0, keepdims=True)
            centroid.append(prototype)

        # use the obtained centroid to predict the lable to single data
        centroid = np.concatenate(centroid) # [-1, 768]
        
        estimate_k = centroid.shape[0]
        print(f"Estimated number of global clusters: {estimate_k}")
        act_protos = estimate_k
        global_prototypes = torch.Tensor(centroid).to(self.device)

        # note that pseudo_labels contains -1 for unassigned examples
        # pseudo_labels = preds_unsingle.astype(np.intp)
        # 방법 1: -1로 초기화하여 클러스터링되지 않은 샘플 처리
        sorted_cluster_labels = [-1] * len(cluster_feature)
        for idx in idx2label:
            sorted_cluster_labels[idx] = idx2label[idx]
        sorted_cluster_labels = np.array(sorted_cluster_labels)

        return sorted_cluster_labels, act_protos, global_prototypes
    
    def evaluate(self, epoch: int, local_models=None, local_datasets: List[torch.utils.data.Dataset] = None) -> Dict:

        results = self.evaler.eval(model=self.model, epoch=epoch)
        all_acc = results["all_acc"]
        new_acc = results["new_acc"]
        old_acc = results["old_acc"]
        cluster_acc = results["cluster_acc"]
        cluster_old_acc = results["cluster_old_acc"]
        cluster_new_acc = results["cluster_new_acc"]
        confusion_matrix = results['conf_matrix']
        all_feats = results['feats']
        # all_p_feats = results['feats_proj']
        targets = results['targets']
        infomap_acc = results['infomap_acc']
        infomap_old_acc = results['infomap_old_acc']
        infomap_new_acc = results['infomap_new_acc']

        wandb_dict = {
            f"{self.args.dataset.name}/global/all_acc": all_acc,
            f"{self.args.dataset.name}/global/old_acc": old_acc,
            f"{self.args.dataset.name}/global/new_acc": new_acc,
            f"{self.args.dataset.name}/global/cluster_acc": cluster_acc,
            f"{self.args.dataset.name}/global/cluster_old_acc": cluster_old_acc,
            f"{self.args.dataset.name}/global/cluster_new_acc": cluster_new_acc,
            f"{self.args.dataset.name}/global/infomap_acc": infomap_acc,
            f"{self.args.dataset.name}/global/infomap_old_acc": infomap_old_acc,
            f"{self.args.dataset.name}/global/infomap_new_acc": infomap_new_acc,
            }
        
        logger.warning(f'[Epoch {epoch}] Train Unlabelled ALL Acc: {all_acc:.2f}%, OLD Acc: {old_acc:.2f}%, NEW Acc: {new_acc:.2f}%')
        logger.warning(f'[Epoch {epoch}] Train Unlabelled Cluster Acc: {cluster_acc:.2f}%, OLD Acc: {cluster_old_acc:.2f}%, NEW Acc: {cluster_new_acc:.2f}%')
        logger.warning(f'[Epoch {epoch}] Train Unlabelled Infomap Acc: {infomap_acc:.2f}%, OLD Acc: {infomap_old_acc:.2f}%, NEW Acc: {infomap_new_acc:.2f}%')

        plt.close()

        self.wandb_log(wandb_dict, step=epoch)

        if self.args.confusion.freq > 0 and epoch % self.args.confusion.freq == 0:
            conf_results = self.evaler.plot_confusion_matrix(confusion_matrix)
            self.wandb_log(conf_results, step=epoch)

        if self.args.server_umap.freq > 0 and epoch % self.args.server_umap.freq == 0:
            if self.args.server_umap.plot_locals:
                umap_results = self.evaler.visualize_umaps(copy.deepcopy(self.model), local_models, local_datasets, local_ind_map_pred_to_gt_list, epoch=epoch)
            else:
                if self.args.eval.cluster_eval == 'feats':
                    umap_results = self.evaler.visualize_server_umap(copy.deepcopy(self.model), all_feats, targets, epoch)
                elif self.args.eval.cluster_eval == 'feats_proj':
                    umap_results = self.evaler.visualize_server_umap(copy.deepcopy(self.model), all_p_feats, targets, epoch)
            self.wandb_log(umap_results, step=epoch)
        return {
            "all_acc": all_acc,
            "new_acc": new_acc,
            "old_acc": old_acc
        }


    def evaluate_local(self, epoch: int, local_models: List[nn.Module], ) -> Dict:
        local_model = copy.deepcopy(self.model)
        local_model.load_state_dict(local_models[0])
        results = self.evaler.eval(model=local_model, epoch=epoch)
        acc = results["acc"]
        ece = results["ece"]
        entropy = results["entropy"]
        ece_diagram = results["ece_diagram"]
        logger.warning(f'  [Epoch {epoch}] Local Test Accuracy: {acc:.2f}%')

        wandb_dict = {
            f"acc/{self.args.dataset.name}/local": acc,
            f'entropy/{self.args.dataset.name}/local': entropy,
            f'ece/{self.args.dataset.name}/local': ece,
            }

        if epoch % 10 == 0:
            wandb_dict.update({f'ece_diagram/{self.args.dataset.name}/local': wandb.Image(ece_diagram)})

        self.wandb_log(wandb_dict, step=epoch)

        plt.close()

        return {
            "acc": acc,
            
        }

@TRAINER_REGISTRY.register()
class PriorUpdateTrainerClient0(PriorUpdateTrainer):

    def __init__(self,
                 model: nn.Module,
                 client_type: Type,
                 server: Server,
                 evaler_type: Type,
                 datasets: Dict,
                 device: torch.device,
                 args: DictConfig,
                 multiprocessing: Dict = None,
                 **kwargs) -> None:

        super(PriorUpdateTrainerClient0, self).__init__(model, client_type, server, evaler_type, datasets, device, args,
                 multiprocessing, **kwargs)

    
    
    
    
    
    
    def train(self) -> Dict:

        M = max(int(self.participation_rate * self.num_clients), 1)

        if self.args.eval_first:
            self.evaluate(epoch=0, local_datasets=None, local_est_priors=None, local_gt_priors=None,
                              local_ind_map_pred_to_gt_list=None)

        for epoch in range(self.start_round, self.global_rounds):
            self.lr_update(epoch=epoch)

            global_state_dict = copy.deepcopy(self.model.state_dict())
            prev_model_weight = copy.deepcopy(self.model.state_dict())
            
            # Select clients
            selected_client_ids = [3]
            logger.info(f"Global epoch {epoch}, Selected client : {selected_client_ids}")

            current_lr = self.lr
            local_weights = defaultdict(list)
            local_loss_dicts = defaultdict(list)
            local_deltas = defaultdict(list)

            local_models = []
            local_est_priors = []
            local_gt_priors = []
            local_centroids_list = []
            local_labelled_centroids_list = []
            local_labelled_class_set_list = []
            local_ind_map_pred_to_gt_list = []

            # Only fine-tuning fc classifier
            if self.args.get('freeze_backbone'):
                if epoch > self.args.freeze_backbone.epoch:
                    self.model.freeze_backbone()

            # FedACG lookahead momentum
            if self.args.server.get('FedACG'):
                assert(self.args.server.momentum > 0)
                self.model= copy.deepcopy(self.server.FedACG_lookahead(copy.deepcopy(self.model)))
                global_state_dict = copy.deepcopy(self.model.state_dict())

            # Client-side
            start = time.time()
            for i, client_idx in enumerate(selected_client_ids):
                task_queue_input = {
                    # 'model': self.model if self.args.multiprocessing else None,
                    'model': copy.deepcopy(self.model),
                    'client_idx': client_idx,
                    #'lr': current_lr,
                    'local_lr': current_lr,
                    'global_epoch': epoch,
                }

                local_state_dict, local_loss_dict, local_est_prior, local_gt_prior, local_centroids, local_labelled_centroids, local_labelled_class_set, ind_map_pred_to_gt = self.local_update(self.device, task_queue_input)
                for loss_key in local_loss_dict:
                    local_loss_dicts[loss_key].append(local_loss_dict[loss_key])

                # local_state_dict = local_model.state_dict()
                local_est_priors.append(local_est_prior)
                local_gt_priors.append(local_gt_prior)
                local_models.append(local_state_dict)
                local_centroids_list.append(local_centroids)
                local_labelled_centroids_list.append(local_labelled_centroids)
                local_labelled_class_set_list.append(local_labelled_class_set)
                local_ind_map_pred_to_gt_list.append(ind_map_pred_to_gt)

                for param_key in local_state_dict:
                    local_weights[param_key].append(local_state_dict[param_key])
                    #local_deltas[param_key].append(local_state_dict[param_key] - global_state_dict[param_key])

            logger.info(f"Global epoch {epoch}, Train End. Total Time: {time.time() - start:.2f}s")

            # Server-side
            updated_global_state_dict, global_clustering_result_w = self.server.aggregate(local_weights, local_deltas,
                                                              selected_client_ids, copy.deepcopy(self.model), local_centroids_list, local_labelled_centroids_list, local_labelled_class_set_list, current_lr,
                                                              epoch)
            print(self.model.load_state_dict(updated_global_state_dict, strict=False))
            

            if global_clustering_result_w is not None:
                conf_results = self.evaler.plot_confusion_matrix(global_clustering_result_w.astype(int))
                self.wandb_log({
                    f"{self.args.dataset.name}/global/global_centroids_clustering_conf": conf_results['confusion_matrix']
                }, step=epoch)

            if 'NovelClustering' in self.args.server.type:
                if epoch >= self.args.client.start_update:
                    self.model.update_novel_classifier_weights(updated_global_state_dict['global_centroids.weight'])
                else:
                    self.model.update_global_centroids_as_classifier_weights()

            local_datasets = [self.local_dataset_split_ids[client_id] for client_id in selected_client_ids]

            # Logging
            wandb_dict = {loss_key: np.mean(local_loss_dicts[loss_key]) for loss_key in local_loss_dicts}
            wandb_dict['lr'] = self.lr
            
            # try:
            model_device = next(self.model.parameters()).device
            if self.args.eval.freq > 0 and (epoch + 1) % self.args.eval.freq == 0:
                self.evaluate(epoch=epoch, local_models=local_models, local_datasets=local_datasets,
                              local_est_priors=local_est_priors, local_gt_priors=local_gt_priors,
                              local_ind_map_pred_to_gt_list=local_ind_map_pred_to_gt_list)


            if (self.args.save_freq > 0 and (epoch + 1) % self.args.save_freq == 0) or (epoch + 1 == self.args.trainer.global_rounds):
                self.save_model(local_models, prev_global_model_state=prev_model_weight, epoch=epoch)

            self.wandb_log(wandb_dict, step=epoch)
            gc.collect()

        return
    
