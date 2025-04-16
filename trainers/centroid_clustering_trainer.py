from pathlib import Path
from typing import Callable, Dict, Tuple, Union, List, Type, Any
from argparse import Namespace
from collections import defaultdict

import faiss
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import tqdm
import wandb
import gc

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
import torch.nn.functional as F


@TRAINER_REGISTRY.register()
class CCTrainer():

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

        self.args = args
        self.device = device
        self.model = model

        self.checkpoint_path = Path(self.args.checkpoint_path)
        mode = self.args.split.mode 
        if self.args.split.mode == 'dirichlet':
            mode += str(self.args.split.alpha)
        elif self.args.split.mode == 'skew':
            mode += str(self.args.split.class_per_client)
        self.exp_path = self.checkpoint_path / self.args.dataset.name / mode / self.args.exp_name
        logger.info(f"Exp path : {self.exp_path}")

        ### training config
        trainer_args = self.args.trainer
        self.num_clients = trainer_args.num_clients
        self.participation_rate = trainer_args.participation_rate
        self.global_rounds = trainer_args.global_rounds
        # self.local_epochs = trainer_args.local_epochs
        self.lr = trainer_args.local_lr
        self.local_lr_decay = trainer_args.local_lr_decay

        # self.datasets = datasets
        self.datasets = self.get_datasets(datasets)
        self.local_dataset_split_ids = get_local_datasets(self.args, self.datasets['train'], mode=self.args.split.mode)

        test_loader = DataLoader(self.datasets["test"],
                                 batch_size=args.evaler.batch_size if args.evaler.batch_size > 0 else args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)
        eval_device = self.device if not self.args.multiprocessing else torch.device(f'cuda:{self.args.main_gpu}')
        eval_params = {
            "test_loader": test_loader,
            # "gallery_loader": test_loader,
            # "query_loader": test_loader,
            "device": eval_device,
            "args": args,
        }
        self.eval_params = eval_params
        self.eval_device = eval_device

        # self.evaler = evaler_type(test_loader=test_loader, device=eval_device, args=args)
        self.evaler = evaler_type(**eval_params)

        self.clients: List[Client] = [client_type(self.args, client_index=c, evaler=self.evaler) for c in range(self.args.trainer.num_clients)]
        self.server = server
        if self.args.server.momentum > 0:
            self.server.set_momentum(self.model)

        logger.info(f"Trainer: {self.__class__}, client: {client_type}, server: {server.__class__}, evaler: {evaler_type}")

        self.start_round = 0
        if self.args.get('load_model_path'):
            self.load_model()

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
        }
        client.setup(**setup_inputs)
        local_model_state_dict, local_loss_dict, centroids = client.local_train(global_epoch=task['global_epoch'])
        return local_model_state_dict, local_loss_dict, centroids


    def train(self) -> Dict:

        M = max(int(self.participation_rate * self.num_clients), 1)

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
            local_centroids = defaultdict(list)

            local_models = []

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

                local_state_dict, local_loss_dict, centroids_dict = self.local_update(self.device, task_queue_input)
                for loss_key in local_loss_dict:
                    local_loss_dicts[loss_key].append(local_loss_dict[loss_key])

                for centroid_key in centroids_dict:
                    local_centroids[centroid_key].append(centroids_dict[centroid_key])

                # local_state_dict = local_model.state_dict()
                local_models.append(local_state_dict)

                for param_key in local_state_dict:
                    local_weights[param_key].append(local_state_dict[param_key])
                    #local_deltas[param_key].append(local_state_dict[param_key] - global_state_dict[param_key])

            logger.info(f"Global epoch {epoch}, Train End. Total Time: {time.time() - start:.2f}s")

            # Server-side
            updated_global_state_dict = self.server.aggregate(local_weights, local_deltas,
                                                              selected_client_ids, copy.deepcopy(global_state_dict), current_lr,
                                                              epoch)
            print(self.model.load_state_dict(updated_global_state_dict, strict=False))


            ## Classifier aggregation
            aggregated_local_labelled_centroids = self.server.aggregate_local_labelled_centroids(local_weights, local_centroids['local_labelled_centroids'], selected_client_ids)
            all_local_centroids = self.server.get_local_centroids(local_weights, local_centroids[
                'local_centroids'], selected_client_ids)

            aligned_centroids = self.server.aggregate_centroids(local_weights, all_local_centroids, aggregated_local_labelled_centroids,
                                                                selected_client_ids)
            aligned_centroids = F.normalize(aligned_centroids, dim=1)

            # Update classifier weights
            self.model.proj_layer.last_layer.parametrizations.weight.original1.data = aligned_centroids.clone()

            local_datasets = [self.local_dataset_split_ids[client_id] for client_id in selected_client_ids]


            # Logging
            wandb_dict = {loss_key: np.mean(local_loss_dicts[loss_key]) for loss_key in local_loss_dicts}
            wandb_dict['lr'] = self.lr
            
            # try:
            model_device = next(self.model.parameters()).device
            if self.args.eval.freq > 0 and epoch % self.args.eval.freq == 0:
                self.evaluate(epoch=epoch, local_models=local_models, local_datasets=local_datasets)


            if (self.args.save_freq > 0 and (epoch + 1) % self.args.save_freq == 0) or (epoch + 1 == self.args.trainer.global_rounds):
                self.save_model(local_models, epoch=epoch)

            self.wandb_log(wandb_dict, step=epoch)
            gc.collect()

        return

    def lr_update(self, epoch: int) -> None:
        # TODO: adopt other lr policy
        # self.lr = self.lr * (self.lr_decay) ** (epoch)
        if self.args.trainer.lr_scheduler == 'cosanneal':
            self.lr = self.args.trainer.min_lr + 0.5 * (self.args.trainer.local_lr - self.args.trainer.min_lr) * (1 + torch.cos(torch.tensor(epoch * torch.pi / self.args.trainer.global_rounds)))
            self.lr = self.lr.item()
            logger.info(f'Current Lr: {self.lr}')
        else:
            self.lr = self.args.trainer.local_lr * (self.local_lr_decay) ** (epoch)
        return
    

    def save_model(self, local_models, epoch: int = -1, suffix: str = '') -> None:
        
        model_path = self.exp_path / self.args.output_model_path
        if not model_path.parent.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)

        if epoch < self.args.trainer.global_rounds - 1:
            model_path = Path(f"{model_path}.e{epoch+1}")

        if suffix:
            model_path = Path(f"{model_path}.{suffix}")

        # try:
        #     torch.save({'model_state_dict': self.model.state_dict()}, model_path)
        #     logger.warning(f'Saved model at {model_path}')
        # except:
        #     logger.error(f"Fail to save model at {model_path}")
        
        save_checkpoint(self.model, local_models, model_path, epoch, save_torch=True, use_breakpoint=False)
        return
    

    def load_model(self) -> None:
        if self.args.get('load_model_path'):
            saved_dict = torch.load(self.args.load_model_path)
            self.model.load_state_dict(saved_dict['model_state_dict'], strict=False)
            if 'epoch' in saved_dict:
                self.start_round = saved_dict["epoch"]+1
                logger.warning(f'Load model from {self.args.load_model_path}, epoch {saved_dict["epoch"]}')
            else:
                logger.warning(f'Load model from {self.args.load_model_path}')
            
            # model = torch.jit.load(str(self.args.load_model_path), map_location='cpu')
            # logger.info(f"Load model from {self.args.load_model_path}")
            # self.model = model
        return


    def wandb_log(self, log: Dict, step: int = None):
        if self.args.wandb:
            wandb.log(log, step=step)

    def validate(self, epoch: int, ) -> Dict:
        return

    def evaluate(self, epoch: int, local_models, local_datasets: List[torch.utils.data.Dataset] = None) -> Dict:

        results = self.evaler.eval(model=self.model, epoch=epoch)
        all_acc = results["all_acc"]
        new_acc = results["new_acc"]
        old_acc = results["old_acc"]
        all_p_acc = results["all_p_acc"]
        new_p_acc = results["new_p_acc"]
        old_p_acc = results["old_p_acc"]
        confusion_matrix = results['conf_matrix']
        all_feats = results['feats']
        targets = results['targets']

        wandb_dict = {
            f"all_acc/{self.args.dataset.name}": all_acc,
            f"old_acc/{self.args.dataset.name}": old_acc,
            f"new_acc/{self.args.dataset.name}": new_acc,
            f"all_p_acc/{self.args.dataset.name}": all_p_acc,
            f"old_p_acc/{self.args.dataset.name}": old_p_acc,
            f"new_p_acc/{self.args.dataset.name}": new_p_acc,
            }
        
        logger.warning(f'[Epoch {epoch}] Test ALL Acc: {all_acc:.2f}%, OLD Acc: {old_acc:.2f}%, NEW Acc: {new_acc:.2f}%')
        logger.warning(
            f'[Epoch {epoch}] Test ALL on projected feature Acc: {all_p_acc:.2f}%, OLD Acc: {old_p_acc:.2f}%, NEW Acc: {new_p_acc:.2f}%')

        plt.close()

        self.wandb_log(wandb_dict, step=epoch)

        if self.args.confusion.freq > 0 and epoch % self.args.confusion.freq == 0:
            conf_results = self.evaler.plot_confusion_matrix(confusion_matrix)
            self.wandb_log(conf_results, step=epoch)

        if self.args.server_umap.freq > 0 and epoch % self.args.server_umap.freq == 0:
            if self.args.server_umap.plot_locals:
                umap_results = self.evaler.visualize_umaps(copy.deepcopy(self.model), local_models, local_datasets, epoch=epoch)
            else:
                umap_results = self.evaler.visualize_server_umap(copy.deepcopy(self.model), all_feats, targets, epoch)
            self.wandb_log(umap_results, step=epoch)

        return {
            "all_acc": all_acc,
            "new_acc": new_acc,
            "old_acc": old_acc
        }

    def _evaluate_subset(self, epoch: int, class_accs: List, local_datasets: List[torch.utils.data.Dataset] = None) -> Dict:
        seen_acc = major_acc = minor_acc = missing_acc = minor_seen_acc = -1
        wandb_dict = {}

        if local_datasets is not None:
            seen_accs = []
            major_accs, minor_accs, missing_accs, minor_seen_accs = [], [], [], []
            for local_dataset in local_datasets:
                local_classes = [int(i) for i in local_dataset.class_dict.keys()]


                num_classes = len(local_dataset.dataset.classes)
                num_local_classes = len(local_dataset.class_dict.keys())
                major_classes = [int(key) for key in local_dataset.class_dict if local_dataset.class_dict[key] >= len(local_dataset)/num_local_classes]
                minor_seen_classes = [int(key) for key in local_dataset.class_dict if local_dataset.class_dict[key] < len(local_dataset)/num_local_classes]
                missing_classes = [i for i in range(num_classes) if str(i) not in local_dataset.class_dict]
                minor_classes = minor_seen_classes + missing_classes

                seen_accs.append(torch.mean(class_accs[local_classes]))
                major_accs.append(torch.mean(class_accs[major_classes]))
                minor_accs.append(torch.mean(class_accs[minor_classes]))
                missing_accs.append(torch.mean(class_accs[missing_classes]))
                minor_seen_accs.append(torch.mean(class_accs[minor_seen_classes]))

            seen_acc = np.mean(seen_accs)
            major_acc, minor_acc, missing_acc, minor_seen_acc = np.mean(major_accs), np.mean(minor_accs), np.mean(missing_accs), np.mean(minor_seen_accs)
            wandb_dict.update({
                f"seen_acc/{self.args.dataset.name}": seen_acc, #deprecated

                f"acc/{self.args.dataset.name}/seen": seen_acc,
                f"acc/{self.args.dataset.name}/major": major_acc,
                f"acc/{self.args.dataset.name}/minor": minor_acc,
                f"acc/{self.args.dataset.name}/missing": missing_acc,
                f"acc/{self.args.dataset.name}/minor_seen": minor_seen_acc,
                })
            
            logger.info(f'   (Seen: {seen_acc:.2f}%, Major: {major_acc:.2f}%, Minor-Seen: {minor_seen_acc:.2f}%, Minor: {minor_acc:.2f}%)')

        return wandb_dict
    
    def evaluate_retrieval(self, epoch: int, local_datasets: List[torch.utils.data.Dataset] = None) -> Dict:
        retrieval_results = self.evaler.eval_retrieval(model=self.model, local_datasets=local_datasets)
        mAP, cmc = retrieval_results["mAP"], retrieval_results["CMC"]
        logger.warning(f'[Epoch {epoch}] Retrieval mAP: {mAP:.2f}, CMC: {cmc:.2f}%')
        self.wandb_log({
            f'mAP/{self.args.dataset.name}': mAP,
            f'CMC/{self.args.dataset.name}': cmc,
        }, step=epoch)
        return retrieval_results


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
    


    #TODO: move to evaler
    def visualize_landscape(self, global_model: nn.Module, local_models: List[nn.Module], prev_model_weight: Any,  epoch: int, ):

        comm, rank, nproc = None, 0, 1

        w = net_plotter.get_weights(self.model)
        s = copy.deepcopy(self.model.state_dict())
        model_files = copy.deepcopy([local_model for local_model in local_models])

        #--------------------------------------------------------------------------
        # Create projection directions
        #--------------------------------------------------------------------------
        dir_file = setup_PCA_directions_fed(self.args, copy.deepcopy(model_files), w, s, epoch)
        #--------------------------------------------------------------------------
        # projection trajectory to given directions
        #--------------------------------------------------------------------------
        proj_file, (local_xmax,local_xmin,local_ymax,local_ymin) = project_fed(dir_file, w, s,
                                    copy.deepcopy(model_files + [prev_model_weight]), self.args.landscape.dir_type, 'cos')
        
        if self.args.landscape.adaptive_xy_range:
            self.args.landscape.xmax, self.args.landscape.xmin, self.args.landscape.ymax, self.args.landscape.ymin = (np.array([local_xmax,local_xmin,local_ymax,local_ymin])*1.25).tolist()
        
        # if epoch==0:
        #     self.args.landscape.xnum, self.args.landscape.ynum = 3,3
        #--------------------------------------------------------------------------
        # Setup the direction file and the surface file
        #--------------------------------------------------------------------------
        surf_file = plot_surface.name_surface_file_fed(self.args.landscape ,dir_file)
        plot_surface.setup_surface_file(self.args.landscape, surf_file, dir_file)
        # load directions
        d = net_plotter.load_directions(dir_file)
        # calculate the consine similarity of the two directions
        if len(d) == 2 :
            similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
            print('cosine similarity between x-axis and y-axis: %f' % similarity)


        #--------------------------------------------------------------------------
        # Start the computation
        #--------------------------------------------------------------------------
        plot_surface.crunch_fed(surf_file, copy.deepcopy(self.model), w, s, d, 'test_loss', 'test_acc', comm, rank, self.args, evaler = self.evaler, epoch = epoch)

        

        self.wandb_log({f"loss-landscape/{self.args.dataset.name}": wandb.Image(plot_2D.plot_contour_fed(surf_file, dir_file, proj_file, surf_name='test_acc',
                        vmin=0.1, vmax=100, vlevel=5, show=False, adaptive_v = self.args.landscape.adaptive_v)) }, step=epoch)
        #results =  self.visualizer.loss_landscape(global_model=global_model, local_models=local_models, epoch=epoch)
        #self.wandb_log(results~~)
        return
    

    def visualize_umap(self, global_model: nn.Module, local_models: List[nn.Module],  epoch: int, local_datasets: List[torch.utils.data.Dataset] = None):
        umap_results = self.evaler.visualize_umap(global_model=global_model, local_models=local_models, local_datasets=local_datasets, epoch=epoch)
        self.wandb_log(umap_results, step=epoch)

        return

    def evaluate_svd(self, epoch: int, local_models: List[nn.Module], global_model: nn.Module, local_datasets: List[torch.utils.data.Dataset] = None):
        try:
        # if True:
            results = self.evaler.visualize_svd(epoch, local_models, global_model, local_datasets)
            self.wandb_log(results, step=epoch)
        except Exception as e:
            logger.warning(e)



        # activation_results = self.evaler.evaluate_activation_matching(epoch, local_models, global_model, local_datasets)
        # self.wandb_log(activation_results, step=epoch)

        return
    

    def evaluate_minority_collapse(self, prev_model_weight: nn.Module, local_models: List[nn.Module], local_datasets: List[torch.utils.data.Dataset], epoch: int):
        results = self.evaler.evaluate_minority_collapse(prev_model_weight, local_models, local_datasets)
        self.wandb_log(results, step = epoch)
        
        return 
    


    def get_datasets(self, datasets):
        if 'toy' in self.args.split.mode:
            print("Modify testset, trainset according to toy set")
            #For test

            for idx, dataset_key in enumerate(['train', 'test']):
                dataset = datasets[dataset_key]
                num_valid_classes = min(len(dataset.classes), self.args.split.limit_total_classes)
                idxs = np.arange(len(dataset))
                labels = []
                for element in dataset:
                    labels.append(int(element[1]))
                idxs_labels = np.vstack((idxs, labels))
                idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
                selected_classes_idxs = idxs_labels[:,idxs_labels[1]<num_valid_classes][0]
                
                
                modified_set = DatasetSplit(dataset, idxs=selected_classes_idxs)
                modified_set.classes = dataset.classes[:num_valid_classes]
                

                
                datasets[dataset_key] = modified_set

                dist = defaultdict(int)
                for element in modified_set:
                    dist[element[1]]+=1

                print("Distribution of ", dataset_key,": ",  dist)

                # datasets['test'] = total_testset
                # datasets['train'].classes = datasets['train'].classes[:num_valid_classes]

        return datasets

        

    


    # def get_limited_testloader(self, args):
        

    #     dataset = self.datasets['test']
    #     idxs = np.arange(len(dataset))
    #     labels = []
    #     for element in dataset:
    #         labels.append(int(element[1]))
    #     idxs_labels = np.vstack((idxs, labels))
    #     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    #     selected_classes_idxs = idxs_labels[:,idxs_labels[1]<num_valid_classes][0]
        
    #     num_valid_classes = min(len(dataset.classes), args.split.limit_total_classes)

    #     total_testset = DatasetSplit(dataset, idxs=selected_classes_idxs)
    #     total_testset.classes = dataset.classes[:num_valid_classes]
    #     test_loader = DataLoader(total_testset,
    #                             batch_size=args.evaler.batch_size if args.evaler.batch_size > 0 else args.batch_size,
    #                             shuffle=False, num_workers=args.num_workers)
        
    #     num_classes = len(dataset.classes)
    #     num_samples_class = {i:0 for i in range(num_classes)}
    #     for _,labels in test_loader:
    #         for label in labels:
    #             num_samples_class[int(label)]+=1
    
    #     print("total test set samples distribution : ",num_samples_class)        

    #     return test_loader