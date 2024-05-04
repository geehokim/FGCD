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

import pickle, os
import numpy as np


import logging
logger = logging.getLogger(__name__)
# coloredlogs.install(level='INFO', fmt='%(asctime)s %(name)s[%(process)d] %(message)s', datefmt='%m-%d %H:%M:%S')



import time, io, copy

from trainers.build import TRAINER_REGISTRY

from servers import Server
from clients import Client

from utils import DatasetSplit, DatasetSplitMultiViews, get_dataset
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





@TRAINER_REGISTRY.register()
class Trainer():

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


        self.clients: List[Client] = [client_type(self.args, client_index=c, model=copy.deepcopy(self.model)) for c in range(self.args.trainer.num_clients)]
        self.server = server
        if self.args.server.momentum > 0:
            self.server.set_momentum(self.model)

        #self.datasets = datasets
        self.datasets = self.get_datasets(datasets)
        self.local_dataset_split_ids = get_dataset(self.args, self.datasets['train'], mode=self.args.split.mode)

        test_loader = DataLoader(self.datasets["test"],
                                batch_size=args.evaler.batch_size if args.evaler.batch_size > 0 else args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
        #print(test_loader)
        #print(args.evaler.batch_size)
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
        logger.info(f"Trainer: {self.__class__}, client: {client_type}, server: {server.__class__}, evaler: {evaler_type}")

        self.start_round = 0
        if self.args.get('load_model_path'):
            self.load_model()



    def local_update(self, device, task_queue, result_queue):
        if self.args.multiprocessing:
            torch.cuda.set_device(device)
            initalize_random_seed(self.args)

        while True:
            task = task_queue.get()
            if task is None:
                break
            client = self.clients[task['client_idx']]
            # logger.info(f"[C{task['client_idx']}] before dataset split")

            if self.args.dataset.get('num_views'):
                local_dataset = DatasetSplitMultiViews(self.datasets['train'], idxs=self.local_dataset_split_ids[task['client_idx']])
            else:
                local_dataset = DatasetSplit(self.datasets['train'], idxs=self.local_dataset_split_ids[task['client_idx']])
            # logger.info(f"[C{task['client_idx']}] after dataset split")
            setup_inputs = {
                # 'model': copy.deepcopy(task['model']) if self.args.multiprocessing else copy.deepcopy(self.model),
                'model': task['model'],
                'device': device,
                'local_dataset': local_dataset,
                'local_lr': task['local_lr'],
                'global_epoch': task['global_epoch'],
                'trainer': self,
            }
            # logger.info(f"[C{task['client_idx']}] after input setup")
            client.setup(**setup_inputs)
            # client.setup(model=copy.deepcopy(self.model), device=device, local_dataset=local_dataset, init_lr=self.lr)
            # logger.info(f"[C{task['client_idx']}] after setup")
            # Local Training
            local_model, local_loss_dict = client.local_train(global_epoch=task['global_epoch'])
            result_queue.put((local_model, local_loss_dict))
            if not self.args.multiprocessing:
                break

    def train(self) -> Dict:

        result_queue = mp.Manager().Queue()

        M = max(int(self.participation_rate * self.num_clients), 1)

        if self.args.multiprocessing:
            ngpus_per_node = torch.cuda.device_count()
            task_queues = [mp.Queue() for _ in range(M)]
            processes = [mp.get_context('spawn').Process(target=self.local_update, args=(
                i % ngpus_per_node, task_queues[i], result_queue)) for i in range(M)]

            # start all processes
            for p in processes:
                p.start()


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
                    'model': self.model.state_dict(),
                    'client_idx': client_idx,
                    #'lr': current_lr,
                    'local_lr': current_lr,
                    'global_epoch': epoch,
                }
                if self.args.multiprocessing:
                    task_queues[i].put(task_queue_input)
                    # logger.info(f"[C{client_idx}] put queue")
                else:
                    task_queue = mp.Queue()
                    task_queue.put(task_queue_input)
                    self.local_update(self.device, task_queue, result_queue)

                    local_state_dict, local_loss_dict = result_queue.get()
                    for loss_key in local_loss_dict:
                        local_loss_dicts[loss_key].append(local_loss_dict[loss_key])

                    # local_state_dict = local_model.state_dict()
                    local_models.append(local_state_dict)

                    for param_key in local_state_dict:
                        local_weights[param_key].append(local_state_dict[param_key])
                        local_deltas[param_key].append(local_state_dict[param_key] - global_state_dict[param_key])

            if self.args.multiprocessing:
                for _ in range(len(selected_client_ids)):
                    # Retrieve results from the queue
                    result = result_queue.get()
                    local_state_dict, local_loss_dict = result
                    for loss_key in local_loss_dict:
                        local_loss_dicts[loss_key].append(local_loss_dict[loss_key])

                    local_models.append(local_state_dict)

                    # If you want to save gpu memory, make sure that weights are not allocated to GPU
                    #local_state_dict = local_model.state_dict()
                    for param_key in local_state_dict:
                        local_weights[param_key].append(local_state_dict[param_key])
                        local_deltas[param_key].append(local_state_dict[param_key] - global_state_dict[param_key])

            logger.info(f"Global epoch {epoch}, Train End. Total Time: {time.time() - start:.2f}s")


            # Server-side
            updated_global_state_dict = self.server.aggregate(local_weights, local_deltas,
                                                              selected_client_ids, copy.deepcopy(global_state_dict), current_lr)
            self.model.load_state_dict(updated_global_state_dict)

            local_datasets = [DatasetSplit(self.datasets['train'], idxs=self.local_dataset_split_ids[client_id]) for client_id in selected_client_ids]

            # Logging
            wandb_dict = {loss_key: np.mean(local_loss_dicts[loss_key]) for loss_key in local_loss_dicts}
            wandb_dict['lr'] = self.lr
            
            # try:
            model_device = next(self.model.parameters()).device
            if self.args.eval.freq > 0 and epoch % self.args.eval.freq == 0:
                self.evaluate(epoch=epoch, local_datasets=local_datasets)
                
            if self.args.analysis:
                if self.args.eval.retrieval_freq > 0 and epoch % self.args.eval.retrieval_freq == 0:
                    self.evaluate_retrieval(epoch=epoch, local_datasets=local_datasets)

                if epoch > 0 and self.args.eval.local_freq > 0 and epoch % self.args.eval.local_freq == 0:
                    self.evaluate_local(epoch=epoch, local_models=local_models)

                if epoch > 0 and self.args.svd.freq > 0 and epoch % self.args.svd.freq == 0:
                # if self.args.svd.freq > 0 and epoch > 0 and epoch % self.args.svd.freq == 0:
                    self.evaluate_svd(epoch=epoch, local_models=local_models, global_model=self.model, local_datasets=local_datasets)

                # if epoch > 0 and self.args.umap.freq > 0 and epoch % self.args.umap.freq == 0:
                if self.args.umap.freq > 0 and epoch % self.args.umap.freq == 0:
                    self.visualize_umap(global_model=self.model, local_models=local_models, local_datasets=local_datasets, epoch=epoch)
                    # self.wandb_log(self.evaler.visualize_umap(global_model=self.model, local_models=local_models, epoch=epoch), step=epoch)

                if self.args.landscape.freq > 0 and epoch % self.args.landscape.freq == 0:
                    self.visualize_landscape(global_model=self.model, local_models=local_models, prev_model_weight=prev_model_weight, epoch=epoch)

                if self.args.collapse.freq > 0 and epoch % self.args.collapse.freq == 0:
                    self.evaluate_minority_collapse(prev_model_weight=prev_model_weight, local_models=local_models, local_datasets=local_datasets, epoch=epoch)

            # except:
            #     self.model.to(model_device)
            #     self.model.train()
            #     pass

            if (self.args.save_freq > 0 and (epoch + 1) % self.args.save_freq == 0) or (epoch + 1 == self.args.trainer.global_rounds):
                self.save_model(epoch=epoch)

            self.wandb_log(wandb_dict, step=epoch)
            gc.collect()


        if self.args.multiprocessing:
            # Terminate Processes
            terminate_processes(task_queues, processes)

        return

    def lr_update(self, epoch: int) -> None:
        # TODO: adopt other lr policy
        # self.lr = self.lr * (self.lr_decay) ** (epoch)
        self.lr = self.args.trainer.local_lr * (self.local_lr_decay) ** (epoch)
        return
    

    def save_model(self, epoch: int = -1, suffix: str = '') -> None:
        
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
        
        save_checkpoint(self.model, model_path, epoch, save_torch=True, use_breakpoint=False)        
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

    def evaluate(self, epoch: int, local_datasets: List[torch.utils.data.Dataset] = None) -> Dict:

        results = self.evaler.eval(model=self.model, epoch=epoch)
        acc = results["acc"]
        entropy = results["entropy"]
        ece = results["ece"]
        ece_diagram = results["ece_diagram"]

        wandb_dict = {
            f"acc/{self.args.dataset.name}": acc,
            f"confusion_matrix/{self.args.dataset.name}" : results["confusion_matrix"] if "confusion_matrix" in results else None,
            f'entropy/{self.args.dataset.name}': entropy,
            f'ece/{self.args.dataset.name}': ece,
            }

        if epoch % 10 == 0:
            wandb_dict.update({f'ece_diagram/{self.args.dataset.name}': wandb.Image(ece_diagram)})

        # if self.args.get('debugs'):
        #     breakpoint()
        
        logger.warning(f'[Epoch {epoch}] Test Accuracy: {acc:.2f}%, Rel Entropy: {entropy:.3f}, ECE: {100*ece:.2f}')

        # Local major/minor accuracies
        class_accs = results["class_acc"]
        subset_results = self._evaluate_subset(epoch=epoch, class_accs=class_accs, local_datasets=local_datasets)
        wandb_dict.update(subset_results)

        plt.close()

        # seen_acc = major_acc = minor_acc = missing_acc = minor_seen_acc = -1
        # if local_datasets is not None:
        #     seen_accs = []
        #     major_accs, minor_accs, missing_accs, minor_seen_accs = [], [], [], []
        #     for local_dataset in local_datasets:
        #         local_classes = [int(i) for i in local_dataset.class_dict.keys()]


        #         num_classes = len(local_dataset.dataset.classes)
        #         num_local_classes = len(local_dataset.class_dict.keys())
        #         major_classes = [int(key) for key in local_dataset.class_dict if local_dataset.class_dict[key] >= len(local_dataset)/num_local_classes]
        #         minor_seen_classes = [int(key) for key in local_dataset.class_dict if local_dataset.class_dict[key] < len(local_dataset)/num_local_classes]
        #         missing_classes = [i for i in range(num_classes) if str(i) not in local_dataset.class_dict]
        #         minor_classes = minor_seen_classes + missing_classes

        #         class_accs = results["class_acc"]
        #         seen_accs.append(torch.mean(class_accs[local_classes]))
        #         major_accs.append(torch.mean(class_accs[major_classes]))
        #         minor_accs.append(torch.mean(class_accs[minor_classes]))
        #         missing_accs.append(torch.mean(class_accs[missing_classes]))
        #         minor_seen_accs.append(torch.mean(class_accs[minor_seen_classes]))

        #     seen_acc = np.mean(seen_accs)
        #     major_acc, minor_acc, missing_acc, minor_seen_acc = np.mean(major_accs), np.mean(minor_accs), np.mean(missing_accs), np.mean(minor_seen_accs)
        #     wandb_dict.update({
        #         f"seen_acc/{self.args.dataset.name}": seen_acc, #deprecated

        #         f"acc/{self.args.dataset.name}/seen": seen_acc,
        #         f"acc/{self.args.dataset.name}/major": major_acc,
        #         f"acc/{self.args.dataset.name}/minor": minor_acc,
        #         f"acc/{self.args.dataset.name}/missing": missing_acc,
        #         f"acc/{self.args.dataset.name}/minor_seen": minor_seen_acc,
        #         })
        

        self.wandb_log(wandb_dict, step=epoch)
        return {
            "acc": acc
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