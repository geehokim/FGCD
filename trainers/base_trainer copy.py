from pathlib import Path
from typing import Callable, Dict, Tuple, Union, List, Type
from argparse import Namespace
from collections import defaultdict

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import tqdm
import wandb

import pickle, os
import numpy as np

import logging
logger = logging.getLogger(__name__)


import time, io, copy

from trainers.build import TRAINER_REGISTRY

from servers import Server
from clients import Client

from utils import DatasetSplit, get_dataset
from utils.logging_utils import AverageMeter

from torch.utils.data import DataLoader

from utils import terminate_processes, initalize_random_seed
from omegaconf import DictConfig,OmegaConf




from loss_landscape import net_plotter, plot_2D, plot_surface
from loss_landscape.projection import setup_PCA_directions_fed, project_fed
import loss_landscape.projection as proj
from loss_landscape.plot_surface import crunch_fed



from utils.visualize import log_models_Umap




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
        self.exp_path = self.checkpoint_path / self.args.dataset.name / self.args.exp_name


        ### training config
        trainer_args = self.args.trainer
        self.num_clients = trainer_args.num_clients
        self.participation_rate = trainer_args.participation_rate
        self.global_rounds = trainer_args.global_rounds
        # self.local_epochs = trainer_args.local_epochs
        self.lr = trainer_args.local_lr
        self.local_lr_decay = trainer_args.local_lr_decay


        self.clients: List[Client] = [client_type(self.args, client_index=c) for c in range(self.args.trainer.num_clients)]
        self.server = server


        self.datasets = datasets
        self.local_dataset_split_ids = get_dataset(self.args, self.datasets['train'], mode=self.args.split.mode)

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
        self.eval_device = eval_device
        
        # self.evaler = evaler_type(test_loader=test_loader, device=eval_device, args=args)
        self.evaler = evaler_type(**eval_params)
        logger.info(f"Trainer: {self.__class__}, client: {client_type}, server: {server.__class__}, evaler: {evaler_type}")



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
            local_dataset = DatasetSplit(self.datasets['train'], idxs=self.local_dataset_split_ids[task['client_idx']])
            # logger.info(f"[C{task['client_idx']}] after dataset split")
            setup_inputs = {
                'model': copy.deepcopy(task['model']) if self.args.multiprocessing else copy.deepcopy(self.model),
                'device': device,
                'local_dataset': local_dataset,
                'init_lr': task['lr'],
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


    def update_local_inform(self, global_state_dict, result, local_inform):
        self._update_local_inform_base(global_state_dict, result, local_inform)


    def _update_local_inform_base(self, global_state_dict, result, local_inform):
        local_model, local_loss_dict = result
        for loss_key in local_loss_dict:
            local_inform['local_loss_dicts'][loss_key].append(local_loss_dict[loss_key])

        # If you want to save gpu memory, make sure that weights are not allocated to GPU
        local_state_dict = local_model.state_dict()
        local_inform['local_models'].append(local_state_dict)

        for param_key in local_state_dict:
            local_inform['local_weights'][param_key].append(local_state_dict[param_key])
            local_inform['local_deltas'][param_key].append(local_state_dict[param_key] - global_state_dict[param_key])  

    def get_local_result(self, task_queue_input, result_queue):
        task_queue = mp.Queue()
        task_queue.put(task_queue_input)
        self.local_update(self.device, task_queue, result_queue)

        result = result_queue.get()
        return result



    def create_local_inform(self):
        local_inform = {}
        self._create_local_inform_base(local_inform)
        return local_inform


    def _create_local_inform_base(self, local_inform):
        local_inform['local_weights'] = defaultdict(list)
        local_inform['local_loss_dicts'] = defaultdict(list)
        local_inform['local_deltas'] = defaultdict(list)
        local_inform['local_models'] = []


    def create_taskqueues_processes(self, M, result_queue):
        ngpus_per_node = torch.cuda.device_count()
        task_queues_list = []
        task_queues_list.append(self._create_taskqueues_base(M))
        processes_list = []
        processes_list.append(self._create_processes_base(M, result_queue, ngpus_per_node, task_queues_list[0]))
        return task_queues_list, processes_list

    def _create_taskqueues_base(self, M):
        task_queues = [mp.Queue() for _ in range(M)]
        return task_queues

    def _create_processes_base(self, M, result_queue, ngpus_per_node, task_queues):
        processes = [mp.get_context('spawn').Process(target=self.local_update, args=(
            i % ngpus_per_node, task_queues[i], result_queue)) for i in range(M)]

        # start all processes
        for p in processes:
            p.start()

        return processes


    def put_task_queue_input(self, task_queues_list, task_queue_input):
        task_queues_list[0].put(task_queue_input)


    def do_local_client(self, task_queue_input, result_queue, global_state_dict, local_inform):
        self._do_localclient_base(task_queue_input, result_queue, global_state_dict, local_inform)

    def _do_localclient_base(self, task_queue_input, result_queue, global_state_dict, local_inform):

        result = self.get_local_result(task_queue_input, result_queue)
        self.update_local_inform(global_state_dict, result, local_inform)



    def do_local_client_mp(self, result_queue, global_state_dict, local_inform):
        self._do_localclient_mp_base(result_queue, global_state_dict, local_inform)


    def _do_localclient_base_mp(self, result_queue, global_state_dict, local_inform):

        result = result_queue.get()
        self.update_local_inform(global_state_dict, result, local_inform)


    def train(self) -> Dict:

        result_queue = mp.Manager().Queue()

        M = max(int(self.participation_rate * self.num_clients), 1)

        if self.args.multiprocessing:
            # ngpus_per_node = torch.cuda.device_count()
            # task_queues = [mp.Queue() for _ in range(M)]
            # processes = [mp.get_context('spawn').Process(target=self.local_update, args=(
            #     i % ngpus_per_node, task_queues[i], result_queue)) for i in range(M)]

            # # start all processes
            # for p in processes:
            #     p.start()

            task_queues_list, processes_list = self.create_taskqueues_processes(M, result_queue)

        for epoch in range(self.global_rounds):
            self.lr_update(epoch=epoch)

            global_state_dict = copy.deepcopy(self.model.state_dict())
            prev_model_weight = copy.deepcopy(self.model.state_dict())
            # Select clients
            if self.participation_rate < 1.:
                selected_client_ids = np.random.choice(range(self.num_clients), M, replace=False)
            else:
                selected_client_ids = range(len(self.clients))
            logger.info(f"Global epoch {epoch}, Selected client : {selected_client_ids}")

            current_lr = self.lr
            # local_inform = {}
            # local_inform['local_weights'] = defaultdict(list)
            # local_inform['local_loss_dicts'] = defaultdict(list)
            # local_inform['local_deltas'] = defaultdict(list)
            # local_inform['local_models'] = []
            local_inform = self.create_local_inform()

            # Client-side
            start = time.time()
            for i, client_idx in enumerate(selected_client_ids):
                task_queue_input = {
                    'model': self.model if self.args.multiprocessing else None,
                    'client_idx': client_idx,
                    'lr': current_lr,
                    'global_epoch': epoch,
                }
                if self.args.multiprocessing:
                    #task_queues[i].put(task_queue_input)
                    self.put_task_queue_input(task_queues_list, task_queue_input)
                    # logger.info(f"[C{client_idx}] put queue")
                else:
                    # task_queue = mp.Queue()
                    # task_queue.put(task_queue_input)
                    # self.local_update(self.device, task_queue, result_queue)

                    # local_model, local_loss_dict = result_queue.get()
                    
                    
                    # result = self.get_local_result(task_queue_input, result_queue)
                    # self.update_local_inform(global_state_dict, result, local_inform)
                    
                    self.do_local_client(task_queue_input, result_queue, global_state_dict, local_inform)
                    
                    # for loss_key in local_loss_dict:
                    #     local_loss_dicts[loss_key].append(local_loss_dict[loss_key])

                    # # If you want to save gpu memory, make sure that weights are not allocated to GPU
                    # local_state_dict = local_model.state_dict()
                    # local_models.append(local_state_dict)

                    # for param_key in local_state_dict:
                    #     local_weights[param_key].append(local_state_dict[param_key])
                    #     local_deltas[param_key].append(local_state_dict[param_key] - global_state_dict[param_key])

            if self.args.multiprocessing:
                for _ in range(len(selected_client_ids)):
                    # Retrieve results from the queue


                    self.do_local_client_mp(result_queue, global_state_dict, local_inform)

                    # result = result_queue.get()
                    # self.update_local_inform(global_state_dict, result, local_inform)




                    # for loss_key in local_loss_dict:
                    #     local_loss_dicts[loss_key].append(local_loss_dict[loss_key])

                    # # If you want to save gpu memory, make sure that weights are not allocated to GPU
                    # local_state_dict = local_model.state_dict()
                    # for param_key in local_state_dict:
                    #     local_weights[param_key].append(local_state_dict[param_key])
                    #     local_deltas[param_key].append(local_state_dict[param_key] - global_state_dict[param_key])
            logger.info(f"Global epoch {epoch}, Train End. Total Time: {time.time() - start:.2f}s")

            # client = self.clients[client_idx]
            # #dataset = self.datasets['train']
            # #local_dataset_split_ids = get_dataset(self.args, dataset, mode=self.args.mode)
            # local_dataset = DatasetSplit(dataset, idxs=local_dataset_split_ids[client_idx])
            # client.setup(model=copy.deepcopy(self.model), device=self.device, local_dataset=local_dataset, init_lr=current_lr)
            #
            # # Local Training
            # local_model, local_loss_dict = client.local_train()
            #
            # for loss_key in local_loss_dict:
            #     local_loss_dicts[loss_key].append(local_loss_dict[loss_key])
            #
            # # If you want to save gpu memory, make sure that weights are not allocated to GPU
            # local_state_dict = local_model.state_dict()
            # for param_key in local_state_dict:
            #     local_weights[param_key].append(local_state_dict[param_key])
            #     local_deltas[param_key].append(local_state_dict[param_key] - global_state_dict[param_key])

            # Server-side
            updated_global_state_dict = self.server.aggregate(local_inform['local_weights'], local_inform['local_deltas'],
                                                              client_ids=selected_client_ids)
            self.model.load_state_dict(updated_global_state_dict)

            # Logging
            wandb_dict = {loss_key: np.mean(local_inform['local_loss_dicts'][loss_key]) for loss_key in local_inform['local_loss_dicts']}
            wandb_dict['lr'] = self.lr
            
            if self.args.eval.freq > 0 and epoch % self.args.eval.freq == 0:
                self.evaluate(epoch=epoch)


            if self.args.umap.freq > 0 and epoch % self.args.umap.freq == 0:
                self.evaler.visualize_umap(global_model=self.model, local_models=local_inform['local_models'], epoch=epoch)

            if self.args.landscape.freq > 0 and epoch % self.args.landscape.freq == 0:

                self.visualize_landscape(global_model=self.model, local_models=local_inform['local_models'], epoch=epoch)
                comm, rank, nproc = None, 0, 1

                w = net_plotter.get_weights(self.model)
                s = copy.deepcopy(self.model.state_dict())
                model_files = copy.deepcopy(local_inform['local_models'])

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



            self.wandb_log(wandb_dict, step=epoch)


        if self.args.multiprocessing:
            # Terminate Processes
            for task_queues,processes in zip(task_queues_list, processes_list):
                terminate_processes(task_queues, processes)

        return

    def lr_update(self, epoch: int) -> None:
        # TODO: adopt other lr policy
        # self.lr = self.lr * (self.lr_decay) ** (epoch)
        self.lr = self.args.trainer.local_lr * (self.local_lr_decay) ** (epoch)
        return
    
    #TODO
    def save_model(self, epoch: int = -1, suffix: str = '') -> None:
        logger.info("Not implemented")
        
        model_path = self.exp_path / self.args.output_model_path
        if not model_path.parent.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)

        if epoch < self.args.trainer.global_rounds - 1:
            model_path = Path(f"{model_path}.e{epoch+1}")

        if suffix:
            model_path = Path(f"{model_path}.{suffix}")
        
        # save_checkpoint(self.model, model_path, epoch, save_torch=True)        
        return


    def wandb_log(self, log: Dict, step: int = None):
        if self.args.wandb:
            wandb.log(log, step=step)

    def validate(self, epoch: int, ) -> Dict:
        return

    def evaluate(self, epoch: int, ) -> Dict:

        '''
        Return: accuracy of global test data
        '''
        # eval_device = self.device if not self.args.multiprocessing else torch.device(f'cuda:{self.args.main_gpu}')
        # self.model.eval()
        # self.model.to(eval_device)
        # correct, total = 0, 0

        # with torch.no_grad():
        #     for images, labels in self.loaders["test"]:
        #         images, labels = images.to(eval_device), labels.to(eval_device)
        #         results = self.model(images)
        #         # _, predicted = torch.max(results.data, 1) 
        #         _, predicted = torch.max(results["logit"].data, 1) # if errors occur, use ResNet18_base instead of ResNet18_GFLN
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()

        # acc = 100. * correct / float(total)
        # logger.warning(f'[Epoch {epoch}] Test Accuracy: {acc:.2f}%')

        # self.model.to('cpu')
        # self.model.train()

        # self.wandb_log({f"acc/{self.args.dataset.name}": acc}, step=epoch)
        # return acc

        # self.model.to(self.eval_device)
        results = self.evaler.eval(model=self.model, epoch=epoch)
        acc = results["acc"]
        logger.warning(f'[Epoch {epoch}] Test Accuracy: {acc:.2f}%')
        # self.model.to()

        self.wandb_log({f"acc/{self.args.dataset.name}": acc}, step=epoch)
        return {
            "acc": acc
        }


    #TODO
    def visualize_landscape(self, global_model: nn.Module, local_models: List[nn.Module], epoch: int, ):
        #results =  self.visualizer.loss_landscape(global_model=global_model, local_models=local_models, epoch=epoch)
        #self.wandb_log(results~~)
        return
    
    #TODO
    def visualize_umap(self, prev_model_weight, current_weight, local_weight, wandb_dict ):
        #results =  self.visualizer.umap(model=global_model, epoch=epoch)
        #self.wandb_log(results~~)
        # self.evaler.visualize_umap()

        # log_models_Umap(model = copy.deepcopy(model),
        #                 models_dict_list = models_state_dict_list,
        #                 testloader = testloader, args = args, names_list = names_list,
        #                 num_of_sample_per_class = 100,
        #                 draw_classes = 10, 
        #                 drawing_options = drawing_option
        #     )
        models_state_dict_list = []
        models_state_dict_list.append(prev_model_weight)
        models_state_dict_list.append(current_weight)
        models_state_dict_list.append(local_weight[0])
        models_state_dict_list.append(local_weight[1])
        
        
        names_list = ["pastglobal","global", "local0", "local1"]
        drawing_option = [
            [True, True, False, False],
            [True, False, True, False],
            [True, False, False, True],
            [False, True, True, False],
            [False, True, False, True],
            [False, False, True, True],
            [True, False, True, True],
            [False, True, True, True],
            [True, True, True, True],
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
        ]
        wandb_dict.update(log_models_Umap(model = copy.deepcopy(self.model), models_dict_list = models_state_dict_list, testloader = self.evaler.testloader, args = self.args, names_list = names_list, num_of_sample_per_class = 100, draw_classes = 10, drawing_options = drawing_option
        ))


        wandb_dict.update(log_models_Umap(model = copy.deepcopy(self.model), models_dict_list = models_state_dict_list, testloader = testloader, args = self.args, names_list = names_list, num_of_sample_per_class = 100, draw_classes = 4, drawing_options = drawing_option
        ))




        return wandb_dict




    

    
