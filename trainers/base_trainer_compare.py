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

from utils import DatasetSplit, get_dataset, cal_cos
from utils.logging_utils import AverageMeter

from torch.utils.data import DataLoader

from utils import terminate_processes, initalize_random_seed
from omegaconf import DictConfig,OmegaConf
from clients.build import get_client_type_compare



from loss_landscape import net_plotter, plot_2D, plot_surface
from loss_landscape.projection import setup_PCA_directions_fed, project_fed
import loss_landscape.projection as proj
from loss_landscape.plot_surface import crunch_fed



import collections


from trainers.base_trainer import Trainer

@TRAINER_REGISTRY.register()
class Trainer_compare(Trainer):

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

        # self.args = args
        # self.device = device
        # self.model = model

        # self.checkpoint_path = Path(self.args.checkpoint_path)
        # self.exp_path = self.checkpoint_path / self.args.dataset.name / self.args.exp_name



        # ### training config
        # trainer_args = self.args.trainer
        # self.num_clients = trainer_args.num_clients
        # self.participation_rate = trainer_args.participation_rate
        # self.global_rounds = trainer_args.global_rounds
        # # self.local_epochs = trainer_args.local_epochs
        # self.lr = trainer_args.local_lr
        # self.local_lr_decay = trainer_args.local_lr_decay

        

        # self.clients: List[Client] = [client_type(self.args, client_index=c) for c in range(self.args.trainer.num_clients)]
        
        # self.server = server


        # self.datasets = datasets
        # self.local_dataset_split_ids = get_dataset(self.args, self.datasets['train'], mode=self.args.split.mode)

        # test_loader = DataLoader(self.datasets["test"],
        #                          batch_size=args.evaler.batch_size if args.evaler.batch_size > 0 else args.batch_size,
        #                          shuffle=False, num_workers=args.num_workers)
        # eval_device = self.device if not self.args.multiprocessing else torch.device(f'cuda:{self.args.main_gpu}')
        # eval_params = {
        #     "test_loader": test_loader,
        #     # "gallery_loader": test_loader,
        #     # "query_loader": test_loader,
        #     "device": eval_device,
        #     "args": args,
        # }
        # self.eval_device = eval_device
        # self.evaler_type = evaler_type
        # self.eval_params = eval_params
        # # self.evaler = evaler_type(test_loader=test_loader, device=eval_device, args=args)
        # self.evaler = evaler_type(**eval_params)
        # logger.info(f"Trainer: {self.__class__}, client: {client_type}, server: {server.__class__}, evaler: {evaler_type}")

        super().__init__(model, client_type, server, evaler_type, datasets, device, args, multiprocessing, **kwargs)
        self.evaler_type = evaler_type
        client_type_compare = get_client_type_compare(args)


        if self.args.compare_base_is_FedAvg:
            self.args.client = self.args.client_compare
        self.clients_compare: List[Client] = [client_type_compare(self.args, client_index=c) for c in range(self.args.trainer.num_clients)]
        



    
    def local_update(self, device, task_queue, result_queue, compare = False):
        if self.args.multiprocessing:
            torch.cuda.set_device(device)
            initalize_random_seed(self.args)

        while True:
            task = task_queue.get()
            if task is None:
                break

            if compare:
                client = self.clients_compare[task['client_idx']]
            else:
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
            
            with torch.no_grad():
                if self.args.minority_collapse:
                    #check_data_distribution(dataloader,class_num:int=10,default_dist:torch.tensor=None)     get_l2norm(statedict)      cal_cos(statedict1, statedict2, eps = 1e-12)
                    num_classes = len(local_dataset.dataset.classes)
                    num_samples_per_class = local_dataset.class_dict
                    major_dict = {}
                    threshold = len(local_dataset) / num_classes
                    for i in range(num_classes):
                        if str(i) not in num_samples_per_class.keys():
                            num_samples_per_class[str(i)] = 0
                            major_dict[str(i)] = False

                        else:
                            num_samples_per_class[str(i)] /= len(local_dataset)
                            if num_samples_per_class[str(i)] >= threshold:
                                major_dict[str(i)] = True
                            elif num_samples_per_class[str(i)] < threshold * 0.5:
                                major_dict[str(i)] = False


                    #num_samples = 

                    
                    #fc_weight_dict = {'fc.weight', local_model['fc.weight']}
                    minor_fc_weight_dict = {}
                    minor_fc_bias_dict = {}
                    local_model_dict = local_model.state_dict()
                    for key in major_dict.keys():
                        if  major_dict[key]==False:
                            minor_fc_weight_dict[key] = local_model_dict['fc.weight'][int(key)]
                            minor_fc_bias_dict[key] = local_model_dict['fc.bias'][int(key)]

                    
                    minor_fc_weight_cos = 0
                    minor_fc_bias_cos = 0
                    count = 0
                    
                    minor_fc_weight_dict_keys = list(minor_fc_weight_dict.keys())
                    for i in range(len(minor_fc_weight_dict_keys)):
                        for j in range(i+1, len(minor_fc_weight_dict_keys)):
                            count += 1
                            
                            key1 = minor_fc_weight_dict_keys[i]
                            key2 = minor_fc_weight_dict_keys[j]
                            
                            weight_state1 = {'weight': minor_fc_weight_dict[key1]}
                            weight_state2 = {'weight': minor_fc_weight_dict[key2]}
                            minor_fc_weight_cos += cal_cos(weight_state1, weight_state2)

                            bias_state1 = {'bias': minor_fc_bias_dict[key1]}
                            bias_state2 = {'bias': minor_fc_bias_dict[key2]}
                            minor_fc_bias_cos += cal_cos(bias_state1, bias_state2)

                    minor_fc_weight_cos /= count
                    minor_fc_bias_cos /= count
                    
                    local_loss_dict['minor_fc_weight_cos'] = minor_fc_weight_cos
                    local_loss_dict['minor_fc_bias_cos'] = minor_fc_bias_cos

                    #fc_bias_dict = {'fc.bias', local_model['fc.bias']}



            if self.args.multiprocessing:
                result_queue.put((local_model, local_loss_dict, task['client_idx']))
            else:
                result_queue.put((local_model, local_loss_dict))
            if not self.args.multiprocessing:
                break

    def train(self) -> Dict:

        result_queue = mp.Manager().Queue()
        result_queue_compare = mp.Manager().Queue()

        M = max(int(self.participation_rate * self.num_clients), 1)

        if self.args.multiprocessing:
            ngpus_per_node = torch.cuda.device_count()
            task_queues = [mp.Queue() for _ in range(M)]
            task_queues_compare = [mp.Queue() for _ in range(M)]
            processes_original = [mp.get_context('spawn').Process(target=self.local_update, args=(
                i % ngpus_per_node, task_queues[i], result_queue)) for i in range(M)]


            #for loss landscape visualization
            processes_compare = [mp.get_context('spawn').Process(target=self.local_update, args=(
                i % ngpus_per_node, task_queues_compare[i], result_queue_compare, True)) for i in range(M)]

            #processes = processes_original + processes_compare
            # start all processes
            for p in processes_original:
                p.start()

            for p_compare in processes_compare:
                p_compare.start()

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

            local_weights = defaultdict(list)
            local_loss_dicts = defaultdict(list)
            local_deltas = defaultdict(list)

            local_models = []




            local_weights_compare = defaultdict(list)
            local_loss_dicts_compare = defaultdict(list)
            local_deltas_compare = defaultdict(list)

            local_models_compare = []
            local_evalers = []
            # Client-side
            start = time.time()
            for i, client_idx in enumerate(selected_client_ids):
                #print("client_idx-put : ",client_idx)
                task_queue_input = {
                    'model': self.model if self.args.multiprocessing else None,
                    'client_idx': client_idx,
                    'lr': current_lr,
                    'global_epoch': epoch,
                }
                if self.args.multiprocessing:
                    task_queues[i].put(task_queue_input)
                    if self.args.landscape.freq>0 and epoch % self.args.landscape.freq == 0:
                        task_queues_compare[i].put(task_queue_input)
                    # logger.info(f"[C{client_idx}] put queue")
                else:
                    task_queue = mp.Queue()
                    task_queue.put(task_queue_input)
                    self.local_update(self.device, task_queue, result_queue)

                    local_model, local_loss_dict = result_queue.get()
                    for loss_key in local_loss_dict:
                        local_loss_dicts[loss_key].append(local_loss_dict[loss_key])

                    # If you want to save gpu memory, make sure that weights are not allocated to GPU
                    
                    #for compare
                    local_state_dict = local_model.state_dict()
                    local_models.append(local_state_dict)

                    for param_key in local_state_dict:           
                        local_weights[param_key].append(local_state_dict[param_key])
                        local_deltas[param_key].append(local_state_dict[param_key] - global_state_dict[param_key])




                    #### compare
                    if self.args.landscape.freq>0 and epoch % self.args.landscape.freq == 0:
                        task_queue = mp.Queue()
                        task_queue.put(task_queue_input)

                        if self.args.landscape.visualize_local_loss:
                            this_local_eval_params = copy.deepcopy(self.eval_params)
                            this_local_traindata = DatasetSplit(self.datasets['train'], idxs=self.local_dataset_split_ids[client_idx])
                            this_local_trainloader = DataLoader(this_local_traindata, batch_size=self.args.evaler.batch_size, shuffle=True) 
                            this_local_eval_params['test_loader'] = this_local_trainloader
                            local_evalers.append(self.evaler_type(**this_local_eval_params))


                        self.local_update(self.device, task_queue, result_queue, compare = True)

                        local_model_compare, local_loss_dict_compare = result_queue.get()
                        for loss_key in local_loss_dict_compare:
                            local_loss_dicts_compare[loss_key].append(local_loss_dict_compare[loss_key])

                        # If you want to save gpu memory, make sure that weights are not allocated to GPU
                        local_state_dict_compare = local_model_compare.state_dict()
                        local_models_compare.append(local_state_dict_compare)

                        for param_key in local_state_dict_compare:
                            local_weights_compare[param_key].append(local_state_dict_compare[param_key])
                            local_deltas_compare[param_key].append(local_state_dict_compare[param_key] - global_state_dict[param_key])

            if self.args.multiprocessing:
                local_models = {}
                local_models_compare = {}
                local_evalers = {}



                for i, client_idx in enumerate(selected_client_ids):
                    # Retrieve results from the queue
                    result = result_queue.get()
                    local_model, local_loss_dict, client_idx1 = result
                    for loss_key in local_loss_dict:
                        local_loss_dicts[loss_key].append(local_loss_dict[loss_key])

                    # If you want to save gpu memory, make sure that weights are not allocated to GPU
                    local_state_dict = local_model.state_dict()
                    for param_key in local_state_dict:
                        local_weights[param_key].append(local_state_dict[param_key])
                        local_deltas[param_key].append(local_state_dict[param_key] - global_state_dict[param_key])

                    #for compare
                    local_models[client_idx1] = (local_state_dict)



                    #compare
                    if self.args.landscape.freq>0 and epoch % self.args.landscape.freq == 0:
                        if self.args.landscape.visualize_local_loss:
                            this_local_eval_params = copy.deepcopy(self.eval_params)
                            this_local_traindata = DatasetSplit(self.datasets['train'], idxs=self.local_dataset_split_ids[client_idx])
                            this_local_trainloader = DataLoader(this_local_traindata, batch_size=self.args.evaler.batch_size, shuffle=True) 
                            this_local_eval_params['test_loader'] = this_local_trainloader
                            local_evalers[client_idx]= (self.evaler_type(**this_local_eval_params))



                        result_compare = result_queue_compare.get()
                        local_model_compare, local_loss_dict_compare, client_idx2 = result_compare
                        for loss_key in local_loss_dict_compare:
                            local_loss_dicts_compare[loss_key].append(local_loss_dict_compare[loss_key])

                        # If you want to save gpu memory, make sure that weights are not allocated to GPU
                        local_state_dict_compare = local_model_compare.state_dict()
                        local_models_compare[client_idx2] = (local_state_dict_compare)

                        for param_key in local_state_dict_compare:
                            local_weights_compare[param_key].append(local_state_dict_compare[param_key])
                            local_deltas_compare[param_key].append(local_state_dict_compare[param_key] - global_state_dict[param_key])



                        #print("clientidx, clientidx1, clientidx2 :", client_idx,client_idx1,client_idx2)                    

                local_models, local_models_compare, local_evalers = [list(collections.OrderedDict(sorted(x.items())).values()) for x in [local_models, local_models_compare, local_evalers]]



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
            updated_global_state_dict = self.server.aggregate(local_weights, local_deltas,
                                                              client_ids=selected_client_ids)
            
            if self.args.landscape.freq>0 and epoch % self.args.landscape.freq == 0:
                updated_global_state_dict_compare = self.server.aggregate(local_weights_compare, local_deltas_compare,
                                                              client_ids=selected_client_ids)

            self.model.load_state_dict(updated_global_state_dict)

            # Logging
            wandb_dict = {loss_key: np.mean(local_loss_dicts[loss_key]) for loss_key in local_loss_dicts}
            wandb_dict['lr'] = self.lr
            
            if self.args.eval.freq>0 and epoch % self.args.eval.freq == 0:
                self.evaluate(epoch=epoch)


            if self.args.landscape.freq>0 and epoch % self.args.landscape.freq == 0:
                
                comm, rank, nproc = None, 0, 1

                w = net_plotter.get_weights(self.model)
                s = copy.deepcopy(self.model.state_dict())
                

                model_files = copy.deepcopy(local_models + local_models_compare)

                #--------------------------------------------------------------------------
                # Create projection directions
                #--------------------------------------------------------------------------
                dir_file = setup_PCA_directions_fed(self.args, copy.deepcopy(model_files), w, s, epoch)
                #--------------------------------------------------------------------------
                # projection trajectory to given directions
                #--------------------------------------------------------------------------
                proj_file, (local_xmax,local_xmin,local_ymax,local_ymin) = project_fed(dir_file, w, s,
                                            copy.deepcopy(model_files  + [updated_global_state_dict_compare]+ [prev_model_weight]), self.args.landscape.dir_type, 'cos')
                
                if self.args.landscape.adaptive_xy_range:
                    self.args.landscape.xmax, self.args.landscape.xmin, self.args.landscape.ymax, self.args.landscape.ymin = (np.array([local_xmax,local_xmin,local_ymax,local_ymin])*self.args.landscape.grid_size).tolist()
                
                 # load directions
                d = net_plotter.load_directions(dir_file)
                # calculate the consine similarity of the two directions
                if len(d) == 2 :
                    similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
                    print('cosine similarity between x-axis and y-axis: %f' % similarity)



                
                #--------------------------------------------------------------------------
                # Setup the direction file and the surface file
                #--------------------------------------------------------------------------
                surf_file = plot_surface.name_surface_file_fed(self.args.landscape ,dir_file)
                plot_surface.setup_surface_file(self.args.landscape, surf_file, dir_file)
               



                #--------------------------------------------------------------------------
                # Start the computation
                #--------------------------------------------------------------------------
                plot_surface.crunch_fed(surf_file, copy.deepcopy(self.model), w, s, d, 'test_loss', 'test_acc', comm, rank, self.args, evaler = [self.evaler] + local_evalers, epoch = epoch)

               

                self.wandb_log({f"loss-landscape/{self.args.dataset.name}": wandb.Image(plot_2D.plot_contour_fed(surf_file, dir_file, proj_file, surf_name='test_loss',
                                vmin=0.1, vmax=100, vlevel=5, show=False, adaptive_v = self.args.landscape.adaptive_v, compare = True, plot_num = len([self.evaler] + local_evalers))) }, step=epoch)





                ####### measure the distance between global model

                # def distance_globalmodel(global_model_dict, model_dicts):
                #     result = 0
                #     for model_dict in model_dicts:
                #         for key in model_dict.keys():
                #             result+=sum(((global_model_dict[key] - model_dict[key])**2)/len(model_dicts)).item()

                #     return result
                

                # self.wandb_log({f"distance_global_local":distance_globalmodel(self.model.state_dict(),local_models)})
                # self.wandb_log({f"distance_global_prev":distance_globalmodel(self.model.state_dict(),[prev_model_weight])})

            self.wandb_log(wandb_dict, step=epoch)


        if self.args.multiprocessing:
            # Terminate Processes
            terminate_processes(task_queues, processes_original)
            terminate_processes(task_queues_compare, processes_compare)

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

