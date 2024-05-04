#!/usr/bin/env python
# coding: utf-8
import copy
import time

import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from sklearn.manifold import TSNE

from utils import *
from utils.metrics import evaluate
from utils.visualize import __log_test_metric__, umap_allmodels, cka_allmodels, log_fisher_diag
from models import build_encoder, get_model
from typing import Callable, Dict, Tuple, Union, List


from servers.build import SERVER_REGISTRY

@SERVER_REGISTRY.register()
class Server():

    def __init__(self, args):
        self.args = args
        return
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C
        return local_weights
    

@SERVER_REGISTRY.register()
class ServerM(Server):    
    
    def set_momentum(self, model):

        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])


        self.global_delta = global_delta
        self.global_momentum = global_momentum

    @torch.no_grad()
    def FedACG_lookahead(self, model):
        sending_model_dict = copy.deepcopy(model.state_dict())
        for key in self.global_momentum.keys():
            sending_model_dict[key] += self.args.server.momentum * self.global_momentum[key]

        model.load_state_dict(sending_model_dict)
        return copy.deepcopy(model)
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C
        if self.args.server.momentum>0:
            # print("self.args.server.get('FedACG'): ",self.args.server.get('FedACG'))

            if not self.args.server.get('FedACG'): 
                for param_key in local_weights:               
                    local_weights[param_key] += self.args.server.momentum * self.global_momentum[param_key]
                    
            for param_key in local_deltas:
                self.global_delta[param_key] = sum(local_deltas[param_key])/C
                self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + self.global_delta[param_key]
            

        return local_weights


@SERVER_REGISTRY.register()
class ServerAdam(Server):    
    
    def set_momentum(self, model):

        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        global_v = copy.deepcopy(model.state_dict())
        for key in global_v.keys():
            global_v[key] = torch.zeros_like(global_v[key]) + (self.args.server.tau * self.args.server.tau)



        self.global_delta = global_delta
        self.global_momentum = global_momentum
        self.global_v = global_v

    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        #model_dict = model.state_dict()
        for param_key in local_deltas:
            self.global_delta[param_key] = sum(local_deltas[param_key])/C
            self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + (1-self.args.server.momentum) * self.global_delta[param_key]
            self.global_v[param_key] = self.args.server.beta * self.global_v[param_key] + (1-self.args.server.beta) * (self.global_delta[param_key] * self.global_delta[param_key])

        for param_key in model_dict.keys():
            model_dict[param_key] += current_lr *  self.global_momentum[param_key] / ( (self.global_v[param_key]**0.5) + self.args.server.tau)
            
        return model_dict


    
# @SERVER_REGISTRY.register()
# class Server_AvgM():

#     def __init__(self, args):
#         self.args = args
#         return
    
#     def aggregate(self, local_weights, local_deltas, client_ids):
#         C = len(client_ids)
        
#         for param_key in local_weights:
#             local_weights[param_key] = sum(local_weights[param_key])/C

#         return local_weights




        

def GlobalUpdate(args,device,trainset,testloader,LocalUpdate):
    print("GlobalUpdate args")
    print(args)
    model = build_encoder(args)
    wandb.watch(model)
    dataset = get_dataset(args, trainset, args.mode)
    loss_train = []
    acc_train = []
    this_lr = args.lr
    this_alpha = args.alpha
    m = max(int(args.participation_rate * args.num_of_clients), 1)
    print("Participated clients every round : ", m)

    if args.load_from_saved_dict != "":
        saved_dict = torch.load(args.load_from_saved_dict)
        model.load_state_dict(saved_dict['model_state_dict'])
        print("Model succesfully load dict from :", args.load_from_saved_dict)

    if args.multiprocessing:
        # initialize shared list
        global_list = mp.Manager().list()
        global_list.append((args, dataset, trainset, testloader))
        global_list.append(model)

        # initialize shared queue, and task specific queue (not shared)
        ngpus_per_node = torch.cuda.device_count()
        queues = [mp.Queue() for _ in range(m)]
        result_queue = mp.Manager().Queue()
        processes = [mp.get_context('spawn').Process(target=train, args=(
        i % ngpus_per_node, LocalUpdate, queues[i], result_queue, global_list)) for i in range(m)]

        # start all processes
        for p in processes:
            p.start()

    for epoch in range(args.global_epochs):
        wandb_dict={}
        num_of_data_clients=[]
        local_weight = []
        local_loss = []
        local_delta = []
        global_weight = copy.deepcopy(model.state_dict())
        if (epoch==0) or (args.participation_rate<1) :
            selected_user = np.random.choice(range(args.num_of_clients), m, replace=False)
        else:
            pass 
        print(f"This is global {epoch} epoch")
        start = time.time()
        for i, user in enumerate(selected_user):
            if args.multiprocessing:
                queues[i].put((copy.deepcopy(model), user, this_lr, this_alpha, epoch))
            else:
                num_of_data_clients.append(len(dataset[user]))

                local_update_class = None

                local_setting = LocalUpdate(args=args, lr=this_lr, local_epoch=args.local_epochs, device=device,
                                            batch_size=args.batch_size, dataset=trainset, idxs=dataset[user], alpha=this_alpha,
                                            testloader=testloader, user_index=user, participation_index=i,
                                            )
                weight, loss = local_setting.train(net=copy.deepcopy(model).to(device) , current_global_epoch = epoch   )
                #weight, loss_dict = local_setting.train(net=copy.deepcopy(model).to(device))
                local_weight.append(copy.deepcopy(weight))
                local_loss.append(copy.deepcopy(loss))
                delta = {}
                for key in weight.keys():
                    delta[key] = weight[key] - global_weight[key]

                # If you want to save gpu memory, make sure that local_delta not allocated to GPU
                local_delta.append(delta)
                client_ldr_train = DataLoader(DatasetSplit(trainset, dataset[user]), batch_size=args.batch_size, shuffle=True)

        if args.multiprocessing:
            for _ in range(len(selected_user)):
                # Retrieve results from the queue
                result = result_queue.get()
                weight, loss, delta, num_of_data = result
                local_weight.append(copy.deepcopy(weight))
                local_loss.append(copy.deepcopy(loss))
                local_delta.append(delta)
                num_of_data_clients.append(num_of_data)

        print("One round training time: ", time.time() - start)

        #log_local_sim
        if args.log_local_sim:
            length = len(local_delta)
            sim_all = []
            for key in local_delta[0].keys():
                global_delta = 0
                local_delta_layer = []
                for i in local_delta:
                    global_delta += i[key]
                    local_delta_layer.append(i[key].view(-1))
                global_delta /= length
                local_delta_layer = torch.stack(local_delta_layer, dim=0)

                global_delta = F.normalize(global_delta.view(-1).unsqueeze(0), 2, dim=1)
                local_delta_layer = F.normalize(local_delta_layer, 2, dim=1)
                sim = (local_delta_layer @ global_delta.T).squeeze().mean()
                sim_all.append(sim)
                wandb_dict[key + "_cosine_similarity"] = sim.item()
            sim_all = torch.stack(sim_all).mean()
            wandb_dict["all_layers_mean" + "_cosine_similarity"] = sim_all.item()                 

        ## Update Model
        total_num_of_data_clients=sum(num_of_data_clients)        
        FedAvg_weight = copy.deepcopy(local_weight[0])
        for key in FedAvg_weight.keys():
            for i in range(len(local_weight)):
                if i==0:
                    FedAvg_weight[key]*=num_of_data_clients[i]
                else:                       
                    FedAvg_weight[key] += local_weight[i][key]*num_of_data_clients[i]
            #print(key)
            FedAvg_weight[key] /= total_num_of_data_clients
        prev_model_weight = copy.deepcopy(model.state_dict())
        current_model_weight = copy.deepcopy(FedAvg_weight)
        model.load_state_dict(FedAvg_weight)
        
        if type(local_loss[0]) == dict:
            loss_avg = copy.deepcopy(local_loss[0])
            for idx,el in enumerate(local_loss):
                if idx == 0:
                    continue
                else:
                    for key in loss_avg.keys():
                        try:
                            loss_avg[key] += el[key]
                        except:
                            print("fail to aggregate :", key, ", in clien :", idx)
            for key in loss_avg.keys():
                loss_avg[key] /= len(local_loss)
            print(' num_of_data_clients : ',num_of_data_clients)
            for i, key in enumerate(loss_avg.keys()):
                if i < 5:
                    print(key, loss_avg[key])
                wandb_dict[args.mode + '_'+key]= loss_avg[key]    
        else:
            loss_avg = sum(local_loss) / len(local_loss)
            print(' Average loss {:.3f}'.format(loss_avg))
            wandb_dict[args.mode + '_loss']= loss_avg

        if args.analysis:
            checkpoint_path = './data/saved_model/fed/CIFAR10/centralized/Fedavg/_best.pth'
            cosinesimilarity=calculate_cosinesimilarity_from_optimal(args, checkpoint_path, current_model_weight, prev_model_weight)
            wandb_dict[args.mode + "_cosinesimilarity"] = cosinesimilarity
        if (args.CKA==True) and (epoch%args.CKA_freq==0):
            wandb_dict = cka_allmodels(prev_model_weight,FedAvg_weight,local_weight,model,wandb_dict,testloader,args,epoch)

        if (args.umap==True) and (epoch%args.umap_freq==0):
            wandb_dict = umap_allmodels(prev_model_weight,FedAvg_weight,local_weight,wandb_dict,model,testloader,args)

        if args.log_fisher:
            if epoch % args.log_fisher_freq == 0:
                log_fisher_diag(args, model, trainset, testloader, device, this_lr, wandb_dict)

        ## Evaluate
        if epoch % args.print_freq == 0:
            acc = evaluate(args, model, testloader, device)
            acc_train.append(acc)

        model.train()
        wandb_dict[args.mode + "_acc"]=acc_train[-1]

        if (args.log_test_metric) and (epoch % args.log_test_metric_freq == 0):
            prev_model = copy.deepcopy(model)
            prev_model.load_state_dict(prev_model_weight)

            local_model0 = copy.deepcopy(model)
            local_model0.load_state_dict(local_weight[0])
            
            local_model1 = copy.deepcopy(model)
            local_model1.load_state_dict(local_weight[1])

            model_list = [model, prev_model, local_model0, local_model1]
            model_name_list = ['g', 'p', "a", "b"]

            #G - Pg
            i,j = 0,1
            wandb_dict.update(__log_test_metric__(model_list[i],model_list[j],model_name_list[i],model_name_list[j], args, testloader, device))
            wandb_dict.update(__log_test_metric__(model_list[j],model_list[i],model_name_list[j],model_name_list[i], args, testloader, device))
            #G - L
            i,j = 0,2
            wandb_dict.update(__log_test_metric__(model_list[i],model_list[j],model_name_list[i],model_name_list[j], args, testloader, device))
            wandb_dict.update(__log_test_metric__(model_list[j],model_list[i],model_name_list[j],model_name_list[i], args, testloader, device))
            #Pg - L
            i,j = 1,2
            wandb_dict.update(__log_test_metric__(model_list[i],model_list[j],model_name_list[i],model_name_list[j], args, testloader, device))
            wandb_dict.update(__log_test_metric__(model_list[j],model_list[i],model_name_list[j],model_name_list[i], args, testloader, device))
            #L - L
            i,j = 2,3
            wandb_dict.update(__log_test_metric__(model_list[i],model_list[j],model_name_list[i],model_name_list[j], args, testloader, device))
            wandb_dict.update(__log_test_metric__(model_list[j],model_list[i],model_name_list[j],model_name_list[i], args, testloader, device))

        wandb_dict['lr']=this_lr
        wandb.log(wandb_dict, step = epoch)

        ## Update learning rate
        this_lr *= args.learning_rate_decay
        if args.alpha_mul_epoch == True:
            this_alpha = args.alpha * (epoch + 1)
        elif args.alpha_divide_epoch == True:
            this_alpha = args.alpha / (epoch + 1)

        if args.save_every >= 0 and epoch%args.save_every ==0:
            try:
                torch.save({'model_state_dict': model.state_dict()},
                                '{}/{}.pth'.format(args.LOG_DIR, args.mode + "_" + ((str(args.dirichlet_alpha) + "_") if args.mode!='iid' else "") +"_globalepoch"+str(epoch)))
                print('model saved, epoch : ', epoch)
            except:
                print("Fail to save model at "+ str(epoch) +"epoch. Keep running the code")

    #Terminate Processes
    terminate_processes(queues, processes)



def train(gpu, LocalUpdate, task_queue, result_queue, global_list):
    start = time.time()
    torch.cuda.set_device(gpu)
    while True:
        task = task_queue.get()
        if task is None:
            break

        model, user, this_lr, this_alpha, global_epoch = task
        args, dataset, trainset, testloader = global_list[0]

        # Initialize random seed only once
        if global_epoch == 0:
            initalize_random_seed(args)
        num_of_data = len(dataset[user])

        local_setting = LocalUpdate(args=args, lr=this_lr, local_epoch=args.local_epochs, device=gpu,
                                    batch_size=args.batch_size, dataset=trainset, idxs=dataset[user], alpha=this_alpha,
                                    testloader=testloader, user_index=user, participation_index=gpu,
                                    )
        weight, loss = local_setting.train(net=copy.deepcopy(model), current_global_epoch=global_epoch)

        delta = {}
        for key in weight.keys():
            delta[key] = weight[key] - model.state_dict()[key]
        result_queue.put((weight, loss, delta, num_of_data))
    end = time.time()
    print("Time for 1 client : ", end - start)


