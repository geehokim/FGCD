#!/usr/bin/env python
# coding: utf-8
import copy
import time

import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from sklearn.manifold import TSNE
from sklearn.cluster import BisectingKMeans

from utils import *
from utils.metrics import evaluate
from utils.visualize import __log_test_metric__, umap_allmodels, cka_allmodels, log_fisher_diag
from models import build_encoder, get_model
from typing import Callable, Dict, Tuple, Union, List
from utils import linear_assignment
from finch import FINCH
import wandb
import torch.nn.functional as F
import sys
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans


from servers.build import SERVER_REGISTRY


# Sinkhorn Knopp
def sknopp(cZ, lamd=25, max_iters=100):
    with torch.no_grad():
        N_samples, N_centroids = cZ.shape # cZ is [N_samples, N_centroids]
        probs = F.softmax(cZ * lamd, dim=1).T # probs should be [N_centroids, N_samples]

        r = torch.ones((N_centroids, 1), device=probs.device) / N_centroids # desired row sum vector
        c = torch.ones((N_samples, 1), device=probs.device) / N_samples # desired col sum vector

        inv_N_centroids = 1. / N_centroids
        inv_N_samples = 1. / N_samples

        err = 1e3
        for it in range(max_iters):
            r = inv_N_centroids / (probs @ c)  # (N_centroids x N_samples) @ (N_samples, 1) = N_centroids x 1
            c_new = inv_N_samples / (r.T @ probs).T  # ((1, N_centroids) @ (N_centroids x N_samples)).t() = N_samples x 1
            if it % 10 == 0:
                err = torch.nansum(torch.abs(c / c_new - 1))
            c = c_new
            if (err < 1e-2):
                break

        # inplace calculations.
        probs *= c.squeeze()
        probs = probs.T # [N_samples, N_centroids]
        probs *= r.squeeze()

        return probs * N_samples # Soft assignments



def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)
    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

######### Progress bar #########
term_width = 150 
TOTAL_BAR_LENGTH = 30.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')
    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)
    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')
    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))
    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

@SERVER_REGISTRY.register()
class Server():

    def __init__(self, args):
        self.args = args
        return

    def aggregate(self, local_weights, local_deltas, local_optimizer_state_dicts, client_ids, model_dict, local_centroids_list=None, local_labelled_centroids_list=None, local_labelled_class_set_list=None, current_lr=0, epoch=0,
                  local_novel_class_mask_list=None):
        C = len(client_ids)
        local_act_protos = [ i.item() for i in local_weights['proj_layer.act_protos'] ]
        local_prototypes = [ l_weight[:local_act_protos[i]] for i, l_weight in enumerate(local_weights['proj_layer.local_prototypes']) ]

        for param_key in local_weights:            
            local_weights[param_key] = sum(local_weights[param_key]) / C
        
        
        # for param_key in local_optimizer_state_dicts:
        #     momentum_buffers = []
        #     for state in local_optimizer_state_dicts[param_key]:
        #         momentum_buffers.append(state['momentum_buffer'])
        #     momentum_buffers = torch.stack(momentum_buffers)
        #     local_optimizer_state_dicts[param_key] = {'momentum_buffer': momentum_buffers.sum(dim=0) / C}

        return local_weights, local_prototypes, None

@SERVER_REGISTRY.register()
class ServerNovelClustering(Server):

    def __init__(self, args):
        super().__init__(args)

    def aggregate(self, local_weights, local_deltas, client_ids, global_model, local_novel_cluster_means_list=None, local_novel_cluster_targets_list=None, local_labelled_class_set_list=None, current_lr=0, epoch=0,
                  local_novel_class_mask_list=None):
        C = len(client_ids)
        local_classifier_weights = None
        for param_key in local_weights:
            if 'original1' in param_key:
                local_classifier_weights = copy.deepcopy(local_weights[param_key])
            local_weights[param_key] = sum(local_weights[param_key]) / C
        
        #if len(local_novel_cluster_means_list) > 0:
        if None not in local_novel_cluster_means_list:
            print("global clustering")
            # Filter out None values and concatenate only tensors
            valid_means = [means for means in local_novel_cluster_means_list if means is not None]
            valid_targets = [targets for targets in local_novel_cluster_targets_list if targets is not None]
            novel_cluster_means = torch.cat(local_novel_cluster_means_list, dim=0)
            novel_cluster_targets = torch.cat(local_novel_cluster_targets_list, dim=0)
            # global_clustering happens on the server
            if self.args.server.clustering_after_aggregation:
                global_model.global_centroids.weight.data.copy_(local_weights['proj_layer.last_layer.parametrizations.weight.original1'][len(self.args.dataset.seen_classes):])
                
            global_model.global_clustering(novel_cluster_means, current_lr)
            local_weights['global_centroids.weight'] = global_model.global_centroids.weight.data.clone()
            w = None
            # local_classifier_weights = torch.stack(local_classifier_weights)
            # feat_dim = local_classifier_weights.size(-1)
            # local_classifier_weights_novel = local_classifier_weights[:, len(self.args.dataset.seen_classes):, :].reshape(-1, feat_dim)
            # local_novel_class_mask = torch.cat(local_novel_class_mask_list, dim=0).cpu()
            # print('local_novel_class_mask: ', local_novel_class_mask)
            # filtered_classifier_weights = local_classifier_weights_novel[local_novel_class_mask]
            # global_model.global_clustering(filtered_classifier_weights, current_lr)
            # local_weights['global_centroids.weight'] = global_model.global_centroids.weight.data.clone()
            # w = None
        else:
            w = None

        return local_weights, w
    

    

@SERVER_REGISTRY.register()
class ServerClusteringClassifierWeights(Server):

    def __init__(self, args):
        super().__init__(args)

    def aggregate(self, local_weights, local_deltas, client_ids, global_model, local_novel_cluster_means_list=None, local_novel_cluster_targets_list=None, local_labelled_class_set_list=None, current_lr=0, epoch=0,
                  local_novel_class_mask_list=None):
        C = len(client_ids)
        local_classifier_weights = None
        for param_key in local_weights:
            if 'original1' in param_key:
                local_classifier_weights = copy.deepcopy(local_weights[param_key])
            local_weights[param_key] = sum(local_weights[param_key]) / C
        
        local_classifier_weights = torch.cat(local_classifier_weights, dim=0)
        global_model.global_clustering_all(local_classifier_weights)
        local_weights['global_centroids_all.weight'] = global_model.global_centroids_all.weight.data.clone()
        w = None

        return local_weights, w
    
    

@SERVER_REGISTRY.register()
class CCServer():

    def __init__(self, args):
        self.args = args
        self.server_config = args.server
        return

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, epoch=0):
        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key]) / C
        return local_weights

    def aggregate_local_labelled_centroids(self, local_weights, local_labelled_centroids_list, client_ids):

        # local_labelled_centroids
        all_local_labelled_centroids = {}
        for i in range(len(self.args.dataset.seen_classes)):
            all_local_labelled_centroids[i] = []
        for local_labelled_centroids in local_labelled_centroids_list:
            for class_id in local_labelled_centroids:
                all_local_labelled_centroids[class_id].append(local_labelled_centroids[class_id])
        aggregated_labelled_centroids = []
        for i in range(len(self.args.dataset.seen_classes)):
            total = 0
            tmp_feat = 0
            for (num, feat) in all_local_labelled_centroids[i]:
                total += num
                tmp_feat += num * feat
            tmp_feat = tmp_feat / total
            aggregated_labelled_centroids.append(tmp_feat)
        aggregated_labelled_centroids = torch.stack(aggregated_labelled_centroids, dim=0)

        return aggregated_labelled_centroids

    def get_local_centroids(self, local_weights, local_centroids_list, client_ids):

        all_local_centroids = []
        for local_centroids in local_centroids_list:
            cts = list(local_centroids.values())
            cts = torch.stack(cts, dim=0)
            all_local_centroids.append(cts)

        all_local_centroids = torch.cat(all_local_centroids, dim=0)
        return all_local_centroids

    def aggregate_centroids(self, local_weights, all_local_centroids, aggregated_local_labelled_centroids,
                                                                client_ids):
        num_classes = len(self.args.dataset.seen_classes) + len(self.args.dataset.unseen_classes)
        if self.server_config.centroid_aggre_type == 'local_centroids_SK':
            target_centroids = nn.Linear(768, num_classes)
            N = all_local_centroids.shape[0]  # Z has dimensions [m_size * n_clients, D]
            # Optimizer setup
            optimizer = torch.optim.SGD(target_centroids.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            train_loss = 0.
            total_rounds = 500
            angular_criterion = AngularPenaltySMLoss()
            for round_idx in range(total_rounds):
                with torch.no_grad():
                    # Cluster assignments from Sinkhorn Knopp
                    SK_assigns = sknopp(target_centroids(all_local_centroids))
                    # print(SK_assigns.size())
                    # print(SK_assigns)
                # Zero grad
                optimizer.zero_grad()
                # Predicted cluster assignments [N, N_centroids] = local centroids [N, D] x global centroids [D, N_centroids]
                probs1 = F.softmax(target_centroids(F.normalize(all_local_centroids, dim=1)) / 0.1, dim=1)
                ## 增加 Prototype距离 ##
                # cos_output = self.centroids(F.normalize(Z1, dim=1))
                # SK_target = np.argmax(SK_assigns.cpu().numpy(), axis=1)
                # angular_loss = angular_criterion(cos_output, SK_target)
                # print("angular_loss: ", angular_loss)
                ######################
                # Match predicted assignments with SK assignments
                cos_loss = F.cosine_similarity(SK_assigns, probs1, dim=-1).mean()
                loss = - cos_loss  # + angular_loss
                print("F.cosine_similarity: ", cos_loss)
                # Train
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    # self.centroids.weight.copy_(self.centroids.weight.data.clone()) # Not Normalize centroids
                    target_centroids.weight.copy_(F.normalize(target_centroids.weight.data.clone(), dim=1))  # Normalize centroids
                    train_loss += loss.item()
            result_centroids = target_centroids.weight.data.clone()
        elif self.server_config.centroid_aggre_type == 'local_centroids_finch':
            c, num_clust, req_c = FINCH(all_local_centroids.numpy(), req_clust=num_classes, distance='cosine', verbose=False)
            new_cws = {}
            for i in range(num_classes):
                new_cws[i] = []
            for i, idx in enumerate(req_c):
                new_cws[idx].append(all_local_centroids[i])
            aggregated_cws = []
            for i in new_cws:
                mean_centroids = torch.stack(new_cws[i]).mean(0)
                aggregated_cws.append(mean_centroids)

            aggregated_cws = torch.stack(aggregated_cws)
            aggregated_cws = F.normalize(aggregated_cws, dim=-1)
            result_centroids = aggregated_cws
        else:
            raise NotImplementedError

        aggregated_local_labelled_centroids = F.normalize(aggregated_local_labelled_centroids, dim=1)
        sim_mat = torch.matmul(aggregated_local_labelled_centroids.cpu(), result_centroids.T)
        row_ind, col_ind = linear_assignment(sim_mat.max() - sim_mat)

        # 주어진 인덱스
        indices = torch.tensor(col_ind)

        # 주어진 인덱스에 해당하는 행을 선택
        selected_rows = result_centroids[indices]

        # 나머지 인덱스 생성
        remaining_indices = torch.tensor([i for i in range(num_classes) if i not in indices])

        # 나머지 행 선택
        remaining_rows = result_centroids[remaining_indices]

        # 선택된 행과 나머지 행을 연결
        aligned_centroids = torch.cat([selected_rows, remaining_rows], dim=0)
        return aligned_centroids


    

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
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, epoch=0):
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
class FedACG_evalpoint_accel(Server):    
    
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
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, epoch=0):
        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C
        if self.args.server.momentum>0:
            # print("self.args.server.get('FedACG'): ",self.args.server.get('FedACG'))

 
                    
            for param_key in local_deltas:
                self.global_delta[param_key] = sum(local_deltas[param_key])/C
                self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + self.global_delta[param_key]


            for param_key in local_weights:               
                local_weights[param_key] += self.args.server.momentum * self.global_momentum[param_key]   

        return local_weights


@SERVER_REGISTRY.register()
class SNAG(Server):    
    
    def set_momentum(self, model):

        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])


        past_delta = copy.deepcopy(model.state_dict())
        for key in past_delta.keys():
            past_delta[key] = torch.zeros_like(past_delta[key])


        self.global_delta = global_delta
        self.global_momentum = global_momentum
        self.past_delta = past_delta

    @torch.no_grad()
    def FedACG_lookahead(self, model):
        sending_model_dict = copy.deepcopy(model.state_dict())
        for key in self.global_momentum.keys():
            sending_model_dict[key] += self.args.server.momentum * self.global_momentum[key]

        model.load_state_dict(sending_model_dict)
        return copy.deepcopy(model)
    
    @torch.no_grad()
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, epoch=0):
        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C
        if self.args.server.momentum>0:
            # print("self.args.server.get('FedACG'): ",self.args.server.get('FedACG'))
                    
            for param_key in local_deltas:
                self.global_delta[param_key] = sum(local_deltas[param_key])/C
                if self.args.server.get('MNAG'):
                    self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + self.global_delta[param_key] * (1 - self.args.server.momentum) + self.past_delta[param_key] * self.args.server.momentum
                else:
                    self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + self.global_delta[param_key] * (1 + self.args.server.momentum) - self.past_delta[param_key] * self.args.server.momentum
                
                #print diff between past and current delta
                print("diff between past and current delta: ", torch.norm(self.global_delta[param_key] - self.past_delta[param_key]))
                self.past_delta[param_key] = (self.global_delta[param_key]).detach().clone()
            
            if not self.args.server.get('FedACG'): 
                for param_key in local_weights:               
                    local_weights[param_key] += self.global_momentum[param_key] - self.global_delta[param_key]

        return local_weights


@SERVER_REGISTRY.register()
class SNAG_check(Server):    
    
    def set_momentum(self, model):

        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])


        past_delta = copy.deepcopy(model.state_dict())
        for key in past_delta.keys():
            past_delta[key] = torch.zeros_like(past_delta[key])


        self.global_delta = global_delta
        self.global_momentum = global_momentum
        self.past_delta = past_delta

    @torch.no_grad()
    def FedACG_lookahead(self, model):
        sending_model_dict = copy.deepcopy(model.state_dict())
        for key in self.global_momentum.keys():
            sending_model_dict[key] += self.args.server.momentum * self.global_momentum[key]

        model.load_state_dict(sending_model_dict)
        return copy.deepcopy(model)
    
    @torch.no_grad()
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C
        if self.args.server.momentum>0:
            # print("self.args.server.get('FedACG'): ",self.args.server.get('FedACG'))
                    
            for param_key in local_deltas:
                self.global_delta[param_key] = sum(local_deltas[param_key])/C
                if self.args.server.get('MNAG'):
                    self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + self.global_delta[param_key] * (1 - self.args.server.momentum) + self.past_delta[param_key] * self.args.server.momentum
                else:
                    self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + self.global_delta[param_key]
                
                # #print diff between past and current delta
                # print("diff between past and current delta: ", torch.norm(self.global_delta[param_key] - self.past_delta[param_key]))
                # self.past_delta[param_key] = (self.global_delta[param_key]).detach().clone()
            
            if not self.args.server.get('FedACG'): 
                for param_key in local_weights:               
                    local_weights[param_key] += self.global_momentum[param_key] * self.args.server.momentum

        return local_weights


@SERVER_REGISTRY.register()
class ServerDyn(Server):    
    
    def set_momentum(self, model):
        #global_momentum is h^t in FedDyn paper
        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])


        self.global_delta = global_delta
        self.global_momentum = global_momentum

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        #breakpoint()
        C = len(client_ids)
        for param_key in self.global_momentum:
            self.global_momentum[param_key] -= self.args.client.dyn.alpha / self.args.trainer.num_clients * sum(local_deltas[param_key])
            local_weights[param_key] = sum(local_weights[param_key])/C - 1/self.args.client.dyn.alpha * self.global_momentum[param_key]
        #print("self.args.num_clients: ",self.args.trainer.num_clients )
        #print(self.args.num_clients)
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
        server_lr = self.args.trainer.global_lr
        #model_dict = model.state_dict()
        print("server_lr:", server_lr)
        for param_key in local_deltas:
            self.global_delta[param_key] = sum(local_deltas[param_key])/C
            self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + (1-self.args.server.momentum) * self.global_delta[param_key]
            self.global_v[param_key] = self.args.server.beta * self.global_v[param_key] + (1-self.args.server.beta) * (self.global_delta[param_key] * self.global_delta[param_key])

        for param_key in model_dict.keys():
            model_dict[param_key] += server_lr *  self.global_momentum[param_key] / ( (self.global_v[param_key]**0.5) + self.args.server.tau)
            
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


