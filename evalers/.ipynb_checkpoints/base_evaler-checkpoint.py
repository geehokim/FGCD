from pathlib import Path
from typing import Callable, Dict, Tuple, Union, List, Type
from argparse import Namespace
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import tqdm
import wandb
import gc

import pickle, os
import numpy as np

import logging
logger = logging.getLogger(__name__)


import time, io, copy

from evalers.build import EVALER_REGISTRY

from servers import Server
from clients import Client

from utils import DatasetSplit, get_dataset
from utils.logging_utils import AverageMeter

from torch.utils.data import DataLoader

from utils import terminate_processes, initalize_random_seed, cal_cos, get_local_classes
from omegaconf import DictConfig

import umap.umap_ as umap
from mlxtend.plotting import plot_confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt

import netcal
from netcal.metrics import ECE
from netcal import metrics, scaling, presentation
from matplotlib.backends.backend_pdf import PdfPages


def all_gather_nd(x):
    return x

# @EVALER_REGISTRY.register()
# class Evaler():

    # def __init__(self,
    #              test_loader: torch.utils.data.DataLoader,
    #             device: torch.device,
    #             args: DictConfig,
    #             distance_metric: str = 'cosine',
    #             **kwargs) -> None:

    #     self.args = args
    #     self.device = device

    #     # self.gallery_loader = gallery_loader
    #     # self.query_loader = query_loader
    #     self.test_loader = test_loader



@EVALER_REGISTRY.register()
class Evaler():

    def __init__(self,
                 test_loader: torch.utils.data.DataLoader,
                device: torch.device,
                args: DictConfig,
                gallery_loader: torch.utils.data.DataLoader = None,
                query_loader: torch.utils.data.DataLoader = None,
                distance_metric: str = 'cosine',
                **kwargs) -> None:

        self.args = args
        self.device = device

        self.test_loader = test_loader
        self.gallery_loader = gallery_loader
        self.query_loader = query_loader
        self.criterion = nn.CrossEntropyLoss(reduction = 'none')

    # Setting test loaders for evaluating local loss
    def set_local_test_loaders(self):
        return

    @torch.no_grad()
    def eval(self, model: nn.Module, epoch: int, device: torch.device = None, **kwargs) -> Dict:
        # eval_device = self.device if not self.args.multiprocessing else torch.device(f'cuda:{self.args.main_gpu}')

        model.eval()
        model_device = next(model.parameters()).device
        if device is None:
            device = self.device
        model.to(device)
        #print(device)
        loss, correct, total = 0, 0, 0

        if type(self.test_loader.dataset) == DatasetSplit:
            C = len(self.test_loader.dataset.dataset.classes)
        else:
            C = len(self.test_loader.dataset.classes)

        class_loss, class_correct, class_total = torch.zeros(C), torch.zeros(C), torch.zeros(C)
        # saved_preds, saved_labels = None, None

        entropies = []

        logits_all, labels_all = [], []
        #print(device)


        with torch.no_grad():
            #for images, labels in self.loaders["test"]:
            #print(f'total_iteration: {len(self.test_loader)}')
            for idx, (images, labels) in enumerate(self.test_loader):
                #print(f'iteration: {idx}')
                images, labels = images.to(device), labels.to(device)

                results = model(images)
                _, predicted = torch.max(results["logit"].data, 1) # if errors occur, use ResNet18_base instead of ResNet18_GFLN
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                bin_labels = labels.bincount()
                # print(class_total.size())
                # print(bin_labels.size())
                class_total[:bin_labels.size(0)] += bin_labels.cpu()
                bin_corrects = labels[(predicted == labels)].bincount()
                class_correct[:bin_corrects.size(0)] += bin_corrects.cpu()


                this_loss = self.criterion(results["logit"], labels)
                loss += this_loss.sum().cpu()

                for class_idx, bin_label in enumerate(bin_labels):
                    class_loss[class_idx] += this_loss[(labels.cpu() == class_idx)].sum().cpu()


                logits_all.append(results["logit"].data.cpu())
                labels_all.append(labels.cpu())
                # print('test processing')

                # score = F.softmax(results["logit"], 1).detach()
                # entropy = torch.distributions.Categorical(score).entropy()
                # uniform_entropy = torch.distributions.Categorical(torch.ones_like(score)).entropy()
                # entropy_ = entropy / uniform_entropy
                # entropies.append(entropy_.cpu().numpy())


                # if idx==0:
                #     saved_labels = (labels.cpu())
                #     saved_preds = (predicted.cpu())
                # else:
                #     saved_labels = torch.cat((saved_labels,labels.cpu()),  dim=0)
                #     saved_preds = torch.cat((saved_preds,predicted.cpu()), dim=0)

        logits_all = torch.cat(logits_all)
        labels_all = torch.cat(labels_all)
        # saved_labels = saved_labels.tolist()
        # saved_preds = saved_preds.tolist()

        scores = F.softmax(logits_all, 1)
        entropy = torch.distributions.Categorical(scores).entropy().mean()
        uniform_entropy = torch.distributions.Categorical(torch.ones_like(scores)).entropy().mean()
        entropy_ = entropy / uniform_entropy
        print('entropy processing')

        confidences, pred_labels = scores.max(1)
        corrects = (pred_labels == labels_all)


        n_bins = 10
        ece = ECE(n_bins, detection=True)
        # miscalibration = ece.measure(calibrated, matched, uncertainty="mean")
        ece_score = ece.measure(confidences.numpy(), corrects.numpy())

            
        # entropy = np.mean(entropies)

        acc = 100. * correct / float(total)
        class_acc = 100. * class_correct / class_total
        
        loss = loss / float(total)
        class_loss = class_loss / class_total

        # if self.args.get('debugs'):
        diagram = netcal.presentation.ReliabilityDiagram(15)
        # diagram.plot(confidences.numpy(), corrects.numpy()).savefig(self.checkpoint_dir / 'ece_{}_e{:04d}.png'.format(desc, epoch))
        ece_diagram = diagram.plot(confidences.numpy(), corrects.numpy())
            # breakpoint()

        # logger.warning(f'[Epoch {epoch}] Test Accuracy: {acc:.2f}%')
        model.to(model_device)
        model.train()
        results = {
            "acc": acc,
            'class_acc': class_acc,
            'loss': loss,
            'class_loss' : class_loss,
            'entropy': entropy_,
            'ece': ece_score,
            'ece_diagram': ece_diagram,
        }
        # if self.args.get('wandb'):
        #     results["confusion_matrix"] = wandb.plot.confusion_matrix(probs=None,
        #                 y_true=saved_labels, preds=saved_preds,
        #                 class_names=range(len(self.test_loader.dataset.classes)))
        
        return results


    #----- UMAP ----#
    @torch.no_grad()
    def visualize_umap(self, global_model: nn.Module, local_models: List[nn.Module], local_datasets: List[torch.utils.data.Dataset], epoch: int):

        #num_of_sample_per_class: int = 100, draw_classes: int = -1
        test_loader = self.test_loader
        classes = test_loader.dataset.classes

        # if self.args.get('debug_test_loader'):
        #     test_loader =  DataLoader(local_datasets[0], batch_size=self.args.batch_size, shuffle=False,
        #                            num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)

        
        
        # names_list = ["global", "local0", "local1"]
        names_list = ["global", "local"]
        local_model0 = copy.deepcopy(global_model)
        local_model0.load_state_dict(local_models[0])
        # local_model1 = copy.deepcopy(global_model)
        # local_model1.load_state_dict(local_models[1])
        
        # models_list = [global_model] + [local_model0] + [local_model1]
        models_list = [global_model,local_model0 ]# + [local_model0]
        eval_result_dict = {}
        for name in names_list:
            eval_result_dict[name] = {}
            eval_result_dict[name]['saved_features'] = {}
            eval_result_dict[name]['labels_num_for_each_class'] = {}
            eval_result_dict[name]['saved_embeddings'] = {}

            for class_idx, each_class in enumerate(classes):
                eval_result_dict[name]['saved_features'][str(class_idx)] = torch.tensor([])
                eval_result_dict[name]['labels_num_for_each_class'][str(class_idx)] = 0
                eval_result_dict[name]['saved_embeddings'][str(class_idx)] = torch.tensor([])


        drawing_options = [
            # [True, False, False],
            # [False, True, False],
            # [False, False, True]
            [True, True],
            [True, False],
            [False, True],
            #[True]
        ]

        umap_args = self.args.umap

        
        # assert umap_args.samples_per_class <= float(len(test_loader.dataset)/len(classes))
        color_cycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
        marker_list = ['o','P','X','^']* 100
        
        opacity_max = 1
        opacity_min = 0.2
        
        wandb_dict = {}


        if umap_args.draw_classes < 0 :
            draw_classes = len(classes)
        else:
            draw_classes = min(umap_args.draw_classes,len(classes))

        for name, model in zip(names_list, models_list):
            results = self.extract_features(model=model, loader=test_loader)

            this_features = results['features']
            this_labels = results['labels']

            # if self.args.get('debug_test_loader'):
            #     this_features = results['layer4']
                # breakpoint()


            for feat, label in zip(this_features, this_labels):
                if label < draw_classes:
                    if self.args.get('debug_test_loader') or eval_result_dict[name]['labels_num_for_each_class'][str(label.item())] < umap_args.samples_per_class:
                        eval_result_dict[name]['saved_features'][str(label.item())] = torch.cat((eval_result_dict[name]['saved_features'][str(label.item())], feat.cpu().reshape(1,feat.size(0))), dim = 0)
                        eval_result_dict[name]['labels_num_for_each_class'][str(label.item())] += 1


        #post_processing
        all_features = torch.tensor([])
        num_of_samples_for_each_model = {}
        #end_idx = 0
        for name in names_list:
            num_of_samples_for_each_model[name] = 0
            for class_idx, each_class in enumerate(classes):
                all_features= torch.cat((all_features, eval_result_dict[name]['saved_features'][str(class_idx)]))
                num_of_samples_for_each_model[name] += eval_result_dict[name]['labels_num_for_each_class'][str(class_idx)]
        
        reducer = umap.UMAP(random_state=0, n_components=umap_args.umap_dim, metric='cosine', n_neighbors=umap_args.n_neighbors, min_dist = umap_args.min_dist, spread=umap_args.spread)
        embedding = reducer.fit_transform((all_features))

        idx = 0
        for name in names_list:
            this_model_embeddings = embedding[idx: idx+num_of_samples_for_each_model[name]]
            this_class_idx_start = 0
            for class_idx, each_class in enumerate(classes):
                eval_result_dict[name]['saved_embeddings'][str(class_idx)] = this_model_embeddings[this_class_idx_start:this_class_idx_start + eval_result_dict[name]['labels_num_for_each_class'][str(class_idx)]]
                this_class_idx_start += eval_result_dict[name]['labels_num_for_each_class'][str(class_idx)]
                # print(name, class_idx)
                # print(eval_result_dict[name]['saved_embeddings'][str(class_idx)].shape)
            idx += num_of_samples_for_each_model[name]
        # embedding_seperate_model = [embedding[len(sorted_label)*j:len(sorted_label) *(j+1)] for j in range(len(models_dict_list))]


        ##################### plot ground truth #######################

        for drawing_option in drawing_options:
            all_names = "umap"
            for model_option, name in zip(drawing_option,names_list):
                if model_option:
                    all_names += "_" + str(name)

            f = plt.figure(figsize=(10, 10))

            # if args.umap_dim == 3:
            #     ax = plt.axes(projection=('3d'))
            # else:
            #     ax = plt.axes()
            this_draw_num = float(sum(drawing_option))
            this_opacity_gap = (opacity_max-opacity_min)/max((this_draw_num - 1),1)
            #breakpoint()
            for i in range(draw_classes):
                first = True
                #y_i = (y_test == i)
                count = -1
                for j in range(len(drawing_option)):
                    if drawing_option[j]:
                        try:
                            this_name = names_list[j]
                            count += 1
                            this_embedding = eval_result_dict[this_name]['saved_embeddings'][str(i)]
                            #scatter_input = [this_embedding[y_i, k] for k in range(args.umap_dim)]
                            plt.scatter(this_embedding[:,0],this_embedding[:,1], color = color_cycle[i], marker =marker_list[count], alpha = 0.2)#opacity_max - this_opacity_gap*count) #, label=classes[i] if first else None
                            plt.xticks([])  # Remove x-axis ticks
                            plt.yticks([])  # Remove y-axis ticks 
                            first = False
                        except:
                            breakpoint()
            #plt.legend(loc=4)
            plt.gca().invert_yaxis()
            plt.box(False)
            fig_name = all_names
            wandb_dict[fig_name] = wandb.Image(plt) #filedir +args.set + args.mode + args.additional_experiment_name+this_name+'.png')
            os.makedirs("umap", exist_ok=True)
            #plt.savefig(f"checkpoints/umap/{self.args.exp_name}.pdf", bbox_inches='tight', pad_inches=0)
            #breakpoint()
            pp = PdfPages(f"umap/{self.args.exp_name + fig_name}.pdf")
            pp.savefig(f, bbox_inches = 'tight', pad_inches = 0.0)
            pp.close()
            plt.close()
    
        return wandb_dict


    @torch.no_grad()
    def visualize_prototype(self, epoch: int, local_models: List[nn.Module], global_model: nn.Module, local_datasets: List[torch.utils.data.Dataset], device: torch.device = None, **kwargs):
        stride = self.args.svd.stride
        logger.info("Visualize_prototype")
        wandb_dict = {}
        global_model.eval()

        model_device = next(global_model.parameters()).device
        if device is None:
            device = self.device
        global_model.to(device)


        # local_model, local_dataset = local_models[0].eval(), local_datasets[0]
        # local_model.to(device)


        

        global_feature_results, local_feature_results = defaultdict(list), {}
        for idx, local_model in enumerate(local_models):
            local_model.eval()
            local_model.to(device)
            local_feature_results[idx] = (defaultdict(list))

        labels = []

        num_classes = len(self.test_loader.dataset.classes)
        num_samples = len(self.test_loader.dataset)
        num_samples_per_class = int(num_samples / num_classes / stride)

        with torch.no_grad():
            # for images, labels in self.loaders["test"]:
            for images, label in self.test_loader:
                images, label = images.to(device), label.to(device)

                labels.append(label)
                global_results = global_model(images)
                

                for key in global_results:
                    feat = global_results[key]

                    if len(feat.shape) == 4:
                        # if "flatten_feature" in self.args.analysis_mode:
                        #     feat = feat.view(feat.shape[0], -1)      
                        # else:
                        #     feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                        feat = feat.view(feat.shape[0], -1)   

                    # if "flatten_feature" in self.args.analysis_mode:
                    #     feat = feat.view(feat.shape(0), -1)      
                    # else:
                    #     if len(feat.shape) == 4:
                    #         feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                    global_feature_results[key].append(feat.cpu())

                for idx, local_model in enumerate(local_models):

                    local_results = local_model(images)
                    for key in local_results:
                        feat = local_results[key]

                        if len(feat.shape) == 4:
                            # if "flatten_feature" in self.args.analysis_mode:
                            #     feat = feat.view(feat.shape[0], -1)      
                            # else:
                            #     feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                            feat = feat.view(feat.shape[0], -1)   

                        # if "flatten_feature" in self.args.analysis_mode:
                        #     feat = feat.view(feat.shape(0), -1)  
                        # else:                    
                        #     if len(feat.shape) == 4:
                        #         feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                        local_feature_results[idx][key].append(feat.cpu())

            labels = torch.cat(labels)
            idxs = labels.argsort()
            labels = labels[idxs]
            for key in global_feature_results:
                global_feature_results[key] = torch.cat(global_feature_results[key])[idxs, :][::stride]
            for idx in local_feature_results:
                el = local_feature_results[idx]
                for key in el:
                    el[key] = torch.cat(el[key])[idxs, :][::stride]


            labels = labels[::stride]
    


        norm_of_mean_of_local_prototypes = []
        var_of_local_prototypes = []
        var_of_mean_of_local_prototypes = 0
        local_prototypes = []
        local_prototypes_means = []
        local_var_sample_prototype = []

        global_prototype, global_prototype_mean, global_var_sample_prototype= self._get_prototypes(global_feature_results['feature'].cpu(), labels, num_classes)
        norm_of_mean_of_global_prototypes = torch.norm(global_prototype_mean)
        var_of_global_prototypes = torch.norm((global_prototype - global_prototype_mean), dim = 1).mean()


        for idx in local_feature_results:
            el = local_feature_results[idx]
            this_local_prototype, this_local_prototype_mean, this_local_var_sample_prototype = self._get_prototypes(el['feature'].cpu(), labels, num_classes)
            assert(len(this_local_prototype) == num_classes)
            local_prototypes.append(this_local_prototype)
            local_prototypes_means.append(this_local_prototype_mean)
            
            
            local_var_sample_prototype.append(this_local_var_sample_prototype)
            norm_of_mean_of_local_prototypes.append(torch.norm(this_local_prototype_mean))
            var_of_local_prototypes.append(torch.norm((this_local_prototype - this_local_prototype_mean), dim = 1).mean())

        local_prototypes_means = torch.cat(local_prototypes_means)
        mean_local_prototypes_mean = local_prototypes_means.mean(dim=0, keepdim=True)
        var_of_mean_of_local_prototypes = torch.norm((local_prototypes_means - mean_local_prototypes_mean), dim =1).mean()
        #breakpoint()
        wandb_dict['norm_of_mean_of_global_prototypes/feature'] = norm_of_mean_of_global_prototypes
        wandb_dict['var_of_global_prototypes/feature'] = var_of_global_prototypes
        wandb_dict['global_var_sample_prototype/feature'] = global_var_sample_prototype

        wandb_dict['norm_of_mean_of_local_prototypes/feature'] = sum(norm_of_mean_of_local_prototypes)/len(norm_of_mean_of_local_prototypes)
        wandb_dict['var_of_local_prototypes/feature'] = sum(var_of_local_prototypes)/len(var_of_local_prototypes)
        wandb_dict['var_of_mean_of_local_prototypes/feature'] = var_of_mean_of_local_prototypes
        wandb_dict['local_var_sample_prototype/feature'] = sum(local_var_sample_prototype) / len(local_var_sample_prototype)

        

        #umap class prototype
        # color_cycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
        # umap_args = self.args.umap

        # reducer = umap.UMAP(random_state=0, n_components=umap_args.umap_dim, metric='cosine')
        # embedding = reducer.fit_transform(torch.cat(local_prototypes))
        # plt.figure(figsize=(50, 50))
        

        # for idx in range(len(local_feature_results)):
        #     this_embedding = embedding[num_classes*idx : num_classes*(idx+1)]
        #     for class_idx, prototype_embedding in enumerate(this_embedding):
        #         label = str(idx) + '_' + str(class_idx)
        #         #breakpoint()
        #         plt.text(this_embedding[class_idx,0],this_embedding[class_idx,1], label, color = color_cycle[idx]) #, label=classes[i] if first else None
        #         plt.xticks([])  # Remove x-axis ticks
        #         plt.yticks([])  # Remove y-axis ticks 

        #     #breakpoint()
        #     proto_mean_embedding = this_embedding.mean(axis=0)
        #     label = str(idx) + '_mean'
        #     #breakpoint()
        #     plt.text(proto_mean_embedding[0],proto_mean_embedding[1], label ,color = color_cycle[idx]) #, label=classes[i] if first else None
        #     plt.xticks([])  # Remove x-axis ticks
        #     plt.yticks([])  # Remove y-axis ticks 

        # plt.legend(loc=4)
        # plt.gca().invert_yaxis()
        # wandb_dict['Umap_clients_prototypes'] = wandb.Image(plt) #filedir +args.set + args.mode + args.additional_experiment_name+this_name+'.png')
        # plt.close()

        global_model.to(model_device)
        global_model.train()

        for local_model in local_models:
            local_model.to(model_device)
            local_model.train()

        return wandb_dict


    @torch.no_grad()
    def _get_prototypes(self, features, labels, num_classes):
        prototypes = torch.zeros((num_classes, features.size(1)))
        for i in range(num_classes):
            prototypes[i] = features[labels==i].mean(0)

        valid_num_classes = labels.bincount().nonzero().size(0)
        prototypes_ = prototypes[~prototypes[:, 0].isnan(), :]
        valid_num_classes = prototypes_.size(0)
        valid_class_counts = labels.bincount()[labels.bincount().nonzero()]
        prototypes_mean =  prototypes_.mean(0, keepdim=True)

        var_sample_prototype = torch.zeros((num_classes))
        for i in range(num_classes):
            var_sample_prototype[i] = torch.norm((features[labels==i] - prototypes[i]), dim =1).mean()
        var_sample_prototype = var_sample_prototype.mean()

        return prototypes_, prototypes_mean, var_sample_prototype

    



    # ----- SVD, feature norm, class norm ----#
    @torch.no_grad()
    def visualize_svd(self, epoch: int, local_models: List[nn.Module], global_model: nn.Module, local_datasets: List[torch.utils.data.Dataset], device: torch.device = None):

        stride = self.args.svd.stride
        
        logger.info("Visualize SVD")
        wandb_dict = {}
        global_model_ = copy.deepcopy(global_model)
        global_model_.eval()

        model_device = next(global_model_.parameters()).device
        if device is None:
            device = self.device
        global_model_.to(device)



        local_model_dict, local_dataset = local_models[0], local_datasets[0]
        local_model = copy.deepcopy(global_model)
        local_model.load_state_dict(local_model_dict)
        local_model.to(device)
        local_model.eval()

        global_feature_results, local_feature_results = defaultdict(list), defaultdict(list)
        labels = []

        num_classes = len(self.test_loader.dataset.classes)
        num_samples = len(self.test_loader.dataset)
        num_samples_per_class = int(num_samples / num_classes / stride)

        with torch.no_grad():
            # for images, labels in self.loaders["test"]:
            for images, label in self.test_loader:
                images, label = images.to(device), label.to(device)

                labels.append(label)
                global_results = global_model_(images)
                local_results = local_model(images)

                for key in global_results:
                    feat = global_results[key]

                    if len(feat.shape) == 4:
                        if "flatten_feature" in self.args.analysis_mode:
                            feat = feat.view(feat.shape[0], -1)      
                        else:
                            feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)


                    # if "flatten_feature" in self.args.analysis_mode:
                    #     feat = feat.view(feat.shape(0), -1)      
                    # else:
                    #     if len(feat.shape) == 4:
                    #         feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                    global_feature_results[key].append(feat.cpu())

                for key in local_results:
                    feat = local_results[key]

                    if len(feat.shape) == 4:
                        if "flatten_feature" in self.args.analysis_mode:
                            feat = feat.view(feat.shape[0], -1)      
                        else:
                            feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)


                    # if "flatten_feature" in self.args.analysis_mode:
                    #     feat = feat.view(feat.shape(0), -1)  
                    # else:                    
                    #     if len(feat.shape) == 4:
                    #         feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                    local_feature_results[key].append(feat.cpu())

            labels = torch.cat(labels)
            idxs = labels.argsort()
            labels = labels[idxs]

            for key in global_feature_results:
                global_feature_results[key] = torch.cat(global_feature_results[key])[idxs, :][::stride]
            for key in local_feature_results:
                local_feature_results[key] = torch.cat(local_feature_results[key])[idxs, :][::stride]


            labels = labels[::stride]

            # for key in ['layer4', 'feature']:
                
            #     ug, sg, vg = torch.svd(torch.cov(global_feature_results[key].T))
            #     ul, sl, vl = torch.svd(torch.cov(local_feature_results[key].T))

            #     corresponding_angle = F.cosine_similarity(ug, ul)
            #     print(f"{key}, top10: {corresponding_angle[:10].mean()}, bottom10: {corresponding_angle[-10:].mean()}")

            # if not self.args.get('debugs'):
            global_singular_results = self._get_singular_values(global_feature_results, labels=labels, desc='global', epoch=epoch)
            local_singular_results = self._get_singular_values(local_feature_results, labels=labels, local_dataset=local_dataset, desc='local', epoch=epoch)
            wandb_dict.update(global_singular_results)
            wandb_dict.update(local_singular_results)

            if "flatten_feature" in self.args.analysis_mode:
                zero_mean = ('zero_mean' in self.args.analysis_mode)
                print("Prototype 0 mean? :", zero_mean)
                global_raw_features_prototype_cossim_results = self._get_raw_features_prototype_cossim(global_feature_results, labels=labels, desc='global', epoch=epoch, zero_mean = zero_mean)
                local_raw_features_prototype_cossim_results = self._get_raw_features_prototype_cossim(local_feature_results, labels=labels, local_dataset=local_dataset, desc='local', epoch=epoch, zero_mean = zero_mean)

                wandb_dict.update(global_raw_features_prototype_cossim_results)
                wandb_dict.update(local_raw_features_prototype_cossim_results)


            
            norm_gf = global_feature_results['feature'].view(num_classes, num_samples_per_class, -1).norm(dim=2).mean(1)
            norm_lf = local_feature_results['feature'].view(num_classes, num_samples_per_class, -1).norm(dim=2).mean(1)

            classes = range(num_classes)
            data = list(zip(classes, norm_gf))
            wandb_dict["feature_norm/global"] = plot_bar(data, "label", "value", "feature norm global_model")

            data = list(zip(classes, norm_lf))
            wandb_dict["feature_norm/local"] = plot_bar(data, "label", "value", "feature norm local_model")

            g_s = global_model_.state_dict()
            l_s = local_model.state_dict()
            for key in g_s.keys():
                norm_g = g_s[key].norm()
                norm_l = l_s[key].norm()
                wandb_dict[f"weight_norm/global/{key}"] = norm_g
                wandb_dict[f"weight_norm/local/{key}"] = norm_l



        #### Local dataset
        # local_loader = DataLoader(local_dataset, batch_size=self.args.batch_size, shuffle=True)

        # global_feature_results, local_feature_results = defaultdict(list), defaultdict(list)
        # labels = []

        # with torch.no_grad():
        #     for images, label in local_loader:
        #         images, label = images.to(device), label.to(device)

        #         labels.append(label)
        #         global_results = global_model(images)
        #         local_results = local_model(images)

        #         for key in global_results:
        #             feat = global_results[key]

        #             if len(feat.shape) == 4:
        #                 feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)

        #             global_feature_results[key].append(feat.cpu())

        #         for key in local_results:
        #             feat = local_results[key]

        #             if len(feat.shape) == 4:
        #                 feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)

        #             local_feature_results[key].append(feat.cpu())


        #     labels = torch.cat(labels)
        #     idxs = labels.argsort()
        #     labels = labels[idxs][::stride]

        #     for key in global_feature_results:
        #         global_feature_results[key] = torch.cat(global_feature_results[key])[idxs, :][::stride]
        #     for key in local_feature_results:
        #         local_feature_results[key] = torch.cat(local_feature_results[key])[idxs, :][::stride]

        #     local_train_singular_results = self._get_singular_values(local_feature_results, labels=labels, local_dataset=local_dataset, desc='local_train', epoch=epoch)
        #     wandb_dict.update(local_train_singular_results)

        # global_model.to(model_device)
        # local_model.to(model_device)
        # global_model.train()
        # local_model.train()
        global_model_.cpu()
        local_model.cpu()
        del global_model_
        del local_model
        gc.collect()

        return wandb_dict




    def _get_singular_values(self, feature_results, labels, local_dataset=None, desc='', epoch=-1):

        # def __lalign(x, y, alpha=2):
        #     return (x - y).norm(dim=1).pow(alpha).mean()
        
        def __lunif(x, t=2):
            sq_pdist = torch.pdist(x, p=2).pow(2)
            return sq_pdist.mul(-t).exp().mean().log()

        def __get_covariance_results(features, labels, num_classes):

            total_cov = torch.cov(features.T)
            # feat_ = features.reshape(num_classes, -1, features.size(1))

            prototypes = torch.zeros((num_classes, features.size(1)))
            for i in range(num_classes):
                prototypes[i] = features[labels==i].mean(0)

            # within_diff = (feat_ - feat_.mean(1, keepdim=True)).reshape(-1, features.size(1))
            # within_cov = torch.mm(within_diff.T, within_diff) / (features.size(0)-1)
            within_diff = features - prototypes[labels]
            within_cov = torch.mm(within_diff.T, within_diff) / (features.size(0)-1)

            # prototypes = feat_.mean(1) # C * dim
            # btn_diff = prototypes - prototypes.mean(0, keepdim=True)
            # btn_cov = torch.mm(btn_diff.T, btn_diff) / feat_.size(0)
            
            '''
            btn_diff = prototypes_ - prototypes_.mean(0, keepdim=True)
            btn_cov = torch.mm(btn_diff.T, btn_diff) / feat_.size(0)

            p1 = prototypes_.unsqueeze(1).repeat(1, num_classes, 1)
            p2 = prototypes_.unsqueeze(0).repeat(num_classes, 1, 1)
            proto_cosines = F.cosine_similarity(p1, p2, dim=2)

            proto_cosines_ = torch.masked_select(proto_cosines, (1-torch.eye(num_classes)).bool())
            mean_cosines = proto_cosines_.nanmean()
            var_cosines =  proto_cosines_.var()
            collapse_error = F.mse_loss(proto_cosines_, torch.ones(num_classes*(num_classes-1))*(-1/(num_classes-1)))
            '''

            if self.args.get('debugb') and local_dataset is not None:
                within_cov_classes = [0] * num_classes
                for i in range(num_classes):
                    within_diff_i = within_diff[labels==i]
                    within_cov_i = torch.mm(within_diff_i.T, within_diff_i) / (within_diff_i.size(0)-1)
                    # within_cov_i = torch.mm(within_diff_i.T, within_diff_i) / (within_diff_i.size(0))
                    within_cov_classes[i] = torch.trace(within_cov_i)

                within_cov_classes = torch.stack(within_cov_classes)
                breakpoint()

            # try:
            valid_num_classes = labels.bincount().nonzero().size(0)
            prototypes_ = prototypes[~prototypes[:, 0].isnan(), :]
            valid_num_classes = prototypes_.size(0)
            valid_class_counts = labels.bincount()[labels.bincount().nonzero()]

            btn_diff = prototypes_ - prototypes_.mean(0, keepdim=True)
            # btn_cov = torch.mm(btn_diff.T, btn_diff) / valid_num_classes
            btn_cov = torch.cov(btn_diff.T, fweights=valid_class_counts.squeeze().cpu())

            p1 = prototypes_.unsqueeze(1).repeat(1, valid_num_classes, 1)
            p2 = prototypes_.unsqueeze(0).repeat(valid_num_classes, 1, 1)
            proto_cosines = F.cosine_similarity(p1, p2, dim=2)

            proto_cosines_ = torch.masked_select(proto_cosines, (1-torch.eye(valid_num_classes)).bool())
            mean_cosines = proto_cosines_.mean()
            var_cosines =  proto_cosines_.var()
            # if valid_num_classes > 1:
            try:
                collapse_error = F.mse_loss(proto_cosines_, torch.ones(valid_num_classes*(valid_num_classes-1))*(-1/(valid_num_classes-1)))
            except:
                collapse_error = 0

            vci = 1 - torch.trace(torch.mm(torch.linalg.pinv(total_cov), btn_cov))/torch.linalg.matrix_rank(btn_cov)
            # vci = 0

            # except:
            #     print(f"Error during getting btn_cov on {desc}")
            #     # breakpoint()
            #     btn_cov = total_cov * 0
            #     vci = proto_cosines = mean_cosines = var_cosines = collapse_error = 0


            # if self.args.get('debugs'):
            #     print(valid_num_classes, torch.abs(total_cov[0] - btn_cov[0] - within_cov[0]).mean())
            #     if vci < 0 or torch.abs(total_cov[0] - within_cov[0] - btn_cov[0]).mean() > 1e-3:
            #         breakpoint()

            results = {
                "total_cov": torch.trace(total_cov),
                "within_cov": torch.trace(within_cov),
                "between_cov": torch.trace(btn_cov),

                "collapse": mean_cosines,
                "collapse_error": collapse_error,
                "collapse_var": var_cosines,

                "proto_cosine_sims": proto_cosines,
                "vci": vci,
            }

            # if local_dataset is not None:
            #     local_classes = get_local_classes(local_dataset=local_dataset)
                
            #     minor_inds, major_inds = [], []
            #     for minor_c in local_classes["minor"]:
            #         minor_inds.append(((labels==minor_c).nonzero()).squeeze())
            #     minor_inds = torch.cat(minor_inds)
            #     for major_c in local_classes["major"]:
            #         major_inds.append(((labels==major_c).nonzero()).squeeze())
            #     major_inds = torch.cat(major_inds)

            #     # minor_label_indices = [int(label) for label in labels if label in local_classes["minor"]]
            #     within_diff_minor = within_diff[minor_inds]
            #     within_cov_minor = torch.mm(within_diff_minor.T, within_diff_minor) / (within_diff_minor.size(0)-1)
            #     total_cov_minor = torch.cov(features[minor_inds].T)
            #     btn_cov_minor = total_cov_minor - within_cov_minor

            #     # btn_diff = prototypes_[local_classes["minor"]] - prototypes_[local_classes["minor"]].mean(0, keepdim=True)
            #     # btn_cov_minor = torch.cov(btn_diff.T)
            #     # major_inds = torch.arange(labels.size(0)) - minor_inds

            #     within_diff_major = within_diff[major_inds]
            #     within_cov_major = torch.mm(within_diff_major.T, within_diff_major) / (within_diff_major.size(0)-1)
            #     total_cov_major = torch.cov(features[major_inds].T)
            #     btn_cov_major = total_cov_major - within_cov_major

            #     vci_minor = 1 - torch.trace(torch.mm(torch.linalg.pinv(total_cov_minor), btn_cov_minor))/torch.linalg.matrix_rank(btn_cov_minor)
            #     vci_major = 1 - torch.trace(torch.mm(torch.linalg.pinv(total_cov_major), btn_cov_major))/torch.linalg.matrix_rank(btn_cov_major)


            #     results.update({
            #         "within_cov/minor": torch.trace(within_cov_minor),
            #         "between_cov/minor": torch.trace(btn_cov_minor),
            #         "total_cov/minor": torch.trace(total_cov_minor),
            #         "vci/minor": vci_minor,

            #         "within_cov/major": torch.trace(within_cov_major),
            #         "between_cov/major": torch.trace(btn_cov_major),
            #         "total_cov/major": torch.trace(total_cov_major),
            #         "vci/major": vci_major,
            #     })


            uniform_loss = __lunif(features)

            align_loss = []
            for c in range(num_classes):
                feat_c = features[labels==c]
                if feat_c.size(0) > 0:
                    # if self.args.get('debugs'):
                    #     breakpoint()
                    if feat_c.size(0) > 30:
                        rind = torch.randperm(feat_c.size(0))[:30]
                        feat_c = feat_c[rind]
                    n = feat_c.size(0)
                    f1 = feat_c.unsqueeze(1).repeat(1, n, 1)
                    f2 = feat_c.unsqueeze(0).repeat(n, 1, 1)
                    align = (f1 - f2).norm(dim=2).pow(2)
                    align_loss.append(torch.masked_select(align, (1-torch.eye(n)).bool()).mean())
            
            align_loss = np.mean(align_loss)
            

            results.update({
                'uniform_loss': uniform_loss,
                'align_loss': align_loss,
            })

            return results


        def __get_collapse_error(cosine_sims, num_classes):
            N = cosine_sims.size(0)
            cosine_sims_ = torch.masked_select(cosine_sims, (1-torch.eye(N)).bool())
            mean_cosines = cosine_sims_.mean()
            var_cosines = cosine_sims_.var()
            # collapse_error = F.mse_loss(cosine_sims_, torch.ones(N*(N-1))*(-1/(num_classes-1)))
            return {
                "collapse": mean_cosines,
                # "collapse_error": collapse_error,
                "collapse_var": var_cosines,
            }

        singular_results = {}


        for key in feature_results:
            # if key not in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'feature']:
            #     continue

            ### Trace of Covariance Matrix
            num_classes = len(labels.bincount())

            # cov_results = __get_covariance_results(feature_results[key], labels, num_classes)
            # singular_results.update({
            #     f"trace/{self.args.dataset.name}/total/{desc}/{key}": cov_results["total_cov"], 
            #     f"trace/{self.args.dataset.name}/within_class/{desc}/{key}": cov_results["within_cov"],
            #     f"trace/{self.args.dataset.name}/between_class/{desc}/{key}": cov_results["between_cov"],
            #     f"collapse/{self.args.dataset.name}/prototype/{desc}/{key}": cov_results["collapse"],
            #     # f"collapse_error/{self.args.dataset.name}/prototype/{desc}/{key}": cov_results["collapse_error"],
            #     f"collapse_var/{self.args.dataset.name}/prototype/{desc}/{key}": cov_results["collapse_var"],
            #     # f"rep/{self.args.dataset.name}/uniform/{desc}/{key}": cov_results["uniform_loss"],
            #     # f"rep/{self.args.dataset.name}/align/{desc}/{key}": cov_results["align_loss"],
            #     # f"rep/{self.args.dataset.name}/vci/{desc}/{key}": cov_results["vci"],
            # })

            feat_norm = torch.nn.functional.normalize(feature_results[key], dim=1)
            cov_norm_results = __get_covariance_results(feat_norm, labels, num_classes)
            singular_results.update({
                f"trace_norm/{self.args.dataset.name}/total/{desc}/{key}": cov_norm_results["total_cov"],
                f"trace_norm/{self.args.dataset.name}/within_class/{desc}/{key}": cov_norm_results["within_cov"],
                f"trace_norm/{self.args.dataset.name}/between_class/{desc}/{key}": cov_norm_results["between_cov"],
                f"collapse_norm/{self.args.dataset.name}/prototype/{desc}/{key}": cov_norm_results["collapse"],
                # f"collapse_norm_error/{self.args.dataset.name}/prototype/{desc}/{key}": cov_norm_results["collapse_error"],
                f"collapse_norm_var/{self.args.dataset.name}/prototype/{desc}/{key}": cov_norm_results["collapse_var"],
                f"rep/{self.args.dataset.name}/uniform/{desc}/{key}": cov_norm_results["uniform_loss"],
                f"rep/{self.args.dataset.name}/align/{desc}/{key}": cov_norm_results["align_loss"],
                f"rep/{self.args.dataset.name}/vci/{desc}/{key}": cov_norm_results["vci"],
            })

            if local_dataset is not None:

                # singular_results.update({
                #     f"trace_norm/{self.args.dataset.name}/total_minor/{desc}/{key}": cov_norm_results["total_cov/minor"],
                #     f"trace_norm/{self.args.dataset.name}/within_class_minor/{desc}/{key}": cov_norm_results["within_cov/minor"],
                #     f"trace_norm/{self.args.dataset.name}/between_class_minor/{desc}/{key}": cov_norm_results["between_cov/minor"],
                #     f"trace_norm/{self.args.dataset.name}/total_major/{desc}/{key}": cov_norm_results["total_cov/major"],
                #     f"trace_norm/{self.args.dataset.name}/within_class_major/{desc}/{key}": cov_norm_results["within_cov/major"],
                #     f"trace_norm/{self.args.dataset.name}/between_class_major/{desc}/{key}": cov_norm_results["between_cov/major"],

                #     f"rep/{self.args.dataset.name}/vci_minor/{desc}/{key}": cov_norm_results["vci/minor"],
                #     f"rep/{self.args.dataset.name}/vci_major/{desc}/{key}": cov_norm_results["vci/major"],

                # })


                try:
                    major_classes = [int(key) for key in local_dataset.class_dict if local_dataset.class_dict[key] >= len(local_dataset)/num_classes]
                    minor_seen_classes = [int(key) for key in local_dataset.class_dict if local_dataset.class_dict[key] < len(local_dataset)/num_classes]
                    # minor_classes = [i for i in range(num_classes) if i not in major_classes]
                    missing_classes = [i for i in range(num_classes) if str(i) not in local_dataset.class_dict]

                    # if self.args.get('debugs'):
                    #     print(major_classes, minor_classes, missing_classes)
                    minor_classes = minor_seen_classes + missing_classes

                    proto_cosines = cov_norm_results["proto_cosine_sims"]

                    major_cosines = proto_cosines[major_classes][:, major_classes]
                    minor_cosines = proto_cosines[minor_classes][:, minor_classes]
                    missing_cosines = proto_cosines[missing_classes][:, missing_classes]
                    minor_seen_cosines = proto_cosines[minor_seen_classes][:, minor_seen_classes]

                    # major_results = self._get_collapse_error(major_cosines, num_classes)
                    # minor_results = self._get_collapse_error(minor_cosines, num_classes)
                    # missing_results = self._get_collapse_error(missing_cosines, num_classes)
                    major_results = __get_collapse_error(major_cosines, num_classes)
                    minor_results = __get_collapse_error(minor_cosines, num_classes)
                    missing_results = __get_collapse_error(missing_cosines, num_classes)
                    minor_seen_results = __get_collapse_error(minor_seen_cosines, num_classes)


                    major_minor_cosines = proto_cosines[major_classes][:, minor_classes]
                    major_minor_results = {'collapse': major_minor_cosines.mean()}
                    major_minor_seen_cosines = proto_cosines[major_classes][:, minor_seen_classes]
                    major_minor_seen_results = {'collapse': major_minor_seen_cosines.mean()}

                    singular_results.update({
                        f"collapse/{self.args.dataset.name}/prototype_major/{desc}/{key}": major_results['collapse'],
                        f"collapse/{self.args.dataset.name}/prototype_minor/{desc}/{key}": minor_results['collapse'],
                        f"collapse/{self.args.dataset.name}/prototype_missing/{desc}/{key}": missing_results['collapse'],
                        f"collapse/{self.args.dataset.name}/prototype_minor_seen/{desc}/{key}": minor_seen_results['collapse'],
                        f"collapse/{self.args.dataset.name}/prototype_major_minor/{desc}/{key}": major_minor_results['collapse'],
                        f"collapse/{self.args.dataset.name}/prototype_major_minor_seen/{desc}/{key}": major_minor_seen_results['collapse'],
                    })
                    # print(singular_results)
                except:
                    print(f"Error during getting minor collapse on {desc}")


            ### Singular Values
            covariance_matrix = torch.cov(feature_results[key].T)
            eigen_vals = torch.linalg.svdvals(covariance_matrix)
            log_eigen_vals = torch.log(eigen_vals)
            data = list(zip(range(len(log_eigen_vals)), log_eigen_vals))

            probs = eigen_vals/torch.abs(eigen_vals).sum()
            effective_rank = torch.exp(torch.distributions.Categorical(probs).entropy())

            ## Effective rank of relu-ed feature
            relu_covariance_matrix = torch.cov(F.relu(feature_results[key]).T)
            relu_eigen_vals = torch.linalg.svdvals(relu_covariance_matrix)
            relu_probs = relu_eigen_vals/torch.abs(relu_eigen_vals).sum()
            relu_effective_rank = torch.exp(torch.distributions.Categorical(relu_probs).entropy())
            
            singular_results.update({
                # f"singular/{self.args.dataset.name}/graph/{desc}/{key}": image,
                f"singular/{self.args.dataset.name}/rank1/{desc}/{key}": log_eigen_vals[0],
                # f"singular/{self.args.dataset.name}/rank10/{desc}/{key}": log_eigen_vals[9],
                f"singular/{self.args.dataset.name}/median/{desc}/{key}": log_eigen_vals[len(log_eigen_vals)//2],
                f"singular/{self.args.dataset.name}/avg/{desc}/{key}": torch.mean(log_eigen_vals),
                f"singular/{self.args.dataset.name}/effective_rank/{desc}/{key}": effective_rank,
                f"singular/{self.args.dataset.name}/effective_rank/{desc}/relu_{key}": relu_effective_rank,
            })

            if self.args.get('wandb'):
                table = wandb.Table(data=data, columns=["x", "y"])
                singular_results.update({
                    f"singular/{self.args.dataset.name}/plot/{desc}/{key}": wandb.plot.line(table, "x", "y", title=f"Singular/{desc}/{key}"),
                })

            ### Singular Values on sample dimension

            # covariance_matrix = torch.cov(feature_results[key])

            # eigen_vals = torch.linalg.svdvals(covariance_matrix)
            # log_eigen_vals = torch.log(eigen_vals)
            # data = list(zip(range(len(log_eigen_vals)), log_eigen_vals))  
            # probs = eigen_vals/torch.abs(eigen_vals).sum()
            # effective_rank = torch.exp(torch.distributions.Categorical(probs).entropy())     

            # singular_results.update({
            #     f"sample_singular/{self.args.dataset.name}/rank1/{desc}/{key}": log_eigen_vals[0],
            #     f"sample_singular/{self.args.dataset.name}/median/{desc}/{key}": log_eigen_vals[len(log_eigen_vals)//2],
            #     f"sample_singular/{self.args.dataset.name}/avg/{desc}/{key}": torch.mean(log_eigen_vals),
            #     f"sample_singular/{self.args.dataset.name}/effective_rank/{desc}/{key}": effective_rank,
            #     f"sample_singular/{self.args.dataset.name}/effective_rank/{desc}/relu_{key}": relu_effective_rank,
            # })   

            # if self.args.get('wandb'):
            #     table = wandb.Table(data=data, columns=["x", "y"])
            #     singular_results.update({
            #         f"sample_singular/{self.args.dataset.name}/plot/{desc}/{key}": wandb.plot.line(table, "x", "y", title=f"Singular(sample)/{desc}/{key}"),
            #     })
          

            ### Singular Values of normalizing covariance matrix
            covariance_matrix = torch.cov(feature_results[key].T) / torch.var(feature_results[key].T)
            eigen_vals = torch.linalg.svdvals(covariance_matrix)
            probs = eigen_vals/torch.abs(eigen_vals).sum()
            effective_rank = torch.exp(torch.distributions.Categorical(probs).entropy())
            log_eigen_vals = torch.log(eigen_vals)


            data = list(zip(range(len(log_eigen_vals)), log_eigen_vals))
            
            singular_results.update({
                f"singular_normal/{self.args.dataset.name}/rank1/{desc}/{key}": log_eigen_vals[0],
                # f"singular_normal/{self.args.dataset.name}/rank10/{desc}/{key}": log_eigen_vals[9],
                f"singular_normal/{self.args.dataset.name}/avg/{desc}/{key}": torch.mean(log_eigen_vals),
                f"singular_normal/{self.args.dataset.name}/median/{desc}/{key}": log_eigen_vals[len(log_eigen_vals)//2],
                f"singular_normal/{self.args.dataset.name}/effective_rank/{desc}/{key}": effective_rank,
            })             
            
            if self.args.get('wandb'):
                table = wandb.Table(data=data, columns=["x", "y"])
                singular_results.update({
                    f"singular_normal/{self.args.dataset.name}/plot/{desc}/{key}": wandb.plot.line(table, "x", "y", title=f"Singular_normal/{desc}/{key}"),
                })


            # if self.args.get('debugc') or self.args.get('intra_rank'):
            prototypes = torch.zeros((num_classes, feature_results[key].size(1)))
            for i in range(num_classes):
                prototypes[i] = feature_results[key][labels==i].mean(0)
            feature_class_norm = feature_results[key] - prototypes[labels]
            covariance_matrix_ = torch.cov(feature_class_norm.T) / torch.var(feature_class_norm.T)
            eigen_vals_ = torch.linalg.svdvals(covariance_matrix_)
            probs_ = eigen_vals_/torch.abs(eigen_vals_).sum()
            effective_rank_ = torch.exp(torch.distributions.Categorical(probs_).entropy())
            singular_results.update({
                f"singular_normal_class_norm/{self.args.dataset.name}/effective_rank/{desc}/{key}": effective_rank_,
            })
                # breakpoint()

            if self.args.get('debuga'):
                breakpoint()


            ## Correlation coefficient matrix
            correlation_matrix = torch.corrcoef(feature_results[key].T)
            identity_matrix = torch.eye(len(correlation_matrix))
            distance = (identity_matrix - correlation_matrix).norm()
            singular_results.update({
                 f"distance_correlation_identity/{self.args.dataset.name}/{desc}/{key}": distance,
                 }) 
            
        return singular_results





    def _get_raw_features_prototype_cossim(self, feature_results, labels, local_dataset=None, desc='', epoch=-1, zero_mean = False):

        def __cossim_results(features, num_classes, zero_mean = False):

            feat_ = features.reshape(num_classes, -1, features.size(1))
            prototypes = feat_.mean(1)
            prototypes_mean = prototypes.mean(0) 
            prototypes_mean_norm = prototypes_mean.norm()
            if zero_mean:
                prototypes -= prototypes_mean
            p1 = prototypes.unsqueeze(1).repeat(1, num_classes, 1)
            p2 = prototypes.unsqueeze(0).repeat(num_classes, 1, 1)
            proto_cosines = F.cosine_similarity(p1, p2, dim=2)
            proto_cosines_ = torch.masked_select(proto_cosines, (1-torch.eye(num_classes)).bool())
            mean_cosines = proto_cosines_.mean()
            var_cosines = proto_cosines_.var()
            collapse_error = F.mse_loss(proto_cosines_, torch.ones(num_classes*(num_classes-1))*(-1/(num_classes-1)))
            
            
            return {
                "raw_features_prototypes_mean_norm": prototypes_mean_norm,
                "raw_features_collapse": mean_cosines,
                "raw_features_cossim_var": var_cosines,
                "raw_features_collapse_error": collapse_error,
                "raw_features_cossim": mean_cosines,
                "raw_features_proto_cosine_sims": proto_cosines,
            }

        raw_features_prototype_cossim_results = {}
        num_classes = len(labels.bincount())

        for key in feature_results:
            # cossim_results = self._get_raw_features_prototype_cossim_results(feature_results[key], num_classes, zero_mean = zero_mean)
            cossim_results = __cossim_results(feature_results[key], num_classes, zero_mean = zero_mean)
            #print(cossim_results)
            raw_features_prototype_cossim_results.update({
                f"prototype_mean_norm_raw_feature{'_zero_mean' if zero_mean else ''}/{self.args.dataset.name}/prototype/{desc}/{key}": cossim_results["raw_features_prototypes_mean_norm"],
                f"collapse_raw_feature{'_zero_mean' if zero_mean else ''}/{self.args.dataset.name}/prototype/{desc}/{key}": cossim_results["raw_features_collapse"],
                f"collapse_var_raw_feature{'_zero_mean' if zero_mean else ''}/{self.args.dataset.name}/prototype/{desc}/{key}": cossim_results["raw_features_cossim_var"],
                f"collapse_error_raw_feature{'_zero_mean' if zero_mean else ''}/{self.args.dataset.name}/prototype/{desc}/{key}": cossim_results["raw_features_collapse_error"],
            })


        if local_dataset is not None:
            major_classes = [int(key) for key in local_dataset.class_dict if local_dataset.class_dict[key] >= len(local_dataset)/num_classes]
            # minor_classes = [int(key) for key in local_dataset.class_dict if local_dataset.class_dict[key] < len(local_dataset)/num_classes]
            minor_classes = [i for i in range(num_classes) if i not in major_classes]
            missing_classes = [i for i in range(num_classes) if str(i) not in local_dataset.class_dict]

            # if self.args.get('debugs'):
            #     print(major_classes, minor_classes, missing_classes)
            # non_major_classes = minor_classes + missing_classes

            proto_cosines = cossim_results["raw_features_proto_cosine_sims"]

            major_cosines = proto_cosines[major_classes][:, major_classes]
            minor_cosines = proto_cosines[minor_classes][:, minor_classes]
            missing_cosines = proto_cosines[missing_classes][:, missing_classes]
            # non_major_cosines = proto_cosines[non_major_classes][:, non_major_classes]

            major_results = self._get_collapse_error(major_cosines, num_classes)
            minor_results = self._get_collapse_error(minor_cosines, num_classes)
            missing_results = self._get_collapse_error(missing_cosines, num_classes)

            raw_features_prototype_cossim_results.update({
                f"collapse_raw_feature{'_zero_mean' if zero_mean else ''}/{self.args.dataset.name}/prototype_major/{desc}/{key}": major_results['collapse'],
                f"collapse_raw_feature{'_zero_mean' if zero_mean else ''}/{self.args.dataset.name}/prototype_minor/{desc}/{key}": minor_results['collapse'],
                f"collapse_raw_feature{'_zero_mean' if zero_mean else ''}/{self.args.dataset.name}/prototype_missing/{desc}/{key}": missing_results['collapse'],
                # f"collapse/{self.args.dataset.name}/prototype_nonmajor/{desc}/{key}": non_major_results['collapse'],
                # f"collapse_error/{self.args.dataset.name}/prototype_major/{desc}/{key}": major_results['collapse_error'],
            })

        return raw_features_prototype_cossim_results




    def visualize_loss_landscape(self):
        return



   # ----- SVD, feature norm, class norm ----#
    @torch.no_grad()
    def evaluate_activation_matching(self, epoch: int, local_models: List[nn.Module], global_model: nn.Module, local_datasets: List[torch.utils.data.Dataset], device: torch.device = None):
        stride = self.args.svd.stride
        
        logger.info("evaluate_activation_matching")
        wandb_dict = {}
        global_model.eval()
        model_device = next(global_model.parameters()).device
        if device is None:
            device = self.device
        global_model.to(device)
        global_feature_results = defaultdict(list)

        participated_clients_local_feature_results = []

        for idx, (local_model, local_dataset) in enumerate(zip(local_models, local_datasets)):
            if idx > 1:
                break

            local_model = local_model.eval()
            local_model.to(device)

            # global_features = []
            # local_features = []
            local_feature_results = defaultdict(list)
            labels = []

            num_classes = len(self.test_loader.dataset.classes)
            num_samples = len(self.test_loader.dataset)
            num_samples_per_class = int(num_samples / num_classes / stride)

            with torch.no_grad():
                # for images, labels in self.loaders["test"]:
                for images, label in self.test_loader:
                    images, label = images.to(device), label.to(device)

                    labels.append(label)
                    
                    local_results = local_model(images)
                    for key in local_results:
                        feat = local_results[key]
                        
                        if len(feat.shape) == 4:
                            if "flatten_feature" in self.args.analysis_mode and '4' in key:
                                feat = feat.view(feat.shape[0], -1)      
                            else:
                                feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)


                        local_feature_results[key].append(feat.cpu())

                    if idx==0:
                        global_results = global_model(images)
                        for key in global_results:
                            feat = global_results[key]


                            if len(feat.shape) == 4:
                                if "flatten_feature" in self.args.analysis_mode and '4' in key:
                                    feat = feat.view(feat.shape[0], -1)      
                                else:
                                    feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                            
                                
                            global_feature_results[key].append(feat.cpu())

                    # global_features.append(global_results['feature'].cpu())
                    # local_features.append(local_results['feature'].cpu())
                    

                # global_features = torch.cat(global_features)
                # local_features = torch.cat(local_features)
                labels = torch.cat(labels)
                idxs = labels.argsort()
                labels = labels[idxs]

                if idx==0:
                    for key in global_feature_results:
                        global_feature_results[key] = torch.cat(global_feature_results[key])[idxs, :][::stride]
                for key in local_feature_results:
                    local_feature_results[key] = torch.cat(local_feature_results[key])[idxs, :][::stride]


                labels = labels[::stride]
                # labels = torch.cat(labels)
                # idxs = labels.argsort()
                # labels = labels[idxs]
                # global_features = global_features[idxs, :]
                # local_features = local_features[idxs, :]

                # sampled_gf = global_features[::stride]
                # sampled_lf = local_features[::stride]
                # sampled_gt = labels[::stride]
                # breakpoint()


                participated_clients_local_feature_results.append(local_feature_results)

            local_model.to(model_device)    
            local_model.train()

            if idx==0:
                global_model.to(model_device)
                global_model.train()
        

        samesample_activation_cossim_results = self._get_samesample_activation_cossim(global_feature_results, participated_clients_local_feature_results, desc="")
        wandb_dict.update(samesample_activation_cossim_results)

        return wandb_dict



    @torch.no_grad()
    def _get_samesample_activation_cossim(self, global_feature_results, participated_clients_local_feature_results, desc = ""):
        same_sample_activation_cossim_results = {}

        global_local_cossim_results =  defaultdict(list)
        local_local_cossim_results = defaultdict(list)

        global_local_cossim_results_relu = defaultdict(list)
        local_local_cossim_results_relu =  defaultdict(list)

        global_local_activate_region_results =  defaultdict(list)
        local_local_activate_region_results =  defaultdict(list)


        # for key in global_feature_results:
        #     global_feature_results[key] = torch.stack(global_feature_results[key], dim=0)
            

        #     for local_feature_results in enumerate(participated_clients_local_feature_results):
        #         local_feature_results[key] = torch.stack(local_feature_results[key], dim=0)


        for i, local_feature_results in enumerate(participated_clients_local_feature_results):
            
            for key in global_feature_results:
                global_feature_results_relu = F.relu(global_feature_results[key])
                ge0_global_feature_results = 1.0 * ((global_feature_results_relu)>0)
                local_feature_results_relu = F.relu(local_feature_results[key])
                ge0_local_feature_results = 1.0 * ((local_feature_results_relu)>0)


                global_local_cossim_results[key].append(F.cosine_similarity(global_feature_results[key], local_feature_results[key]).mean())
                global_local_cossim_results_relu[key].append(F.cosine_similarity(global_feature_results_relu, local_feature_results_relu).mean())
                global_local_activate_region_results[key].append(F.cosine_similarity(ge0_global_feature_results, ge0_local_feature_results).mean())

                for j in range(i+1, len(participated_clients_local_feature_results)):
                    another_local_feature_results = participated_clients_local_feature_results[j]
                    another_local_feature_results_relu = F.relu(another_local_feature_results[key])
                    ge0_another_local_feature_results = 1.0 * (another_local_feature_results_relu>0)
                    
                    local_local_cossim_results[key].append(F.cosine_similarity(another_local_feature_results[key], local_feature_results[key]).mean())
                    local_local_cossim_results_relu[key].append(F.cosine_similarity(another_local_feature_results_relu, local_feature_results_relu).mean())
                    local_local_activate_region_results[key].append(F.cosine_similarity(ge0_another_local_feature_results, ge0_local_feature_results).mean())


        for key in global_feature_results:
            global_local_cossim_results[key] = sum(global_local_cossim_results[key]) / len(global_local_cossim_results[key])
            global_local_cossim_results_relu[key] = sum(global_local_cossim_results_relu[key]) / len(global_local_cossim_results_relu[key])
            global_local_activate_region_results[key] = sum(global_local_activate_region_results[key]) / len(global_local_activate_region_results[key])
            
            if len(local_local_cossim_results[key]) > 0:
                local_local_cossim_results[key] = sum(local_local_cossim_results[key]) / len(local_local_cossim_results[key])
                local_local_cossim_results_relu[key] = sum(local_local_cossim_results_relu[key]) / len(local_local_cossim_results_relu[key])
                local_local_activate_region_results[key] = sum(local_local_activate_region_results[key]) / len(local_local_activate_region_results[key])
            else:
                local_local_cossim_results[key] = 0.
                local_local_cossim_results_relu[key] = 0.
                local_local_activate_region_results[key] = 0.

            same_sample_activation_cossim_results.update({
                # f"activation_samesample/{self.args.dataset.name}/global_local_cossim/{desc}/{key}": global_local_cossim_results[key],
                # f"activation_samesample/{self.args.dataset.name}/global_local_cossim_relu/{desc}/{key}": global_local_cossim_results_relu[key],
                # f"activation_samesample/{self.args.dataset.name}/global_local_activate_region_aligns/{desc}/{key}": global_local_activate_region_results[key],

                # f"activation_samesample/{self.args.dataset.name}/local_local_cossim/{desc}/{key}": local_local_cossim_results[key],
                # f"activation_samesample/{self.args.dataset.name}/local_local_cossim_relu/{desc}/{key}": local_local_cossim_results_relu[key],
                # f"activation_samesample/{self.args.dataset.name}/local_local_activate_region_aligns/{desc}/{key}": local_local_activate_region_results[key],

                # Key update (compatibility를 위해 당분간 기존 키도 남겨둠)
                f"matching/{self.args.dataset.name}/feature_cossim/global_local/{key}": global_local_cossim_results[key],
                f"matching/{self.args.dataset.name}/relu_feature_cossim/global_local/{key}": global_local_cossim_results_relu[key],
                f"matching/{self.args.dataset.name}/activation_align/global_local/{key}": global_local_activate_region_results[key],

                f"matching/{self.args.dataset.name}/feature_cossim/local_local/{key}": local_local_cossim_results[key],
                f"matching/{self.args.dataset.name}/relu_feature_cossim/local_local/{key}": local_local_cossim_results_relu[key],
                f"matching/{self.args.dataset.name}/activation_align/local_local/{key}": local_local_activate_region_results[key],
            })             

        return same_sample_activation_cossim_results








    '''
    def _get_collapse_error(self, cosine_sims, num_classes):
        N = cosine_sims.size(0)
        cosine_sims_ = torch.masked_select(cosine_sims, (1-torch.eye(N)).bool())
        mean_cosines = cosine_sims_.mean()
        var_cosines = cosine_sims_.var()
        collapse_error = F.mse_loss(cosine_sims_, torch.ones(N*(N-1))*(-1/(num_classes-1)))
        return {
            "collapse": mean_cosines,
            "collapse_error": collapse_error,
            "collapse_var": var_cosines,
        }


    def _get_covariance_results(self, features, num_classes):

        total_cov = torch.cov(features.T)
        feat_ = features.reshape(num_classes, -1, features.size(1))
        within_diff = (feat_ - feat_.mean(1, keepdim=True)).reshape(-1, features.size(1))
        within_cov = torch.mm(within_diff.T, within_diff) / (features.size(0)-1)

        prototypes = feat_.mean(1) # C * dim

        btn_diff = prototypes - prototypes.mean(0, keepdim=True)
        btn_cov = torch.mm(btn_diff.T, btn_diff) / feat_.size(0)


        p1 = prototypes.unsqueeze(1).repeat(1, num_classes, 1)
        p2 = prototypes.unsqueeze(0).repeat(num_classes, 1, 1)
        proto_cosines = F.cosine_similarity(p1, p2, dim=2)

        # collapse_results = self._get_collapse_error(proto_cosines, num_classes)

        proto_cosines_ = torch.masked_select(proto_cosines, (1-torch.eye(num_classes)).bool())
        mean_cosines = proto_cosines_.mean()
        var_cosines =  proto_cosines_.var()
        collapse_error = F.mse_loss(proto_cosines_, torch.ones(num_classes*(num_classes-1))*(-1/(num_classes-1)))

        return {
            "total_cov": torch.trace(total_cov),
            "within_cov": torch.trace(within_cov),
            "between_cov": torch.trace(btn_cov),

            "collapse": mean_cosines,
            "collapse_error": collapse_error,
            "collapse_var": var_cosines,

            "proto_cosine_sims": proto_cosines,
        }
    '''


    @torch.no_grad()
    # def minority_collapse(self, localmodel, )
    # def _evaluate_minority_collapse(self, prev_model_weight: nn.Module.state_dict(), local_models: List[nn.Module], local_datasets: List[torch.utils.data.Dataset]):
    def evaluate_minority_collapse(self, prev_model_weight: nn.Module, local_models: List[nn.Module], local_datasets: List[torch.utils.data.Dataset]):
        results = {}
        # results[f'fc_weight_collapse/{self.args.dataset.name}/minor/local'] = []
        # results[f'fc_bias_collapse/{self.args.dataset.name}/minor/local'] = []

        weight_collapses, bias_collapses = [], []
        weight_aligns, bias_aligns = [], []
        global_model_dict = prev_model_weight#global_model.state_dict()

        for local_model, local_dataset in zip(local_models, local_datasets):
            # local_dataset = loader.dataset
            # if 'toy' in self.args.split.mode:
            #     num_classes = 3
            # else:
            num_classes = len(local_dataset.dataset.classes)
            num_samples_per_class = dict(local_dataset.class_dict)
            major_dict = {}
            threshold = len(local_dataset) / num_classes
            for i in range(num_classes):
                if str(i) not in num_samples_per_class.keys():
                    num_samples_per_class[str(i)] = 0
                    major_dict[str(i)] = False

                else:
                    #num_samples_per_class[str(i)] /= len(local_dataset)
                    if num_samples_per_class[str(i)] >= threshold:
                        major_dict[str(i)] = True
                    elif num_samples_per_class[str(i)] < threshold:
                        major_dict[str(i)] = False


            #num_samples = 

            
            #fc_weight_dict = {'fc.weight', local_model['fc.weight']}
            minor_fc_weight_dict = {}
            minor_fc_bias_dict = {}
            local_model_dict = local_model
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

            if count > 0:
                minor_fc_weight_cos /= count
                minor_fc_bias_cos /= count
            
            # local_loss_dict['minor_fc_weight_cos'] = minor_fc_weight_cos
            weight_collapses.append(minor_fc_weight_cos)
            bias_collapses.append(minor_fc_bias_cos)
            # local_loss_dict['minor_fc_bias_cos'] = minor_fc_bias_cos

            
            this_weight_aligns, this_bias_aligns = 0, 0
            for k in range(num_classes):
                local_weight_state = {'weight': local_model_dict['fc.weight'][k]}
                global_weight_state = {'weight': global_model_dict['fc.weight'][k]}
                this_weight_aligns += cal_cos(local_weight_state, global_weight_state)

                local_bias_state = {'bias': local_model_dict['fc.bias'][k]}
                global_bias_state = {'bias': global_model_dict['fc.bias'][k]}
                this_bias_aligns += cal_cos(local_bias_state, global_bias_state)

            this_weight_aligns /= num_classes
            this_bias_aligns /= num_classes

            weight_aligns.append(this_weight_aligns)
            bias_aligns.append(this_bias_aligns)

        results[f'fc_weight_aligns/{self.args.dataset.name}'] = sum(weight_aligns) / len(weight_aligns)
        results[f'fc_bias_aligns/{self.args.dataset.name}'] = sum(bias_aligns) / len(bias_aligns)
        results[f'fc_weight_collapse/{self.args.dataset.name}/minor/local'] = sum(weight_collapses)/len(weight_collapses)
        results[f'fc_bias_collapse/{self.args.dataset.name}/minor/local'] = sum(bias_collapses)/len(bias_collapses)


        return results
        








    #----- Retrieval ----#

    @torch.no_grad()
    def extract_features(self, model, loader, device: torch.device = None):
        model_device = next(model.parameters()).device
        if device is None:
            device = self.device
        model.to(device)
        features, labels = [], []
        # layer4s = []
        features_ = defaultdict(list)
        with torch.no_grad():
            for g_data, g_label in tqdm.tqdm(loader, desc='loader'):
                g_data = g_data.to(self.device, non_blocking=True)
                g_label = g_label.to(self.device, non_blocking=True)

                g_outs = model(g_data)
                features.append(g_outs["feature"].squeeze())

                # for key in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5']:
                for key in ['layer4']:
                    feat_key = g_outs[key]
                    # feat_local = features_[key].detach()
                    if len(feat_key.shape) == 4:
                        feat_key = F.adaptive_avg_pool2d(feat_key, 1).squeeze(-1).squeeze(-1)
                    features_[key].append(feat_key)
                
                labels.append(g_label)

        features = torch.cat(features)
        labels = torch.cat(labels)

        for key in features_:
            features_[key] = torch.cat(features_[key])


        ## DDPs
        features = all_gather_nd(features)
        labels = all_gather_nd(labels)
        results = {
            "features": features,
            "labels": labels,
        }
        results.update({key: features_[key] for key in features_})
        model.to(model_device)
        #model.train()
        return results

    @torch.no_grad()
    def extract_multi_features(self, model, model2, loader, extract_types=[], desc=''):
        model.eval()
        model2.eval()

        features, features2 = [], []
        labels = []
        scores, scores2 = [], []
        print_extract_type, print_extract_type2 = False, False

        print(desc, extract_types)
        with torch.no_grad():
            for data, label in tqdm.tqdm(loader):
                data = data.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)

                score, score2 = None, None
                outs = model(data)
                
                # if len(out) == 2:
                #     score, feature = out
                # else:
                #     print("len(out) is not 2")
                #     feature = out

                outs2 = model2(data)

                # if len(out2) == 2:
                #     score2, feature2 = out2
                # else:
                #     print("len(out2) is not 2")
                #     feature2 = out2

                all_feature = all_gather_nd(outs["layer4"])
                all_feature2 = all_gather_nd(feature2)
                all_label = all_gather_nd(label)

                features.append(all_feature.squeeze().cpu())
                features2.append(all_feature2.squeeze().cpu())
                labels.append(all_label.cpu())
                
                if score is not None:
                    all_score = all_gather_nd(score.max(1)[0].unsqueeze(1))
                    scores.append(all_score.cpu())
                # else:
                #     scores.append(0)

                if score2 is not None:
                    all_score2 = all_gather_nd(score2.max(1)[0].unsqueeze(1))
                    scores2.append(all_score2.cpu())
                # else:
                #     scores2.append(0)

                # del feature, feature2, data, label, score, score2
                del feature, feature2, data


        results = {
            "features": torch.cat(features),
            "features2": torch.cat(features2),
            "labels": torch.cat(labels),
            "scores": torch.cat(scores) if len(scores) > 0 else None,
            "scores2": torch.cat(scores2) if len(scores2) > 0 else None,
        }
        return results

    def calculate_rank(self, query_feats, gallery_feats, topk):
        import faiss

        logger.info(f"query_feats shape: {query_feats.shape}, gallery_feats shape: {gallery_feats.shape}")
        num_q, feat_dim = query_feats.shape
        faiss_index = faiss.IndexFlatIP(feat_dim)
        
        faiss.normalize_L2(gallery_feats)
        faiss_index.add(gallery_feats)
        # logger.info("=> begin faiss search")
        faiss.normalize_L2(query_feats)
        ranked_scores, ranked_gallery_indices = faiss_index.search(query_feats, topk)
        # logger.info("=> end faiss search")
        return ranked_scores, ranked_gallery_indices
    

    def calculate_mAP(self, ranked_indices, query_ids=None, gallery_ids=None, verbose=True, topk=0):
        n = gallery_ids.shape[0]
        m = query_ids.shape[0]

        num_q = ranked_indices.shape[0]
        average_precision = np.zeros(num_q, dtype=float)
        # query_gts = matches = gallery_ids[indices] == query_ids[:, np.newaxis]

        if list(query_ids) == list(gallery_ids):
            ranked_indices = ranked_indices[:, 1:]

        # for i in range(num_q):
        iterator = tqdm.tqdm(range(m), desc="mAP") if verbose else range(m)
        for i in iterator:
            query_gt_i = (query_ids[i] == gallery_ids).nonzero()[0]
            retrieved_indices = np.where(np.in1d(ranked_indices[i], query_gt_i))[0]
            # retrieved_indices = np.where(np.in1d(ranked_gallery_indices[i], np.array(query_gts[i])))[0]
            if retrieved_indices.shape[0] > 0:
                retrieved_indices = np.sort(retrieved_indices)
                # gts_all_count = min(len(query_gts[i]), topk)
                gts_all_count = min(len(query_gt_i), topk) if topk > 0 else len(query_gt_i)
                for j, index in enumerate(retrieved_indices):
                    average_precision[i] += (j + 1) * 1.0 / (index + 1)
                average_precision[i] /= gts_all_count

        return np.mean(average_precision)


    def calculate_CMC(self, ranked_indices, query_ids, gallery_ids, topk=5,
            single_gallery_shot=False, first_match_break=True, per_class=True, verbose=False,):
        
        matches = gallery_ids[ranked_indices] == query_ids[:, np.newaxis]

        n = gallery_ids.shape[0]
        m = query_ids.shape[0]

        indices_ = ranked_indices
        matches_ = matches
        gallery_ids_ = gallery_ids
        query_ids_ = query_ids

        ret = np.zeros(topk)

        if per_class:
            ret_per_class = {cls: np.zeros(topk) for cls in set(gallery_ids_)}
            num_valid_queries_per_class = {cls: 0 for cls in set(gallery_ids_)}

        num_valid_queries = 0

        iterator = tqdm.tqdm(range(m), desc="cmc") if verbose else range(m)
        for i in iterator:
            if list(query_ids_) == list(gallery_ids_):
                # If query set is part of gallery set
                valid = np.arange(n)[indices_[i]] != np.arange(m)[i]
            else:
                valid = None

            if not np.any(matches_[i, valid]):
                continue

            if single_gallery_shot:
                repeat = 10
                gids = gallery_ids_[indices_[i][valid]]
                inds = np.where(valid)[0]
                ids_dict = defaultdict(list)

                for j, x in zip(inds, gids):
                    ids_dict[x].append(j)
            else:
                repeat = 1

            for _ in range(repeat):
                if single_gallery_shot:
                    # Randomly choose one instance for each id
                    sampled = valid & _unique_sample(ids_dict, len(valid))
                    index = np.nonzero(matches_[i, sampled])[0]
                else:
                    index = np.nonzero(matches_[i, valid])[0] # filter only the true gallery images for query i (99 images in CIFAR-100)

                delta = 1.0 / (len(index) * repeat)

                # topk 이미지 중에 찾은게 있는지 (index 값들 중에 topk보다 작은 숫자가 최대한 앞에 있어야 함)
                for j, k in enumerate(index):
                    if k - j >= topk:
                        break

                    if first_match_break:
                        ret[k - j] += 1
                        if per_class:
                            ret_per_class[query_ids_[i]][k - j] += 1
                        break

                    ret[k - j] += delta
                    if per_class:
                        ret_per_class[query_ids_[i]][k - j] += delta
                
            num_valid_queries += 1

            if per_class:
                num_valid_queries_per_class[query_ids_[i]] += 1

        if num_valid_queries == 0:
            raise RuntimeError("No valid query")

        results = {
            "topk": ret.cumsum() / num_valid_queries,
            # "dist": top1_results,
        }

        if per_class:
            results["topk_class"] = {
                cls: ret_class.cumsum() / num_valid_queries_per_class[cls]
                for cls, ret_class in ret_per_class.items()
            }

        return results


    # def eval_retrieval(self, gallery_model, query_model=None, **kwargs):
    def eval_retrieval(self, model, local_datasets: List[torch.utils.data.Dataset] = None, **kwargs):

        # gallery_model.eval()
        # gallery_results = self.extract_features(gallery_model, self.test_loader)

        # if query_model is None:
        #     query_results = gallery_results
        # else:
        #     query_model.eval()
        #     query_results = self.extract_features(query_model, self.query_loader)
        # self.model.eval()
        model.eval()
        feature_results = self.extract_features(model, self.test_loader)
        ranked_scores, ranked_gallery_indices = self.calculate_rank(feature_results['features'].cpu().numpy(),
                                                                    feature_results['features'].cpu().numpy(), 
                                                                    topk=feature_results['features'].size(0))

        map_scores = self.calculate_mAP(ranked_gallery_indices, feature_results['labels'].cpu().numpy(), feature_results['labels'].cpu().numpy(), verbose=True)
        cmc_results = self.calculate_CMC(ranked_gallery_indices, feature_results['labels'].cpu().numpy(), feature_results['labels'].cpu().numpy(), topk=5, verbose=True)
        cmc_scores = cmc_results["topk"][0]

        # if local_datasets is not None:
        #     breakpoint()

        #     map_scores_local = []
        #     cmc_scores_local = []

        #     for local_dataset in local_datasets:
        #         local_classes = get_local_classes(local_dataset)
                
        #         # seen_features = 
                
        #         ranked_scores, ranked_gallery_indices = self.calculate_rank(feature_results['features'].cpu().numpy(),
        #                                                             feature_results['features'].cpu().numpy(), 
        #                                                             topk=feature_results['features'].size(0))

        #         map_scores = self.calculate_mAP(ranked_gallery_indices, feature_results['labels'].cpu().numpy(), feature_results['labels'].cpu().numpy(), verbose=True)
        #         cmc_results = self.calculate_CMC(ranked_gallery_indices, feature_results['labels'].cpu().numpy(), feature_results['labels'].cpu().numpy(), topk=5, verbose=True)
        #         cmc_scores = cmc_results["topk"][0]

        #     pass
        
        # ranked_scores, ranked_gallery_indices = self.calculate_rank(query_results['features'].cpu().numpy(),
        #                                                             gallery_results['features'].cpu().numpy(), 
        #                                                             topk=gallery_results['features'].size(0))

        # map_scores = self.calculate_mAP(ranked_gallery_indices, query_results['labels'].cpu().numpy(), gallery_results['labels'].cpu().numpy(), verbose=True)
        # cmc_results = self.calculate_CMC(ranked_gallery_indices, query_results['labels'].cpu().numpy(), gallery_results['labels'].cpu().numpy(), topk=5, verbose=True)
        

        results = {
            "mAP" : 100. * map_scores,
            "CMC" : 100. * cmc_scores,
        }
        return results



def plot_line(data, x_label, y_label, title):
    fig, ax = plt.subplots()
    x, y = zip(*data)
    ax.plot(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()
    image = wandb.Image(fig)
    plt.close()

    return image


def plot_bar(data, x_label, y_label, title):
    fig, ax = plt.subplots()
    x, y = zip(*data)
    ax.bar(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()
    image = wandb.Image(fig)
    plt.close()
    return image