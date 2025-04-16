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

from evalers.build import EVALER_REGISTRY

from servers import Server
from clients import Client

from datasets.data_utils  import DatasetSplit
from utils.logging_utils import AverageMeter

from torch.utils.data import DataLoader

from utils import terminate_processes, initalize_random_seed
from omegaconf import DictConfig
import faiss

import umap.umap_ as umap
from mlxtend.plotting import plot_confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt

from evalers import Evaler

import gc

@EVALER_REGISTRY.register()
class VisualizeEvaler(Evaler):

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


    @torch.no_grad()
    def eval(self, model: nn.Module, epoch: int, **kwargs) -> Dict:
        # eval_device = self.device if not self.args.multiprocessing else torch.device(f'cuda:{self.args.main_gpu}')
        
        model.eval()
        model_device = next(model.parameters()).device
        model.to(self.device)
        loss, correct, total = 0, 0, 0

        if type(self.test_loader.dataset) == DatasetSplit:
            C = len(self.test_loader.dataset.dataset.classes)
        else:
            C = len(self.test_loader.dataset.classes)

        class_loss, class_correct, class_total = torch.zeros(C), torch.zeros(C), torch.zeros(C)

        with torch.no_grad():
            # for images, labels in self.loaders["test"]:
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                results = model(images)
                _, predicted = torch.max(results["logit"].data, 1) # if errors occur, use ResNet18_base instead of ResNet18_GFLN
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                bin_labels = labels.bincount()
                class_total[:bin_labels.size(0)] += bin_labels.cpu()
                bin_corrects = labels[(predicted == labels)].bincount()
                class_correct[:bin_corrects.size(0)] += bin_corrects.cpu()

                #calculate CE loss

                this_loss = self.criterion(results["logit"], labels)
                loss += this_loss.sum().cpu()

                for class_idx, bin_label in enumerate(bin_labels):
                    class_loss[class_idx] += this_loss[(labels.cpu() == class_idx)].sum().cpu()


        acc = 100. * correct / float(total)
        class_acc = 100. * class_correct / class_total
        
        loss = loss / float(total)
        class_loss = class_loss / class_total

        # logger.warning(f'[Epoch {epoch}] Test Accuracy: {acc:.2f}%')
        # model.to(model_device)
        # model.train()
        results = {
            "acc": acc,
            'class_acc': class_acc,
            'loss': loss,
            'class_loss' : class_loss
        }
        model.to('cpu')
        del model
        gc.collect()
        
        return results



    def visualize_umap(self, global_model: nn.Module, local_models: List[nn.Module], epoch: int, ):

        # if args.set == 'CIFAR10':
        #     classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # elif args.set == 'MNIST':
        #     classes=['0','1','2','3','4','5','6','7','8','9']
        # elif args.set == 'CIFAR100':
        #     classes= testloader.dataset.classes
        # else:
        #     raise Exception("Not valid args.set")      

        classes = self.test_loader.dataset.classes

        umap_args = self.args.umap
        
        assert umap_args.samples_per_class <= float(len(self.test_loader.dataset)/len(classes))
        #color_cycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']* 100
        # NUM_COLORS = min(len(classes),draw_classes)#100
        # cm = plt.get_cmap('gist_rainbow')
        # color_cycle =[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
        color_cycle  = plt.cm.get_cmap('tab20').colors[:100]
        marker_list = ['o','P','X','^']* 100
        #marker_list = ['o','o','o','o']
        # opacity_max = 1
        # opacity_min = 0.2
        #opacity_list = [1, 0.8, 0.6, 0.4]
        wandb_dict = {}

        # if draw_classes == None:
        #     draw_classes = len(classes)
        # else:
        #     draw_classes = min(draw_classes, len(classes))
        draw_classes = len(classes)

        # if drawing_options == None:
        #     drawing_options = [[True for model in models_dict_list]]
        '''
        saved_features = []
        saved_preds =[]
        for model_dict,name in zip(models_dict_list,names_list):
            model.load_state_dict(model_dict)
            model.eval()
            device = next(model.parameters()).device
            first = True
            with torch.no_grad():
                for data in testloader:
                    # activation = {}
                    # model.layer4.register_forward_hook(get_activation('layer4', activation))
                    images, labels = data[0].to(device), data[1].to(device)
                    all_out = model(images, return_feature = True)
                    outputs = all_out[-1]

                    _, predicted = torch.max(outputs.data, 1)
                    if feat_lev != 4:
                        this_feat = F.adaptive_avg_pool2d(all_out[feat_lev],1)
                    else:
                        this_feat = all_out[feat_lev]
                    if first:
                        features = this_feat.view(len(images),-1)
                        saved_labels = labels
                        saved_pred = predicted
                        first = False
                    else:
                        features = torch.cat((features, this_feat.view(len(images),-1)   ))
                        saved_labels = torch.cat((saved_labels, labels))
                        saved_pred = torch.cat((saved_pred, predicted))

                saved_labels = saved_labels.cpu()
                saved_pred = saved_pred.cpu()

                #breakpoint()
                f1 = metrics.f1_score(saved_labels, saved_pred, average='weighted')
                acc = metrics.accuracy_score(saved_labels, saved_pred)
                #print(len(labels), len(saved_labels))
                #cm = metrics.confusion_matrix(saved_labels, saved_pred)
                wandb_dict[name + " f1"] = f1
                wandb_dict[name + " acc"] = 100 * acc

                
                draw_critic = (saved_labels< draw_classes)
                saved_labels = saved_labels[draw_critic]
                saved_pred = saved_pred[draw_critic]
                saved_preds.append(saved_pred)
                features = features[draw_critic]
                sorted_feature, sorted_label = divide_features_classwise(features, saved_labels.cpu(), num_of_sample_per_class = num_of_sample_per_class, draw_classes = draw_classes)
                sorted_feature = concat_all(sorted_feature)
                saved_features.append(sorted_feature)

                sorted_label = concat_all(sorted_label)

        
        y_test = np.asarray(sorted_label)
        all_feature = torch.cat(saved_features)
        # all_feature = concat_all(saved_features)
        # for idx,a in enumerate(saved_features):
        #     print(idx, len(a))
        '''

        results = self.extract_features(model=global_model, loader=self.test_loader)

        reducer = umap.UMAP(random_state=0, n_components=umap_args.umap_dim, metric='cosine')
        embedding = reducer.fit_transform(results['features'].cpu())
        # embedding_seperate_model = [embedding[len(sorted_label)*j:len(sorted_label) *(j+1)] for j in range(len(models_dict_list))]


        ##################### plot ground truth #######################

        # for drawing_option in drawing_options:
        if True:
            all_names = "umap"
            # for model_option, name in zip(drawing_option, names_list):
            #     if model_option:
            #         all_names += "_" + str(name)

            plt.figure(figsize=(10, 10))

            # if args.umap_dim == 3:
            #     ax = plt.axes(projection=('3d'))
            # else:
            #     ax = plt.axes()
            # this_draw_num = float(sum(drawing_option))
            # this_opacity_gap = (opacity_max-opacity_min)/max((this_draw_num - 1),1)
            
            for c in range(draw_classes):
                first = True
                label_c = (results['labels'] == c)
                count = -1
                # for j in range(len(drawing_option)):
                if True:
                    #breakpoint()
                    # if drawing_option[j]:
                    if True:
                        try:
                            count += 1
                            # this_embedding = embedding_seperate_model[j]
                            #scatter_input = [this_embedding[y_i, k] for k in range(args.umap_dim)]
                            plt.scatter(embedding[label_c, 0], embedding[label_c, 1],
                                        color=color_cycle[c],
                                        marker=marker_list[count],
                                        alpha=1,
                            )
                                        # alpha = opacity_max - this_opacity_gap*count) #, label=classes[i] if first else None
                            plt.xticks([])  # Remove x-axis ticks
                            plt.yticks([])  # Remove y-axis ticks 
                            # first = False
                        except:
                            breakpoint()
            #plt.legend(loc=4)
            plt.legend(loc=4)
            plt.gca().invert_yaxis()
            #plt.show()
            this_name = all_names 
            # + "_truelabels_class" + str(draw_classes) + "feat" + str(feat_lev)
            #plt.savefig(filedir + args.set + args.mode+args.additional_experiment_name+this_name)
            #breakpoint()
            wandb_dict[this_name] = wandb.Image(plt) #filedir +args.set + args.mode + args.additional_experiment_name+this_name+'.png')
            plt.close()
    
            
            
            
            ############### plot model predicted class ###########################
            # plt.figure(figsize=(20, 20))

            # if args.umap_dim == 3:
            #     ax = plt.axes(projection=('3d'))
            # else:
            #     ax = plt.axes()

            # for i in range(draw_classes):
            #     y_i =(np.asarray(saved_pred.cpu()) == i)
            #     scatter_input = [embedding[y_i, k] for k in range(args.umap_dim)]
            #     ax.scatter(*scatter_input, label=classes[i])
            # plt.legend(loc=4)
            # plt.gca().invert_yaxis()

            # wandb_dict[all_names + "model predicted class"] = wandb.Image(plt)
            # plt.close()        
            
        # for model in models_dict_list:  
        #     model.train()
        return wandb_dict
