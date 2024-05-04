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

from utils import terminate_processes, initalize_random_seed
from omegaconf import DictConfig
import faiss

import umap.umap_ as umap
from mlxtend.plotting import plot_confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt



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
        loss, correct, total = 0, 0, 0

        if type(self.test_loader.dataset) == DatasetSplit:
            C = len(self.test_loader.dataset.dataset.classes)
        else:
            C = len(self.test_loader.dataset.classes)

        class_loss, class_correct, class_total = torch.zeros(C), torch.zeros(C), torch.zeros(C)

        with torch.no_grad():
            # for images, labels in self.loaders["test"]:
            for images, labels in self.test_loader:
                images, labels = images.to(device), labels.to(device)

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
        model.to(model_device)
        model.train()
        results = {
            "acc": acc,
            'class_acc': class_acc,
            'loss': loss,
            'class_loss' : class_loss
        }
        
        return results


    #----- UMAP ----#
    @torch.no_grad()
    def visualize_umap(self, global_model: nn.Module, local_models: List[nn.Module], epoch: int):

        #num_of_sample_per_class: int = 100, draw_classes: int = -1
        classes = self.test_loader.dataset.classes
        
        names_list = ["global", "local0", "local1"]
        models_list = [global_model] + local_models[:2]
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
            # [True, True, False, False],
            # [True, False, True, False],
            # [True, False, False, True],
            # [False, True, True, False],
            # [False, True, False, True],
            # [False, False, True, True],
            # [True, False, True, True],
            # [False, True, True, True],
            # [True, True, True, True],
            [True, False, False],
            [False, True, False],
            [False, False, True]
        ]




        

        umap_args = self.args.umap
        
        assert umap_args.samples_per_class <= float(len(self.test_loader.dataset)/len(classes))
        color_cycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
        #color_cycle  = plt.cm.get_cmap('tab20').colors[:100]
        marker_list = ['o','P','X','^']* 100
        
        opacity_max = 1
        opacity_min = 0.2
        
        wandb_dict = {}


        if umap_args.draw_classes < 0 :
            draw_classes = len(classes)
        else:
            draw_classes = min(umap_args.draw_classes,len(classes))



        for name, model in zip(names_list, models_list):
            results = self.extract_features(model=model, loader=self.test_loader)
            this_features = results['features']
            this_labels = results['labels']
            #print("Feature shape :", this_features.shape)

            for feat,label in zip(results['features'], results['labels']):
                if label < draw_classes:
                    if eval_result_dict[name]['labels_num_for_each_class'][str(label.item())] < umap_args.samples_per_class :
                        eval_result_dict[name]['saved_features'][str(label.item())] = torch.cat((eval_result_dict[name]['saved_features'][str(label.item())], feat.cpu().reshape(1,feat.size(0))), dim = 0)
                        eval_result_dict[name]['labels_num_for_each_class'][str(label.item())] += 1


        #post_processing
        all_features = torch.tensor([])
        num_of_samples_for_each_model = {}
        #end_idx = 0
        for name in names_list:
            num_of_samples_for_each_model[name] = 0
            for class_idx, each_class in enumerate(classes):
                all_features= torch.cat((all_features,eval_result_dict[name]['saved_features'][str(class_idx)]))
                num_of_samples_for_each_model[name] += eval_result_dict[name]['labels_num_for_each_class'][str(class_idx)]


        

        #print("all_features shape : ", (all_features.shape))
        reducer = umap.UMAP(random_state=0, n_components=umap_args.umap_dim, metric='cosine')
        embedding = reducer.fit_transform((all_features))
        #print("embedding shape : ", (embedding.shape))

        idx = 0
        for name in names_list:
            this_model_embeddings = embedding[idx: idx + num_of_samples_for_each_model[name]]
            this_class_idx_start = 0
            for class_idx, each_class in enumerate(classes):
                eval_result_dict[name]['saved_embeddings'][str(class_idx)] = this_model_embeddings[this_class_idx_start : this_class_idx_start + eval_result_dict[name]['labels_num_for_each_class'][str(class_idx)]]
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

            plt.figure(figsize=(10, 10))

            # if args.umap_dim == 3:
            #     ax = plt.axes(projection=('3d'))
            # else:
            #     ax = plt.axes()
            this_draw_num = float(sum(drawing_option))
            this_opacity_gap = (opacity_max-opacity_min)/max((this_draw_num - 1),1)
            
            for i in range(draw_classes):
                first = True
                #y_i = (y_test == i)
                count = -1
                for j in range(len(drawing_option)):
                    #breakpoint()
                    if drawing_option[j]:
                        try:
                            this_name = names_list[j]
                            count += 1
                            this_embedding = eval_result_dict[this_name]['saved_embeddings'][str(i)]
                            #scatter_input = [this_embedding[y_i, k] for k in range(args.umap_dim)]
                            plt.scatter(this_embedding[:,0],this_embedding[:,1], color = color_cycle[i], marker =marker_list[count], alpha = opacity_max - this_opacity_gap*count) #, label=classes[i] if first else None
                            plt.xticks([])  # Remove x-axis ticks
                            plt.yticks([])  # Remove y-axis ticks 
                            first = False
                        except:
                            breakpoint()
            #plt.legend(loc=4)
            plt.legend(loc=4)
            plt.gca().invert_yaxis()
            #plt.show()
            fig_name = all_names
            # + "_truelabels_class" + str(draw_classes) + "feat" + str(feat_lev)
            #plt.savefig(filedir + args.set + args.mode+args.additional_experiment_name+this_name)
            #breakpoint()
            wandb_dict[fig_name] = wandb.Image(plt) #filedir +args.set + args.mode + args.additional_experiment_name+this_name+'.png')
            plt.close()
    
            
        return wandb_dict

    # ----- SVD, feature norm, class norm ----#
    @torch.no_grad()
    def visualize_svd(self, epoch: int, local_models: List[nn.Module], global_model: nn.Module, device: torch.device = None, stride=100):

        wandb_dict = {}
        global_model.eval()
        local_model = local_models[0].eval()
        model_device = next(global_model.parameters()).device
        if device is None:
            device = self.device
        global_model.to(device)
        local_model.to(device)

        global_features = []
        local_features = []
        labels = []
        num_classes = len(self.test_loader.dataset.classes)
        num_samples = len(self.test_loader.dataset)
        num_samples_per_class = int(num_samples / num_classes / stride)

        with torch.no_grad():
            # for images, labels in self.loaders["test"]:
            for images, label in self.test_loader:
                images, label = images.to(device), label.to(device)

                results = global_model(images)
                global_features.append(results['feature'].cpu())
                results = local_model(images)
                local_features.append(results['feature'].cpu())
                labels.append(label)
            global_features = torch.cat(global_features)
            local_features = torch.cat(local_features)
            labels = torch.cat(labels)

            idxs = labels.argsort()
            labels = labels[idxs]
            global_features = global_features[idxs, :]
            local_features = local_features[idxs, :]

            sampled_gf = global_features[::stride]
            sampled_lf = local_features[::stride]
            sampled_gt = labels[::stride]

            covariance_matrix = torch.cov(sampled_gf)
            eigen_vals = torch.log(torch.linalg.svdvals(covariance_matrix))

            data = list(zip(range(len(eigen_vals)), eigen_vals))
            wandb_dict["Singular values global_model"] = plot_line(data, "x", "y", "Singular values global_model")


            covariance_matrix = torch.cov(sampled_lf)
            eigen_vals = torch.log(torch.linalg.svdvals(covariance_matrix))

            data = list(zip(range(len(eigen_vals)), eigen_vals))
            wandb_dict["Singular values local_model"] = plot_line(data, "x", "y", "Singular values local_model")




            norm_gf = sampled_gf.view(num_classes, num_samples_per_class, -1).norm(dim=2).mean(1)
            norm_lf = sampled_lf.view(num_classes, num_samples_per_class, -1).norm(dim=2).mean(1)

            classes = range(num_classes)
            ## Feature norm of global model
            data = list(zip(classes, norm_gf))
            wandb_dict["feature norm global_model"] = plot_bar(data, "label", "value", "feature norm global_model")

            ## Feature norm of local model
            data = list(zip(classes, norm_lf))
            wandb_dict["feature norm local_model"] = plot_bar(data, "label", "value", "feature norm local_model")

        global_model.to(model_device)
        local_model.to(model_device)
        global_model.train()
        
#         import pdb; pdb.set_trace()
#         print(123)


        return wandb_dict


    def visualize_loss_landscape(self):
        return















    #----- Retrieval ----#



    @torch.no_grad()
    def extract_features(self, model, loader, device: torch.device = None):
        model_device = next(model.parameters()).device
        if device is None:
            device = self.device
        model.to(device)
        features, labels = [], []
        with torch.no_grad():
            for g_data, g_label in tqdm.tqdm(loader, desc='loader'):
                g_data = g_data.to(self.device, non_blocking=True)
                g_label = g_label.to(self.device, non_blocking=True)


                g_outs = model(g_data)
                features.append(g_outs["feature"].squeeze())
                labels.append(g_label)

        features = torch.cat(features)
        labels = torch.cat(labels)

        ## DDP
        features = all_gather_nd(features)
        labels = all_gather_nd(labels)
        results = {
            "features": features,
            "labels": labels,
        }
        model.to(model_device)
        model.train()
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


    def eval_retrieval(self, gallery_model, query_model, **kwargs):

        gallery_model.eval()
        query_model.eval()

        gallery_results = self.extract_features(gallery_model, self.gallery_loader)
        query_results = self.extract_features(query_model, self.query_loader)

        # if get_rank() != 0:
        #     return None, None
        
        ranked_scores, ranked_gallery_indices = self.calculate_rank(query_results['features'].cpu().numpy(),
                                                                    gallery_results['features'].cpu().numpy(), 
                                                                    topk=gallery_results['features'].size(0))

        map_scores = self.calculate_mAP(ranked_gallery_indices, query_results['labels'].cpu().numpy(), gallery_results['labels'].cpu().numpy(), verbose=True)
        cmc_results = self.calculate_CMC(ranked_gallery_indices, query_results['labels'].cpu().numpy(), gallery_results['labels'].cpu().numpy(), topk=5, verbose=True)
        cmc_scores = cmc_results["topk"]

        results = {}

def plot_line(data, x_label, y_label, title):
    fig, ax = plt.subplots()
    x, y = zip(*data)
    ax.plot(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()
    return wandb.Image(fig)

def plot_bar(data, x_label, y_label, title):
    fig, ax = plt.subplots()
    x, y = zip(*data)
    ax.bar(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()
    return wandb.Image(fig)