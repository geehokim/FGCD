import torch
import copy
from torch.utils.data import DataLoader
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
import gc
from finch import FINCH
from utils import linear_assignment, OOD_FINCH
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score, silhouette_score
from utils import extract_local_features
from validclust import dunn
from s_dbw import S_Dbw
from sklearn.metrics import calinski_harabasz_score
import torch.nn.functional as F
from scipy.stats import chi2
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from torchmetrics.clustering import DunnIndex
import time
import math
import torch.nn as nn


# def apply_label_noise(y, num_classes, noise_rate):
    
#     select_index = (torch.rand(y.size(0)) < noise_rate).nonzero()
#     if len(select_index) > 0:
#         random_label = torch.randint(0, num_classes, select_index.size()).long()
#         origin_label = y[select_index]
#         random_label += (origin_label == random_label).long()
#         random_label %= num_classes
#         y[select_index] = random_label
#     return y

class EMAScheduler(nn.Module):
    """"
    Following muGCD to implement this, https://arxiv.org/pdf/2311.17055.pdf
    """
    def __init__(self, w_base=0.7, w_t=0.999, max_epoch=200):
        super().__init__()
        self.w_base     = w_base
        self.w_t        = w_t
        self.max_epoch  = max_epoch
    
    def forward(self, cur_epoch):
        if cur_epoch > self.max_epoch:
            return self.w_t
        return self.w_t - (1-self.w_base)*(math.cos((math.pi*cur_epoch)/self.max_epoch)+1)/2

def normalize_score(score, min_val, max_val):
    return (score - min_val) / (max_val - min_val)

def estimate_num_clusters(feats_labelled, targets_labelled, feats_unlabelled, targets_unlabelled, k_mode='combined', total_classes=10, tolerance=10, evaler=None):
    print(f'Estimating the number of clusters....')
    # 클러스터링 결과를 저장할 리스트
    clustering_results = []

    # 클러스터링 알고리즘 선택
    algorithm = 'finch'  # 예를 들어, 'finch' 알고리즘을 사용합니다.

    # Combine labelled and unlabelled features
    feats_combined = np.vstack([feats_labelled, feats_unlabelled])
    targets_combined = np.concatenate([targets_labelled, targets_unlabelled])

    num_seen_classes = len(set(targets_labelled))

    best_k = num_seen_classes
    best_k_combined = num_seen_classes
    best_ari_labelled = -1
    best_ari_unlabelled = -1
    best_ari_combined = -1
    best_ari_labelled_sil_unlabelled = -1
    best_sil_combined = -1
    best_s_dbw_index = 100000
    best_calinski_score = -1
    best_score = -1

    results = {
        'labelled': [],
        'unlabelled': [],
        'combined': [],
        'ari_sil_combined': [],
        'sil_combined': [],
        's_dbw_index': [],
        'dunn_index': [],
        'calinski_harabasz_score': []
    }
    class_set_labelled = np.unique(targets_labelled)
    client_mask = np.zeros(len(targets_combined), dtype=bool)
    # client_seen_classes = set(args.dataset.seen_classes)
    for i, target in enumerate(targets_combined):
        if target in class_set_labelled:
            client_mask[i] = True
    
    current_tolerance = 0
    c, num_clust, clusters_combined = FINCH(feats_combined, verbose=True)

    for i, k in enumerate(num_clust):
        if k < len(class_set_labelled):
            continue
        
        preds = c[:, i]

        start_time = time.time()
        # Calculate accuracies for this client
        all_acc, old_acc, new_acc, w, _ = evaler.log_accs_from_preds(
            y_true=targets_combined,
            y_pred=preds,
            mask=client_mask,
            T=0,
            eval_funcs=['v3'],
            save_name=f'Local Clustering Client {0}'
        )
        end_time = time.time()
        print(f'Time taken to calculate accuracies: {end_time - start_time:.4f} seconds')
        # preds_labelled = preds[:len(targets_labelled)]
        # ari_combined = adjusted_rand_score(targets_combined, preds)
        start_time = time.time()
        ari_labelled = adjusted_rand_score(targets_labelled, preds[:len(targets_labelled)])
        end_time = time.time()
        print(f'Time taken to calculate ari_labelled: {end_time - start_time:.4f} seconds')

        # start_time = time.time()
        # sil_combined = silhouette_score(feats_combined, preds)
        # end_time = time.time()
        # print(f'Time taken to calculate sil_combined: {end_time - start_time:.4f} seconds')
        # sil_unlabelled = silhouette_score(feats_unlabelled, preds[len(targets_labelled):])
        # dunn_index = DunnIndex(p=2)
        # dunn_index.update(torch.from_numpy(feats_combined), torch.from_numpy(preds))
        # dunn_index_value = dunn_index.compute()
        

        # score = 0.5 * ari_labelled + 0.5 * sil_combined
        score = ari_labelled
        
        # print(f'k={k} - score={score:.4f} | sil_combined={sil_combined:.4f} | sil_unlabelled={sil_unlabelled:.4f} | dunn_index={dunn_index_value:.4f} | ari_labelled={ari_labelled:.4f} | all_acc={all_acc:.4f} | old_acc={old_acc:.4f} | new_acc={new_acc:.4f}')
        print(f'k={k} - score={score:.4f} | ari_labelled={ari_labelled:.4f} | all_acc={all_acc:.4f} | old_acc={old_acc:.4f} | new_acc={new_acc:.4f}')
    
        # Update best k and ARI for combined data
        if score >= best_score:
            best_score = score
            best_k_combined = k
    
    print(f'# of seen classes: {len(class_set_labelled)}')
    for i in range(len(class_set_labelled), total_classes + 1):
        print(f'k={i}')
        _, _, clusters_combined = FINCH(feats_combined, req_clust=i, verbose=False)

        # Calculate accuracies for this client
        all_acc, old_acc, new_acc, w, _ = evaler.log_accs_from_preds(
            y_true=targets_combined,
            y_pred=clusters_combined,
            mask=client_mask,
            T=0,
            eval_funcs=['v3'],
            save_name=f'Local Clustering Client {0}'
        )
        
        ari_labelled = adjusted_rand_score(targets_labelled, clusters_combined[:len(targets_labelled)])
        # sil_combined = silhouette_score(feats_combined, clusters_combined)  
        # sil_unlabelled = silhouette_score(feats_unlabelled, clusters_combined[len(targets_labelled):])
        # dunn_index = DunnIndex(p=2)
        # dunn_index.update(torch.from_numpy(feats_combined), torch.from_numpy(clusters_combined))
        # dunn_index_value = dunn_index.compute()
        
        # score = 0.5 * ari_labelled + 0.5 * sil_combined
        score = ari_labelled
        print(f'k={i} - score={score:.4f} | ari_labelled={ari_labelled:.4f} | all_acc={all_acc:.4f} | old_acc={old_acc:.4f} | new_acc={new_acc:.4f}')

        # Update best k and ARI for combined data
        if score >= best_score:
            best_score = score
            best_k_combined = i
            current_tolerance = 0
        else:
            current_tolerance += 1

        if current_tolerance >= tolerance:
            break

    # print(f'Best - ARI Combined: k={best_k_combined} | ari_combined={best_ari_combined:.4f}')
    print(f'Best - Score: k={best_k_combined} | score={best_score:.4f}')
    if best_k_combined >= total_classes:
        return total_classes
    else:
        return best_k_combined

    # for k in range(num_seen_classes+1, total_classes+1):
    #     # Perform FINCH clustering for each dataset
    #     # _, _, clusters_labelled = FINCH(feats_labelled, req_clust=k, verbose=False)
    #     # _, _, clusters_unlabelled = FINCH(feats_unlabelled, req_clust=k, verbose=False)
    #     c, num_clust, clusters_combined = FINCH(feats_combined, req_clust=k, verbose=False)

    #     # # Calculate ARI score for labelled data
    #     # ari_labelled = adjusted_rand_score(targets_labelled, clusters_labelled)
    #     # # Calculate ARI score for unlabelled data
    #     # ari_unlabelled = adjusted_rand_score(targets_unlabelled, clusters_unlabelled)
    #     # Calculate ARI score for combined data
    #     ari_combined = adjusted_rand_score(targets_combined, clusters_combined)


    #     # # Calculate silhouette score for unlabelled data
    #     # sil_unlabelled = silhouette_score(feats_unlabelled, clusters_unlabelled)
    #     # Calculate silhouette score for combined data
    #     sil_combined = silhouette_score(feats_combined, clusters_combined)

    #     # Calculate s_dbw index for combined data
    #     # s_dbw_index = S_Dbw(feats_combined, clusters_combined)

    #     # Calculate dunn index for combined data
    #     #dunn_index_value = dunn(feats_combined, clusters_combined)
    #     # dunn_index_value = 0

    #     # Calculate calinski_harabasz_score for combined data
    #     calinski_harabasz_score_value = calinski_harabasz_score(feats_combined, clusters_combined)

    #     # sil_unlabelled = normalize_score(sil_unlabelled, -1, 1)
    #     # sil_combined = normalize_score(sil_combined, -1, 1)
    #     # ari_labelled_normalized = normalize_score(ari_labelled, -1, 1)

    #     # ari_sil_combined = 0.5 * ari_labelled_normalized + 0.5 * sil_unlabelled
    #     # #print(f'k={k} - sil unlabelled: {sil_unlabelled:.4f} | sil combined: {sil_combined:.4f}')

    #     # if calinski_harabasz_score_value >= best_calinski_score:
    #     #     best_calinski_score = calinski_harabasz_score_value
    #     #     best_k_calinski_harabasz_score = k
            

    #     # if s_dbw_index <= best_s_dbw_index:
    #     #     best_s_dbw_index = s_dbw_index
    #     #     best_k_s_dbw_index = k
    #     if calinski_harabasz_score_value >= best_calinski_score:
    #         best_calinski_score = calinski_harabasz_score_value
    #         best_k_calinski_harabasz_score = k
    #         current_tolerance = 0
    #     else:
    #         current_tolerance += 1


    #     if sil_combined >= best_sil_combined:
    #         best_sil_combined = sil_combined
    #         best_k_sil_combined = k
    #     #     current_tolerance = 0
    #     # else:
    #     #     current_tolerance += 1

    #     # if ari_sil_combined >= best_ari_labelled_sil_unlabelled:
    #     #     best_ari_labelled_sil_unlabelled = ari_sil_combined
    #     #     best_k_ari_sil_combined = k

    #     # # Update best k and ARI for labelled data
    #     # if ari_labelled >= best_ari_labelled:
    #     #     best_ari_labelled = ari_labelled
    #     #     best_k_labelled = k

    #     # # Update best k and ARI for unlabelled data
    #     # if ari_unlabelled >= best_ari_unlabelled:
    #     #     best_ari_unlabelled = ari_unlabelled
    #     #     best_k_unlabelled = k

    #     # Update best k and ARI for combined data
    #     if ari_combined >= best_ari_combined:
    #         best_ari_combined = ari_combined
    #         best_k_combined = k
            

    #     # results['labelled'].append([k, ari_labelled])
    #     # results['unlabelled'].append([k, ari_unlabelled])
    #     results['combined'].append([k, ari_combined])
    #     # results['ari_sil_combined'].append([k, ari_sil_combined])
    #     results['sil_combined'].append([k, sil_combined])
    #     # results['s_dbw_index'].append([k, s_dbw_index])
    #     # results['dunn_index'].append([k, dunn_index_value])
    #     results['calinski_harabasz_score'].append([k, calinski_harabasz_score_value])

    #     if current_tolerance >= tolerance:
    #         break

    # for i in range(len(results['combined'])):
    #     k = results['combined'][i][0]
    #     result_combined = results['combined'][i][1]
    #     # result_s_dbw_index = results['s_dbw_index'][i][1]
    #     result_calinski_harabasz_score = results['calinski_harabasz_score'][i][1]
    #     result_sil_combined = results['sil_combined'][i][1]

    #     print(f'k={k} - combined={result_combined:.4f} | calinski_harabasz_score={result_calinski_harabasz_score:.4f} | sil_combined={result_sil_combined:.4f}')
    # # Log best results for each dataset
    # print(f'Best - ARI Combined: k={best_k_combined} | calinski_harabasz_score={best_k_calinski_harabasz_score:.4f} | sil_combined={best_k_sil_combined:.4f}')

    # if k_mode == 'combined':
    #     return best_k_combined
    # elif k_mode == 'calinski':
    #     return best_k_calinski_harabasz_score
    # elif k_mode == 'sil':
    #     return best_k_sil_combined
    # else:
    #     assert False

def calculate_accuracies(clusters, targets):
    # This is a placeholder function. You need to implement the actual accuracy calculations.
    # The implementation will depend on how you define "seen" and "unseen" classes.
    all_acc = adjusted_rand_score(targets, clusters)
    seen_acc = 0  # Calculate seen accuracy
    unseen_acc = 0  # Calculate unseen accuracy
    return all_acc, seen_acc, unseen_acc
    


def update_prior(args, prior_dist, model, loader, evaler, num_classes, epoch, device, K=None):
    # model.eval()
    # # model_device = next(model.parameters()).device
    # model.to(device)
    # test_transform = copy.deepcopy(evaler.test_loader.dataset.transform)

    # labelled_dataset = copy.deepcopy(loader.dataset.labelled_dataset)
    # labelled_dataset.transform = test_transform
    # unlabelled_dataset = copy.deepcopy(loader.dataset.unlabelled_dataset)
    # unlabelled_dataset.transform = test_transform

    # all_feats_labelled = []
    # all_feats_proj_labelled = []
    # all_logits_labelled = []
    # targets_labelled = np.array([])
    # mask = np.array([])

    # labelled_loader = DataLoader(labelled_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # all_feats_unlabelled = []
    # all_feats_proj_unlabelled = []
    # all_logits_unlabelled = []
    # targets_unlabelled = np.array([])

    # unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    # print('Collating logits...')
    # preds = []
    # # First extract feats labelled
    # for batch_idx, (images, label, _) in enumerate(labelled_loader):
    #     images = images.to(device)

    #     # Pass features through base model and then additional learnable transform (linear layer)
    #     with torch.no_grad():
    #         feats, feats_proj, logits = model(images, return_all=True)

    #     preds.append(logits.argmax(1).cpu().numpy())
    #     targets_labelled = np.append(targets_labelled, label.cpu().numpy().astype(int))
    #     all_feats_labelled.append(feats.cpu().clone())
    #     all_feats_proj_labelled.append(feats_proj.cpu().clone())
    #     all_logits_labelled.append(logits)
    #     # mask = np.append(mask, np.array([True if x.item() in range(len(args.dataset.seen_classes))
    #     #                                  else False for x in label]))

    # # First extract feats unlabelled
    # for batch_idx, (images, label, _) in enumerate(unlabelled_loader):
    #     images = images.to(device)

    #     # Pass features through base model and then additional learnable transform (linear layer)
    #     with torch.no_grad():
    #         feats, feats_proj, logits = model(images, return_all=True)

    #     preds.append(logits.argmax(1).cpu().numpy())
    #     targets_unlabelled = np.append(targets_unlabelled, label.cpu().numpy().astype(int))
    #     all_feats_unlabelled.append(feats.cpu().clone())
    #     all_feats_proj_unlabelled.append(feats_proj.cpu().clone())
    #     all_logits_unlabelled.append(logits)
    #     mask = np.append(mask, np.array([True if x.item() in range(len(args.dataset.seen_classes))
    #                                      else False for x in label]))

    # preds = np.concatenate(preds)
    # feats_labelled = torch.cat(all_feats_labelled, dim=0)
    # feats_proj_labelled = torch.cat(all_feats_proj_labelled, dim=0)
    # #targets_labelled = torch.cat(targets_labelled, dim=0)

    # feats_unlabelled = torch.cat(all_feats_unlabelled, dim=0)
    # feats_proj_unlabelled = torch.cat(all_feats_proj_unlabelled, dim=0)
    # #targets_unlabelled = torch.cat(targets_unlabelled, dim=0)

    # all_feats = torch.cat([feats_labelled, feats_unlabelled], dim=0)
    # all_feats_proj = torch.cat([feats_proj_labelled, feats_proj_unlabelled], dim=0)
    # all_targets = np.concatenate([targets_labelled, targets_unlabelled])

    # logits_labelled = torch.cat(all_logits_labelled, dim=0)
    # logits_unlabelled = torch.cat(all_logits_unlabelled, dim=0)
    # all_logits = torch.cat([logits_labelled, logits_unlabelled], dim=0)
    # mask = mask.astype(bool)
    
    # soft_preds = (all_logits / args.client.soft_preds_temp).softmax(dim=1)
    start_time = time.time()
    feats_labelled, feats_proj_labelled, logits_labelled, targets_labelled, feats_unlabelled, feats_proj_unlabelled, logits_unlabelled, targets_unlabelled, mask = extract_local_features(args, model, loader, evaler, device)
    end_time = time.time()
    print(f'Time taken to extract local features: {end_time - start_time:.4f} seconds')

    targets_unlabelled = targets_unlabelled.astype(int)
    targets_labelled = targets_labelled.astype(int)
    all_feats = torch.cat([feats_labelled, feats_unlabelled], dim=0)
    all_feats_proj = torch.cat([feats_proj_labelled, feats_proj_unlabelled], dim=0)
    all_targets = np.concatenate([targets_labelled, targets_unlabelled])
    all_logits = torch.cat([logits_labelled, logits_unlabelled], dim=0)


    if args.client.est_num_clusters:
        ## Estimate num_clusters
        if args.client.clust_feats == 'feats':
            num_clusters = estimate_num_clusters(feats_labelled.numpy(), targets_labelled, feats_unlabelled.numpy(), targets_unlabelled, k_mode=args.client.k_mode, total_classes=num_classes, tolerance=args.client.tolerance, evaler=evaler)
        elif args.client.clust_feats == 'feats_proj':
            num_clusters = estimate_num_clusters(feats_proj_labelled.numpy(), targets_labelled, feats_proj_unlabelled.numpy(), targets_unlabelled, k_mode=args.client.k_mode, total_classes=num_classes, tolerance=args.client.tolerance)
    else:
        num_clusters = K

    
    ## Perform local clustering
    if args.client.update_alg =='finch':
        if args.client.clust_feats == 'feats':
            c, num_clust, req_c = FINCH(all_feats.numpy(), req_clust=num_clusters, distance='cosine', verbose=False)
        elif args.client.clust_feats == 'feats_proj':
            c, num_clust, req_c = FINCH(all_feats_proj.numpy(), req_clust=num_clusters, distance='cosine', verbose=False)
        req_c = req_c.astype(int)
        targets = all_targets.astype(int)        

    elif args.client.update_alg =='ood_finch':

        lambda_param = args.client.lambda_param
        similarities = cosine_similarity(feats_unlabelled, feats_labelled)
        # k-th nearest neighbor 찾기
        kth_similarities = np.sort(similarities, axis=1)[:, -1]
        ood_scores = 1 - kth_similarities

        c, num_clust, req_c = OOD_FINCH(feats_unlabelled.numpy(), req_clust=num_clusters, ood_scores=ood_scores, lambda_param=lambda_param, distance='cosine', verbose=False, labeled_features=feats_labelled.numpy())
        req_c = req_c.astype(int)
        targets = targets_unlabelled.astype(int)

    elif args.client.update_alg =='semi_finch':

        orig_dist = metrics.pairwise.pairwise_distances(all_feats.numpy(), all_feats.numpy(), metric='cosine')
        orig_dist_copy = copy.deepcopy(orig_dist)

        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)

        orig_dist_labelled = orig_dist_copy[:len(targets_labelled)]
        for cls in np.unique(targets_labelled):
           indices = np.where(targets_labelled == cls)[0]
           cls_dist = orig_dist_labelled[indices]
           cls_rank = np.argmax(cls_dist, axis=1)
           initial_rank[indices] = cls_rank

        c, num_clust, req_c = FINCH(all_feats.numpy(), initial_rank=initial_rank, req_clust=num_clusters, distance='cosine', verbose=False)
        targets = all_targets.astype(int)
        req_c = req_c.astype(int)
    else:
        assert False

    ## Eval Clustering results
    all_acc, old_acc, new_acc, w, ind_map_gt = evaler.split_cluster_acc_v2(targets, req_c, mask)

    prior = torch.zeros(num_classes)
    total = 0
    
    # update prior for labelled data
    for i in range(len(targets_labelled)):
        prior[int(targets_labelled[i])] += 1
        total += 1
    

    # Update prior for unlabelled data
    if args.client.align_gt:
        # Align Class idxes with ground truths
        ind_map_pred_to_gt = {j: i for i, j in ind_map_gt.items()}
        ind_map_pred_to_gt = dict(sorted(ind_map_pred_to_gt.items()))
    
    else:
        if args.client.align_type == 'sample':
            preds_unlabelled = logits_unlabelled.argmax(1).cpu().numpy()
            concat_targets = np.concatenate([targets_labelled, preds_unlabelled]).astype(int)
            D = num_classes
            w = np.zeros((D, D), dtype=int)
            for i in range(concat_targets.size):
                w[req_c[i], concat_targets[i]] += 1

            ind = linear_assignment(w.max() - w)
            ind = np.vstack(ind).T
            ind_map = {i: j for i, j in ind}
            ind_map_pred_to_gt = dict(sorted(ind_map.items()))
            print(f'Prior update process:')
            print(f'ind_map_pred_to_gt_logit: {ind_map_pred_to_gt}')

        elif args.client.align_type == 'centroid':

            # Get the classifier weights
            classifier_weights = model.proj_layer.last_layer.parametrizations.weight.original1.data.clone().cpu()
            
            targets_labelled = torch.LongTensor(targets_labelled)
            if args.client.clust_feats == 'feats':
                #feats_labelled = F.normalize(feats_labelled, dim=1, p=2)
                feats_labelled = feats_labelled
            elif args.client.clust_feats == 'feats_proj':
                feats_labelled = feats_proj_labelled

            unique_labels_labelled = torch.unique(targets_labelled).tolist()                
            class_centroids_labelled = [feats_labelled[targets_labelled == label].mean(dim=0) for label in unique_labels_labelled]
            class_centroids_labelled = torch.stack(class_centroids_labelled)

            # Align class indexes using cluster prototypes and classifier weights
            # Calculate prototype for each cluster
            cluster_set = set(req_c)
            cluster_prototypes = []
            #cluster_stds = []
            for cluster_ind in range(len(cluster_set)):

                # get the mean of features for the cluster
                if args.client.clust_feats == 'feats':
                    cluster_feats = all_feats[req_c == cluster_ind]
                elif args.client.clust_feats == 'feats_proj':
                    cluster_feats = all_feats_proj[req_c == cluster_ind]
                cluster_mean = cluster_feats.mean(dim=0)
                cluster_prototypes.append(cluster_mean)

                # Calculate standard deviation for the cluster
                #cluster_std = cluster_feats.std(dim=0)
                # for numerical stability
                #cluster_std[cluster_std == 0] = 1e-8
                #cluster_stds.append(cluster_std)

            cluster_prototypes = torch.stack(cluster_prototypes)
            #cluster_stds = torch.stack(cluster_stds)

            seen_class_similarities = F.cosine_similarity(class_centroids_labelled.unsqueeze(1), cluster_prototypes.unsqueeze(0), dim=2)


            # Initialize the mapping dictionary
            ind_map_gt_to_pred = {}
            assigned_clusters = set()
            remaining_clusters = [i for i in range(len(cluster_set))]
            not_assign_list = [i for i in range(num_classes)]

            ## Filter out the classes that have wasserstein distance less than threshold
            threshold = args.client.align_threshold
            for i, label in enumerate(unique_labels_labelled):
                class_centroid = class_centroids_labelled[i]
                simiilarities_between_clusters = seen_class_similarities[i]
                cluster_ind = torch.argmax(simiilarities_between_clusters).item()
                print(f'class {label} - top1 cluster index: {cluster_ind}')
                if cluster_ind in assigned_clusters:
                    continue
                ## Estimate distribution of the cluster
                if args.client.clust_feats == 'feats':
                    cluster_feats = all_feats[req_c == cluster_ind]
                elif args.client.clust_feats == 'feats_proj':
                    cluster_feats = all_feats_proj[req_c == cluster_ind]
                gmm_cluster = GaussianMixture(n_components=1).fit(cluster_feats.cpu().clone().numpy())
                cluster_mean = gmm_cluster.means_[0]
                cluster_var = np.diag(gmm_cluster.covariances_[0])
                diff = class_centroid.cpu().clone().numpy() - cluster_mean
                inv_var = 1 / cluster_var
                mean_var = np.sqrt(np.sum(cluster_var))
                wasserstein_dist = np.sqrt(np.sum((diff ** 2) * inv_var))
                print(f'class {label} - wasserstein distance: {wasserstein_dist} - mean_var: {mean_var} - threshold: {args.client.align_threshold * mean_var}')
                if wasserstein_dist <= threshold * mean_var:
                    ind_map_gt_to_pred[label] = cluster_ind
                    assigned_clusters.add(cluster_ind)
                    remaining_clusters.remove(cluster_ind)


            # Calculate pairwise cosine similarity matrix between classifier weights and cluster prototypes 
            cosine_similarity_matrix = F.cosine_similarity(classifier_weights.unsqueeze(1), 
                                                        cluster_prototypes.unsqueeze(0), dim=2)
            
            # Get the number of seen classes
            num_seen_classes = len(args.dataset.seen_classes)
            
            

            # # Handle labelled set classes
            # for i, label in enumerate(filtered_labels_labelled):
            #     class_centroid = class_centroids_labelled[i]
            #     similarities = F.cosine_similarity(class_centroid.unsqueeze(0), cluster_prototypes)
            #     sorted_indices = torch.argsort(similarities, descending=True)
            #     for idx in sorted_indices:
            #         best_cluster_idx = idx.item()
            #         if best_cluster_idx in remaining_clusters:
            #             # # calculate z-score
            #             # z_score = (class_centroid - cluster_prototypes[best_cluster_idx]) / cluster_stds[best_cluster_idx]
            #             # D_approx = torch.sqrt((z_score ** 2).sum())

            #             # # 카이제곱 분포를 사용하여 확률로 변환
            #             # degrees_of_freedom = len(class_centroid)
            #             # probability = chi2.cdf(D_approx ** 2, degrees_of_freedom)

            #             # # cosine similarity를 softmax해서 확률로 변환
            #             # sim_cluster_classifier = F.cosine_similarity(cluster_prototypes[best_cluster_idx].unsqueeze(0), classifier_weights, dim=1)
            #             # probabilities = torch.softmax(sim_cluster_classifier / args.client.assoc_temp, dim=0)
            #             # probability = probabilities[label.item()]
                        
                        
            #             # probabilities = torch.softmax(similarities / curr_assoc_temp, dim=0)
            #             # probability = probabilities[best_cluster_idx]

            #             # Sample a value from a uniform distribution between 0 and 1
            #             # sampled_value = torch.rand(1).item()
            #             # print(f'probabilities: {probabilities}')
            #             # print(f'class {label.item()} - cluster {best_cluster_idx} - similarity: {similarities[best_cluster_idx]}  - Probability: {probability} - sampled_value: {sampled_value}')
            #             # if sampled_value <= probability:
            #             #     ind_map_gt_to_pred[label.item()] = best_cluster_idx
            #             #     assigned_clusters.add(best_cluster_idx)
            #             #     remaining_clusters.remove(best_cluster_idx)
            #             #     break
                        
            #             ind_map_gt_to_pred[label] = best_cluster_idx
            #             assigned_clusters.add(best_cluster_idx)
            #             remaining_clusters.remove(best_cluster_idx)
            #             break
                        

                        
            # Handle unseen classes
            remaining_clusters_copy = copy.deepcopy(set(remaining_clusters))
            unseen_class_weights = set(range(num_seen_classes, num_classes))

            # Hungarian matching for unseen classes
            if remaining_clusters_copy and unseen_class_weights:
                remaining_clusters_copy = list(remaining_clusters_copy)
                unseen_class_weights = list(unseen_class_weights)
                cost_matrix = torch.zeros((len(unseen_class_weights), len(remaining_clusters_copy)))
                for i, weight_idx in enumerate(unseen_class_weights):
                    for j, cluster_idx in enumerate(remaining_clusters_copy):
                        cost_matrix[i, j] = 1 - cosine_similarity_matrix[weight_idx, cluster_idx]
                row_ind, col_ind = linear_assignment(cost_matrix.cpu().numpy())
                for i, j in zip(row_ind, col_ind):
                    ind_map_gt_to_pred[unseen_class_weights[i]] = remaining_clusters_copy[j]
                    assigned_clusters.add(remaining_clusters_copy[j])
                    remaining_clusters.remove(remaining_clusters_copy[j])

            # Greedy matching for remaining seen classes
            remaining_seen_weights = set(range(num_seen_classes)) - set(ind_map_gt_to_pred.keys())
            if remaining_seen_weights:
                remaining_seen_weights = list(remaining_seen_weights)
                for weight_idx in remaining_seen_weights:
                    similarities = cosine_similarity_matrix[weight_idx]
                    sorted_indices = torch.argsort(similarities, descending=True)
                    for idx in sorted_indices:
                        if idx.item() in remaining_clusters:
                            ind_map_gt_to_pred[weight_idx] = idx.item()
                            remaining_clusters.remove(idx.item())
                            break
            
            print(f"ind_map_gt_to_pred not sorted!: {ind_map_gt_to_pred}")
            assert len(remaining_clusters) == 0


            # for i in range(num_seen_classes):
            #     sorted_indices = torch.argsort(cosine_similarity_matrix[i], descending=True)
            #     for idx in sorted_indices:
            #         if idx.item() not in assigned_clusters:
            #             if cosine_similarity_matrix[i, idx.item()] >= threshold:
            #                 ind_map_gt_to_pred[i] = idx.item()
            #                 assigned_clusters.add(idx.item())
            #                 break
            #             else:
            #                 break

            # # Then, handle unseen classes
            # for i in range(num_seen_classes, num_classes):
            #     sorted_indices = torch.argsort(cosine_similarity_matrix[i], descending=True)
            #     for idx in sorted_indices:
            #         if idx.item() not in assigned_clusters:
            #             ind_map_gt_to_pred[i] = idx.item()
            #             assigned_clusters.add(idx.item())
            #             break
                        
            # # If there are any unassigned clusters, assign them to the remaining classifier weights
            # remaining_clusters = set(range(len(cluster_prototypes))) - assigned_clusters
            # remaining_weights = set(range(num_classes)) - set(ind_map_gt_to_pred.keys())

            # # Then, handle remaining classes (seen but below threshold)
            # for i in list(remaining_weights):
            #     sorted_indices = torch.argsort(cosine_similarity_matrix[i], descending=True)
            #     for idx in sorted_indices:
            #         if idx.item() not in assigned_clusters:
            #             ind_map_gt_to_pred[i] = idx.item()
            #             assigned_clusters.add(idx.item())
            #             break
            
            # Ensure the mapping is sorted
            ind_map_pred_to_gt = {j: i for i, j in ind_map_gt_to_pred.items()}
            ind_map_pred_to_gt = dict(sorted(ind_map_pred_to_gt.items()))

            print(f'ind_map_pred_to_gt_proto: {ind_map_pred_to_gt}')

            
        elif args.client.align_type == 'centroid_sinkhorn':
            # Get the classifier weights
            classifier_weights = model.proj_layer.last_layer.parametrizations.weight.original1.data.clone().cpu()
            
            targets_labelled = torch.LongTensor(targets_labelled)
            if args.client.clust_feats == 'feats':
                feats_labelled = feats_labelled
            elif args.client.clust_feats == 'feats_proj':
                feats_labelled = feats_proj_labelled

            unique_labels_labelled = torch.unique(targets_labelled).tolist()                
            class_centroids_labelled = [feats_labelled[targets_labelled == label].mean(dim=0) for label in unique_labels_labelled]
            class_centroids_labelled = torch.stack(class_centroids_labelled)

            # Align class indexes using cluster prototypes and classifier weights
            # Calculate prototype for each cluster
            cluster_set = set(req_c)
            cluster_prototypes = []
            
            for cluster_ind in range(len(cluster_set)):

                # get the mean of features for the cluster
                if args.client.clust_feats == 'feats':
                    cluster_feats = all_feats[req_c == cluster_ind]
                elif args.client.clust_feats == 'feats_proj':
                    cluster_feats = all_feats_proj[req_c == cluster_ind]
                cluster_mean = cluster_feats.mean(dim=0)
                cluster_prototypes.append(cluster_mean)

            cluster_prototypes = torch.stack(cluster_prototypes)
            #cluster_stds = torch.stack(cluster_stds)

            seen_class_similarities = F.cosine_similarity(class_centroids_labelled.unsqueeze(1), cluster_prototypes.unsqueeze(0), dim=2)


            # Initialize the mapping dictionary
            ind_map_gt_to_pred = {}
            assigned_clusters = set()
            remaining_clusters = [i for i in range(len(cluster_set))]
            not_assign_list = [i for i in range(num_classes)]

            ## Filter out the classes that have wasserstein distance less than threshold
            threshold = args.client.align_threshold
            for i, label in enumerate(unique_labels_labelled):
                class_centroid = class_centroids_labelled[i]
                simiilarities_between_clusters = seen_class_similarities[i]
                cluster_ind = torch.argmax(simiilarities_between_clusters).item()
                print(f'class {label} - top1 cluster index: {cluster_ind}')
                if cluster_ind in assigned_clusters:
                    continue
                ## Estimate distribution of the cluster
                if args.client.clust_feats == 'feats':
                    cluster_feats = all_feats[req_c == cluster_ind]
                elif args.client.clust_feats == 'feats_proj':
                    cluster_feats = all_feats_proj[req_c == cluster_ind]
                gmm_cluster = GaussianMixture(n_components=1).fit(cluster_feats.cpu().clone().numpy())
                cluster_mean = gmm_cluster.means_[0]
                cluster_var = np.diag(gmm_cluster.covariances_[0])
                diff = class_centroid.cpu().clone().numpy() - cluster_mean
                inv_var = 1 / cluster_var
                mean_var = np.sqrt(np.sum(cluster_var))
                wasserstein_dist = np.sqrt(np.sum((diff ** 2) * inv_var))
                print(f'class {label} - wasserstein distance: {wasserstein_dist} - mean_var: {mean_var} - threshold: {args.client.align_threshold * mean_var}')
                if wasserstein_dist <= threshold * mean_var:
                    ind_map_gt_to_pred[label] = cluster_ind
                    assigned_clusters.add(cluster_ind)
                    remaining_clusters.remove(cluster_ind)


            # Calculate pairwise cosine similarity matrix between classifier weights and cluster prototypes 
            cosine_similarity_matrix = F.cosine_similarity(classifier_weights.unsqueeze(1), 
                                                        cluster_prototypes.unsqueeze(0), dim=2)
            
            # Get the number of seen classes
            num_seen_classes = len(args.dataset.seen_classes)
                        
            # Handle unseen classes
            remaining_clusters_copy = copy.deepcopy(set(remaining_clusters))
            unseen_class_weights = set(range(num_seen_classes, num_classes))

            

            # Hungarian matching for unseen classes
            if remaining_clusters_copy and unseen_class_weights:
                remaining_clusters_copy = list(remaining_clusters_copy)
                unseen_class_weights = list(unseen_class_weights)
                cost_matrix = torch.zeros((len(unseen_class_weights), len(remaining_clusters_copy)))
                for i, weight_idx in enumerate(unseen_class_weights):
                    for j, cluster_idx in enumerate(remaining_clusters_copy):
                        cost_matrix[i, j] = 1 - cosine_similarity_matrix[weight_idx, cluster_idx]
                row_ind, col_ind = linear_assignment(cost_matrix.cpu().numpy())
                for i, j in zip(row_ind, col_ind):
                    ind_map_gt_to_pred[unseen_class_weights[i]] = remaining_clusters_copy[j]
                    assigned_clusters.add(remaining_clusters_copy[j])
                    remaining_clusters.remove(remaining_clusters_copy[j])

            # Greedy matching for remaining seen classes
            remaining_seen_weights = set(range(num_seen_classes)) - set(ind_map_gt_to_pred.keys())
            if remaining_seen_weights:
                remaining_seen_weights = list(remaining_seen_weights)
                for weight_idx in remaining_seen_weights:
                    similarities = cosine_similarity_matrix[weight_idx]
                    sorted_indices = torch.argsort(similarities, descending=True)
                    for idx in sorted_indices:
                        if idx.item() in remaining_clusters:
                            ind_map_gt_to_pred[weight_idx] = idx.item()
                            remaining_clusters.remove(idx.item())
                            break
            
            print(f"ind_map_gt_to_pred not sorted!: {ind_map_gt_to_pred}")
            assert len(remaining_clusters) == 0
            
            # Ensure the mapping is sorted
            ind_map_pred_to_gt = {j: i for i, j in ind_map_gt_to_pred.items()}
            ind_map_pred_to_gt = dict(sorted(ind_map_pred_to_gt.items()))

            print(f'ind_map_pred_to_gt_proto: {ind_map_pred_to_gt}')

        
        else:
            assert False
    
    true_ind_map_pred_to_gt = {j: i for i, j in ind_map_gt.items()}
    true_ind_map_pred_to_gt = dict(sorted(true_ind_map_pred_to_gt.items()))
    # Filter true_ind_map_pred_to_gt based on labelled and unlabelled class sets
    local_classes_set = set(all_targets)
    
    filtered_true_ind_map_pred_to_gt = {}
    for pred, gt in true_ind_map_pred_to_gt.items():
        if gt in local_classes_set:
            filtered_true_ind_map_pred_to_gt[pred] = gt
    
    true_ind_map_pred_to_gt = filtered_true_ind_map_pred_to_gt
    print(f'true_ind_map_pred_to_gt: {true_ind_map_pred_to_gt}')
    print(f'ind_map_pred_to_gt: {ind_map_pred_to_gt}')
    
    aligned_preds = np.array([ind_map_pred_to_gt[i] for i in req_c[len(targets_labelled):]])
    
    #aligned_preds = np.array([ind_map_pred_to_gt_logit[i] for i in req_c[len(targets_labelled):]])
    # aligned_preds = np.array([true_ind_map_pred_to_gt[i] for i in req_c])

    # Plot confusion matrix between aligned_preds and targets
    D = max(aligned_preds.max(), targets.max()) + 1
    aligned_w = np.zeros((D, D), dtype=int)
    for i in range(len(aligned_preds)):
        aligned_w[aligned_preds[i], targets_unlabelled[i]] += 1

    ind = linear_assignment(aligned_w.max() - aligned_w)
    ind = np.vstack(ind).T
    ind_map_pred_to_gt_after_alignment = {i: j for i, j in ind}
    print(f'ind_map_pred_to_gt_after_alignment: {ind_map_pred_to_gt_after_alignment}')


    # Make Calss dict
    class_dict = defaultdict(int)
    for pred in aligned_preds:
        class_dict[str(pred)] += 1

    # Update prior preds
    for cls in class_dict:
        if not args.client.label_smoothing:
            prior[int(cls)] += class_dict[cls]
        else:
            smooth_max = args.client.smooth_max
            smooth_values = torch.ones(num_classes) * (1 - smooth_max) / (num_classes - 1)
            smooth_values[int(cls)] = smooth_max
            prior += smooth_values * class_dict[cls]

        total += class_dict[cls]
    prior = prior.float() / total
    
    # elif args.client.update_alg == 'gmm':
    #     if K is not None:
    #         num_clusters = K
    #     else:
    #         raise ValueError("Number of clusters (K) must be specified for GMM")

    #     # GMM 클러스터링 수행
    #     gmm = GaussianMixture(n_components=num_clusters, random_state=0)
    #     gmm.fit(all_feats.numpy())
        
    #     # 클러스터 레이블 추출
    #     req_c = gmm.predict(all_feats.numpy())
    #     soft_preds = gmm.predict_proba(all_feats.numpy())
    #     print(soft_preds)

    #     req_c = req_c.astype(int)
    #     targets = targets.astype(int)
        
    #     if args.client.align_gt:
    #         D = int(max(req_c.max(), targets.max()) + 1)
    #         w = np.zeros((D, D), dtype=int)
    #         for i in range(len(targets)):
    #             w[req_c[i], targets[i]] += 1

    #         # Use linear_sum_assignment
    #         ind = linear_assignment(w.max() - w)
    #         ind = np.vstack(ind).T

    #         ind_map_pred_to_gt = {i: j for i, j in ind}
    #         aligned_preds = [ind_map_pred_to_gt[i] for i in req_c]

    #         # Make Class dict
    #         class_dict = defaultdict(int)
    #         for pred in aligned_preds:
    #             class_dict[str(pred)] += 1

    #         # Update prior preds
    #         prior = torch.zeros(num_classes)
    #         total = 0
    #         for cls in class_dict:
    #             if not args.client.label_smoothing:
    #                 prior[int(cls)] += class_dict[cls]
    #             else:
    #                 smooth_max = args.client.smooth_max
    #                 smooth_values = torch.ones(num_classes) * (1 - smooth_max) / (num_classes - 1)
    #                 smooth_values[int(cls)] = smooth_max
    #                 prior += smooth_values * class_dict[cls]

    #             total += class_dict[cls]
    #         prior = prior.float() / total
    
    # elif args.client.update_alg == 'agglomerative':
    #     if K is not None:
    #         num_clusters = K
    #     else:
    #         raise False

    #     # Perform agglomerative clustering
    #     linked = linkage(all_feats.numpy(), method=args.client.agglo_method)

    #     # Use the distance threshold that results in the desired number of clusters
    #     distances = linked[:, 2]
    #     threshold = distances[-num_clusters]
    #     req_c = fcluster(linked, t=threshold, criterion='distance')

    #     req_c = req_c.astype(int)
    #     targets = targets.astype(int)
    #     if args.client.align_gt:
    #         D = int(max(req_c.max(), targets.max()) + 1)
    #         w = np.zeros((D, D), dtype=int)
    #         for i in range(len(targets)):
    #             w[req_c[i], targets[i]] += 1

    #         ind = linear_assignment(w.max() - w)
    #         ind = np.vstack(ind).T

    #         ind_map_pred_to_gt = {i: j for i, j in ind}
    #         aligned_preds = [ind_map_pred_to_gt[i] for i in req_c]

    #         ## Make Calss dict
    #         class_dict = defaultdict(int)
    #         for pred in aligned_preds:
    #             class_dict[str(pred)] += 1

    #         ## Update prior preds
    #         prior = torch.zeros(num_classes)
    #         total = 0
    #         for cls in class_dict:
    #             if not args.client.label_smoothing:
    #                 prior[int(cls)] += class_dict[cls]
    #             else:
    #                 smooth_max = args.client.smooth_max
    #                 smooth_values = torch.ones(num_classes) * (1 - smooth_max) / (num_classes - 1)
    #                 smooth_values[int(cls)] = smooth_max
    #                 prior += smooth_values * class_dict[cls]

    #             total += class_dict[cls]
    #         prior = prior.float() / total

    # elif args.client.update_alg == 'model_prediction_hard':
    #     if args.client.align_gt:
    #         all_acc, old_acc, new_acc, _, ind_map = evaler.log_accs_from_preds(y_true=targets, y_pred=preds,
    #                                                                                 mask=mask,
    #                                                                                 T=epoch, eval_funcs=['v2'],
    #                                                                                 save_name='Test Acc')
    #         print(ind_map)
    #         ## Align Class idxes
    #         total_ind_map = {i: i for i in range(num_classes)}
    #         for true_cls, pred_cls in ind_map.items():
    #             total_ind_map[pred_cls] = true_cls
    #         aligned_preds = []
    #         for pred in preds:
    #             aligned_preds.append(total_ind_map[pred])
    #     else:
    #         aligned_preds = preds

    #     ## Make Calss dict
    #     class_dict = defaultdict(int)
    #     for pred in aligned_preds:
    #         class_dict[str(pred)] += 1

    #     ## Update prior preds
    #     prior = torch.zeros(num_classes)
    #     total = 0
    #     for cls in class_dict:
    #         if not args.client.label_smoothing:
    #             prior[int(cls)] += class_dict[cls]
    #         else:
    #             smooth_max = args.client.smooth_max
    #             smooth_values = torch.ones(num_classes) * (1 - smooth_max) / (num_classes - 1)
    #             smooth_values[int(cls)] = smooth_max
    #             prior += smooth_values * class_dict[cls]

    #         total += class_dict[cls]
    #     prior = prior.float() / total

    # elif args.client.update_alg == 'model_prediction_soft':
    #     if args.client.align_gt:
    #         all_acc, old_acc, new_acc, _, ind_map = evaler.log_accs_from_preds(y_true=targets, y_pred=preds,
    #                                                                                 mask=mask,
    #                                                                                 T=epoch, eval_funcs=['v2'],
    #                                                                                 save_name='Test Acc')
    #         print(ind_map)
    #         ## Align Class idxes
    #         total_ind_map = {i: i for i in range(num_classes)}
    #         for true_cls, pred_cls in ind_map.items():
    #             total_ind_map[pred_cls] = true_cls
    #         idxs = list(total_ind_map.values())
    #         aligned_preds = soft_preds[:, idxs]
    #     else:
    #         aligned_preds = preds

    #     prior = aligned_preds.mean(dim=0)

    # else:
    #     assert False

    prior = prior.cpu()



    gc.collect()

    return num_clusters, prior, all_acc, old_acc, new_acc, aligned_w, aligned_preds, ind_map_pred_to_gt, req_c[len(targets_labelled):]



def update_prior_ovc(args, prior_dist, model, loader, evaler, num_classes, epoch, device, K=None):
    # model.eval()
    # # model_device = next(model.parameters()).device
    # model.to(device)
    # test_transform = copy.deepcopy(evaler.test_loader.dataset.transform)

    # labelled_dataset = copy.deepcopy(loader.dataset.labelled_dataset)
    # labelled_dataset.transform = test_transform
    # unlabelled_dataset = copy.deepcopy(loader.dataset.unlabelled_dataset)
    # unlabelled_dataset.transform = test_transform

    # all_feats_labelled = []
    # all_feats_proj_labelled = []
    # all_logits_labelled = []
    # targets_labelled = np.array([])
    # mask = np.array([])

    # labelled_loader = DataLoader(labelled_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # all_feats_unlabelled = []
    # all_feats_proj_unlabelled = []
    # all_logits_unlabelled = []
    # targets_unlabelled = np.array([])

    # unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    # print('Collating logits...')
    # preds = []
    # # First extract feats labelled
    # for batch_idx, (images, label, _) in enumerate(labelled_loader):
    #     images = images.to(device)

    #     # Pass features through base model and then additional learnable transform (linear layer)
    #     with torch.no_grad():
    #         feats, feats_proj, logits = model(images, return_all=True)

    #     preds.append(logits.argmax(1).cpu().numpy())
    #     targets_labelled = np.append(targets_labelled, label.cpu().numpy().astype(int))
    #     all_feats_labelled.append(feats.cpu().clone())
    #     all_feats_proj_labelled.append(feats_proj.cpu().clone())
    #     all_logits_labelled.append(logits)
    #     # mask = np.append(mask, np.array([True if x.item() in range(len(args.dataset.seen_classes))
    #     #                                  else False for x in label]))

    # # First extract feats unlabelled
    # for batch_idx, (images, label, _) in enumerate(unlabelled_loader):
    #     images = images.to(device)

    #     # Pass features through base model and then additional learnable transform (linear layer)
    #     with torch.no_grad():
    #         feats, feats_proj, logits = model(images, return_all=True)

    #     preds.append(logits.argmax(1).cpu().numpy())
    #     targets_unlabelled = np.append(targets_unlabelled, label.cpu().numpy().astype(int))
    #     all_feats_unlabelled.append(feats.cpu().clone())
    #     all_feats_proj_unlabelled.append(feats_proj.cpu().clone())
    #     all_logits_unlabelled.append(logits)
    #     mask = np.append(mask, np.array([True if x.item() in range(len(args.dataset.seen_classes))
    #                                      else False for x in label]))

    # preds = np.concatenate(preds)
    # feats_labelled = torch.cat(all_feats_labelled, dim=0)
    # feats_proj_labelled = torch.cat(all_feats_proj_labelled, dim=0)
    # #targets_labelled = torch.cat(targets_labelled, dim=0)

    # feats_unlabelled = torch.cat(all_feats_unlabelled, dim=0)
    # feats_proj_unlabelled = torch.cat(all_feats_proj_unlabelled, dim=0)
    # #targets_unlabelled = torch.cat(targets_unlabelled, dim=0)

    # all_feats = torch.cat([feats_labelled, feats_unlabelled], dim=0)
    # all_feats_proj = torch.cat([feats_proj_labelled, feats_proj_unlabelled], dim=0)
    # all_targets = np.concatenate([targets_labelled, targets_unlabelled])

    # logits_labelled = torch.cat(all_logits_labelled, dim=0)
    # logits_unlabelled = torch.cat(all_logits_unlabelled, dim=0)
    # all_logits = torch.cat([logits_labelled, logits_unlabelled], dim=0)
    # mask = mask.astype(bool)
    
    # soft_preds = (all_logits / args.client.soft_preds_temp).softmax(dim=1)

    feats_labelled, feats_proj_labelled, logits_labelled, targets_labelled, feats_unlabelled, feats_proj_unlabelled, logits_unlabelled, targets_unlabelled, mask = extract_local_features(args, model, loader, evaler, device)

    targets_unlabelled = targets_unlabelled.astype(int)
    targets_labelled = targets_labelled.astype(int)
    all_feats = torch.cat([feats_labelled, feats_unlabelled], dim=0)
    all_feats_proj = torch.cat([feats_proj_labelled, feats_proj_unlabelled], dim=0)
    all_targets = np.concatenate([targets_labelled, targets_unlabelled])
    all_logits = torch.cat([logits_labelled, logits_unlabelled], dim=0)


    if args.client.est_num_clusters:
        ## Estimate num_clusters
        if args.client.clust_feats == 'feats':
            num_clusters = estimate_num_clusters(feats_labelled.numpy(), targets_labelled, feats_unlabelled.numpy(), targets_unlabelled, k_mode=args.client.k_mode, total_classes=num_classes, tolerance=args.client.tolerance)
        elif args.client.clust_feats == 'feats_proj':
            num_clusters = estimate_num_clusters(feats_proj_labelled.numpy(), targets_labelled, feats_proj_unlabelled.numpy(), targets_unlabelled, k_mode=args.client.k_mode, total_classes=num_classes, tolerance=args.client.tolerance)
    else:
        num_clusters = K

    
    ## Perform local clustering
    if args.client.update_alg =='finch':
        if args.client.clust_feats == 'feats':
            c, num_clust, req_c = FINCH(all_feats.numpy(), req_clust=num_clusters, distance='cosine', verbose=False)
        elif args.client.clust_feats == 'feats_proj':
            c, num_clust, req_c = FINCH(all_feats_proj.numpy(), req_clust=num_clusters, distance='cosine', verbose=False)
        req_c = req_c.astype(int)
        targets = all_targets.astype(int)        

    elif args.client.update_alg =='ood_finch':

        lambda_param = args.client.lambda_param
        similarities = cosine_similarity(feats_unlabelled, feats_labelled)
        # k-th nearest neighbor 찾기
        kth_similarities = np.sort(similarities, axis=1)[:, -1]
        ood_scores = 1 - kth_similarities

        c, num_clust, req_c = OOD_FINCH(feats_unlabelled.numpy(), req_clust=num_clusters, ood_scores=ood_scores, lambda_param=lambda_param, distance='cosine', verbose=False, labeled_features=feats_labelled.numpy())
        req_c = req_c.astype(int)
        targets = targets_unlabelled.astype(int)

    elif args.client.update_alg =='semi_finch':

        orig_dist = metrics.pairwise.pairwise_distances(all_feats.numpy(), all_feats.numpy(), metric='cosine')
        orig_dist_copy = copy.deepcopy(orig_dist)

        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)

        orig_dist_labelled = orig_dist_copy[:len(targets_labelled)]
        for cls in np.unique(targets_labelled):
           indices = np.where(targets_labelled == cls)[0]
           cls_dist = orig_dist_labelled[indices]
           cls_rank = np.argmax(cls_dist, axis=1)
           initial_rank[indices] = cls_rank

        c, num_clust, req_c = FINCH(all_feats.numpy(), initial_rank=initial_rank, req_clust=num_clusters, distance='cosine', verbose=False)
        targets = all_targets.astype(int)
        req_c = req_c.astype(int)
    else:
        assert False

    ## Eval Clustering results
    all_acc, old_acc, new_acc, w, ind_map_gt = evaler.split_cluster_acc_v2(targets, req_c, mask)

    prior = torch.zeros(num_classes)
    total = 0
    
    # update prior for labelled data
    for i in range(len(targets_labelled)):
        prior[int(targets_labelled[i])] += 1
        total += 1
    

    # Update prior for unlabelled data
    if args.client.align_gt:
        # Align Class idxes with ground truths
        ind_map_pred_to_gt = {j: i for i, j in ind_map_gt.items()}
        ind_map_pred_to_gt = dict(sorted(ind_map_pred_to_gt.items()))
    
    else:
        if args.client.align_type == 'sample':
            preds_unlabelled = logits_unlabelled.argmax(1).cpu().numpy()
            concat_targets = np.concatenate([targets_labelled, preds_unlabelled]).astype(int)
            D = num_classes
            w = np.zeros((D, D), dtype=int)
            for i in range(concat_targets.size):
                w[req_c[i], concat_targets[i]] += 1

            ind = linear_assignment(w.max() - w)
            ind = np.vstack(ind).T
            ind_map = {i: j for i, j in ind}
            ind_map_pred_to_gt = dict(sorted(ind_map.items()))
            print(f'Prior update process:')
            print(f'ind_map_pred_to_gt_logit: {ind_map_pred_to_gt}')

        elif args.client.align_type == 'centroid':

            # Get the classifier weights
            classifier_weights = model.proj_layer.last_layer.parametrizations.weight.original1.data.clone().cpu()
            
            targets_labelled = torch.LongTensor(targets_labelled)
            if args.client.clust_feats == 'feats':
                #feats_labelled = F.normalize(feats_labelled, dim=1, p=2)
                feats_labelled = feats_labelled
            elif args.client.clust_feats == 'feats_proj':
                feats_labelled = feats_proj_labelled

            unique_labels_labelled = torch.unique(targets_labelled).tolist()                
            class_centroids_labelled = [feats_labelled[targets_labelled == label].mean(dim=0) for label in unique_labels_labelled]
            class_centroids_labelled = torch.stack(class_centroids_labelled)

            # Align class indexes using cluster prototypes and classifier weights
            # Calculate prototype for each cluster
            cluster_set = set(req_c)
            cluster_prototypes = []
            #cluster_stds = []
            for cluster_ind in range(len(cluster_set)):

                # get the mean of features for the cluster
                if args.client.clust_feats == 'feats':
                    cluster_feats = all_feats[req_c == cluster_ind]
                elif args.client.clust_feats == 'feats_proj':
                    cluster_feats = all_feats_proj[req_c == cluster_ind]
                cluster_mean = cluster_feats.mean(dim=0)
                cluster_prototypes.append(cluster_mean)

                # Calculate standard deviation for the cluster
                #cluster_std = cluster_feats.std(dim=0)
                # for numerical stability
                #cluster_std[cluster_std == 0] = 1e-8
                #cluster_stds.append(cluster_std)

            cluster_prototypes = torch.stack(cluster_prototypes)
            #cluster_stds = torch.stack(cluster_stds)

            seen_class_similarities = F.cosine_similarity(class_centroids_labelled.unsqueeze(1), cluster_prototypes.unsqueeze(0), dim=2)


            # Initialize the mapping dictionary
            ind_map_gt_to_pred = {}
            assigned_clusters = set()
            remaining_clusters = [i for i in range(len(cluster_set))]
            not_assign_list = [i for i in range(num_classes)]

            ## Filter out the classes that have wasserstein distance less than threshold
            threshold = args.client.align_threshold
            for i, label in enumerate(unique_labels_labelled):
                class_centroid = class_centroids_labelled[i]
                simiilarities_between_clusters = seen_class_similarities[i]
                cluster_ind = torch.argmax(simiilarities_between_clusters).item()
                print(f'class {label} - top1 cluster index: {cluster_ind}')
                if cluster_ind in assigned_clusters:
                    continue
                ## Estimate distribution of the cluster
                if args.client.clust_feats == 'feats':
                    cluster_feats = all_feats[req_c == cluster_ind]
                elif args.client.clust_feats == 'feats_proj':
                    cluster_feats = all_feats_proj[req_c == cluster_ind]
                gmm_cluster = GaussianMixture(n_components=1).fit(cluster_feats.cpu().clone().numpy())
                cluster_mean = gmm_cluster.means_[0]
                cluster_var = np.diag(gmm_cluster.covariances_[0])
                diff = class_centroid.cpu().clone().numpy() - cluster_mean
                inv_var = 1 / cluster_var
                mean_var = np.sqrt(np.sum(cluster_var))
                wasserstein_dist = np.sqrt(np.sum((diff ** 2) * inv_var))
                print(f'class {label} - wasserstein distance: {wasserstein_dist} - mean_var: {mean_var} - threshold: {args.client.align_threshold * mean_var}')
                if wasserstein_dist <= threshold * mean_var:
                    ind_map_gt_to_pred[label] = cluster_ind
                    assigned_clusters.add(cluster_ind)
                    remaining_clusters.remove(cluster_ind)


            # Calculate pairwise cosine similarity matrix between classifier weights and cluster prototypes 
            cosine_similarity_matrix = F.cosine_similarity(classifier_weights.unsqueeze(1), 
                                                        cluster_prototypes.unsqueeze(0), dim=2)
            
            # Get the number of seen classes
            num_seen_classes = len(args.dataset.seen_classes)
            
            

            # # Handle labelled set classes
            # for i, label in enumerate(filtered_labels_labelled):
            #     class_centroid = class_centroids_labelled[i]
            #     similarities = F.cosine_similarity(class_centroid.unsqueeze(0), cluster_prototypes)
            #     sorted_indices = torch.argsort(similarities, descending=True)
            #     for idx in sorted_indices:
            #         best_cluster_idx = idx.item()
            #         if best_cluster_idx in remaining_clusters:
            #             # # calculate z-score
            #             # z_score = (class_centroid - cluster_prototypes[best_cluster_idx]) / cluster_stds[best_cluster_idx]
            #             # D_approx = torch.sqrt((z_score ** 2).sum())

            #             # # 카이제곱 분포를 사용하여 확률로 변환
            #             # degrees_of_freedom = len(class_centroid)
            #             # probability = chi2.cdf(D_approx ** 2, degrees_of_freedom)

            #             # # cosine similarity를 softmax해서 확률로 변환
            #             # sim_cluster_classifier = F.cosine_similarity(cluster_prototypes[best_cluster_idx].unsqueeze(0), classifier_weights, dim=1)
            #             # probabilities = torch.softmax(sim_cluster_classifier / args.client.assoc_temp, dim=0)
            #             # probability = probabilities[label.item()]
                        
                        
            #             # probabilities = torch.softmax(similarities / curr_assoc_temp, dim=0)
            #             # probability = probabilities[best_cluster_idx]

            #             # Sample a value from a uniform distribution between 0 and 1
            #             # sampled_value = torch.rand(1).item()
            #             # print(f'probabilities: {probabilities}')
            #             # print(f'class {label.item()} - cluster {best_cluster_idx} - similarity: {similarities[best_cluster_idx]}  - Probability: {probability} - sampled_value: {sampled_value}')
            #             # if sampled_value <= probability:
            #             #     ind_map_gt_to_pred[label.item()] = best_cluster_idx
            #             #     assigned_clusters.add(best_cluster_idx)
            #             #     remaining_clusters.remove(best_cluster_idx)
            #             #     break
                        
            #             ind_map_gt_to_pred[label] = best_cluster_idx
            #             assigned_clusters.add(best_cluster_idx)
            #             remaining_clusters.remove(best_cluster_idx)
            #             break
                        

                        
            # Handle unseen classes
            remaining_clusters_copy = copy.deepcopy(set(remaining_clusters))
            unseen_class_weights = set(range(num_seen_classes, num_classes))

            # Hungarian matching for unseen classes
            if remaining_clusters_copy and unseen_class_weights:
                remaining_clusters_copy = list(remaining_clusters_copy)
                unseen_class_weights = list(unseen_class_weights)
                cost_matrix = torch.zeros((len(unseen_class_weights), len(remaining_clusters_copy)))
                for i, weight_idx in enumerate(unseen_class_weights):
                    for j, cluster_idx in enumerate(remaining_clusters_copy):
                        cost_matrix[i, j] = 1 - cosine_similarity_matrix[weight_idx, cluster_idx]
                row_ind, col_ind = linear_assignment(cost_matrix.cpu().numpy())
                for i, j in zip(row_ind, col_ind):
                    ind_map_gt_to_pred[unseen_class_weights[i]] = remaining_clusters_copy[j]
                    assigned_clusters.add(remaining_clusters_copy[j])
                    remaining_clusters.remove(remaining_clusters_copy[j])

            # Greedy matching for remaining seen classes
            remaining_seen_weights = set(range(num_seen_classes)) - set(ind_map_gt_to_pred.keys())
            if remaining_seen_weights:
                remaining_seen_weights = list(remaining_seen_weights)
                for weight_idx in remaining_seen_weights:
                    similarities = cosine_similarity_matrix[weight_idx]
                    sorted_indices = torch.argsort(similarities, descending=True)
                    for idx in sorted_indices:
                        if idx.item() in remaining_clusters:
                            ind_map_gt_to_pred[weight_idx] = idx.item()
                            remaining_clusters.remove(idx.item())
                            break
            
            print(f"ind_map_gt_to_pred not sorted!: {ind_map_gt_to_pred}")
            assert len(remaining_clusters) == 0


            # for i in range(num_seen_classes):
            #     sorted_indices = torch.argsort(cosine_similarity_matrix[i], descending=True)
            #     for idx in sorted_indices:
            #         if idx.item() not in assigned_clusters:
            #             if cosine_similarity_matrix[i, idx.item()] >= threshold:
            #                 ind_map_gt_to_pred[i] = idx.item()
            #                 assigned_clusters.add(idx.item())
            #                 break
            #             else:
            #                 break

            # # Then, handle unseen classes
            # for i in range(num_seen_classes, num_classes):
            #     sorted_indices = torch.argsort(cosine_similarity_matrix[i], descending=True)
            #     for idx in sorted_indices:
            #         if idx.item() not in assigned_clusters:
            #             ind_map_gt_to_pred[i] = idx.item()
            #             assigned_clusters.add(idx.item())
            #             break
                        
            # # If there are any unassigned clusters, assign them to the remaining classifier weights
            # remaining_clusters = set(range(len(cluster_prototypes))) - assigned_clusters
            # remaining_weights = set(range(num_classes)) - set(ind_map_gt_to_pred.keys())

            # # Then, handle remaining classes (seen but below threshold)
            # for i in list(remaining_weights):
            #     sorted_indices = torch.argsort(cosine_similarity_matrix[i], descending=True)
            #     for idx in sorted_indices:
            #         if idx.item() not in assigned_clusters:
            #             ind_map_gt_to_pred[i] = idx.item()
            #             assigned_clusters.add(idx.item())
            #             break
            
            # Ensure the mapping is sorted
            ind_map_pred_to_gt = {j: i for i, j in ind_map_gt_to_pred.items()}
            ind_map_pred_to_gt = dict(sorted(ind_map_pred_to_gt.items()))

            print(f'ind_map_pred_to_gt_proto: {ind_map_pred_to_gt}')

            
        else:
            assert False
    
    true_ind_map_pred_to_gt = {j: i for i, j in ind_map_gt.items()}
    true_ind_map_pred_to_gt = dict(sorted(true_ind_map_pred_to_gt.items()))
    # Filter true_ind_map_pred_to_gt based on labelled and unlabelled class sets
    local_classes_set = set(all_targets)
    
    filtered_true_ind_map_pred_to_gt = {}
    for pred, gt in true_ind_map_pred_to_gt.items():
        if gt in local_classes_set:
            filtered_true_ind_map_pred_to_gt[pred] = gt
    
    true_ind_map_pred_to_gt = filtered_true_ind_map_pred_to_gt
    print(f'true_ind_map_pred_to_gt: {true_ind_map_pred_to_gt}')
    print(f'ind_map_pred_to_gt: {ind_map_pred_to_gt}')
    
    aligned_preds = np.array([ind_map_pred_to_gt[i] for i in req_c[len(targets_labelled):]])
    
    #aligned_preds = np.array([ind_map_pred_to_gt_logit[i] for i in req_c[len(targets_labelled):]])
    # aligned_preds = np.array([true_ind_map_pred_to_gt[i] for i in req_c])

    # Plot confusion matrix between aligned_preds and targets
    D = max(aligned_preds.max(), targets.max()) + 1
    aligned_w = np.zeros((D, D), dtype=int)
    for i in range(len(aligned_preds)):
        aligned_w[aligned_preds[i], targets_unlabelled[i]] += 1

    ind = linear_assignment(aligned_w.max() - aligned_w)
    ind = np.vstack(ind).T
    ind_map_pred_to_gt_after_alignment = {i: j for i, j in ind}
    print(f'ind_map_pred_to_gt_after_alignment: {ind_map_pred_to_gt_after_alignment}')


    # Make Calss dict
    class_dict = defaultdict(int)
    for pred in aligned_preds:
        class_dict[str(pred)] += 1

    # Update prior preds
    for cls in class_dict:
        if not args.client.label_smoothing:
            prior[int(cls)] += class_dict[cls]
        else:
            smooth_max = args.client.smooth_max
            smooth_values = torch.ones(num_classes) * (1 - smooth_max) / (num_classes - 1)
            smooth_values[int(cls)] = smooth_max
            prior += smooth_values * class_dict[cls]

        total += class_dict[cls]
    prior = prior.float() / total
    
    # elif args.client.update_alg == 'gmm':
    #     if K is not None:
    #         num_clusters = K
    #     else:
    #         raise ValueError("Number of clusters (K) must be specified for GMM")

    #     # GMM 클러스터링 수행
    #     gmm = GaussianMixture(n_components=num_clusters, random_state=0)
    #     gmm.fit(all_feats.numpy())
        
    #     # 클러스터 레이블 추출
    #     req_c = gmm.predict(all_feats.numpy())
    #     soft_preds = gmm.predict_proba(all_feats.numpy())
    #     print(soft_preds)

    #     req_c = req_c.astype(int)
    #     targets = targets.astype(int)
        
    #     if args.client.align_gt:
    #         D = int(max(req_c.max(), targets.max()) + 1)
    #         w = np.zeros((D, D), dtype=int)
    #         for i in range(len(targets)):
    #             w[req_c[i], targets[i]] += 1

    #         # Use linear_sum_assignment
    #         ind = linear_assignment(w.max() - w)
    #         ind = np.vstack(ind).T

    #         ind_map_pred_to_gt = {i: j for i, j in ind}
    #         aligned_preds = [ind_map_pred_to_gt[i] for i in req_c]

    #         # Make Class dict
    #         class_dict = defaultdict(int)
    #         for pred in aligned_preds:
    #             class_dict[str(pred)] += 1

    #         # Update prior preds
    #         prior = torch.zeros(num_classes)
    #         total = 0
    #         for cls in class_dict:
    #             if not args.client.label_smoothing:
    #                 prior[int(cls)] += class_dict[cls]
    #             else:
    #                 smooth_max = args.client.smooth_max
    #                 smooth_values = torch.ones(num_classes) * (1 - smooth_max) / (num_classes - 1)
    #                 smooth_values[int(cls)] = smooth_max
    #                 prior += smooth_values * class_dict[cls]

    #             total += class_dict[cls]
    #         prior = prior.float() / total
    
    # elif args.client.update_alg == 'agglomerative':
    #     if K is not None:
    #         num_clusters = K
    #     else:
    #         raise False

    #     # Perform agglomerative clustering
    #     linked = linkage(all_feats.numpy(), method=args.client.agglo_method)

    #     # Use the distance threshold that results in the desired number of clusters
    #     distances = linked[:, 2]
    #     threshold = distances[-num_clusters]
    #     req_c = fcluster(linked, t=threshold, criterion='distance')

    #     req_c = req_c.astype(int)
    #     targets = targets.astype(int)
    #     if args.client.align_gt:
    #         D = int(max(req_c.max(), targets.max()) + 1)
    #         w = np.zeros((D, D), dtype=int)
    #         for i in range(len(targets)):
    #             w[req_c[i], targets[i]] += 1

    #         ind = linear_assignment(w.max() - w)
    #         ind = np.vstack(ind).T

    #         ind_map_pred_to_gt = {i: j for i, j in ind}
    #         aligned_preds = [ind_map_pred_to_gt[i] for i in req_c]

    #         ## Make Calss dict
    #         class_dict = defaultdict(int)
    #         for pred in aligned_preds:
    #             class_dict[str(pred)] += 1

    #         ## Update prior preds
    #         prior = torch.zeros(num_classes)
    #         total = 0
    #         for cls in class_dict:
    #             if not args.client.label_smoothing:
    #                 prior[int(cls)] += class_dict[cls]
    #             else:
    #                 smooth_max = args.client.smooth_max
    #                 smooth_values = torch.ones(num_classes) * (1 - smooth_max) / (num_classes - 1)
    #                 smooth_values[int(cls)] = smooth_max
    #                 prior += smooth_values * class_dict[cls]

    #             total += class_dict[cls]
    #         prior = prior.float() / total

    # elif args.client.update_alg == 'model_prediction_hard':
    #     if args.client.align_gt:
    #         all_acc, old_acc, new_acc, _, ind_map = evaler.log_accs_from_preds(y_true=targets, y_pred=preds,
    #                                                                                 mask=mask,
    #                                                                                 T=epoch, eval_funcs=['v2'],
    #                                                                                 save_name='Test Acc')
    #         print(ind_map)
    #         ## Align Class idxes
    #         total_ind_map = {i: i for i in range(num_classes)}
    #         for true_cls, pred_cls in ind_map.items():
    #             total_ind_map[pred_cls] = true_cls
    #         aligned_preds = []
    #         for pred in preds:
    #             aligned_preds.append(total_ind_map[pred])
    #     else:
    #         aligned_preds = preds

    #     ## Make Calss dict
    #     class_dict = defaultdict(int)
    #     for pred in aligned_preds:
    #         class_dict[str(pred)] += 1

    #     ## Update prior preds
    #     prior = torch.zeros(num_classes)
    #     total = 0
    #     for cls in class_dict:
    #         if not args.client.label_smoothing:
    #             prior[int(cls)] += class_dict[cls]
    #         else:
    #             smooth_max = args.client.smooth_max
    #             smooth_values = torch.ones(num_classes) * (1 - smooth_max) / (num_classes - 1)
    #             smooth_values[int(cls)] = smooth_max
    #             prior += smooth_values * class_dict[cls]

    #         total += class_dict[cls]
    #     prior = prior.float() / total

    # elif args.client.update_alg == 'model_prediction_soft':
    #     if args.client.align_gt:
    #         all_acc, old_acc, new_acc, _, ind_map = evaler.log_accs_from_preds(y_true=targets, y_pred=preds,
    #                                                                                 mask=mask,
    #                                                                                 T=epoch, eval_funcs=['v2'],
    #                                                                                 save_name='Test Acc')
    #         print(ind_map)
    #         ## Align Class idxes
    #         total_ind_map = {i: i for i in range(num_classes)}
    #         for true_cls, pred_cls in ind_map.items():
    #             total_ind_map[pred_cls] = true_cls
    #         idxs = list(total_ind_map.values())
    #         aligned_preds = soft_preds[:, idxs]
    #     else:
    #         aligned_preds = preds

    #     prior = aligned_preds.mean(dim=0)

    # else:
    #     assert False

    prior = prior.cpu()

    if args.client.over_clustering:
        c, num_clust, req_c = FINCH(all_feats.numpy(), req_clust=num_clusters * 2, distance='cosine', verbose=False)

        cluster_set = set(req_c)
        cluster_prototypes = []
        #cluster_stds = []
        for cluster_ind in range(len(cluster_set)):

            # get the mean of features for the cluster
            if args.client.clust_feats == 'feats':
                cluster_feats = all_feats[req_c == cluster_ind]
            elif args.client.clust_feats == 'feats_proj':
                cluster_feats = all_feats_proj[req_c == cluster_ind]
            cluster_mean = cluster_feats.mean(dim=0)
            cluster_prototypes.append(cluster_mean)

            # Calculate standard deviation for the cluster
            #cluster_std = cluster_feats.std(dim=0)
            # for numerical stability
            #cluster_std[cluster_std == 0] = 1e-8
            #cluster_stds.append(cluster_std)

        cluster_prototypes = torch.stack(cluster_prototypes)
    else:
        cluster_prototypes = None


    gc.collect()

    return num_clusters, prior, all_acc, old_acc, new_acc, aligned_w, aligned_preds, ind_map_pred_to_gt, req_c[len(targets_labelled):], cluster_prototypes






def update_prior_with_clustering_results(args, prior_dist, model, loader, evaler, num_classes, epoch, device, K=None):

    feats_labelled, feats_proj_labelled, logits_labelled, targets_labelled, feats_unlabelled, feats_proj_unlabelled, logits_unlabelled, targets_unlabelled, mask = extract_local_features(args, model, loader, evaler, device)

    targets_unlabelled = targets_unlabelled.astype(int)
    targets_labelled = targets_labelled.astype(int)
    all_feats = torch.cat([feats_labelled, feats_unlabelled], dim=0)
    all_feats_proj = torch.cat([feats_proj_labelled, feats_proj_unlabelled], dim=0)
    all_targets = np.concatenate([targets_labelled, targets_unlabelled])
    all_logits = torch.cat([logits_labelled, logits_unlabelled], dim=0)


    if args.client.est_num_clusters:
        ## Estimate num_clusters
        if args.client.clust_feats == 'feats':
            num_clusters = estimate_num_clusters(feats_labelled.numpy(), targets_labelled, feats_unlabelled.numpy(), targets_unlabelled, k_mode=args.client.k_mode, total_classes=num_classes, tolerance=args.client.tolerance)
        elif args.client.clust_feats == 'feats_proj':
            num_clusters = estimate_num_clusters(feats_proj_labelled.numpy(), targets_labelled, feats_proj_unlabelled.numpy(), targets_unlabelled, k_mode=args.client.k_mode, total_classes=num_classes, tolerance=args.client.tolerance)
    else:
        num_clusters = K

    
    ## Perform local clustering
    if args.client.update_alg =='finch':
        if args.client.clust_feats == 'feats':
            c, num_clust, req_c = FINCH(all_feats.numpy(), req_clust=num_clusters, distance='cosine', verbose=False)
        elif args.client.clust_feats == 'feats_proj':
            c, num_clust, req_c = FINCH(all_feats_proj.numpy(), req_clust=num_clusters, distance='cosine', verbose=False)
        req_c = req_c.astype(int)
        targets = all_targets.astype(int)        

    elif args.client.update_alg =='ood_finch':

        lambda_param = args.client.lambda_param
        similarities = cosine_similarity(feats_unlabelled, feats_labelled)
        # k-th nearest neighbor 찾기
        kth_similarities = np.sort(similarities, axis=1)[:, -1]
        ood_scores = 1 - kth_similarities

        c, num_clust, req_c = OOD_FINCH(feats_unlabelled.numpy(), req_clust=num_clusters, ood_scores=ood_scores, lambda_param=lambda_param, distance='cosine', verbose=False, labeled_features=feats_labelled.numpy())
        req_c = req_c.astype(int)
        targets = targets_unlabelled.astype(int)

    elif args.client.update_alg =='semi_finch':

        orig_dist = metrics.pairwise.pairwise_distances(all_feats.numpy(), all_feats.numpy(), metric='cosine')
        orig_dist_copy = copy.deepcopy(orig_dist)

        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)

        orig_dist_labelled = orig_dist_copy[:len(targets_labelled)]
        for cls in np.unique(targets_labelled):
           indices = np.where(targets_labelled == cls)[0]
           cls_dist = orig_dist_labelled[indices]
           cls_rank = np.argmax(cls_dist, axis=1)
           initial_rank[indices] = cls_rank

        c, num_clust, req_c = FINCH(all_feats.numpy(), initial_rank=initial_rank, req_clust=num_clusters, distance='cosine', verbose=False)
        targets = all_targets.astype(int)
        req_c = req_c.astype(int)
    else:
        assert False

    ## Eval Clustering results
    all_acc, old_acc, new_acc, w, ind_map_gt = evaler.split_cluster_acc_v2(targets, req_c, mask)

    prior = torch.zeros(num_classes)
    total = 0
    
    # update prior for labelled data
    for i in range(len(targets_labelled)):
        prior[int(targets_labelled[i])] += 1
        total += 1
    

    # Update prior for unlabelled data
    if args.client.align_gt:
        # Align Class idxes with ground truths
        ind_map_pred_to_gt = {j: i for i, j in ind_map_gt.items()}
        ind_map_pred_to_gt = dict(sorted(ind_map_pred_to_gt.items()))
    
    else:
        if args.client.align_type == 'sample':
            preds_unlabelled = logits_unlabelled.argmax(1).cpu().numpy()
            concat_targets = np.concatenate([targets_labelled, preds_unlabelled]).astype(int)
            D = num_classes
            w = np.zeros((D, D), dtype=int)
            for i in range(concat_targets.size):
                w[req_c[i], concat_targets[i]] += 1

            ind = linear_assignment(w.max() - w)
            ind = np.vstack(ind).T
            ind_map = {i: j for i, j in ind}
            ind_map_pred_to_gt = dict(sorted(ind_map.items()))
            print(f'Prior update process:')
            print(f'ind_map_pred_to_gt_logit: {ind_map_pred_to_gt}')

        elif args.client.align_type == 'centroid':

            # Get the classifier weights
            classifier_weights = model.proj_layer.last_layer.parametrizations.weight.original1.data.clone().cpu()
            
            targets_labelled = torch.LongTensor(targets_labelled)
            if args.client.clust_feats == 'feats':
                #feats_labelled = F.normalize(feats_labelled, dim=1, p=2)
                feats_labelled = feats_labelled
            elif args.client.clust_feats == 'feats_proj':
                feats_labelled = feats_proj_labelled

            unique_labels_labelled = torch.unique(targets_labelled).tolist()                
            class_centroids_labelled = [feats_labelled[targets_labelled == label].mean(dim=0) for label in unique_labels_labelled]
            class_centroids_labelled = torch.stack(class_centroids_labelled)

            # Align class indexes using cluster prototypes and classifier weights
            # Calculate prototype for each cluster
            cluster_set = set(req_c)
            cluster_prototypes = []
            #cluster_stds = []
            for cluster_ind in range(len(cluster_set)):

                # get the mean of features for the cluster
                if args.client.clust_feats == 'feats':
                    cluster_feats = all_feats[req_c == cluster_ind]
                elif args.client.clust_feats == 'feats_proj':
                    cluster_feats = all_feats_proj[req_c == cluster_ind]
                cluster_mean = cluster_feats.mean(dim=0)
                cluster_prototypes.append(cluster_mean)

                # Calculate standard deviation for the cluster
                #cluster_std = cluster_feats.std(dim=0)
                # for numerical stability
                #cluster_std[cluster_std == 0] = 1e-8
                #cluster_stds.append(cluster_std)

            cluster_prototypes = torch.stack(cluster_prototypes)
            #cluster_stds = torch.stack(cluster_stds)

            seen_class_similarities = F.cosine_similarity(class_centroids_labelled.unsqueeze(1), cluster_prototypes.unsqueeze(0), dim=2)


            # Initialize the mapping dictionary
            ind_map_gt_to_pred = {}
            assigned_clusters = set()
            remaining_clusters = [i for i in range(len(cluster_set))]
            not_assign_list = [i for i in range(num_classes)]

            ## Filter out the classes that have wasserstein distance less than threshold
            threshold = args.client.align_threshold
            for i, label in enumerate(unique_labels_labelled):
                class_centroid = class_centroids_labelled[i]
                simiilarities_between_clusters = seen_class_similarities[i]
                cluster_ind = torch.argmax(simiilarities_between_clusters).item()
                print(f'class {label} - top1 cluster index: {cluster_ind}')
                ## Estimate distribution of the cluster
                if args.client.clust_feats == 'feats':
                    cluster_feats = all_feats[req_c == cluster_ind]
                elif args.client.clust_feats == 'feats_proj':
                    cluster_feats = all_feats_proj[req_c == cluster_ind]
                gmm_cluster = GaussianMixture(n_components=1).fit(cluster_feats.cpu().clone().numpy())
                cluster_mean = gmm_cluster.means_[0]
                cluster_var = np.diag(gmm_cluster.covariances_[0])
                diff = class_centroid.cpu().clone().numpy() - cluster_mean
                inv_var = 1 / cluster_var
                mean_var = np.sqrt(np.sum(cluster_var))
                wasserstein_dist = np.sqrt(np.sum((diff ** 2) * inv_var))
                print(f'class {label} - wasserstein distance: {wasserstein_dist} - mean_var: {mean_var} - threshold: {args.client.align_threshold * mean_var}')
                if wasserstein_dist <= threshold * mean_var:
                    ind_map_gt_to_pred[label] = cluster_ind
                    assigned_clusters.add(cluster_ind)
                    remaining_clusters.remove(cluster_ind)


            # Calculate pairwise cosine similarity matrix between classifier weights and cluster prototypes 
            cosine_similarity_matrix = F.cosine_similarity(classifier_weights.unsqueeze(1), 
                                                        cluster_prototypes.unsqueeze(0), dim=2)
            
            # Get the number of seen classes
            num_seen_classes = len(args.dataset.seen_classes)
            

                        
            # Handle unseen classes
            remaining_clusters_copy = copy.deepcopy(set(remaining_clusters))
            unseen_class_weights = set(range(num_seen_classes, num_classes))

            # Hungarian matching for unseen classes
            if remaining_clusters_copy and unseen_class_weights:
                remaining_clusters_copy = list(remaining_clusters_copy)
                unseen_class_weights = list(unseen_class_weights)
                cost_matrix = torch.zeros((len(unseen_class_weights), len(remaining_clusters_copy)))
                for i, weight_idx in enumerate(unseen_class_weights):
                    for j, cluster_idx in enumerate(remaining_clusters_copy):
                        cost_matrix[i, j] = 1 - cosine_similarity_matrix[weight_idx, cluster_idx]
                row_ind, col_ind = linear_assignment(cost_matrix.cpu().numpy())
                for i, j in zip(row_ind, col_ind):
                    ind_map_gt_to_pred[unseen_class_weights[i]] = remaining_clusters_copy[j]
                    assigned_clusters.add(remaining_clusters_copy[j])
                    remaining_clusters.remove(remaining_clusters_copy[j])

            # Greedy matching for remaining seen classes
            remaining_seen_weights = set(range(num_seen_classes)) - set(ind_map_gt_to_pred.keys())
            if remaining_seen_weights:
                remaining_seen_weights = list(remaining_seen_weights)
                for weight_idx in remaining_seen_weights:
                    similarities = cosine_similarity_matrix[weight_idx]
                    sorted_indices = torch.argsort(similarities, descending=True)
                    for idx in sorted_indices:
                        if idx.item() in remaining_clusters:
                            ind_map_gt_to_pred[weight_idx] = idx.item()
                            remaining_clusters.remove(idx.item())
                            break
            
            print(f"ind_map_gt_to_pred not sorted!: {ind_map_gt_to_pred}")
            assert len(remaining_clusters) == 0
            
            # Ensure the mapping is sorted
            ind_map_pred_to_gt = {j: i for i, j in ind_map_gt_to_pred.items()}
            ind_map_pred_to_gt = dict(sorted(ind_map_pred_to_gt.items()))

            print(f'ind_map_pred_to_gt_proto: {ind_map_pred_to_gt}')

            
        else:
            assert False
    
    true_ind_map_pred_to_gt = {j: i for i, j in ind_map_gt.items()}
    true_ind_map_pred_to_gt = dict(sorted(true_ind_map_pred_to_gt.items()))
    # Filter true_ind_map_pred_to_gt based on labelled and unlabelled class sets
    local_classes_set = set(all_targets)
    
    filtered_true_ind_map_pred_to_gt = {}
    for pred, gt in true_ind_map_pred_to_gt.items():
        if gt in local_classes_set:
            filtered_true_ind_map_pred_to_gt[pred] = gt
    
    print(f'ind_map_pred_to_gt: {ind_map_pred_to_gt}')
    true_ind_map_pred_to_gt = filtered_true_ind_map_pred_to_gt
    print(f'true_ind_map_pred_to_gt: {true_ind_map_pred_to_gt}')
    
    aligned_preds = np.array([ind_map_pred_to_gt[i] for i in req_c[len(targets_labelled):]])
    
    #aligned_preds = np.array([ind_map_pred_to_gt_logit[i] for i in req_c[len(targets_labelled):]])
    # aligned_preds = np.array([true_ind_map_pred_to_gt[i] for i in req_c])

    # Plot confusion matrix between aligned_preds and targets
    D = max(aligned_preds.max(), targets.max()) + 1
    aligned_w = np.zeros((D, D), dtype=int)
    for i in range(len(aligned_preds)):
        aligned_w[aligned_preds[i], targets_unlabelled[i]] += 1

    ind = linear_assignment(aligned_w.max() - aligned_w)
    ind = np.vstack(ind).T
    ind_map_pred_to_gt_after_alignment = {i: j for i, j in ind}
    print(f'ind_map_pred_to_gt_after_alignment: {ind_map_pred_to_gt_after_alignment}')


    # Make Calss dict
    class_dict = defaultdict(int)
    for pred in aligned_preds:
        class_dict[str(pred)] += 1

    # Update prior preds
    for cls in class_dict:
        if not args.client.label_smoothing:
            prior[int(cls)] += class_dict[cls]
        else:
            smooth_max = args.client.smooth_max
            smooth_values = torch.ones(num_classes) * (1 - smooth_max) / (num_classes - 1)
            smooth_values[int(cls)] = smooth_max
            prior += smooth_values * class_dict[cls]

        total += class_dict[cls]
    prior = prior.float() / total

    prior = prior.cpu()
    
    gc.collect()

    return num_clusters, prior, all_acc, old_acc, new_acc, aligned_w, aligned_preds, ind_map_pred_to_gt, req_c[len(targets_labelled):]


def update_prior_with_novel_centroids(args, prior_dist, model, loader, evaler, num_classes, epoch, device, K=None):
    feats_labelled, feats_proj_labelled, logits_labelled, targets_labelled, feats_unlabelled, feats_proj_unlabelled, logits_unlabelled, targets_unlabelled, mask = extract_local_features(args, model, loader, evaler, device)

    targets_unlabelled = targets_unlabelled.astype(int)
    targets_labelled = targets_labelled.astype(int)
    all_feats = torch.cat([feats_labelled, feats_unlabelled], dim=0)
    all_feats_proj = torch.cat([feats_proj_labelled, feats_proj_unlabelled], dim=0)
    all_targets = np.concatenate([targets_labelled, targets_unlabelled])
    all_logits = torch.cat([logits_labelled, logits_unlabelled], dim=0)


    if args.client.est_num_clusters:
        ## Estimate num_clusters
        if args.client.clust_feats == 'feats':
            num_clusters = estimate_num_clusters(feats_labelled.numpy(), targets_labelled, feats_unlabelled.numpy(), targets_unlabelled, k_mode=args.client.k_mode, total_classes=num_classes, tolerance=args.client.tolerance)
        elif args.client.clust_feats == 'feats_proj':
            num_clusters = estimate_num_clusters(feats_proj_labelled.numpy(), targets_labelled, feats_proj_unlabelled.numpy(), targets_unlabelled, k_mode=args.client.k_mode, total_classes=num_classes, tolerance=args.client.tolerance)
    else:
        num_clusters = K

    
    ## Perform local clustering
    if args.client.update_alg =='finch':
        if args.client.clust_feats == 'feats':
            c, num_clust, req_c = FINCH(all_feats.numpy(), req_clust=num_clusters, distance='cosine', verbose=False)
        elif args.client.clust_feats == 'feats_proj':
            c, num_clust, req_c = FINCH(all_feats_proj.numpy(), req_clust=num_clusters, distance='cosine', verbose=False)
        req_c = req_c.astype(int)
        targets = all_targets.astype(int)        

    elif args.client.update_alg =='ood_finch':

        lambda_param = args.client.lambda_param
        similarities = cosine_similarity(feats_unlabelled, feats_labelled)
        # k-th nearest neighbor 찾기
        kth_similarities = np.sort(similarities, axis=1)[:, -1]
        ood_scores = 1 - kth_similarities

        c, num_clust, req_c = OOD_FINCH(feats_unlabelled.numpy(), req_clust=num_clusters, ood_scores=ood_scores, lambda_param=lambda_param, distance='cosine', verbose=False, labeled_features=feats_labelled.numpy())
        req_c = req_c.astype(int)
        targets = targets_unlabelled.astype(int)

    elif args.client.update_alg =='semi_finch':

        orig_dist = metrics.pairwise.pairwise_distances(all_feats.numpy(), all_feats.numpy(), metric='cosine')
        orig_dist_copy = copy.deepcopy(orig_dist)

        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)

        orig_dist_labelled = orig_dist_copy[:len(targets_labelled)]
        for cls in np.unique(targets_labelled):
           indices = np.where(targets_labelled == cls)[0]
           cls_dist = orig_dist_labelled[indices]
           cls_rank = np.argmax(cls_dist, axis=1)
           initial_rank[indices] = cls_rank

        c, num_clust, req_c = FINCH(all_feats.numpy(), initial_rank=initial_rank, req_clust=num_clusters, distance='cosine', verbose=False)
        targets = all_targets.astype(int)
        req_c = req_c.astype(int)
    else:
        assert False

    ## Eval Clustering results
    all_acc, old_acc, new_acc, w, ind_map_gt = evaler.split_cluster_acc_v2(targets, req_c, mask)

    prior = torch.zeros(num_classes)
    total = 0
    
    # update prior for labelled data
    for i in range(len(targets_labelled)):
        prior[int(targets_labelled[i])] += 1
        total += 1
    

    cluster_set = set(req_c)
    # Update prior for unlabelled data
    if args.client.align_gt:
        # Align Class idxes with ground truths
        ind_map_pred_to_gt = {j: i for i, j in ind_map_gt.items()}
        ind_map_pred_to_gt = dict(sorted(ind_map_pred_to_gt.items()))


        cluster_prototypes = []
        #cluster_stds = []
        for cluster_ind in range(len(cluster_set)):
            # get the mean of features for the cluster
            if args.client.clust_feats == 'feats':
                cluster_feats = all_feats[req_c == cluster_ind]

                # req_c_unlabelled = req_c[len(targets_labelled):]
                # cluster_feats = feats_unlabelled[req_c_unlabelled == cluster_ind]

            elif args.client.clust_feats == 'feats_proj':
                cluster_feats = all_feats_proj[req_c == cluster_ind]
            cluster_mean = cluster_feats.mean(dim=0)
            cluster_prototypes.append(cluster_mean)

        cluster_prototypes = torch.stack(cluster_prototypes)
    
    else:
        if args.client.align_type == 'sample':
            preds_unlabelled = logits_unlabelled.argmax(1).cpu().numpy()
            concat_targets = np.concatenate([targets_labelled, preds_unlabelled]).astype(int)
            D = num_classes
            w = np.zeros((D, D), dtype=int)
            for i in range(concat_targets.size):
                w[req_c[i], concat_targets[i]] += 1

            ind = linear_assignment(w.max() - w)
            ind = np.vstack(ind).T
            ind_map = {i: j for i, j in ind}
            ind_map_pred_to_gt = dict(sorted(ind_map.items()))
            print(f'Prior update process:')
            print(f'ind_map_pred_to_gt_logit: {ind_map_pred_to_gt}')

        elif args.client.align_type == 'centroid':

            # Get the classifier weights
            classifier_weights = model.proj_layer.last_layer.parametrizations.weight.original1.data.clone().cpu()
            
            targets_labelled = torch.LongTensor(targets_labelled)
            if args.client.clust_feats == 'feats':
                #feats_labelled = F.normalize(feats_labelled, dim=1, p=2)
                feats_labelled = feats_labelled
            elif args.client.clust_feats == 'feats_proj':
                feats_labelled = feats_proj_labelled


            unique_labels_labelled = torch.unique(targets_labelled).tolist()                
            class_centroids_labelled = [feats_labelled[targets_labelled == label].mean(dim=0) for label in unique_labels_labelled]
            class_centroids_labelled = torch.stack(class_centroids_labelled)

            # Align class indexes using cluster prototypes and classifier weights
            # Calculate prototype for each cluster
            
            cluster_prototypes = []
            #cluster_stds = []
            for cluster_ind in range(len(cluster_set)):
                # get the mean of features for the cluster
                if args.client.clust_feats == 'feats':
                    cluster_feats = all_feats[req_c == cluster_ind]
                elif args.client.clust_feats == 'feats_proj':
                    cluster_feats = all_feats_proj[req_c == cluster_ind]
                cluster_mean = cluster_feats.mean(dim=0)
                cluster_prototypes.append(cluster_mean)

            cluster_prototypes = torch.stack(cluster_prototypes)
            #cluster_stds = torch.stack(cluster_stds)

            seen_class_similarities = F.cosine_similarity(class_centroids_labelled.unsqueeze(1), cluster_prototypes.unsqueeze(0), dim=2)


            # Initialize the mapping dictionary
            ind_map_gt_to_pred = {}
            assigned_clusters = set()
            remaining_clusters = [i for i in range(len(cluster_set))]
            not_assign_list = [i for i in range(num_classes)]

            ## Filter out the classes that have wasserstein distance less than threshold
            threshold = args.client.align_threshold
            for i, label in enumerate(unique_labels_labelled):
                class_centroid = class_centroids_labelled[i]
                simiilarities_between_clusters = seen_class_similarities[i]
                cluster_ind = torch.argmax(simiilarities_between_clusters).item()
                print(f'class {label} - top1 cluster index: {cluster_ind}')
                ## Estimate distribution of the cluster
                if args.client.clust_feats == 'feats':
                    cluster_feats = all_feats[req_c == cluster_ind]
                elif args.client.clust_feats == 'feats_proj':
                    cluster_feats = all_feats_proj[req_c == cluster_ind]
                gmm_cluster = GaussianMixture(n_components=1).fit(cluster_feats.cpu().clone().numpy())
                cluster_mean = gmm_cluster.means_[0]
                cluster_var = np.diag(gmm_cluster.covariances_[0])
                diff = class_centroid.cpu().clone().numpy() - cluster_mean
                inv_var = 1 / cluster_var
                mean_var = np.sqrt(np.sum(cluster_var))
                wasserstein_dist = np.sqrt(np.sum((diff ** 2) * inv_var))
                print(f'class {label} - wasserstein distance: {wasserstein_dist} - mean_var: {mean_var} - threshold: {args.client.align_threshold * mean_var}')
                if wasserstein_dist <= threshold * mean_var:
                    ind_map_gt_to_pred[label] = cluster_ind
                    assigned_clusters.add(cluster_ind)
                    remaining_clusters.remove(cluster_ind)

            # Calculate pairwise cosine similarity matrix between classifier weights and cluster prototypes 
            cosine_similarity_matrix = F.cosine_similarity(classifier_weights.unsqueeze(1), 
                                                        cluster_prototypes.unsqueeze(0), dim=2)
            
            # Get the number of seen classes
            num_seen_classes = len(args.dataset.seen_classes)

            # Handle unseen classes
            remaining_clusters_copy = copy.deepcopy(set(remaining_clusters))
            unseen_class_weights = set(range(num_seen_classes, num_classes))

            # Hungarian matching for unseen classes
            if remaining_clusters_copy and unseen_class_weights:
                remaining_clusters_copy = list(remaining_clusters_copy)
                unseen_class_weights = list(unseen_class_weights)
                cost_matrix = torch.zeros((len(unseen_class_weights), len(remaining_clusters_copy)))
                for i, weight_idx in enumerate(unseen_class_weights):
                    for j, cluster_idx in enumerate(remaining_clusters_copy):
                        cost_matrix[i, j] = 1 - cosine_similarity_matrix[weight_idx, cluster_idx]
                row_ind, col_ind = linear_assignment(cost_matrix.cpu().numpy())
                for i, j in zip(row_ind, col_ind):
                    ind_map_gt_to_pred[unseen_class_weights[i]] = remaining_clusters_copy[j]
                    assigned_clusters.add(remaining_clusters_copy[j])
                    remaining_clusters.remove(remaining_clusters_copy[j])

            # Greedy matching for remaining seen classes
            remaining_seen_weights = set(range(num_seen_classes)) - set(ind_map_gt_to_pred.keys())
            if remaining_seen_weights:
                remaining_seen_weights = list(remaining_seen_weights)
                for weight_idx in remaining_seen_weights:
                    similarities = cosine_similarity_matrix[weight_idx]
                    sorted_indices = torch.argsort(similarities, descending=True)
                    for idx in sorted_indices:
                        if idx.item() in remaining_clusters:
                            ind_map_gt_to_pred[weight_idx] = idx.item()
                            remaining_clusters.remove(idx.item())
                            break
            
            print(f"ind_map_gt_to_pred not sorted!: {ind_map_gt_to_pred}")
            assert len(remaining_clusters) == 0
            
            # Ensure the mapping is sorted
            ind_map_pred_to_gt = {j: i for i, j in ind_map_gt_to_pred.items()}
            ind_map_pred_to_gt = dict(sorted(ind_map_pred_to_gt.items()))

            print(f'ind_map_pred_to_gt_proto: {ind_map_pred_to_gt}')
            
        else:
            assert False
        
    true_ind_map_pred_to_gt = {j: i for i, j in ind_map_gt.items()}
    true_ind_map_pred_to_gt = dict(sorted(true_ind_map_pred_to_gt.items()))

    # Filter true_ind_map_pred_to_gt based on labelled and unlabelled class sets
    local_classes_set = set(all_targets)
    
    filtered_true_ind_map_pred_to_gt = {}
    for pred, gt in true_ind_map_pred_to_gt.items():
        if gt in local_classes_set:
            filtered_true_ind_map_pred_to_gt[pred] = gt
    
    true_ind_map_pred_to_gt = filtered_true_ind_map_pred_to_gt
    print(f'true_ind_map_pred_to_gt: {true_ind_map_pred_to_gt}')

    print(f'ind_map_pred_to_gt: {ind_map_pred_to_gt}')
    
    aligned_preds = np.array([ind_map_pred_to_gt[i] for i in req_c[len(targets_labelled):]])
    
    #aligned_preds = np.array([ind_map_pred_to_gt_logit[i] for i in req_c[len(targets_labelled):]])
    # aligned_preds = np.array([true_ind_map_pred_to_gt[i] for i in req_c])

    # Plot confusion matrix between aligned_preds and targets
    D = max(aligned_preds.max(), targets.max()) + 1
    aligned_w = np.zeros((D, D), dtype=int)
    for i in range(len(aligned_preds)):
        aligned_w[aligned_preds[i], targets_unlabelled[i]] += 1

    ind = linear_assignment(aligned_w.max() - aligned_w)
    ind = np.vstack(ind).T
    ind_map_pred_to_gt_after_hungarian = {i: j for i, j in ind}
    print(f'ind_map_pred_to_gt_after_hungarian: {ind_map_pred_to_gt_after_hungarian}')

    # Make Calss dict
    class_dict = defaultdict(int)
    for pred in aligned_preds:
        class_dict[str(pred)] += 1

    # Update prior preds
    for cls in class_dict:
        if not args.client.label_smoothing:
            prior[int(cls)] += class_dict[cls]
        else:
            smooth_max = args.client.smooth_max
            smooth_values = torch.ones(num_classes) * (1 - smooth_max) / (num_classes - 1)
            smooth_values[int(cls)] = smooth_max
            prior += smooth_values * class_dict[cls]

        total += class_dict[cls]
    prior = prior.float() / total
    prior = prior.cpu()

    # Get centroids of novel classes
    cluster_centroids = []
    cluster_targets = []

    for clust_id, classifier_idx in ind_map_pred_to_gt.items():
        if classifier_idx >= len(args.dataset.seen_classes):
            if clust_id in cluster_set:
                cluster_centroids.append(cluster_prototypes[clust_id])
                cluster_targets.append(classifier_idx)

    cluster_centroids = torch.stack(cluster_centroids)
    cluster_targets = torch.LongTensor(cluster_targets)
    #print(f'cluster_centroids: {cluster_centroids}')
    print(f'returning cluster_targets: {cluster_targets}')
    gc.collect()

    return num_clusters, prior, all_acc, old_acc, new_acc, aligned_w, aligned_preds, ind_map_pred_to_gt, cluster_centroids, cluster_targets



def update_prior_threshold(args, prior_dist, model, loader, evaler, num_classes, epoch, device, K=None):


    threshold = args.client.align_threshold

    feats_labelled, feats_proj_labelled, logits_labelled, targets_labelled, feats_unlabelled, feats_proj_unlabelled, logits_unlabelled, targets_unlabelled, mask = extract_local_features(args, model, loader, evaler, device)

    targets_unlabelled = targets_unlabelled.astype(int)
    targets_labelled = targets_labelled.astype(int)
    all_feats = torch.cat([feats_labelled, feats_unlabelled], dim=0)
    all_feats = F.normalize(all_feats, dim=1, p=2)
    all_feats_proj = torch.cat([feats_proj_labelled, feats_proj_unlabelled], dim=0)
    all_targets = np.concatenate([targets_labelled, targets_unlabelled])
    all_logits = torch.cat([logits_labelled, logits_unlabelled], dim=0)


    if args.client.est_num_clusters:
        ## Estimate num_clusters
        if args.client.clust_feats == 'feats':
            num_clusters = estimate_num_clusters(feats_labelled.numpy(), targets_labelled, feats_unlabelled.numpy(), targets_unlabelled, k_mode=args.client.k_mode)
        elif args.client.clust_feats == 'feats_proj':
            num_clusters = estimate_num_clusters(feats_proj_labelled.numpy(), targets_labelled, feats_proj_unlabelled.numpy(), targets_unlabelled, k_mode=args.client.k_mode)
    else:
        num_clusters = K

    # Get class centroids for seen classes
    targets_labelled = torch.LongTensor(targets_labelled)
    feats_labelled = F.normalize(feats_labelled, dim=1, p=2)
    feats_unlabelled = F.normalize(feats_unlabelled, dim=1, p=2)
    unique_labels_labelled = torch.unique(targets_labelled)
    class_centroids_labelled = [feats_labelled[targets_labelled == label].mean(dim=0) for label in unique_labels_labelled]
    class_centroids_labelled = torch.stack(class_centroids_labelled)

    # Calculate cosine similarity between unlabelled features and seen class centroids
    # similarities = cosine_similarity(feats_unlabelled, class_centroids_labelled)
    
    # print(f'similarities: {similarities}')

    # # Filter examlpes based on the threshold
    # vals, indices = torch.max(torch.from_numpy(similarities), dim=1)
    # seen_mask = vals >= threshold
    # unseen_mask = ~seen_mask

    # # Get preds from filtered seen features
    # feats_filtered_seen = feats_unlabelled[seen_mask]
    # targets_filtered_seen = targets_unlabelled[seen_mask]
    # preds_filtered_seen = indices[seen_mask]

    # # Filter unseen features
    # feats_filtered_unseen = feats_unlabelled[unseen_mask]
    # targets_filtered_unseen = targets_unlabelled[unseen_mask]
    # preds_filtered_unseen = indices[unseen_mask]

    similarities = cosine_similarity(feats_unlabelled, feats_labelled)
    
    print(f'similarities: {similarities}')

    # Filter examlpes based on the threshold
    vals, indices = torch.max(torch.from_numpy(similarities), dim=1)
    seen_mask = vals >= threshold
    unseen_mask = ~seen_mask

    # Get preds from filtered seen features
    feats_filtered_seen = feats_unlabelled[seen_mask]
    targets_filtered_seen = targets_unlabelled[seen_mask]
    preds_filtered_seen = targets_labelled[indices[seen_mask]]

    # Filter unseen features
    feats_filtered_unseen = feats_unlabelled[unseen_mask]
    targets_filtered_unseen = targets_unlabelled[unseen_mask]
    preds_filtered_unseen = targets_labelled[indices[unseen_mask]]

    # Run FINCH on filtered unseen features
    c, num_clust, req_c = FINCH(feats_filtered_unseen.numpy(), req_clust=num_clusters - len(unique_labels_labelled), distance='cosine', verbose=False)

    preds = torch.zeros_like(torch.from_numpy(targets_unlabelled)).long()
    preds[seen_mask] = preds_filtered_seen
    preds[unseen_mask] = torch.from_numpy(req_c + len(args.dataset.seen_classes)).long()


    # Evaluate clustering results
    all_acc, old_acc, new_acc, w, ind_map_gt = evaler.split_cluster_acc_v3(targets_unlabelled, preds.numpy(), mask[len(targets_labelled):])

    prior = torch.zeros(num_classes)
    total = 0
    
    # update prior for labelled data
    for i in range(len(targets_labelled)):
        prior[int(targets_labelled[i])] += 1
        total += 1

    # Get the classifier weights
    classifier_weights = model.proj_layer.last_layer.parametrizations.weight.original1.data.clone().cpu()

    # Align class indexes using cluster prototypes and classifier weights
    # Calculate prototype for each cluster
    cluster_set = set(req_c)
    cluster_prototypes = []
    cluster_stds = []
    for cluster_ind in range(len(cluster_set)):

        # get the mean of features for the cluster
        cluster_feats = feats_filtered_unseen[req_c == cluster_ind]
        cluster_feats = F.normalize(cluster_feats, dim=1, p=2)
        cluster_mean = cluster_feats.mean(dim=0)
        cluster_prototypes.append(cluster_mean)

        # Calculate standard deviation for the cluster
        cluster_std = cluster_feats.std(dim=0)
        # for numerical stability
        cluster_std[cluster_std == 0] = 1e-8
        cluster_stds.append(cluster_std)

    cluster_prototypes = torch.stack(cluster_prototypes)
    cluster_stds = torch.stack(cluster_stds)

    # Calculate pairwise cosine similarity matrix between classifier weights and cluster prototypes 
    cosine_similarity_matrix = F.cosine_similarity(classifier_weights.unsqueeze(1), 
                                                cluster_prototypes.unsqueeze(0), dim=2)
    
    # Get the number of seen classes
    num_seen_classes = len(args.dataset.seen_classes)
    
    # Initialize the mapping dictionary
    ind_map_gt_to_pred = {}


    # Handle labelled set classes
    for i, label in enumerate(unique_labels_labelled):
        ind_map_gt_to_pred[label.item()] = label.item()


    assigned_clusters = set()
    remaining_clusters = [i for i in range(len(cluster_set))]
    not_assign_list = [i for i in range(num_classes)]

    # Handle unseen classes
    remaining_clusters_copy = copy.deepcopy(set(remaining_clusters))
    unseen_class_weights = set(range(num_seen_classes, num_classes))

    # Hungarian matching for unseen classes
    if remaining_clusters_copy and unseen_class_weights:
        remaining_clusters_copy = list(remaining_clusters_copy)
        unseen_class_weights = list(unseen_class_weights)
        cost_matrix = torch.zeros((len(unseen_class_weights), len(remaining_clusters_copy)))
        for i, weight_idx in enumerate(unseen_class_weights):
            for j, cluster_idx in enumerate(remaining_clusters_copy):
                cost_matrix[i, j] = 1 - cosine_similarity_matrix[weight_idx, cluster_idx]
        row_ind, col_ind = linear_assignment(cost_matrix.cpu().numpy())
        for i, j in zip(row_ind, col_ind):
            ind_map_gt_to_pred[unseen_class_weights[i]] = remaining_clusters_copy[j] + num_seen_classes
            assigned_clusters.add(remaining_clusters_copy[j])
            remaining_clusters.remove(remaining_clusters_copy[j])

    # Greedy matching for remaining seen classes
    remaining_seen_weights = set(range(num_seen_classes)) - set(ind_map_gt_to_pred.keys())
    if remaining_seen_weights:
        remaining_seen_weights = list(remaining_seen_weights)
        for weight_idx in remaining_seen_weights:
            similarities = cosine_similarity_matrix[weight_idx]
            sorted_indices = torch.argsort(similarities, descending=True)
            for idx in sorted_indices:
                if idx.item() in remaining_clusters:
                    ind_map_gt_to_pred[weight_idx] = idx.item() + num_seen_classes
                    remaining_clusters.remove(idx.item())
                    break
    
    print(f"ind_map_gt_to_pred not sorted!: {ind_map_gt_to_pred}")

    # Ensure the mapping is sorted
    ind_map_pred_to_gt = {j: i for i, j in ind_map_gt_to_pred.items()}
    ind_map_pred_to_gt = dict(sorted(ind_map_pred_to_gt.items()))

    print(f'ind_map_pred_to_gt_proto: {ind_map_pred_to_gt}')

    assert len(remaining_clusters) == 0
    
    true_ind_map_pred_to_gt = {j: i for i, j in ind_map_gt.items()}
    true_ind_map_pred_to_gt = dict(sorted(true_ind_map_pred_to_gt.items()))
    # Filter true_ind_map_pred_to_gt based on labelled and unlabelled class sets
    local_classes_set = set(all_targets)
    
    filtered_true_ind_map_pred_to_gt = {}
    for pred, gt in true_ind_map_pred_to_gt.items():
        if gt in local_classes_set:
            filtered_true_ind_map_pred_to_gt[pred] = gt
    
    true_ind_map_pred_to_gt = filtered_true_ind_map_pred_to_gt
    print(f'true_ind_map_pred_to_gt: {true_ind_map_pred_to_gt}')
    
    shifted_req_c = req_c + num_seen_classes
    aligned_preds = np.array([ind_map_pred_to_gt[i] for i in shifted_req_c])
    aligned_preds = np.concatenate([preds_filtered_seen.numpy(), aligned_preds])

    targets = np.concatenate([targets_filtered_seen, targets_filtered_unseen])

    # Plot confusion matrix between aligned_preds and targets
    D = max(aligned_preds.max(), targets.max()) + 1
    aligned_w = np.zeros((D, D), dtype=int)
    for i in range(len(aligned_preds)):
        aligned_w[aligned_preds[i], targets[i]] += 1

    ind = linear_assignment(aligned_w.max() - aligned_w)
    ind = np.vstack(ind).T
    ind_map_pred_to_gt = {i: j for  i, j in ind}
    print(f'ind_map_pred_to_gt: {ind_map_pred_to_gt}')


    # Make Calss dict
    class_dict = defaultdict(int)
    for pred in aligned_preds:
        class_dict[str(pred)] += 1

    # Update prior preds
    for cls in class_dict:
        if not args.client.label_smoothing:
            prior[int(cls)] += class_dict[cls]
        else:
            smooth_max = args.client.smooth_max
            smooth_values = torch.ones(num_classes) * (1 - smooth_max) / (num_classes - 1)
            smooth_values[int(cls)] = smooth_max
            prior += smooth_values * class_dict[cls]

        total += class_dict[cls]
    prior = prior.float() / total
    prior = prior.cpu()
    
    gc.collect()

    return num_clusters, prior, all_acc, old_acc, new_acc, aligned_w, aligned_preds, ind_map_pred_to_gt



def apply_label_noise(y, noise_rate):

    y = y.clone()
    classes = y.unique()
    num_classes = len(classes)
    
    select_index = (torch.rand(y.size(0)) < noise_rate).nonzero()
    if len(select_index) > 0:
        random_label = torch.randint(0, num_classes, select_index.size()).long()
        random_label = classes[random_label]
        origin_label = y[select_index]
        random_label += (origin_label == random_label).long()
        random_label %= num_classes
        y[select_index] = random_label
    return y