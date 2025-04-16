
import os
import pandas as pd
import numpy as np
import torch
from copy import deepcopy



def subsample_instances(dataset, prop_indices_to_subsample=0.5):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices

def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[mask]
    dataset.targets = dataset.targets[mask]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cub = np.array(include_classes) + 1     # CUB classes are indexed 1 --> 200 instead of 0 --> 199
    cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows()) if int(r['target']) in include_classes_cub]

    # target_xform_dict = {}
    # for i, k in enumerate(include_classes):
    #     target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.data['target'])

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.data['target'] == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_cub_datasets(args, trainset, client):

    # Init entire training set
    #whole_training_set = CustomCub2011(root=cub_root, transform=train_transform, train=True)
    client_dataset = subsample_dataset(deepcopy(trainset), np.array(list(client['idxs'])))

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(client_dataset), include_classes=args.dataset.seen_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=0.5)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Get unlabelled data
    unlabelled_indices = set(client_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(trainset), np.array(list(unlabelled_indices)))

    # merged_dataset = MergedDataset(labelled_dataset=deepcopy(train_dataset_labelled),
    #                                unlabelled_dataset=deepcopy(train_dataset_unlabelled),
    #                                class_dict=client['class_dict'])

    return train_dataset_labelled, train_dataset_unlabelled, unlabelled_indices




def get_cub_test_semisup_dataset(args, testset):

    # Get labelled test set which has subsampled classes, then subsample some indices from that
    test_dataset_labelled = subsample_classes(deepcopy(testset), include_classes=args.dataset.seen_classes)
    subsample_indices = subsample_instances(test_dataset_labelled, prop_indices_to_subsample=0.5)
    test_dataset_labelled = subsample_dataset(test_dataset_labelled, subsample_indices)

    # Get unlabelled data
    unlabelled_indices = set(testset.uq_idxs) - set(test_dataset_labelled.uq_idxs)
    test_dataset_unlabelled = subsample_dataset(deepcopy(testset), np.array(list(unlabelled_indices)))

    # merged_dataset = MergedDataset(labelled_dataset=deepcopy(train_dataset_labelled),
    #                                unlabelled_dataset=deepcopy(train_dataset_unlabelled),
    #                                class_dict=client['class_dict'])

    return test_dataset_labelled, test_dataset_unlabelled