import torch
from torchvision import datasets, transforms
import os
from datasets.cifar import cifar_noniid, cifar_dirichlet_balanced, cifar_dirichlet_unbalanced, cifar_iid, cifar_overlap, cifar_toyset, cifar_subclass_dirichlet_balanced
# from datasets.cub import get_cub_datasets
from datasets.utils_cub import get_cub_datasets
import torch.nn as nn
import csv
from typing import List,Dict
import copy
import json
from collections import OrderedDict
from torch.utils.data import  Dataset
import numpy as np
import time
#from datasets import get_cub_datasets
from datasets.base import MergedDataset

from collections import defaultdict
from datasets.utils_cub import subsample_dataset as subsample_dataset_cub
from datasets.imagenet import get_imagenet_datasets
from datasets.imagenet import subsample_dataset as subsample_dataset_imagenet
from datasets.stanford_cars import get_scars_datasets
from datasets.stanford_cars import subsample_dataset as subsample_dataset_scars
from datasets.pets import get_pets_datasets
from datasets.pets import subsample_dataset as subsample_dataset_pets

__all__ = ['DatasetSplit', 'DatasetSplitMultiView', 'DatasetSplitMultiViews', 'get_dataset', 'MultiViewDataInjector', 'GaussianBlur', 'TransformTwice'
                                                                                                            ]

create_dataset_log = False


# class MergedDataset(Dataset):
#     """
#     Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
#     Allows you to iterate over them in parallel
#     """
#     def __init__(self, labelled_dataset, unlabelled_dataset, class_dict=None):
#
#         self.labelled_dataset = labelled_dataset
#         self.unlabelled_dataset = unlabelled_dataset
#         self.target_transform = None
#         if class_dict is not None:
#             self.class_dict = class_dict
#         else:
#             self.class_dict = defaultdict(int)
#             for idx in range(len(self.labelled_dataset)):
#                 _, label, _ = self.labelled_dataset[idx]
#                 if torch.is_tensor(label):
#                     label = str(label.item())
#                 else:
#                     label = str(label)
#
#                 self.class_dict[str(label)] += 1
#             for idx in range(len(self.unlabelled_dataset)):
#                 _, label, _ = self.unlabelled_dataset[idx]
#                 if torch.is_tensor(label):
#                     label = str(label.item())
#                 else:
#                     label = str(label)
#                 if label in self.class_dict:
#                     self.class_dict[str(label)] += 1
#                 else:
#                     self.class_dict[str(label)] = 1
#
#     def __getitem__(self, item):
#
#         if item < len(self.labelled_dataset):
#             img, label, uq_idx = self.labelled_dataset[item]
#             labeled_or_not = 1
#
#         else:
#             #import pdb; pdb.set_trace()
#             img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
#             labeled_or_not = 0
#
#
#         return img, label, uq_idx, np.array([labeled_or_not])
#
#     def __len__(self):
#         return len(self.unlabelled_dataset) + len(self.labelled_dataset)

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

        self.class_dict = {}
        for idx in self.idxs:
            _, label = self.dataset[idx]
            if torch.is_tensor(label):
                label = str(label.item())
            else:
                label = str(label)
            if label in self.class_dict:
                self.class_dict[str(label)] += 1
            else:
                self.class_dict[str(label)] = 1


    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label)
        return image, label
    
    @property
    def num_classes(self):
        return len(self.class_dict.keys())
    
    @property
    def class_ids(self):
        return self.class_dict.keys()
    
    def importance_weights(self, labels, pow=1):
        # total_count = sum(self.class_dict.values())
        class_counts = np.array([self.class_dict[str(label.item())] for label in labels])
        weights = (1/class_counts)**pow
        weights /= weights.mean()
        return weights

class DatasetSplitMultiView(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        (view1, view2), label = self.dataset[self.idxs[item]]
        return torch.tensor(view1), torch.tensor(view2), torch.tensor(label)
    

class DatasetSplitMultiViews(DatasetSplit):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    # def __init__(self, dataset, idxs):
    #     self.dataset = dataset
    #     self.idxs = [int(i) for i in idxs]


    def __getitem__(self, item):
        # (view1, view2), label = self.dataset[self.idxs[item]]
        view1, label = self.dataset[self.idxs[item]]
        view2, label = self.dataset[self.idxs[item]]
        return (torch.tensor(view1), torch.tensor(view2)), torch.tensor(label)

# 저장할 때 사용된 함수를 다시 정의
def get_local_datasets(args, trainset, mode='iid'):
    dataset_name = args.dataset.name
    num_seen_classes = len(args.dataset.seen_classes)
    num_unseen_classes = len(args.dataset.unseen_classes)    
    
    if dataset_name in ['cifar10', 'cifar100', 'cub', 'cub2', 'imagenet', 'scars', 'pets']:
        directory = args.dataset.client_path + '/' + dataset_name + '/' + (
            'un' if args.split.unbalanced == True else '') + 'balanced'
        filepath = directory + '/' + mode +  (
            str(args.split.alpha) if 'dirichlet' in mode else '') +  '_clients' + str(args.trainer.num_clients) + f'_{num_seen_classes}seen' + '.pt'
        check_already_exist = os.path.isfile(filepath) and (os.stat(filepath).st_size != 0)
        create_new_client_data = not check_already_exist or args.split.create_client_dataset
        print('Create new client data: ', create_new_client_data)
    else:
        assert False

    start_time = time.time()
    if create_new_client_data == False:
        # Load Dataset from the path
        try:
            dataset = {}
            # with open(filepath) as f:
            #     # for idx, line in enumerate(f):
            #     #     dataset = eval(line)
            client_dict = torch.load(filepath)

        except:
            print("Have problem to read client data")
    else:
        # Generate Dataset and Save the dataset at the filepath.
        if mode == 'iid':
            dataset = cifar_iid(trainset, args.trainer.num_clients)
        elif mode == 'overlap':
            dataset = cifar_overlap(trainset, args.trainer.num_clients, args.split.overlap_ratio)
        # elif mode[:4] == 'skew' and mode[-5:] == 'class':
        elif mode == 'skew':
            class_per_client = args.split.class_per_client
            dataset = cifar_noniid(trainset, args.trainer.num_clients, class_per_client)
        elif mode == 'dirichlet':
            if args.split.unbalanced == True:
                dataset = cifar_dirichlet_unbalanced(trainset, args.trainer.num_clients, alpha=args.split.alpha)
            else:
                dataset = cifar_dirichlet_balanced(trainset, args.trainer.num_clients, alpha=args.split.alpha)
        elif mode == 'seen_iid_unseen_dirichlet':
            dataset = cifar_subclass_dirichlet_balanced(trainset, args.trainer.num_clients, alpha=args.split.alpha,
                                                               iid_classes=args.dataset.seen_classes, non_iid_classes=args.dataset.unseen_classes)
        elif mode == 'unseen_iid_seen_dirichlet':
            dataset = cifar_subclass_dirichlet_balanced(trainset, args.trainer.num_clients, alpha=args.split.alpha,
                                                        iid_classes=args.dataset.unseen_classes, non_iid_classes=args.dataset.seen_classes)
        else:
            assert False

        client_dict = {}
        filtered_dataset = {}
        start_time = time.time()
        total_unlabelled_indices =  []
        for client_idx, client_dataset_idxs in dataset.items():
            if args.dataset.name in ['cifar10', 'cifar100']:
                # get labelled training set which has subsampled classes, the subsample some indices from that
                client_dataset = subsample_dataset(copy.deepcopy(trainset), np.array(list(client_dataset_idxs)))
                train_dataset_labelled = subsample_classes(copy.deepcopy(client_dataset),
                                                           include_classes=list(range(len(args.dataset.seen_classes))))
                subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=0.5)
                train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)
                unlabelled_indices = set(client_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
                total_unlabelled_indices.extend(list(unlabelled_indices))
                train_dataset_unlabelled = subsample_dataset(copy.deepcopy(trainset),
                                                             np.array(list(unlabelled_indices)))

            elif args.dataset.name in ['cub', 'cub2']:
                client = {
                    'idxs': client_dataset_idxs,
                    'class_dict': None
                }
                train_dataset_labelled, train_dataset_unlabelled, unlabelled_indices = get_cub_datasets(args, trainset, client)
                total_unlabelled_indices.extend(list(unlabelled_indices))
            elif args.dataset.name in ['imagenet']:
                client = {
                    'idxs': client_dataset_idxs,
                    'class_dict': None
                }
                train_dataset_labelled, train_dataset_unlabelled, unlabelled_indices = get_imagenet_datasets(args, trainset, client)
                total_unlabelled_indices.extend(list(unlabelled_indices))
            elif args.dataset.name in ['scars']:
                client = {
                    'idxs': client_dataset_idxs,
                    'class_dict': None
                }
                train_dataset_labelled, train_dataset_unlabelled, unlabelled_indices = get_scars_datasets(args, trainset, client)
                total_unlabelled_indices.extend(list(unlabelled_indices))
            elif args.dataset.name in ['pets']:
                client = {
                    'idxs': client_dataset_idxs,
                    'class_dict': None
                }
                train_dataset_labelled, train_dataset_unlabelled, unlabelled_indices = get_pets_datasets(args, trainset, client)
                total_unlabelled_indices.extend(list(unlabelled_indices))
            else:
                assert False

            # for dset in [train_dataset_labelled, train_dataset_unlabelled]:
            #     dset.target_transform = target_transform

            merged_dataset = MergedDataset(labelled_dataset=copy.deepcopy(train_dataset_labelled),
                                           unlabelled_dataset=copy.deepcopy(train_dataset_unlabelled))

            client_dict[client_idx] = {'idxs': client_dataset_idxs,
                                       'class_dict': merged_dataset.class_dict}
            filtered_dataset[client_idx] = merged_dataset
        
        if args.dataset.name in ['cifar10', 'cifar100']:
            total_train_dataset_unlabelled = subsample_dataset(copy.deepcopy(trainset), np.array(total_unlabelled_indices))
        elif args.dataset.name in ['cub', 'cub2']:
            total_train_dataset_unlabelled = subsample_dataset_cub(copy.deepcopy(trainset), np.array(total_unlabelled_indices))
        elif args.dataset.name in ['imagenet']:
            total_train_dataset_unlabelled = subsample_dataset_imagenet(copy.deepcopy(trainset), np.array(total_unlabelled_indices))
        elif args.dataset.name in ['scars']:
            total_train_dataset_unlabelled = subsample_dataset_scars(copy.deepcopy(trainset), np.array(total_unlabelled_indices))
        elif args.dataset.name in ['pets']:
            total_train_dataset_unlabelled = subsample_dataset_pets(copy.deepcopy(trainset), np.array(total_unlabelled_indices))
        else:
            assert False
            
        filtered_dataset['total_train_dataset_unlabelled'] = total_train_dataset_unlabelled
        end_time = time.time()
        print(f" Data Generation time: {end_time - start_time:.5f} seconds")

        try:
            os.makedirs(directory, exist_ok=True)
            # with open(filepath, 'w') as f:
            #     print(dataset, file=f)
            torch.save(client_dict, filepath)

        except:
            print("Fail to write client data at " + directory)
        return filtered_dataset

    ## filtering labeled and unlabeled datasets
    filtered_dataset = {}
    start_time = time.time()
    total_unlabelled_indices =  []
    for client_idx, client in client_dict.items():
        if args.dataset.name in ['cifar10', 'cifar100']:
            # get labelled training set which has subsampled classes, the subsample some indices from tha
            client_dataset = subsample_dataset(copy.deepcopy(trainset), np.array(list(client['idxs'])))
            train_dataset_labelled = subsample_classes(copy.deepcopy(client_dataset),
                                                       include_classes=list(range(len(args.dataset.seen_classes))))
            subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=0.5)
            train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

            unlabelled_indices = set(client_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
            total_unlabelled_indices.extend(list(unlabelled_indices))
            train_dataset_unlabelled = subsample_dataset(copy.deepcopy(trainset),
                                                         np.array(list(unlabelled_indices)))


        elif args.dataset.name in ['cub', 'cub2']:
            train_dataset_labelled, train_dataset_unlabelled, unlabelled_indices = get_cub_datasets(args, trainset, client)
            total_unlabelled_indices.extend(list(unlabelled_indices))
            # merged_dataset = get_cub_datasets(args, trainset, client)
        elif args.dataset.name in ['imagenet']:
            train_dataset_labelled, train_dataset_unlabelled, unlabelled_indices = get_imagenet_datasets(args, trainset, client)
            total_unlabelled_indices.extend(list(unlabelled_indices))
        elif args.dataset.name in ['scars']:
            train_dataset_labelled, train_dataset_unlabelled, unlabelled_indices = get_scars_datasets(args, trainset, client)
            total_unlabelled_indices.extend(list(unlabelled_indices))
        elif args.dataset.name in ['pets']:
            train_dataset_labelled, train_dataset_unlabelled, unlabelled_indices = get_pets_datasets(args, trainset, client)
            total_unlabelled_indices.extend(list(unlabelled_indices))
        else:
            assert False

        # for dataset in [train_dataset_labelled, train_dataset_unlabelled]:
        #     dataset.target_transform = target_transform
        #     if args.dataset.name in ['cub', 'cub2']:
        #         dataset.align_targets()
        ## Class_dict transform
        # if args.dataset.name in ['cub']:
        #     new_class_dict = {}
        #     class_dict_before_transform = client['class_dict']
        #     for cls_idx in class_dict_before_transform:
        #         new_class_dict[str(target_transform_dict[int(cls_idx)])] = class_dict_before_transform[cls_idx]
        #     client['class_dict'] = new_class_dict

        merged_dataset = MergedDataset(labelled_dataset=copy.deepcopy(train_dataset_labelled),
                                       unlabelled_dataset=copy.deepcopy(train_dataset_unlabelled),
                                       class_dict=client['class_dict'])

        filtered_dataset[client_idx] = merged_dataset
    
    if args.dataset.name in ['cifar10', 'cifar100']:
        total_train_dataset_unlabelled = subsample_dataset(copy.deepcopy(trainset), np.array(total_unlabelled_indices))
    elif args.dataset.name in ['cub', 'cub2']:
        total_train_dataset_unlabelled = subsample_dataset_cub(copy.deepcopy(trainset), np.array(total_unlabelled_indices))
    elif args.dataset.name in ['imagenet']:
        total_train_dataset_unlabelled = subsample_dataset_imagenet(copy.deepcopy(trainset), np.array(total_unlabelled_indices))
    elif args.dataset.name in ['scars']:
        total_train_dataset_unlabelled = subsample_dataset_scars(copy.deepcopy(trainset), np.array(total_unlabelled_indices))
    elif args.dataset.name in ['pets']:
        total_train_dataset_unlabelled = subsample_dataset_pets(copy.deepcopy(trainset), np.array(total_unlabelled_indices))
    else:
        assert False
    filtered_dataset['total_train_dataset_unlabelled'] = total_train_dataset_unlabelled
    
    end_time = time.time()
    print(f" Data Generation Time: {end_time - start_time:.5f} seconds")

    return filtered_dataset



def subsample_dataset(dataset, idxs):

    # Allow for setting in which all empty set of indices is passed
    if len(idxs) > 0:
        start_time = time.time()
        dataset.data = dataset.data[idxs]
        #dataset.targets = np.array(dataset.targets)[idxs].tolist()
        dataset.targets = np.array(dataset.targets)[idxs]
        dataset.uq_idxs = dataset.uq_idxs[idxs]

        return dataset

    else:

        return None


def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

def subsample_instances(dataset, prop_indices_to_subsample=0.5):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices

def get_cifar_test_semisup_dataset(args, test_dataset):
    test_dataset_labelled = subsample_classes(copy.deepcopy(test_dataset),
                                                include_classes=args.dataset.seen_classes)
    subsample_indices = subsample_instances(test_dataset_labelled, prop_indices_to_subsample=0.5)
    test_dataset_labelled = subsample_dataset(test_dataset_labelled, subsample_indices)

    unlabelled_indices = set(test_dataset.uq_idxs) - set(test_dataset_labelled.uq_idxs)
    test_dataset_unlabelled = subsample_dataset(copy.deepcopy(test_dataset),
                                                    np.array(list(unlabelled_indices)))
    return test_dataset_labelled, test_dataset_unlabelled


class MultiViewDataInjector(object):
    def __init__(self, *args):
        self.transforms = args[0]
        self.random_flip = transforms.RandomHorizontalFlip()

    def __call__(self, sample, *with_consistent_flipping):
        if with_consistent_flipping:
            sample = self.random_flip(sample)
        output = [transform(sample) for transform in self.transforms]
        return output

class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]
    
    