import torch
from torchvision import datasets, transforms
from torch.utils.data import  Dataset
import os
from datasets.cifar import cifar_noniid, cifar_dirichlet_balanced,cifar_dirichlet_unbalanced, cifar_iid, cifar_overlap, cifar_toyset
import torch.nn as nn
import csv
from typing import List,Dict
import copy
import json
from collections import OrderedDict
from torch.utils.data import  Dataset
import numpy as np

__all__ = ['DatasetSplit', 'DatasetSplitMultiView', 'DatasetSplitMultiViews', 'get_dataset', 'MultiViewDataInjector', 'GaussianBlur', 'TransformTwice'
                                                                                                            ]

create_dataset_log = False

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


def get_dataset(args, trainset, mode='iid'):
    set = args.dataset.name
    if 'leaf' not in set and 'Shakespeare' not in set:
        if set not in ["iNaturalist"]:
            directory = args.dataset.client_path + '/' + set + '/' + ('un' if args.split.unbalanced==True else '') + 'balanced'
            filepath = directory+'/' + mode + (str(args.split.class_per_client) if mode == 'skew' else '') + (str(args.split.alpha) if mode == 'dirichlet' else '') + (str(args.split.overlap_ratio) if mode == 'overlap' else '') + '_clients' +str(args.trainer.num_clients) +  (("_toyinform_" + str(args.split.toy_noniid_rate) + "_" + str(args.split.limit_total_classes)+ "_" +  str(args.split.limit_number_per_class)) if 'toy' in mode else "")   + '.txt'
        elif set in ["iNaturalist"]:
            directory = args.dataset.client_path + '/' + set
            filepath = directory+'/' + set + '_client_idx.txt'

        check_already_exist = os.path.isfile(filepath) and (os.stat(filepath).st_size != 0)
        create_new_client_data = not check_already_exist or args.split.create_client_dataset
        # if not create_dataset_log:
        #     print("create new client data: " + str(create_new_client_data))
        #     create_dataset_log = True

        if create_new_client_data == False:
            try:
                dataset = {}
                with open(filepath) as f:
                    for idx, line in enumerate(f):
                        dataset = eval(line)
            except:
                print("Have problem to read client data")

        if create_new_client_data == True:
            if set not in ["iNaturalist"]:
                if mode == 'iid':
                    dataset = cifar_iid(trainset, args.trainer.num_clients)
                elif mode == 'overlap':
                    dataset = cifar_overlap(trainset, args.trainer.num_clients, args.split.overlap_ratio)
                # elif mode[:4] == 'skew' and mode[-5:] == 'class':
                elif mode == 'skew':
                    class_per_client = args.split.class_per_client
                    # assert class_per_client * args.trainer.num_clients == trainset.dataset.num_classes
                    # class_per_user = int(mode[4:-5])
                    dataset = cifar_noniid(trainset, args.trainer.num_clients, class_per_client)
                elif mode == 'dirichlet':
                    if args.split.unbalanced==True:
                        dataset = cifar_dirichlet_unbalanced(trainset, args.trainer.num_clients, alpha=args.split.alpha)
                    else:
                        dataset = cifar_dirichlet_balanced(trainset, args.trainer.num_clients, alpha=args.split.alpha)
                elif mode == 'toy_noniid':
                    dataset = cifar_toyset(trainset, args.trainer.num_clients, num_valid_classes=args.split.limit_total_classes, limit_number_per_class = args.split.limit_number_per_class, toy_noniid_rate = args.split.toy_noniid_rate, non_iid = True)
                elif mode == 'toy_iid':
                    dataset = cifar_toyset(trainset, args.trainer.num_clients, num_valid_classes=args.split.limit_total_classes, limit_number_per_class = args.split.limit_number_per_class, toy_noniid_rate = args.split.toy_noniid_rate, non_iid = False)                

                
                else:
                    print("Invalid mode ==> please select in iid, skewNclass, dirichlet")
                    return
            elif set in ["iNaturalist"]:
                mapping_table = trainset.split_csv # '/inaturalist-user-120k/federated_train_user_120k.csv'
                user_id_set = set()
                for row in mapping_table:
                    user_id_set.add(row['user_id'])
                print('total num of iNaturalist clients',len(user_id_set))

                client_data = OrderedDict()
                for idx,id in enumerate(user_id_set):
                    client_data[id] = OrderedDict()
                    client_data[id]['client_idx'] = idx
                    client_data[id]['client_data'] = []

                this_mapping_table = copy.deepcopy(mapping_table)

                client_idx_data = OrderedDict()
                for i in range(len(user_id_set)):
                    client_idx_data[i] = []
                count = 0
                for row in this_mapping_table:

                    this_user_id = row.pop('user_id')
                    #print(this_user_id)
                    row['data_idx'] = count

                    client_data[this_user_id]['client_data'].append(row)
                    this_client_idx = client_data[this_user_id]['client_idx']
                    client_idx_data[this_client_idx].append(count)
                    count+=1
                dataset = dict(client_idx_data)

            try:
                os.makedirs(directory, exist_ok=True)
                with open(filepath, 'w') as f:
                    print(dataset, file=f)

            except:
                print("Fail to write client data at " + directory)

        return dataset
    elif 'leaf' in set:
        return trainset.get_train_idxs()
    elif set == 'Shakespeare':
        return trainset.get_client_dic()
    else:
        assert False



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
