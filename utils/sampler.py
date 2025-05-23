import torch
from torchvision import datasets
from typing import Union, Any, Dict, Tuple, List
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
import pathlib
#from .data_transforms import get_data_transforms

import torch.distributed as dist
import torch.utils.data as data
import torch.utils.data.distributed
from PIL import Image

#from utils.train_utils import get_rank
import tqdm, random, copy
from collections import defaultdict
from operator import itemgetter

#import lmdb, six
import math
from typing import TypeVar, Optional, Iterator
import numpy as np
from torch.utils.data.sampler import Sampler


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]

class RandomMultipleGallerySamplerNoCam(Sampler):
    def __init__(self, data_source, num_instances=4):
        super().__init__(data_source)

        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, _item in enumerate(data_source):
            pid = _item[1]
            # (_, pid, cam)
            if pid < 0:
                continue
            self.index_pid[index] = pid
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)
        self.max_length =  max([len(indices) for indices in self.pid_index.values()])

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])
            # _, i_pid, i_cam = self.data_source[i]
            # _, i_pid, i_cam
            ret.append(i)

            pid_i = self.index_pid[i]
            index = self.pid_index[pid_i]

            select_indexes = No_index(index, i)
            if not select_indexes:
                continue
            if len(select_indexes) >= self.num_instances:
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
            else:
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

            for kk in ind_indexes:
                ret.append(index[kk])

        return iter(ret)

class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if self.length is not None:
            return self.length

        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)


class RandomClasswiseSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances

        self.class_dic = defaultdict(list)
        #print(dir(data_source))
        #print(data_source.dataset)
        #print(len(data_source))
        # for index, (_, target, _) in enumerate(data_source):
        #print(data_source.train_list)
        #for index, (_, target) in tqdm.tqdm(enumerate(data_source.imlist)):
        print("len(data_source) :", len(data_source))
        for index, (_, _, _, _, target) in tqdm.tqdm(enumerate(data_source)):
            #print(index)
            #print("len(target):",len(target))
            #print("sampler target, index :",target, index)
            # self.class_dic[target.item()].append(index)
            self.class_dic[target].append(index)

        # raise ValueError
        
        self.class_ids = list(self.class_dic.keys())
        self.class_ids.sort()
        #self.sorted_ids =(self.class_ids)
        #print("self.class_ids",self.class_ids)
        # for cl in self.class_ids:
        #     print(cl,len(self.class_dic[cl])) 
        self.num_classes = len(self.class_ids)
        self.length = len(data_source)
        #print("length ::", self.length)
        #self.length = len(data_source.imlist)

        self.print_container = False
        self.counter = 0
        self.ret = None
        # self.list_container = None

    def get_counter(self):
        return self.counter

    def reset_counter(self):
        self.counter = 0

    def __iter__(self):
        # print("do_iter")
        self.counter += 1
        list_container = []

        for class_id in self.class_ids:
            indices = copy.deepcopy(self.class_dic[class_id])
            if len(indices) < self.num_instances:
                indices = np.random.choice(indices, size=self.num_instances, replace=True)
            random.shuffle(indices)

            batch_indices = []
            for idx in indices:
                batch_indices.append(idx)
                if len(batch_indices) == self.num_instances:
                    list_container.append(batch_indices)
                    batch_indices = []
                    continue
            # print("batch : ", batch_indices)
            if len(batch_indices) > 0:
                list_container.append(batch_indices)
        # print(list_container)

        random.shuffle(list_container)

        # if not self.print_container:
        #     print(self.class_dic)
        #     print(list_container)
        #     self.print_container = True

        #     raise ValueError

        ret = []
        for batch_indices in list_container:
            ret.extend(batch_indices)

        self.ret = ret
        return iter(ret)


    def __len__(self):
        if self.ret is not None:
            return len(self.ret)
        else:
            return self.length
    

class RandomClasswiseSampler2(RandomClasswiseSampler):

    def __iter__(self):
        # print("do_iter")
        self.counter += 1
        list_container = []

        for class_id in self.class_ids:
            indices = copy.deepcopy(self.class_dic[class_id])
            # if len(indices) < self.num_instances:
                # indices = np.random.choice(indices, size=self.num_instances, replace=True)
                # indices = np.random.choice(indices, size=len(indices), replace=False)
                # print("debug : indices : ", indices)
            random.shuffle(indices)

            # print(indices)
            # breakpoint()

            if len(indices) < self.num_instances:
                list_container.append(indices)
            else:
                batch_indices = []
                for idx in indices:
                    batch_indices.append(idx)
                    if len(batch_indices) == self.num_instances:
                        list_container.append(batch_indices)
                        batch_indices = []
                        continue
                if len(batch_indices) > 0:
                    list_container.append(batch_indices)
            

        random.shuffle(list_container)


        ret = []
        for batch_indices in list_container:
            ret.extend(batch_indices)

        return iter(ret)
