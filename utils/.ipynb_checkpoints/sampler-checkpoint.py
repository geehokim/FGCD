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




class RandomClasswiseSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances

        self.class_dic = defaultdict(list)
        # for index, (_, target, _) in enumerate(data_source):
        for index, (_, target) in tqdm.tqdm(enumerate(data_source.imlist)):
            self.class_dic[target].append(index)

        self.class_ids = list(self.class_dic.keys())
        self.num_classes = len(self.class_ids)

        self.length = len(data_source.imlist)

        self.print_container = False


    def __iter__(self):
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

        random.shuffle(list_container)

        # if not self.print_container:
        #     print(list_container[:10])
        #     self.print_container = True
        

        ret = []
        for batch_indices in list_container:
            ret.extend(batch_indices)

        return iter(ret)

    def __len__(self):
        return self.length