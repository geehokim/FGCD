
from __future__ import print_function
from __future__ import division

import os
import torch
import torchvision
import numpy as np
import PIL.Image
from PIL import Image
from collections import defaultdict
from torch.utils.data import  Dataset

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import torchvision.datasets.accimage as accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, transform = None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        def img_load(index):
            im = PIL.Image.open(self.im_paths[index])
            # convert gray to rgb
            if len(list(im.split())) == 1 : im = im.convert('RGB') 
            if self.transform is not None:
                im = self.transform(im)
            return im

        im = img_load(index)
        target = self.ys[index]

        return im, target

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]


class BaseDataset2(torch.utils.data.Dataset):
    def __init__(self, root, mode, transform = None, loader=default_loader):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []
        self.loader = loader

    def nb_classes(self):
        #assert set(self.ys) == set(self.classes)
        return len(set(self.ys))

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = self.loader(self.im_paths[index])
        if self.transform is not None:
            im = self.transform(im)
        target = self.ys[index]

        return im, target

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]


class MergedDataset(Dataset):
    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """
    def __init__(self, labelled_dataset, unlabelled_dataset, class_dict=None):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None
        if class_dict is not None:
            self.class_dict = class_dict
        else:
            self.class_dict = defaultdict(int)
            for idx in range(len(self.labelled_dataset)):
                _, label, _ = self.labelled_dataset[idx]
                if torch.is_tensor(label):
                    label = str(label.item())
                else:
                    label = str(label)

                self.class_dict[str(label)] += 1
            for idx in range(len(self.unlabelled_dataset)):
                _, label, _ = self.unlabelled_dataset[idx]
                if torch.is_tensor(label):
                    label = str(label.item())
                else:
                    label = str(label)
                if label in self.class_dict:
                    self.class_dict[str(label)] += 1
                else:
                    self.class_dict[str(label)] = 1

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        else:
            #import pdb; pdb.set_trace()
            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0


        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)
    



class MergedDatasetCluster(Dataset):
    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """
    def __init__(self, labelled_dataset, unlabelled_dataset, class_dict=None, cluster_labels=None):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None
        self.class_dict=class_dict
        self.cluster_labels=cluster_labels

    def __getitem__(self, item):
        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        else:
            #import pdb; pdb.set_trace()
            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0

        if labeled_or_not == 0:
            label = self.cluster_labels[item - len(self.labelled_dataset)]
            
        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)
    

class MergedDatasetClusterSemi(Dataset):
    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """
    def __init__(self, labelled_dataset, unlabelled_dataset, class_dict=None, cluster_labels=None):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None
        self.class_dict=class_dict
        self.cluster_labels=cluster_labels

    def __getitem__(self, item):
        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        else:
            #import pdb; pdb.set_trace()
            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0


        label = self.cluster_labels[item]
            
        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)