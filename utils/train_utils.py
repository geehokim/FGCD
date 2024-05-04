import torch
import copy
from torch.utils.data import DataLoader
import numpy as np
import random


# def apply_label_noise(y, num_classes, noise_rate):
    
#     select_index = (torch.rand(y.size(0)) < noise_rate).nonzero()
#     if len(select_index) > 0:
#         random_label = torch.randint(0, num_classes, select_index.size()).long()
#         origin_label = y[select_index]
#         random_label += (origin_label == random_label).long()
#         random_label %= num_classes
#         y[select_index] = random_label
#     return y


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