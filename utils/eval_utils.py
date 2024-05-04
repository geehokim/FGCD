import torch
import copy
from torch.utils.data import DataLoader
import numpy as np
import random


def get_local_classes(local_dataset, majority_threshold=1.):
    local_classes = [int(i) for i in local_dataset.class_dict.keys()]

    num_classes = len(local_dataset.dataset.classes)
    num_local_classes = len(local_dataset.class_dict.keys())
    major_classes = [int(key) for key in local_dataset.class_dict if local_dataset.class_dict[key] >= majority_threshold * len(local_dataset)/num_local_classes]
    minor_seen_classes = [int(key) for key in local_dataset.class_dict if local_dataset.class_dict[key] < majority_threshold * len(local_dataset)/num_local_classes]
    missing_classes = [i for i in range(num_classes) if str(i) not in local_dataset.class_dict]
    minor_classes = minor_seen_classes + missing_classes

    return {
        "seen": local_classes,
        "major": major_classes,
        "minor": minor_classes,
        "missing": missing_classes,
        "minor_seen": minor_seen_classes,
    }