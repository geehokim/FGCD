from utils.registry import Registry
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Normalize, CenterCrop
import yaml
import torch
from utils.data import ContrastiveLearningViewGenerator
from omegaconf import OmegaConf
from datasets.utils_cub import get_cub_test_semisup_dataset
from datasets.data_utils import get_cifar_test_semisup_dataset
from datasets.base import MergedDataset
from datasets.imagenet import get_imagenet_100_datasets_whole
from datasets.stanford_cars import get_scars_dataset_whole
from datasets.pets import get_pets_datasets_whole
import copy
import random
from PIL import ImageFilter
import numpy as np

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
"""
DATASET_REGISTRY.register(CIFAR10)
DATASET_REGISTRY.register(CIFAR100)

import contextlib

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)




__all__ = ['build_dataset', 'build_datasets']

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x



def get_transform(args, train, config=None):


    if 'leaf_femnist' in args.dataset.name:
        transform = transforms.Compose([ToTensor()])
    elif 'leaf_celeba' in args.dataset.name:
        transform = transforms.Compose([
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    elif 'Shakespeare' == args.dataset.name:
        transform = None
    # elif args.dataset.name in ['cifar10', 'cifar100']:
    #     mean = (0.485, 0.456, 0.406)
    #     std = (0.229, 0.224, 0.225)
    #     # mean = (0.4914, 0.4822, 0.4465)
    #     # std = (0.2023, 0.1994, 0.2010)
    #     interpolation = args.dataset.interpolation
    #     crop_pct = args.dataset.crop_pct
    #     image_size = args.dataset.image_size
    #     if train:
    #         if args.dataset.aug == 'normal':
    #             transform = transforms.Compose([
    #                 transforms.Resize(int(image_size / crop_pct), interpolation),
    #                 transforms.RandomCrop(image_size),
    #                 transforms.RandomHorizontalFlip(p=0.5),
    #                 transforms.ColorJitter(),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(
    #                     mean=torch.tensor(mean),
    #                     std=torch.tensor(std))
    #             ])
    #         elif args.dataset.aug == 'strong':
    #             transform = transforms.Compose([
                    
    #                 # transforms.Resize(int(image_size / crop_pct), interpolation),
    #                 # transforms.RandomCrop(image_size),
    #                 transforms.RandomResizedCrop(image_size, scale=(0.3, 1.0), interpolation=interpolation),
    #                 transforms.RandomHorizontalFlip(p=0.5),
    #                 transforms.ColorJitter(),
    #                 transforms.RandomSolarize(threshold=128, p=0.2),
    #                 transforms.GaussianBlur(kernel_size=int(image_size * 0.1) // 2 * 2 + 1, p=0.5),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(
    #                     mean=torch.tensor(mean),
    #                     std=torch.tensor(std))
                    
    #             ])
    #         elif args.dataset.aug == 'weak':
    #             transform = transforms.Compose([
    #                 # transforms.Resize(int(image_size / crop_pct), interpolation),
    #                 # transforms.RandomCrop(image_size),
    #                 transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=interpolation),
    #                 transforms.RandomHorizontalFlip(p=0.5),
    #                 transforms.ColorJitter(),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(
    #                     mean=torch.tensor(mean),
    #                     std=torch.tensor(std))
    #             ])
    #         else:
    #             raise ValueError(f"Invalid augmentation type: {args.dataset.aug}")
    #     else:
    #         transform = transforms.Compose([
    #             transforms.Resize(int(image_size / crop_pct), interpolation),
    #             transforms.CenterCrop(image_size),
    #             transforms.ToTensor(),
    #             transforms.Normalize(
    #                 mean=torch.tensor(mean),
    #                 std=torch.tensor(std))
    #         ])
    elif args.dataset.name in ['cifar10', 'cifar100', 'cub2', 'cub', 'imagenet', 'scars', 'pets']:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        # mean = (0.4914, 0.4822, 0.4465)
        # std = (0.2023, 0.1994, 0.2010)
        interpolation = args.dataset.interpolation
        crop_pct = args.dataset.crop_pct
        image_size = args.dataset.image_size
        if train:
            augs = []
            for aug in args.dataset.aug:
                if aug == 'normal':
                    transform = transforms.Compose([
                        transforms.Resize(int(image_size / crop_pct), interpolation),
                        transforms.RandomCrop(image_size),
                        # transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=interpolation),
                        transforms.RandomHorizontalFlip(p=0.5),
                        # transforms.ColorJitter(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=torch.tensor(mean),
                            std=torch.tensor(std))
                    ])
            #         transform = transforms.Compose([
            #     transforms.Resize(int(image_size / crop_pct), interpolation),
            #     transforms.CenterCrop(image_size),
            #     transforms.ToTensor(),
            #     transforms.Normalize(
            #         mean=torch.tensor(mean),
            #         std=torch.tensor(std))
            # ])
                elif aug == 'strong':
                    transform = transforms.Compose([
                        
                        # transforms.Resize(int(image_size / crop_pct), interpolation),
                        # transforms.RandomCrop(image_size),
                        transforms.RandomResizedCrop(image_size, scale=(0.3, 1.0), interpolation=interpolation),
                        transforms.RandomHorizontalFlip(p=0.5),
                        # transforms.ColorJitter(),
                        transforms.RandomSolarize(threshold=128, p=0.3),
                        # transforms.RandomApply([
                        # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                        #  ], p=0.8),
                        transforms.RandomGrayscale(p=0.3),
                        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                        # transforms.GaussianBlur(kernel_size=5),
                        # transforms.ColorJitter(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=torch.tensor(mean),
                            std=torch.tensor(std))                  
                    ])

                elif aug == 'weak':
                    transform = transforms.Compose([
                        # transforms.Resize(int(image_size / crop_pct), interpolation),
                        # transforms.RandomCrop(image_size),
                        transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=interpolation),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ColorJitter(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=torch.tensor(mean),
                            std=torch.tensor(std))
                    ])
                elif aug == 'imagenet':

                    transform = transforms.Compose([
                        transforms.Resize(int(image_size / crop_pct), interpolation),
                        transforms.RandomCrop(image_size),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ColorJitter(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=torch.tensor(mean),
                            std=torch.tensor(std))
                    ])
                    
                else:
                    raise ValueError(f"Invalid augmentation type: {args.dataset.aug}")
                
                augs.append(transform)
        else:
            transform = transforms.Compose([
                transforms.Resize(int(image_size / crop_pct), interpolation),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std))
            ])
    else:
        color_jitter = transforms.ColorJitter(0.4 * 1, 0.4 * 1, 0.4 * 1, 0.1 * 1)
        normalize = transforms.Normalize(config['mean'],
                                         config['std'])
        imsize = config['imsize']
        if train:
            transform = transforms.Compose(
                [transforms.RandomRotation(10),
                 transforms.RandomCrop(imsize, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 normalize
                 ])
        else:
            transform = transforms.Compose(
                [transforms.CenterCrop(imsize),
                 transforms.ToTensor(),
                 normalize])
    if train:
        if args.client.get('n_views'):
            transform = ContrastiveLearningViewGenerator(base_transforms=augs, n_views=args.client.n_views)
        else:
            pass

    return transform


def build_dataset(args, train=True):
    if args.verbose and train == True:
        print(DATASET_REGISTRY)

    download = args.dataset.download if args.dataset.get('download') else False

    # with open('datasets/configs.yaml', 'r') as f:
    #     dataset_config = yaml.safe_load(f)[args.dataset.name]
    transform = get_transform(args, train)
    dataset = DATASET_REGISTRY.get(args.dataset.name)(root=args.dataset.path, download=download, train=train, transform=transform) if len(args.dataset.path) > 0 else None

    return dataset

def build_datasets(args):
    if args.dataset.name == 'imagenet':
        train_transform = get_transform(args, train=True)
        test_transform = get_transform(args, train=False)
        all_datasets = get_imagenet_100_datasets_whole(args.dataset.path, train_transform, test_transform)
        train_dataset = all_datasets['train']
        test_dataset = all_datasets['test']
    elif args.dataset.name == 'scars':
        train_transform = get_transform(args, train=True)
        test_transform = get_transform(args, train=False)
        all_datasets = get_scars_dataset_whole(args.dataset.path, train_transform, test_transform)
        train_dataset = all_datasets['train']
        test_dataset = all_datasets['test']
    elif args.dataset.name == 'pets':
        train_transform = get_transform(args, train=True)
        test_transform = get_transform(args, train=False)
        all_datasets = get_pets_datasets_whole(args.dataset.path, train_transform, test_transform)
        train_dataset = all_datasets['train']
        test_dataset = all_datasets['test']
    else:
        train_dataset = build_dataset(args, train=True)
        test_dataset = build_dataset(args, train=False)
    
    datasets = {
        "train": train_dataset,
        "test": test_dataset,
    }
    
    if args.split.target_transform:
        # Set target transforms:
        target_transform_dict = {}
        for i, cls in enumerate(list(args.dataset.seen_classes) + list(args.dataset.unseen_classes)):
            target_transform_dict[cls] = i
        target_transform = lambda x: target_transform_dict[x]

        # for dataset_name, dataset in datasets.items():
        #     if dataset is not None:
        #         dataset.target_transform = target_transform
        if args.dataset.name in ['cub', 'cub2', 'scars', 'pets']:
            datasets['test'].align_targets(target_transform)
            datasets['train'].align_targets(target_transform)
        # datasets['test'].target_transform = target_transform

    return datasets