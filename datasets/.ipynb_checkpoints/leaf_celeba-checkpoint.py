from __future__ import print_function
import numpy as np
import torch
import contextlib

import os
import sys
import errno
import numpy as np
from PIL import Image
import torch.utils.data as data
import contextlib
import pickle
from datasets.base import *
import copy
import imageio
import numpy as np
import os
from torchvision import datasets, transforms

from collections import defaultdict
from torch.utils.data import Dataset

from tqdm.autonotebook import tqdm
import json
import PIL.Image
from PIL import Image
from collections import defaultdict
from datasets.build import DATASET_REGISTRY

__all__ = ['leaf_celeba']




@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


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


def build_set(root, split, imgs, noise_type='pairflip', noise_rate=0.5):
    """
       Function to return the lists of paths with the corresponding labels for the images
    Args:
        root (string): Root directory of dataset
        split (str): ['train', 'gallery', 'query'] returns the list pertaining to training images and labels, else otherwise
    Returns:
        return_list: list of 236_comb_fromZeroNoise-tuples with 1st location specifying path and 2nd location specifying the class
    """

    tmp_imgs = imgs

    argidx = np.argsort(tmp_imgs)




def download_and_unzip(URL, root_dir):
    error_message = "Download is not yet implemented. Please, go to {URL} urself."
    raise NotImplementedError(error_message.format(URL))

def _add_channels(img, total_channels=3):
    while len(img.shape) < 3:  # third axis is the channels
        img = np.expand_dims(img, axis=-1)
    while(img.shape[-1]) < 3:
        img = np.concatenate([img, img[:, :, -1:]], axis=-1)
    return img




@DATASET_REGISTRY.register()
class leaf_celeba(data.Dataset):
    def __init__(self, root, train=True, load_transform=None,
                transform=None, download=False, max_samples=None, dataset = 'leaf_celeba'):
        if train:
            self.split = 'train'
        else:
            self.split = 'test'
        #self.preload = preload
        self.transform = transform
        self.transform_results = dict()
        self.dataset = dataset.replace('leaf_',"")
        self.loader = default_loader
            
        
        self.image_size = 84
        self.images_dir = os.path.join(root, 'data', 'celeba', 'data', 'raw', 'img_align_celeba')
        self.users,self.groups,all_data = self.setup_clients(root,self.dataset,split = self.split)
        all_data,self.train_idxs = self.merge_train_data_return_idxs(all_data,self.users)


        assert len(all_data.keys()) == 2 
            
        for i,key in enumerate(all_data.keys()):
            if i==0:
                self.data = all_data[key]
            else:
                self.targets = all_data[key]
        #print(self.targets)
        self.classes = set(tuple(self.targets))  
        print("data len: ",len(self.data))
        print("total client: ",len(self.train_idxs))
        print("self data")
        print(self.data[:5])
        print("self target")
        print(self.targets[:5])
        
        print(len(self.classes))
        
    def __len__(self):
        return len(self.data)
    
    '''
    def _load_image(self, img_name):
        print(os.path.join(self.images_dir, img_name))
        img = Image.open(os.path.join(self.images_dir, img_name))
        img = img.resize((self.image_size, self.image_size)).convert('RGB')
        return img    
    '''
    
    def make_path(self,img_name):
        return os.path.join(self.images_dir, img_name)
    
    def __getitem__(self, idx):
        
        img, target = self.data[idx], self.targets[idx]
        img = self.loader(self.make_path(img))
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(self.targets[idx])
        label = label.type(torch.LongTensor)
        return img, label
    
    def get_train_idxs(self):
        return self.train_idxs
    
    def get_classes(self):
        return self.classes
    
    
    def read_dir(self,data_dir):
        clients = []
        groups = []
        data = defaultdict(lambda : None)

        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith('.json')]
        for f in files:
            file_path = os.path.join(data_dir,f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            clients.extend(cdata['users'])
            if 'hierarchies' in cdata:
                groups.extend(cdata['hierarchies'])
            data.update(cdata['user_data'])

        clients = list(sorted(data.keys()))
        return clients, groups, data


    def merge_test_data(self,test_data,users):
        res  = None
        for u in users:
            if res ==None:
                res = test_data[u]
            else:
                for key in res.keys():
                    res[key].append(test_data[u][key])
        return res
    
    def merge_train_data_return_idxs(self,train_data,users):
        res  = None
        idxs = {}
        pos = 0
        for i,u in enumerate(users):
            if res ==None:
                res = train_data[u]
            else:
                for key in res.keys():
                    res[key].extend(train_data[u][key])
            first_key = list(res.keys())[0]
            partial_data_len = len(train_data[u][first_key])
            idxs[i] = (range(pos,pos + partial_data_len))
            pos = pos + partial_data_len
        return res,idxs       

    '''
    def create_clients(self,users,groups,train_data,test_data,model):
        if len(groups)==0:
            groups=[[] for _ in users]
        clients=[Client(u,g,train_data[u],test_data[u],model)
                 for u,g in zip(users,groups) ]
        return clients
    '''
    def setup_clients(self,root,dataset,split = 'train'):
        #split="test" if not use_val_set else "val"
        #data_dir=os.path.join(root,"data",dataset,"data","train")
        #print(train_data_dir)
        print('root',root)
        data_dir=os.path.join(root,"data",dataset,"data",split)
        print('data_dir',data_dir)
        #print(test_date_dir)
        
        data=self.read_dir(data_dir)
        users,groups,data=data;
        #clients=create_clients(users,groups,train_data,test_data,model)
        return users,groups,data#clients




if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        # resize 256_comb_coteach_OpenNN_CIFAR -> random_crop 224 ==> crop 32, padding 4
        transforms.ToTensor()
    ])

    exs = TinyImageNetDataset('../data/tiny_imagenet', split='train', transform=transform)


    mean = 0
    sq_mean = 0
    for ex in exs:
        mean += ex[0].sum(1).sum(1) / (64 * 64)
        sq_mean += ex[0].pow(2).sum(1).sum(1) / (64 * 64)

    mean /= len(exs)
    sq_mean /= len(exs)

    std = (sq_mean - mean.pow(2)).pow(0.5)

    print(mean)
    print(std)