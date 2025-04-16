import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
from collections import defaultdict

__all__ = ['KLD', 'KLDiv', 'l2norm','get_numclasses','count_label_distribution','check_data_distribution','check_data_distribution_aug','feature_extractor','classifier', 'get_optimizer', 'get_scheduler','freeze_except_fc','unfreeze', 'freeze', 'get_momentum','modeleval','get_l2norm','get_vertical','cal_cos','create_pth_dict','cal_distances_between_models','cal_distance_between_two_models', 'get_major_minor_stat', 'get_avg_data_per_class', 'append_or_not','mean_dict', 'min_dict', 'cal_att_j_div_r']



'''
pytorch example
>>> kl_loss = nn.KLDivLoss(reduction="batchmean")
>>> # input should be a distribution in the log space
>>> input = F.log_softmax(torch.randn(3, 5, requires_grad=True))
>>> # Sample a batch of distributions. Usually this would come from the dataset
>>> target = F.softmax(torch.rand(3, 5))
>>> output = kl_loss(input, target)
'''
def KLD(input_p,input_q,T=1):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    p = F.softmax(input_p/T, dim = 1)
    q = F.log_softmax(input_q/T, dim = 1)
    result = kl_loss(q,p)
    return result

def KLDiv(input_p,input_q,T=1, reduction='none'):
    kl_loss = nn.KLDivLoss(reduction=reduction)
    p = F.softmax(input_p/T, dim = 1)
    q = F.log_softmax(input_q/T, dim = 1)
    result = kl_loss(q,p)
    return result
    
def freeze_except_fc(net):
    for name,par in net.named_parameters():
        if 'fc' not in name:
            par.requires_grad = False


# In[21]:


# def unfreeze(net):
#     for name,par in net.named_parameters():
#         par.requires_grad = True
#
# '''
# def KD(input_p,input_q,T=1):
#     p=F.softmax((input_p/T),dim=1)
#     q=F.softmax((input_q/T),dim=1)
#     result=((p*((p/q).log())).sum())/len(input_p)
#
#     if not torch.isfinite(result):
#         print('==================================================================')
#         print('input_p')
#         print(input_p)
#
#         print('==================================================================')
#         print('input_q')
#         print(input_q)
#         print('==================================================================')
#         print('p')
#         print(p)
#
#         print('==================================================================')
#         print('q')
#         print(q)
#
#
#         print('******************************************************************')
#         print('p/q')
#         print(p/q)
#
#         print('******************************************************************')
#         print('(p/q).log()')
#         print((p/q).log())
#
#         print('******************************************************************')
#         print('(p*((p/q).log())).sum()')
#         print((p*((p/q).log())).sum())
#
#     return result
# '''



def l2norm(x,y):
    z= (((x-y)**2).sum())
    return z/(1+len(x))
class feature_extractor(nn.Module):
            def __init__(self,model,classifier_index=-1):
                super(feature_extractor, self).__init__()
                self.features = nn.Sequential(
                    # stop at conv4
                    *list(model.children())[:classifier_index]
                )
            def forward(self, x):
                x = self.features(x)
                return x


class classifier(nn.Module):
            def __init__(self,model,classifier_index=-1):
                super(classifier, self).__init__()
                self.layers = nn.Sequential(
                    # stop at conv4
                    *list(model.children())[classifier_index:]
                )
            def forward(self, x):
                x = self.layers(x)
                return x

def count_label_distribution(labels,class_num:int=10,default_dist:torch.tensor=None):
    if default_dist!=None:
        default=default_dist
    else:
        default=torch.zeros(class_num)
    data_distribution=default
    for idx,label in enumerate(labels):
        data_distribution[label]+=1 
    data_distribution=data_distribution/data_distribution.sum()
    return data_distribution

def check_data_distribution(dataloader,class_num:int=10,default_dist:torch.tensor=None):
    if default_dist!=None:
        default=default_dist
    else:
        default=torch.zeros(class_num)
    data_distribution=default
    for idx,(images,target) in enumerate(dataloader):
        for i in target:
            data_distribution[i]+=1 
    data_distribution=data_distribution/data_distribution.sum()
    return data_distribution

def check_data_distribution_aug(dataloader,class_num:int=10,default_dist:torch.tensor=None):
    if default_dist!=None:
        default=default_dist
    else:
        default=torch.zeros(class_num)
    data_distribution=default
    for idx,(images, _, target) in enumerate(dataloader):
        for i in target:
            data_distribution[i]+=1
    data_distribution=data_distribution/data_distribution.sum()
    return data_distribution

def get_numclasses(args,trainset = None):
    if args.dataset.name in ['cifar10', "MNIST"]:
        num_classes=10
    elif args.dataset.name in ["cifar100"]:
        num_classes=100
    elif args.dataset.name in ["TinyImageNet"]:
        num_classes=200
    elif args.dataset.name in ["cub", 'cub2']:
        num_classes=200
    elif args.dataset.name in ["iNaturalist"]:
        num_classes=1203
    elif args.dataset.name in ["ImageNet"]:
        num_classes=1000
    elif args.dataset.name in ["leaf_celeba"]:
        num_classes = 2
    elif args.dataset.name in ["leaf_femnist"]:
        num_classes = 62
    elif args.dataset.name in ["Shakespeare"]:
        num_classes=80
    elif args.dataset.name in ["imagenet"]:
        num_classes=100
    elif args.dataset.name in ["scars"]:
        num_classes=196
    elif args.dataset.name in ["pets"]:
        num_classes=37
        
    print("num of classes of ", args.dataset.name," is : ", num_classes)
    return num_classes



def get_optimizer(args, parameters):
    if args.set=='CIFAR10':
        optimizer = optim.SGD(parameters, lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.set=="MNIST":
        optimizer = optim.SGD(parameters, lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.set=="CIFAR100":
        optimizer = optim.SGD(parameters, lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.set in ["iNaturalist"]:
        optimizer = optim.SGD(parameters, lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.set in ["ImageNet"]:
        optimizer = optim.Adam(parameters, lr=0.00001)
    else:
        print("Invalid mode")
        return
    return optimizer



def get_scheduler(optimizer, args):
    if args.set=='CIFAR10':

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                         lr_lambda=lambda epoch: args.learning_rate_decay ** epoch,
        #                         )
    elif args.set=="MNIST":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                         lr_lambda=lambda epoch: args.learning_rate_decay ** (int(epoch/50)),
        #                         )
    elif args.set=="CIFAR100":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                         lr_lambda=lambda epoch: args.learning_rate_decay ** (int(epoch/50)),
        #                         )
    elif args.set in ["iNaturalist","ImageNet"]:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
        
    else:
        print("Invalid mode")
        return
    return scheduler



def get_momentum(args, num_of_data_clients, local_delta, input_global_momentum):
    total_num_of_data_clients=sum(num_of_data_clients)
    global_delta = copy.deepcopy(local_delta[0])
    global_momentum = copy.deepcopy(input_global_momentum)
    for key in global_delta.keys():
        for i in range(len(local_delta)):
            if i==0:
                #global_delta[key] *= num_of_data_clients[i]/local_K[i]
                global_delta[key] *= num_of_data_clients[i]
            else:
                #global_delta[key] += local_delta[i][key]*num_of_data_clients[i]/local_K[i]
                global_delta[key] += local_delta[i][key] * num_of_data_clients[i]
        #global_delta[key] = global_delta[key] / (-1 * total_num_of_data_clients * args.local_epochs * this_lr)
        global_delta[key] = global_delta[key] / (1 * total_num_of_data_clients)
        #global_delta[key] = global_delta[key] / float((-1 * len(local_delta)))
        global_momentum[key] = args.gamma * global_momentum[key] + global_delta[key]
    return global_delta, global_momentum



def modeleval(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:

            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %f %%' % (
            100 * correct / float(total)))
    acc = (100 * correct / float(total))
    model.train()
    return acc



#Code structure are referenced from GSAM

def get_l2norm(statedict):
    result = torch.norm(torch.stack([(statedict[key].clone().detach()).norm(p=2) for key in statedict]) ,p=2)
    return result


def get_vertical(statedict1, statedict2, eps = 1e-12, use_same_direct = False, use_opposite_direct = False):
    inner_prod = 0.0
    for key in statedict1.keys():
        inner_prod += torch.sum(
                    statedict1[key].clone().detach() * statedict2[key].clone().detach()
                )
        
    grad_norm1 = get_l2norm(statedict1)
    grad_norm2 = get_l2norm(statedict2)

    cosine = inner_prod / (grad_norm1 * grad_norm2 + eps)

    grad1_plus_verticalgrad2 = copy.deepcopy(statedict1)
    for key in grad1_plus_verticalgrad2.keys():
        vertical_to_grad1_component = statedict2[key].clone().detach() - cosine * grad_norm2 * statedict1[key].clone().detach() / (grad_norm1 + eps)
        if use_same_direct:
            if cosine > 0:
                vertical_to_grad1_component = statedict2[key].clone().detach() #- cosine * grad_norm2 * statedict1[key].clone().detach() / (grad_norm1 + eps)    
        if use_opposite_direct:
            if cosine < 0:
                vertical_to_grad1_component = statedict2[key].clone().detach()
        grad1_plus_verticalgrad2[key] += vertical_to_grad1_component


    return grad1_plus_verticalgrad2


def cal_cos(statedict1, statedict2, eps = 1e-12): 
    wandb_dict = {}   
    inner_prod = 0.0
    for key in statedict1.keys():
        inner_prod += torch.sum(
                    statedict1[key].clone().detach() * statedict2[key].clone().detach()
                )
        
    grad_norm1 = get_l2norm(statedict1)
    grad_norm2 = get_l2norm(statedict2)

    cosine = inner_prod / (grad_norm1 * grad_norm2 + eps)
    return cosine



def get_prefix_idx(x):
    idx = -4
    while True:
        try:
            int(x[idx-1])
            idx-=1
        except:
            break
    return idx

def get_prefix_num(x):
    idx = get_prefix_idx(x)
    return x[:idx], int(x[idx:-4])

def create_pth_dict(pth_path):
    pth_dir = os.path.dirname(pth_path)
    pth_base = os.path.basename(pth_path)
    pth_prefix,_ = get_prefix_num(pth_base)

    pth_dict = {}

    for filename in os.listdir(pth_dir):
        
        if filename.startswith(pth_prefix):
            _,number = get_prefix_num(filename)
            filepath = os.path.join(pth_dir, filename)
            pth_dict[number] = filepath

    return dict(sorted(pth_dict.items()))



@torch.no_grad()
def cal_distances_between_models(models_dict):
    # models_dict should have the following
    # model_now : nn.Module => current model
    # saved_initial_model: nn.Module => initial model parameters
    # saved_last_model: nn.Module => final model parameters
    # now : int => current iteration or epoch

    if 'distance_initial_last' not in models_dict:
        models_dict['distance_initial_last'] = cal_distance_between_two_models(models_dict['saved_initial_model'], models_dict['saved_last_model'])
    models_dict['distance_current_last'] = cal_distance_between_two_models(models_dict['model_now'], models_dict['saved_last_model'])
    models_dict['ratio_distance_current_last'] = {}
    for key in  models_dict['distance_current_last'].keys():
         models_dict['ratio_distance_current_last'][key] = models_dict['distance_current_last'][key] / models_dict['distance_initial_last'][key]

    return models_dict


@torch.no_grad()
def cal_distance_between_two_models(model1, model2):
    results = {}
    for (name1, child1), (name2, child2) in zip(model1.named_children(),model2.named_children()):
        results[name1] = 0
        for c1, c2 in zip(child1.parameters(), child2.parameters()):
            results[name1] += (((c1-c2)**2).sum()).item()

        results[name1] = (results[name1]**0.5)

    return results


def get_avg_data_per_class(raw_data, labels, num_classes, unique_labels = None):
    if unique_labels == None:
        unique_labels = labels.unique()
    device = labels.device
    avg_result = torch.zeros((num_classes, raw_data.shape[1]), device = device)
    for i, label in enumerate(unique_labels):
        class_mask = (labels == label)
        avg_class = raw_data[class_mask]
        avg_result[label] = avg_class.mean(dim=0)
    return avg_result

def mean_dict(in_x):
    if type(in_x) ==type({}):
        x = list(in_x.values()) 
    else:
        x = in_x
    effective_len = len(x)
    effective_sum = 0
    for el in x:
        if el=={} or el.isnan() or el.isinf() or el==None:
            effective_len -= 1
        else:
            effective_sum += el
    if effective_len==0:
        return torch.tensor(float('NaN'))#{}
    return effective_sum/effective_len




def min_dict(in_x):
    if type(in_x) ==type({}):
        x = list(in_x.values()) 
    else:
        x = in_x
    effective_len = len(x)
    effective_min = []
    for el in x:
        if el=={} or el.isnan() or el.isinf() or el==None:
            effective_len -= 1
        else:
            effective_min.append(el)
    if effective_len==0:
        return torch.tensor(float('NaN'))#{}
    return min(effective_min)

def get_major_minor_stat(class_major, class_minor, data):
    #data should have shape of (classes, classes).
    results = defaultdict(type({}))
    # results['major_self'] = {}
    # results['major_self_minall'] = {}

    # results['major_major'] = {}
    # results['major_major_minall'] = {}
    # results['major_major_minclass'] = {}
    
    # results['major_minor'] = {}
    # results['major_minor_minclass'] = {}
    # results['major_minor'] = {}

    # results['minor_self'] = {}

    # results['minor_major'] = {}
    # results['minor_major'] = {}
    # results['minor_major_minclass'] = {}

    # results['minor_minor'] = {}
    # results['minor_minor'] = {}
    # results['minor_minor_minclass[]'] = {}
    for jc in class_major:
        results['major_self'][jc] = data[jc][jc]
        results['major_major'][jc] = {}
        results['major_minor'][jc] = {}
        for jc2 in class_major:
            if jc2 != jc:
                results['major_major'][jc][jc2] = data[jc][jc2]
        for rc in class_minor:
            results['major_minor'][jc][rc] = data[jc][rc]
        results['major_major_minclass'][jc] = min_dict(results['major_major'][jc])
        results['major_minor_minclass'][jc] = min_dict(results['major_minor'][jc])

    results['major_self_minall'] = min_dict(results['major_self'])
    results['major_major_minall'] = min_dict(results['major_major_minclass'])
    results['major_minor_minall'] = min_dict(results['major_minor_minclass'])

    for rc in class_minor:
        results['minor_self'][rc] = data[rc][rc]
        results['minor_major'][rc] = {}
        results['minor_minor'][rc] = {}
        for jc in class_major:
            results['minor_major'][rc][jc] = data[rc][jc]

        for rc2 in class_minor:
            if rc2 != rc:
                results['minor_minor'][rc][rc2] = data[rc][rc2]
        results['minor_major_minclass'][rc] = min_dict(results['minor_major'][rc])
        results['minor_minor_minclass'][rc] = min_dict(results['minor_minor'][rc])
    results['minor_self_minall'] = min_dict(results['minor_self'])
    results['minor_major_minall'] = min_dict(results['minor_major_minclass'])
    results['minor_minor_minall'] = min_dict(results['minor_minor_minclass'])

    return results



def append_or_not(x, el, one_minus = False):
    if el=={} or el.isnan() or el.isinf() or el==None:
        return
    else:
        if one_minus:
            x.append(1-el)
        else:
            x.append(el)

@torch.no_grad()
def cal_att_j_div_r(input_x):
    input_vector = input_x.detach().clone()
    #input_vector has shape (# Class, 1), and each element accordint to class property of that index.
    # output[r][c] = input_vector[c] / input_vector[r]
    return input_vector.t() / input_vector


def freeze(backbone):
    backbone.eval()
    for m in backbone.parameters():
        m.requires_grad = False
    return backbone

def unfreeze(backbone):
    backbone.train()
    for m in backbone.parameters():
        m.requires_grad = True
    return backbone