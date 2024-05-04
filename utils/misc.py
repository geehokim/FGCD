import torch
import copy
from torch.utils.data import DataLoader
import numpy as np
import random


__all__ = ['compute_fisher_matrix_diag', 'layerwise_normalize', 'normalize_m', 'get_initial_global_prototype', 'update_global_prototype', 'initalize_random_seed', 'terminate_processes']

def terminate_processes(queues, processes):
    # Signal all processes to exit by putting None in each queue
    for queue in queues:
        queue.put(None)

    # Wait for all processes to finish
    for p in processes:
        p.terminate()

def initalize_random_seed(args):
    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    # torch.backends.cudnn.enabled = True
    if args.enable_benchmark:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def compute_fisher_matrix_diag(net, optimizer, trn_loader, device, sampling_type='true'):
    # Store Fisher Information
    fisher = {n: torch.zeros(p.shape).to(device) for n, p in net.named_parameters()
              if p.requires_grad}
    # Compute fisher information for specified number of samples -- rounded to the batch size
    # n_samples_batches = (self.num_samples // trn_loader.batch_size + 1) if self.num_samples > 0 \
    #     else (len(trn_loader.dataset) // trn_loader.batch_size)
    # Do forward and backward pass to compute the fisher information
    model = net
    # for images, targets in itertools.islice(trn_loader, n_samples_batches):
    for batch_idx, (images, labels) in enumerate(trn_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model.basic_forward(images)

        if sampling_type == 'true':
            # Use the labels to compute the gradients based on the CE-loss with the ground truth
            preds = labels.to(device)
        elif sampling_type == 'max_pred':
            # Not use labels and compute the gradients related to the prediction the model has learned
            preds = torch.cat(outputs, dim=1).argmax(1).flatten()
        elif sampling_type == 'multinomial':
            # Use a multinomial sampling to compute the gradients
            probs = torch.nn.functional.softmax(torch.cat(outputs, dim=1), dim=1)
            preds = torch.multinomial(probs, len(labels)).flatten()

        loss = torch.nn.functional.cross_entropy(outputs, preds)
        optimizer.zero_grad()
        loss.backward()
        # Accumulate all gradients from loss with regularization
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.pow(2) * len(labels)
    # Apply mean across all samples
    n_samples = len(trn_loader) * trn_loader.batch_size
    fisher = {n: (p / n_samples) for n, p in fisher.items()}

    return fisher

def layerwise_normalize(fisher):
    fisher = copy.deepcopy(fisher)
    for key in fisher.keys():
        fisher[key] = (fisher[key] - fisher[key].min()) / (fisher[key].max() - fisher[key].min())
    return fisher


def normalize_m(fisher):
    min_value = 100000000
    max_value = 0
    fisher = copy.deepcopy(fisher)
    for key in fisher.keys():
        mi = fisher[key].min()
        ma = fisher[key].max()
        if mi < min_value:
            min_value = mi
        if ma > max_value:
            max_value = ma
    for key in fisher.keys():
        fisher[key] = (fisher[key] - min_value) / (max_value - min_value)
    return fisher


def get_initial_global_prototype(args, global_initial_model, trainset, device='gpu:0'):
    net = global_initial_model
    net = net.to(device)
    net.eval()
    loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    prototypes = []
    prototype_dict = {}
    for i in range(len(trainset.classes)):
        prototype_dict[i] = []
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            features, _ = net(images, return_feature=True)
        for i in range(features.size(0)):
            prototype_dict[labels[i].item()].append(features[i])
    for class_idx in prototype_dict.keys():
        prototypes.append(torch.stack(prototype_dict[class_idx], dim=0).mean(dim=0))
    return torch.stack(prototypes, dim=0)

def update_global_prototype(local_prototypes, prev_global_prototype=None):
    prototypes = [torch.zeros_like(prev_global_prototype[0]) for i in range(prev_global_prototype.size(0))]
    prototype_dict = {}
    for i in range(prev_global_prototype.size(0)):
        prototype_dict[i] = []
    for l_prototype in local_prototypes:
        for i in range(prev_global_prototype.size(0)):
            if len(l_prototype[i]) != 0:
                prototype_dict[i].append(l_prototype[i])
            else:
                continue

    for i in range(prev_global_prototype.size(0)):
        if len(prototype_dict[i]) == 0:
            prototypes[i] = prev_global_prototype[i]
        else:
            prototypes[i] = torch.stack(prototype_dict[i], dim=0).mean(0)

    return torch.stack(prototypes, dim=0)
