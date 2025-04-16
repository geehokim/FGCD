import torch
import copy

from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import random


__all__ = ['compute_fisher_matrix_diag', 'layerwise_normalize', 'normalize_m', 'get_initial_global_prototype', 'update_global_prototype', 'initalize_random_seed', 'terminate_processes', 'extract_feature', 'extract_local_features', 'extract_local_features_unlabelled', 'extract_local_features_only', 'check_bfloat16_support']

def terminate_processes(queues, processes):
    # Signal all processes to exit by putting None in each queue
    for queue in queues:
        queue.put(None)

    # Wait for all processes to finish
    for p in processes:
        p.terminate()

def initalize_random_seed(args):
    random_seed = args.seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    # torch.backends.cudnn.enabled = True
    if args.enable_benchmark:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    


def check_bfloat16_support():
    if not torch.cuda.is_available():
        print("CUDA is not available on this device.")
        return False
    
    device = torch.device("cuda")
    device_capability = torch.cuda.get_device_capability(device)
    
    # Ampere 아키텍처 (7.0) 이상부터 bfloat16 지원
    if device_capability[0] >= 8 or (device_capability[0] == 7 and device_capability[1] >= 5):
        print("This GPU supports bfloat16.")
        return True
    else:
        print("This GPU does not support bfloat16.")
        return False



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


def extract_feature(model, dataset, device='gpu:0', batch_size=1024):
    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    all_feats = []
    all_feats_proj = []
    targets = []
    all_logits = []
    for batch_idx, (images, labels, _) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            features, feats_proj, logits = model(images, return_all=True)
            all_feats.append(features.cpu())
            all_feats_proj.append(feats_proj.cpu())
            targets.append(labels.cpu())
            all_logits.append(logits.cpu())

    all_feats = torch.cat(all_feats, dim=0)
    all_feats_proj = torch.cat(all_feats_proj, dim=0)
    all_logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(targets)

    model = model.cpu()
    del model
    torch.cuda.empty_cache()

    return all_feats, all_feats_proj, targets, all_logits


def extract_local_features(args,model, loader, evaler, device='gpu:0', labelled_test_transform=True):
    model.eval()
    # model_device = next(model.parameters()).device
    model.to(device)
    test_transform = copy.deepcopy(evaler.test_loader.dataset.transform)

    labelled_dataset = copy.deepcopy(loader.dataset.labelled_dataset)
    if labelled_test_transform:
        labelled_dataset.transform = test_transform
    else:
        labelled_dataset.transform = copy.deepcopy(labelled_dataset.transform.base_transforms[0])
    unlabelled_dataset = copy.deepcopy(loader.dataset.unlabelled_dataset)
    unlabelled_dataset.transform = test_transform

    feats_labelled = []
    feats_proj_labelled = []
    logits_labelled = []
    targets_labelled = np.array([])
    mask = np.array([])

    labelled_loader = DataLoader(labelled_dataset, batch_size=args.batch_size * 4, shuffle=False, num_workers=args.num_workers)

    feats_unlabelled = []
    feats_proj_unlabelled = []
    logits_unlabelled = []
    targets_unlabelled = np.array([])

    unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=args.batch_size * 4, shuffle=False, num_workers=args.num_workers)


    print('Collating logits...')
    # First extract feats labelled
    for batch_idx, (images, label, _) in enumerate(labelled_loader):
        images = images.to(device)

        # Pass features through base model and then additional learnable transform (linear layer)
        with torch.no_grad():
            feats, feats_proj, logits = model(images, return_all=True)

        targets_labelled = np.append(targets_labelled, label.cpu().numpy().astype(int))
        feats_labelled.append(feats.cpu().clone())
        feats_proj_labelled.append(feats_proj.cpu().clone())
        logits_labelled.append(logits)
        mask = np.append(mask, np.array([True if x.item() in range(len(args.dataset.seen_classes))
                                          else False for x in label]))

    # First extract feats unlabelled
    for batch_idx, (images, label, _) in enumerate(unlabelled_loader):
        images = images.to(device)

        # Pass features through base model and then additional learnable transform (linear layer)
        with torch.no_grad():
            feats, feats_proj, logits = model(images, return_all=True)

        targets_unlabelled = np.append(targets_unlabelled, label.cpu().numpy().astype(int))
        feats_unlabelled.append(feats.cpu().clone())
        feats_proj_unlabelled.append(feats_proj.cpu().clone())
        logits_unlabelled.append(logits)
        mask = np.append(mask, np.array([True if x.item() in range(len(args.dataset.seen_classes))
                                         else False for x in label]))

    feats_labelled = torch.cat(feats_labelled, dim=0)
    feats_proj_labelled = torch.cat(feats_proj_labelled, dim=0)
    if logits_labelled[0] is not None:
        logits_labelled = torch.cat(logits_labelled, dim=0)
    #targets_labelled = torch.cat(targets_labelled, dim=0)

    feats_unlabelled = torch.cat(feats_unlabelled, dim=0)
    feats_proj_unlabelled = torch.cat(feats_proj_unlabelled, dim=0)
    if logits_unlabelled[0] is not None:
        logits_unlabelled = torch.cat(logits_unlabelled, dim=0)
    #targets_unlabelled = torch.cat(targets_unlabelled, dim=0)

    mask = mask.astype(bool)
    del labelled_loader
    del unlabelled_loader
    
    return feats_labelled, feats_proj_labelled, logits_labelled, targets_labelled, feats_unlabelled, feats_proj_unlabelled, logits_unlabelled, targets_unlabelled, mask

def extract_local_features_unlabelled(args,model, loader, evaler, device='gpu:0'):
    model.eval()
    # model_device = next(model.parameters()).device
    model.to(device)
    test_transform = copy.deepcopy(evaler.test_loader.dataset.transform)

    unlabelled_dataset = copy.deepcopy(loader.dataset.unlabelled_dataset)
    unlabelled_dataset.transform = test_transform

    feats_unlabelled = []
    feats_proj_unlabelled = []
    logits_unlabelled = []
    targets_unlabelled = np.array([])
    mask = np.array([])

    unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=args.batch_size * 4, shuffle=False, num_workers=args.num_workers)

    print('Collating logits...')

    # First extract feats unlabelled
    for batch_idx, (images, label, _) in enumerate(unlabelled_loader):
        images = images.to(device)

        # Pass features through base model and then additional learnable transform (linear layer)
        with torch.no_grad():
            feats, feats_proj, logits = model(images, return_all=True)

        targets_unlabelled = np.append(targets_unlabelled, label.cpu().numpy().astype(int))
        feats_unlabelled.append(feats.cpu().clone())
        feats_proj_unlabelled.append(feats_proj.cpu().clone())
        logits_unlabelled.append(logits)
        mask = np.append(mask, np.array([True if x.item() in range(len(args.dataset.seen_classes))
                                         else False for x in label]))

    # feats_labelled = torch.cat(feats_labelled, dim=0)
    # feats_proj_labelled = torch.cat(feats_proj_labelled, dim=0)
    # logits_labelled = torch.cat(logits_labelled, dim=0)
    #targets_labelled = torch.cat(targets_labelled, dim=0)

    feats_unlabelled = torch.cat(feats_unlabelled, dim=0)
    feats_proj_unlabelled = torch.cat(feats_proj_unlabelled, dim=0)
    logits_unlabelled = torch.cat(logits_unlabelled, dim=0)
    #targets_unlabelled = torch.cat(targets_unlabelled, dim=0)

    mask = mask.astype(bool)
    del unlabelled_loader
    
    return feats_unlabelled, feats_proj_unlabelled, logits_unlabelled, targets_unlabelled, mask



def extract_local_features_only(args,model, loader, evaler, device='gpu:0', labelled_test_transform=True, return_feats_only=True):
    model.eval()
    # model_device = next(model.parameters()).device
    model.to(device)
    test_transform = copy.deepcopy(evaler.test_loader.dataset.transform)

    labelled_dataset = copy.deepcopy(loader.dataset.labelled_dataset)
    if labelled_test_transform:
        labelled_dataset.transform = test_transform
    else:
        labelled_dataset.transform = copy.deepcopy(labelled_dataset.transform.base_transforms[0])
    unlabelled_dataset = copy.deepcopy(loader.dataset.unlabelled_dataset)
    unlabelled_dataset.transform = test_transform

    feats_labelled = []
    feats_proj_labelled = []
    logits_labelled = []
    targets_labelled = np.array([])
    mask = np.array([])

    labelled_loader = DataLoader(labelled_dataset, batch_size=args.batch_size * 4, shuffle=False, num_workers=args.num_workers)

    feats_unlabelled = []
    feats_proj_unlabelled = []
    logits_unlabelled = []
    targets_unlabelled = np.array([])

    unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=args.batch_size * 4, shuffle=False, num_workers=args.num_workers)


    print('Collating logits...')
    # First extract feats labelled
    for batch_idx, (images, label, _) in enumerate(labelled_loader):
        images = images.to(device)

        # Pass features through base model and then additional learnable transform (linear layer)
        with torch.no_grad():
            feats = model(images, return_feats_only=True)

        targets_labelled = np.append(targets_labelled, label.cpu().numpy().astype(int))
        feats_labelled.append(feats.cpu().clone())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.dataset.seen_classes))
                                          else False for x in label]))

    # First extract feats unlabelled
    for batch_idx, (images, label, _) in enumerate(unlabelled_loader):
        images = images.to(device)

        # Pass features through base model and then additional learnable transform (linear layer)
        with torch.no_grad():
            feats = model(images, return_feats_only=True)

        targets_unlabelled = np.append(targets_unlabelled, label.cpu().numpy().astype(int))
        feats_unlabelled.append(feats.cpu().clone())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.dataset.seen_classes))
                                         else False for x in label]))

    feats_labelled = torch.cat(feats_labelled, dim=0)

    feats_unlabelled = torch.cat(feats_unlabelled, dim=0)

    mask = mask.astype(bool)
    del labelled_loader
    del unlabelled_loader
    
    return feats_labelled, targets_labelled, feats_unlabelled, targets_unlabelled, mask


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



