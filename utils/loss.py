import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from utils.helper import *
from torch import nn, autograd
import collections

__all__ = ['ClusterLoss', 'ClusterLossProto', 'MarginLoss', 'FedDecorrLoss', 'MultiLabelCrossEntropyLoss','MetricLoss', 'MetricLoss2', 'UnsupMetricLoss', 'MetricLoss_rel', 'MetricLossSubset', 'TripletLoss', 'IL','CE','IL_negsum', 'DeepInversionFeatureHook', 'LossManager', 'FedLC', 'FedDecorrLoss', 'MetricLoss_djr']

class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5, device=None):
    if device is None:
        return CM.apply(inputs, indexes, features.to(inputs.device), torch.Tensor([momentum]).to(inputs.device))
    else:
        return CM.apply(inputs.to(device), indexes, features.to(device), torch.Tensor([momentum]).to(device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5, device=None):
    if device is None:
        return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))
    else:
        return CM_Hard.apply(inputs.to(device), indexes, features, torch.Tensor([momentum]).to(device))

class ClusterLoss(nn.Module):
    def __init__(self, num_features, num_samples, temperature=0.07, momentum=1, device=None):
        super(ClusterLoss, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.temperature = temperature
        self.register_buffer('prototypes', torch.zeros(self.num_samples, self.num_features))
        self.momentum = momentum
        self.device = device
    # def update_prototypes(self, prototypes, device=None):
    #     prototypes = F.normalize(prototypes, dim=1)
    #     self.prototypes = nn.Parameter(prototypes)
    #     self.device = device

    def forward(self, x, target):
        
        # prototypes = F.normalize(self.prototypes, dim=-1, p=2)
        prototypes = self.prototypes.to(self.device)
        # batch_size, num_view, feat_dim = x.size()
        # x = x.view(batch_size * num_view, feat_dim)
        # target2 = target.detach().clone()
        # target = torch.cat([target, target2], dim=0).to(self.device)
        # outputs = torch.matmul(x, prototypes.T) / self.temperature
        # # outputs = cm(x, target, prototypes, 0.9, self.device)
        # # outputs /= self.temperature
        # loss = F.cross_entropy(outputs, target.long())

        # outputs = torch.matmul(x, prototypes.T) / self.temperature
        outputs = cm(x, target, prototypes, self.momentum, self.device)
        outputs /= self.temperature
        loss = F.cross_entropy(outputs, target.long())
        return loss

class ClusterLossProto(nn.Module):
    def __init__(self, num_features, num_samples, temperature=0.07, device=None):
        super(ClusterLossProto, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.zeros(self.num_samples, self.num_features))
        # self.register_buffer('prototypes', nn.Parameter(torch.zeros(self.num_samples, self.num_features)))
        self.device = device
    
    # def update_prototypes(self, prototypes, device=None):
    #     prototypes = F.normalize(prototypes, dim=1)
    #     self.prototypes = nn.Parameter(prototypes)
    #     self.device = device

    def forward(self, x, target):
        
        prototypes = F.normalize(self.prototypes, dim=-1, p=2)
        prototypes = self.prototypes.to(self.device)
        # batch_size, num_view, feat_dim = x.size()
        # x = x.view(batch_size * num_view, feat_dim)
        # target2 = target.detach().clone()
        # target = torch.cat([target, target2], dim=0).to(self.device)
        outputs = torch.matmul(x, prototypes.T) / self.temperature
        # # outputs = cm(x, target, prototypes, 0.9, self.device)
        # # outputs /= self.temperature
        # loss = F.cross_entropy(outputs, target.long())

        # outputs = torch.matmul(x, prototypes.T) / self.temperature
        # outputs = cm(x, target, prototypes, self.momentum, self.device)
        # outputs /= self.temperature
        loss = F.cross_entropy(outputs, target.long())
        return loss


class MarginLoss(nn.Module):
    """
    Margin Loss
    """
    def __init__(self, m=0.2, weight=None, s=10):
        super(MarginLoss, self).__init__()
        self.m = m
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        x_m = x - self.m * self.s
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(output, target, weight=self.weight)

class MultiLabelCrossEntropyLoss(nn.Module):
    """NLL loss with label smoothing."""

    # def __init__(self, smoothing: float = 0.0):
    def __init__(self, eps: float=0, alpha: float=0.2, topk_pos: int=-1, ver: str=None, temp: float=1., adapt_ce: str=None,
                 p_margin=0, n_margin=0, **kwargs):
        """Construct LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(MultiLabelCrossEntropyLoss, self).__init__()
        self.eps = eps
        self.alpha = alpha
        self.topk_pos = topk_pos
        self.temp = temp
        self.adapt_ce = adapt_ce
        self.p_margin = p_margin
        self.n_margin = n_margin
        self.ver = ver

    def __repr__(self):
        return "MultiLabelCrossEntropyLoss(eps={}, alpha={})".format(self.eps, self.alpha)


    def weighted_softmax(self, x, weights, dim=1):
        maxes = torch.max(x, dim, keepdim=True)[0]
        x_exp = torch.exp(x-maxes)
        x_exp_weighted = x_exp*weights
        x_exp_sum = torch.sum(x_exp_weighted, dim, keepdim=True)

        return x_exp/x_exp_sum


    def get_loss(self, input, targets):
        log_probs = F.log_softmax(input, dim=1)
        loss_ = (-targets * log_probs)
        loss_[loss_==np.inf] = 0.
        loss_[loss_==-np.inf] = 0.
        loss_[loss_.isnan()] = 0.
        loss = loss_.sum(dim=1)

        return loss


    def forward(self, input: torch.Tensor, targets: torch.Tensor, reduction: bool = True, beta: float = None, 
    uncertainty: torch.Tensor = None, class_ratio: torch.Tensor = None, level: float = None, progress: float = None, data_label: torch.Tensor = None) -> torch.Tensor:
        """Apply forward pass.

        :param x: Logits tensor. [N, C]
        :param target: Ground truth target classes. [N] -> [N, C] for {0, 1}
        :return: Loss tensor.
        """
        N, C = input.size()
        E = self.eps

        # breakpoint()

        input[input==np.inf] = -np.inf # to ignore np.inf

        if beta is not None:
            weights = torch.ones_like(targets)
            weights[targets==0] = beta
            input += torch.log(weights)/self.temp


        if self.adapt_ce is not None:
            if self.adapt_ce == 'temp': # sqrt_norm
                # input += torch.log(weights)/self.temp
                # hardness = (input[:, 0]*self.temp+1)/2
                # sample_weights = hardness.unsqueeze(1).detach()
                class_weight = (1 / class_ratio)**0.5
                class_weight /= class_weight.mean()
                input = input * class_weight.unsqueeze(1).to(input)
                
            elif self.adapt_ce == 'positive_margin':
                sim = input * self.temp
                positive_sim = sim[targets == 1]
                clamped_sim = torch.clamp(positive_sim + self.p_margin, max=1)
                sim[targets == 1] = clamped_sim
                input = sim / self.temp

            elif self.adapt_ce == 'temp_lognorm':
                class_weight = 1/torch.log(class_ratio)
                class_weight /= class_weight.mean()
                input = input * class_weight.unsqueeze(1).to(input)

            elif self.adapt_ce == 'pos_temp':
            # input += torch.log(weights)/self.temp
                hardness = (input[:, 0]*self.temp+1)/2
                sample_weights = torch.ones_like(input)
                sample_weights[:, 0] = hardness.detach()
                input = input * sample_weights

            elif self.adapt_ce == 'stoc_smoothing':
                # hardness = (input[:, 0]*self.temp+1)/2
                weight = torch.rand(1)[0]
                input[:, 0] = input[:, 0] * (1-weight) + (1/self.temp)*weight
                # sample_weights = hardness.unsqueeze(1)
                # input = input * sample_weights

            elif self.adapt_ce == 'neg_amplify':
                input[:, 1:] = input[:, 1:] * 2

            elif self.adapt_ce == 'smoothing':
                weight = 0.5
                input[:, :1] = input[:, :1] * (1-weight) + (1/self.temp)*weight

            elif self.adapt_ce == 'neg_smoothing':
                weight = 0.5
                input[:, 1:] = input[:, 1:] * (1-weight) + (1/self.temp)*weight

            elif self.adapt_ce == 'smoothing_L0':
                weight = 0.5 if level == 0 else 0
                input[:, :1] = input[:, :1] * (1-weight) + (1/self.temp)*weight

            elif self.adapt_ce == 'smoothing2':
                weight = 0.25
                # breakpoint()
                input[:, 0] = input[:, 0] * (1-weight) + (1/self.temp)*weight

            elif self.adapt_ce == 'smoothing3':
                # weight = 0.25
                # breakpoint()
                input[:, 0] = (2/self.temp)

            elif self.adapt_ce == 'layerwise':
                weight = 0.5 - (level)/10
                # breakpoint()
                # weight = 2 - level/5 # 0.9, 0.8, 0.7, 0.6, 0.5
                input[:, 0] = input[:, 0] * (1-weight) + (1/self.temp)*weight

            elif self.adapt_ce == 'pos_cond':

                rel_dist = torch.abs((input[:, 1:] - input[:, 0].unsqueeze(1)).detach())

                N = 5
                _, inds = torch.topk(rel_dist, min(N, input.size(1)-1), dim=1, largest=False)
                inds_ = torch.cat((torch.zeros_like(inds)[:, :1], inds+1), dim=1)
                input_ = torch.gather(input, 1, inds_)
                input = input_
                targets = targets[:, :1+N]

            elif self.adapt_ce == 'calibration':
                pos_logit = input[:, :1].mean(1).detach()
                neg_logit = input[:, 1:].mean(1).detach()
                input[:, 0] += (neg_logit - pos_logit).detach()

            elif self.adapt_ce == 'calibration_v2':
                pos_logit = input[:, :1].mean(1).detach()
                neg_logit = input[:, 1:].mean(1).detach()
                delta = 5
                neg_logit_ = neg_logit - delta

                input[:, 0] += (neg_logit_ - pos_logit).clamp(0).detach()


            elif self.adapt_ce == 'balance':
                weight = class_ratio / class_ratio.min()
                breakpoint()
                # cr_inv = 1/torch.sqrt(class_ratio/input.size(0))
                # cr_inv /= cr_inv.mean()
                # sample_weights = cr_inv.unsqueeze(1)
                # input = input * sample_weights
                # weight = class_ratio / (input.size(0)/100)
                # sample_weights = (input[:, 0] > 0).float().detach()
                # input[:, 0] *= sample_weights

            elif self.adapt_ce == 'relu':
                sample_weights = (input[:, 0] > 0).float().detach()
                input[:, 0] *= sample_weights

            elif self.adapt_ce == 'uncertainty':
                weight = uncertainty / 2
                if level == 4:
                    if torch.rand(1).item() < 0.01:
                        print(weight.mean())
                input[:, 0]  = input[:, 0] * (1-weight) + (1/self.temp)*weight

            elif self.adapt_ce == 'uncertainty_v2':
                weight = uncertainty
                if level == 4:
                    if torch.rand(1).item() < 0.01:
                        print(weight.mean())
                input[:, 0]  = input[:, 0] * (1-weight) + (1/self.temp)*weight

            elif self.adapt_ce == 'uncertainty_temp':
                # weight = uncertainty / 2
                # breakpoint()
                # sample_weights = uncertainty.unsqueeze(1).detach()
                # breakpoint()
                # uncertainty = uncertainty or torch.zeros_like(uncertainty)
                confidence = 1 - uncertainty
                sample_weights = confidence.unsqueeze(1).detach()
                if level == 4:
                    if torch.rand(1).item() < 0.01:
                        print(sample_weights.mean())
                input = input * sample_weights


            elif self.adapt_ce == 'progress':
                weight = max(0, (1 - 2*progress)/2)
                # input[:, 0]  = input[:, 0] * (1-weight) + (1/self.temp)*weight
                input[:, 1:]  = input[:, 1:] * (1-weight) + (-1/self.temp)*weight

        # targets *= (1-E)
        # targets[:, self.topk_pos:] = self.topk_pos*E/(C-self.topk_pos)

            
        log_probs = F.log_softmax(input, dim=1)
        loss_ = (-targets * log_probs)
        loss_[loss_==np.inf] = 0.
        loss_[loss_==-np.inf] = 0.
        loss_[loss_.isnan()] = 0.
        loss = loss_.sum(dim=1)
        # if (loss==0).sum() > 0:
        #     breakpoint()
        #breakpoint()

        if self.adapt_ce is not None and 'augment' in self.adapt_ce:
            
            if self.adapt_ce == 'augment_unc':
                input_i = input.clone()
                input_i[:, 0]  = (1/self.temp)
                loss_unsup = self.get_loss(input_i, targets)
                weight = uncertainty / 2
                loss = loss * (1-weight) + loss_unsup * weight

            elif self.adapt_ce == 'augment_unc_v3':
                input_i = input.clone()
                input_i[:, 0]  = (1/self.temp)
                loss_unsup = self.get_loss(input_i, targets)
                weight = uncertainty
                loss = loss + loss_unsup * weight

            elif self.adapt_ce == 'augment_unc_v4':
                input_i = input.clone()
                input_i[:, 0]  = (1/self.temp)
                loss_unsup = self.get_loss(input_i, targets)
                weight = 0.5 + uncertainty
                loss = loss + loss_unsup * weight

            else:
                losses = [loss]
                num_augment = int(self.adapt_ce.split('_')[1])
                for i in range(num_augment):
                    input_i = input.clone()
                    weight = (i+1)/num_augment
                    input_i[:, 0]  = input_i[:, 0] * (1-weight) + (1/self.temp)*weight
                    losses.append(self.get_loss(input_i, targets))

                losses = torch.stack(losses, 0)
                loss = losses.mean(0)

        # input2 = input.clone()
        # input2[targets==1] = 1/self.temp
        # log_probs2 = F.log_softmax(input2, dim=1)
        # loss2_ = (-targets * log_probs2)
        # loss2_[loss2_==np.inf] = 0.
        # loss2_[loss2_==-np.inf] = 0.
        # loss2_[loss2_.isnan()] = 0.
        # loss2 = loss2_.sum(dim=1)
        # loss = loss + loss2

        # breakpoint()


            # loss_pos = -input[targets==1]
            # x_exp = torch.exp(input)
            # # breakpoint()
            # # x_exp_neg = 
            # # x_exp[targets==1] *= 0
            # loss_neg = torch.log(torch.sum(x_exp, dim=1))
            # # weights = torch.ones_like(targets)
            # # weights[targets==0] = beta
            # # softs = self.weighted_softmax(input, weights, dim=1)
            # # log_probs = torch.log(softs)
            # loss = loss_pos + loss_neg
            # breakpoint()
        # breakpoint()
        '''
        with torch.no_grad():

            if self.ver == "mask":
                targets *= (1-E)
                targets[targets==0.] = self.topk_pos*E/(C-self.topk_pos)
            else:
                targets *= (1-E)
                targets[targets==0.] = self.topk_pos*E/(C-self.topk_pos)
        '''

        # loss = (-targets * log_probs).sum(dim=1)

        with torch.no_grad():
            non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)

        if reduction:
            loss = loss.sum() / non_zero_cnt
        else:
            # breakpoint()
            loss = loss

        # if loss.isnan():
        #     breakpoint()

        return loss



class FedDecorrLoss(nn.Module):

    def __init__(self):
        super(FedDecorrLoss, self).__init__()
        self.eps = 1e-8

    def _off_diagonal(self, mat):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = mat.shape
        assert n == m
        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x):
        if len(x.shape) == 4:
            x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        N, C = x.shape
        if N == 1:
            return 0.0

        x = x - x.mean(dim=0, keepdim=True)
        x = x / torch.sqrt(self.eps + x.var(dim=0, keepdim=True))

        corr_mat = torch.matmul(x.t(), x)

        loss = (self._off_diagonal(corr_mat).pow(2)).mean()
        loss = loss / N

        return loss


class MetricLoss(nn.Module):

    def __init__(self, topk_pos=-1, topk_neg=-1, temp=1, eps=0., pair=None, loss_type=None, beta=None,
                 pos_sample_type=None, neg_sample_type=None, adapt_ce=None, pos_loss_type=None, neg_loss_type=None,
                 adapt_sample=None, sampling=None, p_margin=0, n_margin=0,threshold=1,
                 **kwargs):
        super(MetricLoss, self).__init__()
        self.pair = pair
        
        self.topk_pos = self.pair.get('topk_pos') or topk_pos
        self.topk_neg = self.pair.get('topk_neg') or topk_neg
        self.pos_sample_type = self.pair.get('pos_sample_type') or pos_sample_type
        self.neg_sample_type = self.pair.get('neg_sample_type') or neg_sample_type
        
        self.temp = self.pair.get('temp') or temp
        
        self.sims_dict = {}
        
        self.loss_type = self.pair.get('loss_type') or loss_type
        self.pos_loss_type = self.pair.get('pos_loss_type') or pos_loss_type 
        self.neg_loss_type = self.pair.get('neg_loss_type') or neg_loss_type 
        self.beta = self.pair.get('beta') or beta # weight for negative samples. 
        self.adapt_ce = self.pair.get('adapt_ce') or adapt_ce 
        self.adapt_sample = self.pair.get('adapt_sample') or adapt_sample 

        self.sampling = self.pair.get('sampling') or sampling
        self.p_margin = self.pair.get('p_margin') or p_margin
        self.n_margin = self.pair.get('n_margin') or n_margin
        self.threshold = self.pair.get('threshold') or threshold

        self.criterion = MultiLabelCrossEntropyLoss(topk_pos=topk_pos, temp=temp, adapt_ce=self.adapt_ce, eps=eps,
                                                   p_margin=self.p_margin, n_margin=self.n_margin) 


    def __set_num_classes__(self, num_classes):
        self.num_classes = num_classes

    def __repr__(self):
        return "{}(topk_pos={}, topk_neg={}, temp={}, crit={}), pair={}, no sampleno)".format(
            type(self).__name__, self.topk_pos, self.topk_neg, self.temp, self.criterion, self.pair)

    def get_classwise_mask(self, target):
        B = target.size(0)
        classwise_mask = target.expand(B, B).eq(target.expand(B, B).T)
        return classwise_mask
    

    #def get_topk_neg(self, sim, pos_mask=None, topk_neg=None, labels=None, easyneg = True):
    def get_topk_neg(self, sim, pos_mask=None, topk_neg=None, topk_pos=None, labels=None, easyneg = True):
        sim_neg = sim.clone()
        B = sim_neg.size(0)

        neg_loss_type = self.neg_loss_type if self.neg_loss_type else self.loss_type
        #if neg_loss_type == 'unsupervised':
        if 'unsupervised' in neg_loss_type:
            pos_mask = self.get_classwise_mask(torch.arange(B)).to(labels.device) # 1 only if the same sample
        else:
            pos_mask = self.get_classwise_mask(labels)


        if self.neg_sample_type:
            if self.neg_sample_type == 'debug':
                breakpoint()
                
            if self.neg_sample_type == 'all':
                # mining negative samples from all classes except the same instance
                sim_neg[torch.eye(B)==1] = -np.inf
                sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]
            elif self.neg_sample_type == 'intra_class':
                # mining negative samples from the same class
                classwise_mask = self.get_classwise_mask(labels)
                sim_neg[~classwise_mask] = -np.inf
                sim_neg[torch.eye(B)==1] = -np.inf
                # sim_neg[pos_mask==1] = -np.inf
                sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]

            elif self.pos_sample_type == 'center':
                sim_neg[pos_mask==1] = np.nan
                sim_neg = sim_neg.nanmean(1, keepdim=True).repeat(1, topk_neg)


            elif self.pos_sample_type == 'hard_center':
                sim_neg[pos_mask==1] = -np.inf
                sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]

                breakpoint()

            elif self.neg_sample_type == 'inter_class':
                # mining negative samples from the different class only
                classwise_mask = self.get_classwise_mask(labels)
                sim_neg[classwise_mask] = -np.inf
                # sim_neg[pos_mask==1] = -np.inf #This line may be redundant
                sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]

            elif self.neg_sample_type == 'easy':
                
                sim_neg[pos_mask==1] = np.inf
                sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=False)[0]

            elif self.neg_sample_type == 'random':
                # raise NotImplementedError
                # breakpoint()
                sim_neg[pos_mask==1] = np.nan
                random_neg = torch.rand_like(sim_neg)
                random_neg[pos_mask==1] = -np.inf
                random_neg_inds = torch.topk(random_neg, min(topk_neg, B), dim=1, largest=True)[1]
                sim_neg = sim_neg.gather(1, random_neg_inds)
            elif self.neg_sample_type == 'intra_class_thresholding':
                #print("intra_class_thresholding code")
                # mining negative samples from the same class
                classwise_mask = self.get_classwise_mask(labels)
                sim_neg[~classwise_mask] = np.inf
                sim_neg[torch.eye(B)==1] = np.inf
                # sim_neg[pos_mask==1] = -np.inf
                sim_neg = torch.topk(sim_neg, min(topk_pos, B), dim=1, largest=False)[0]
                idx = sim_neg < self.threshold
                sim_neg[idx] = -1
                sim_neg[sim_neg == np.inf] = -1
            else:
                raise ValueError
            # elif self.neg_sample_type == 'pos_cond':
            #     breakpoint()
        elif 'unsupervised_neg' in neg_loss_type and easyneg:
            sim_neg[pos_mask==1] = np.inf
            sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=False)[0]


        else:
            sim_neg[pos_mask==1] = -np.inf
            sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]


        return sim_neg
    

    def get_topk_pos(self, sim, topk_pos=None, labels=None, uncertainty=None):

        sim_pos = sim.clone()

        B = sim.size(0)

        pos_loss_type = self.pos_loss_type if self.pos_loss_type else self.loss_type
        if pos_loss_type == 'unsupervised':
            pos_mask = self.get_classwise_mask(torch.arange(B)).to(labels.device)
        else:
            pos_mask = self.get_classwise_mask(labels)

        inds = None

        if self.pos_sample_type:
            if self.pos_sample_type == 'easy': # high sim
                sim_pos[pos_mask==0] = -np.inf
                sim_pos, inds = torch.topk(sim_pos, topk_pos, dim=1, largest=True)

            elif self.pos_sample_type == 'no_grad':
                sim_pos[pos_mask==0] = np.inf
                sim_pos, inds = torch.topk(sim_pos, topk_pos, dim=1, largest=False)
                sim_pos.fill_(1)

            elif self.pos_sample_type == 'center':
                sim_pos[pos_mask==0] = np.nan
                sim_pos = sim_pos.nanmean(1, keepdim=True).repeat(1, self.topk_pos)

            elif self.pos_sample_type == 'conf_center':
                # breakpoint()
                sim_pos[pos_mask==0] = np.nan
                confs = 1 - uncertainty
                confs_ = confs.unsqueeze(0).repeat(confs.size(0), 1)
                confs_[pos_mask==0] = np.nan
                # confs_.fill_diagonal_(1)
                confs_classmean = confs_.nanmean(1, keepdim=True)
                confs_norm = confs_/confs_classmean
                sim_pos = (confs_norm*sim_pos).nanmean(1, keepdim=True).repeat(1, self.topk_pos)
                # sim_pos2 = sim_pos.nanmean(1, keepdim=True).repeat(1, self.topk_pos)
                
            elif self.pos_sample_type == 'hard_center':
                sim_pos[pos_mask==0] = np.inf
                sim_pos, inds = torch.topk(sim_pos, topk_pos, dim=1, largest=False)
                sim_pos[sim_pos==np.inf] = np.nan

                sim_pos = sim_pos.nanmean(1, keepdim=True).repeat(1, self.topk_pos)

            elif self.pos_sample_type == 'random':
                sim_pos[pos_mask==0] = np.nan
                random_pos = torch.rand_like(sim_pos)
                random_pos[pos_mask==0] = -np.inf
                random_pos_inds = torch.topk(random_pos, topk_pos, dim=1, largest=True)[1]
                sim_pos = sim_pos.gather(1, random_pos_inds)

            elif self.pos_sample_type == 'positive':
                sim_pos[pos_mask==0] = np.inf
                sim_pos[sim_pos < 0] = np.inf
                # breakpoint()
                sim_pos, inds = torch.topk(sim_pos, topk_pos, dim=1, largest=False)

        
        else:
            sim_pos[pos_mask==0] = np.inf
            sim_pos, inds = torch.topk(sim_pos, topk_pos, dim=1, largest=False)
            # breakpoint()


        return sim_pos


    def get_uncertainty(self, sim, topk_pos=None, labels=None):
        # if self.adapt_ce == 'uncertainty':
        sim_pos = sim.clone()
        pos_mask = self.get_classwise_mask(labels)
        sim_pos[pos_mask==0] = np.nan
        sim_pos_ = sim_pos.nanmean(1)

        sim_neg = sim.clone()
        sim_neg[pos_mask==1] = np.nan
        sim_neg_ = sim_neg.nanmean(1)

        conf = (sim_pos_ - sim_neg_).clamp(0, 1).detach()
        unc = 1 - conf
        return unc

    def get_rel_confidence(self, sim_nn, sim_oo, topk_pos=None, labels=None):
        # if self.adapt_ce == 'uncertainty':
        # sim_pos = sim.clone()
        # pos_mask = self.get_classwise_mask(labels)
        # sim_pos[pos_mask==0] = np.nan
        # sim_pos_ = sim_pos.nanmean(1)

        # sim_neg = sim.clone()
        # sim_neg[pos_mask==1] = np.nan
        # sim_neg_ = sim_neg.nanmean(1)

        # conf = (sim_pos_ - sim_neg_).clamp(0, 1).detach()
        # unc = 1 - conf
        nn_unc = self.get_uncertainty(sim_nn, labels=labels)
        oo_unc = self.get_uncertainty(sim_oo, labels=labels)

        nn_conf = 1 - nn_unc
        oo_conf = 1 - oo_unc
        # breakpoint()
        rel_conf = nn_conf - oo_conf

        return rel_conf
        # else:
        #     return None
    def get_avg_class_sim_per_sample(self, raw_data, labels, num_classes, unique_labels = None):
        device = labels.device
        result = torch.zeros((raw_data.shape[0], num_classes),device = device)
        for idx,rd in enumerate(raw_data):
            this_label = labels[idx]
            for i, label in enumerate(unique_labels):
                class_mask = (labels == label)
                if this_label == label:
                    class_mask[idx] = False
                result[idx][label] = (raw_data[idx][class_mask]).mean(dim=0)

        return result





    def forward(self, old_feat, new_feat, target, reduction=True, pair=None, topk_pos=None, topk_neg=None, 
    uncertainty=None, class_ratio=None, level=None, progress=None, name="loss1", djr =False, class_major = None, class_minor = None):
        if old_feat.dim() > 2:
            old_feat = old_feat.squeeze(-1).squeeze(-1)
        if new_feat.dim() > 2:
            new_feat = new_feat.squeeze(-1).squeeze(-1)
        if 'uniquecat' in self.loss_type:
            bin_count = torch.bincount(target)
            unique_classes = torch.unique(target)
            unique_classes = unique_classes[bin_count[unique_classes] == 1]
            is_unique = (target.unsqueeze(1) == unique_classes)
            is_unique = (is_unique.sum(dim=1))>0
            if 'put' in self.loss_type:
                pass
            else:
                old_feat = torch.cat((new_feat, old_feat[is_unique]), dim = 0)
                concat_target = torch.cat((target, target[is_unique]), dim = 0)

        if 'concat' in self.loss_type:
            #new_feat = torch.cat((new_feat, old_feat), dim = 0)
            old_feat = torch.cat((new_feat, old_feat), dim = 0)
            concat_target = torch.cat((target, target), dim = 0)
            #target = torch.cat((target, target), dim = 0)



        B, C = new_feat.size()    
        #B, C = old_feat.size()

        old_feat_ = F.normalize(old_feat, p=2, dim=1)
        new_feat_ = F.normalize(new_feat, p=2, dim=1)

        sims = {}
        all_pair_types = set(self.pair.pos.split(' ') + self.pair.neg.split(' '))

        # sims['oo'] = torch.mm(old_feat_, old_feat_.t()) if 'oo' in all_pair_types else None
        sims['oo'] = torch.mm(old_feat_, old_feat_.t()) 
        sims['no'] = torch.mm(new_feat_, old_feat_.t()) if 'no' in all_pair_types else None
        sims['on'] = torch.mm(old_feat_, new_feat_.t()) if 'on' in all_pair_types else None
        sims['nn'] = torch.mm(new_feat_, new_feat_.t()) if 'nn' in all_pair_types else None
        sims['nd'] = torch.mm(new_feat_, (new_feat_.detach().clone().t())) if 'nd' in all_pair_types else None
        sims['dn'] = torch.mm(new_feat_.detach().clone(), new_feat_.t()) if 'dn' in all_pair_types else None
        sims['dd'] = torch.mm(new_feat_.detach().clone(), (new_feat_.detach().clone().t())) if 'dd' in all_pair_types else None


        if djr:
            unique_labels= target.unique()
            new_feat_norm = new_feat.norm(dim = 1,keepdim = True)
            avg_feat_norm = get_avg_data_per_class(new_feat_norm, target, self.num_classes, unique_labels = unique_labels)

            major_feat_norm = {cl:avg_feat_norm[cl] for cl in class_major}
            minor_feat_norm = {cl:avg_feat_norm[cl] for cl in class_minor}
            feat_norm_stat = {}
            feat_norm_stat['major_self'] = major_feat_norm
            feat_norm_stat['minor_self'] = minor_feat_norm
            feat_norm_stat['major_self_minall'] = min_dict(major_feat_norm)
            feat_norm_stat['minor_self_minall'] = min_dict(minor_feat_norm)


            feat_norm_ratio = avg_feat_norm / (avg_feat_norm.t())
            feat_norm_ratio_stat = get_major_minor_stat(class_major, class_minor, feat_norm_ratio)



            avg_class_sim_per_sample = self.get_avg_class_sim_per_sample(sims['nn'], target, self.num_classes, unique_labels = unique_labels)
            avg_class_sim_per_class = get_avg_data_per_class(avg_class_sim_per_sample,  target, self.num_classes, unique_labels = unique_labels)
            feat_cos_stat = get_major_minor_stat(class_major, class_minor, avg_class_sim_per_class)

            #avg_sims = get_avg_data_per_class(sims['nn'] , target, self.num_classes, unique_labels = unique_labels)
            #breakpoint()


        loss = 0.

        sim_poss, sim_negs = {}, {}
        ind_poss = {}

        if topk_pos is None:
            topk_pos = self.topk_pos

        if topk_neg is None:
            topk_neg = self.topk_neg


        unc = None
        # conf = self.get_rel_confidence(sims['nn'], sims['oo'], labels=target)
        # print(f"conf: {conf.min()}/{conf.max()}/{conf.mean()} at level{level}, prog{progress}")

        neg_loss_type = self.neg_loss_type if self.neg_loss_type else self.loss_type

        if 'cat' in self.loss_type:
            if pair is None:
                pair = self.pair
            if sims['no']==None:
                sims['no'] = torch.mm(new_feat_, old_feat_.t())

            unc = None
            for pair_type in pair['pos'].split(' '):
                if 'put' in self.loss_type:
                    matrix_unique = torch.diag(is_unique)
                    modified = sims[pair_type]
                    modified[matrix_unique] = (sims['no'])[matrix_unique]
                    sim_poss[pair_type] = self.get_topk_pos(modified, topk_pos=topk_pos, labels=target)
                    
                    #sim_poss[pair_type][is_unique] = old_feat[is_unique]
                else:
                    sim_poss[pair_type] = self.get_topk_pos(sims[pair_type], topk_pos=topk_pos, labels=concat_target)
                    if 'nocut' in self.loss_type:
                        pass
                    else:
                        sim_poss[pair_type] = sim_poss[pair_type][:B]
            for pair_type in pair['neg'].split(' '):
                if 'nocut' in self.loss_type:
                    sim_negs[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, labels=concat_target)
                else:
                    sim_negs[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, labels=target)
        else:
            for pair_type in all_pair_types:
                unc = self.get_uncertainty(sims[pair_type], labels=target)
                #if neg_loss_type == 'unsupervised':
                if 'unsupervised_neg' in neg_loss_type:
                    sim_poss[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, labels=target)
                    sim_negs[pair_type] = ((sim_poss[pair_type]).min(dim = 1, keepdim = True)[0]).detach().clone()
                    if 'unsupervised_neg1' in neg_loss_type:
                        pass
                    elif 'unsupervised_neg2' in neg_loss_type:
                        sim_negs[pair_type] *= 0
                    elif 'unsupervised_neg3' in neg_loss_type:
                        sim_negs[pair_type] = sim_negs[pair_type] * 0 - 1
                    elif 'unsupervised_neg4' in neg_loss_type:
                        sim_negs[pair_type] = (sim_negs[pair_type] * 0.9 + (- 1) * 0.1)
                    elif 'unsupervised_neg5' in neg_loss_type:
                        sim_negs[pair_type] = sim_negs[pair_type] * 0 + 1
                    elif 'unsupervised_negpos' in neg_loss_type:
                        sim_negs[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, labels=target, easyneg = False)
                        #print(((sim_poss[pair_type] - sim_negs[pair_type])**2).sum())
                    
                else:
                    
                    sim_poss[pair_type] = self.get_topk_pos(sims[pair_type], topk_pos=topk_pos, labels=target)
                    #sim_negs[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, labels=target)
                    sim_negs[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, topk_pos=topk_pos, labels=target)
                # , sim_pos=sim_poss[pair_type])
                # sim_pos_p, ind_p = self.get_topk_pos(sims[pair_type], topk_pos=topk_pos, labels=target, uncertainty=uncertainty)
                # sim_poss[pair_type] = sim_pos_p
                # ind_poss[pair_type] = ind_p

        if pair is None:
            pair = self.pair

        pair_poss, pair_negs = [], []
        # ind_pair_poss = []
        for pos_name in pair['pos'].split(' '):
            pair_poss.append(sim_poss[pos_name])
            # ind_pair_poss.append(ind_poss[pos_name])

        for neg_name in pair['neg'].split(' '):
            pair_negs.append(sim_negs[neg_name])

        pair_poss = torch.cat(pair_poss, 1) # B*P
        pair_negs = torch.cat(pair_negs, 1) # B*N

        # ind_pair_poss = torch.cat(ind_pair_poss, 1)
        #breakpoint()
        pair_poss_ = pair_poss.unsqueeze(2).repeat(1, 1, 1) # B*P*1
        #pair_negs_ = pair_negs.unsqueeze(1).repeat(1, topk_pos, 1) # B*P*N
        #pair_negs_ = pair_negs.unsqueeze(1).repeat(1, topk_pos * len(pair['pos'].split(' ')) , 1)
        pair_negs_ = pair_negs.unsqueeze(1).repeat(1, pair_poss_.shape[1] , 1)
        pair_all_ = torch.cat((pair_poss_, pair_negs_), 2) # B*P*(N+1)


        binary_zero_labels_ = torch.zeros_like(pair_all_)
        binary_zero_labels_[:, :, 0] = 1


        # class_ratio = target.bincount()[target]
        # if uncertainty is not None:
        #     uncertainty = uncertainty[ind_pair_poss].squeeze()
            # class_ratio = target.bincount()[target]
            # breakpoint()

        # breakpoint()

        loss = self.criterion(pair_all_.reshape(-1, pair_all_.size(2))/self.temp, binary_zero_labels_.reshape(-1, pair_all_.size(2)),
        reduction=reduction, beta=self.beta, uncertainty=unc, class_ratio=class_ratio, level=level, progress=progress, data_label= target)
        loss = loss.reshape(B, -1).mean(1)


        # pair_all = torch.cat((pair_poss, pair_negs), 1)
        # binary_zero_labels = torch.zeros_like(pair_all)
        # binary_zero_labels[:, :pair_poss.size(1)] = 1
        # loss2 = self.criterion(pair_all/self.temp, binary_zero_labels, reduction=reduction, beta=self.beta)
        # loss2 /= self.topk_pos

        if djr:
            return loss, feat_norm_stat, feat_norm_ratio_stat, feat_cos_stat
            
        return loss


class UnsupMetricLoss(nn.Module):

    def __init__(self, topk_pos=-1, topk_neg=-1, temp=1, eps=0., pair=None, loss_type=None, beta=None,
                 pos_sample_type=None, neg_sample_type=None, adapt_ce=None, pos_loss_type=None, neg_loss_type=None,
                 adapt_sample=None, sampling=None, p_margin=0, n_margin=0,
                 **kwargs):
        super(UnsupMetricLoss, self).__init__()
        self.pair = pair
        
        self.topk_pos = self.pair.get('topk_pos') or topk_pos
        self.topk_neg = self.pair.get('topk_neg') or topk_neg
        self.pos_sample_type = self.pair.get('pos_sample_type') or pos_sample_type
        self.neg_sample_type = self.pair.get('neg_sample_type') or neg_sample_type
        
        self.temp = self.pair.get('temp') or temp
        
        self.sims_dict = {}
        
        self.loss_type = self.pair.get('loss_type') or loss_type
        self.pos_loss_type = self.pair.get('pos_loss_type') or pos_loss_type 
        self.neg_loss_type = self.pair.get('neg_loss_type') or neg_loss_type 
        self.beta = self.pair.get('beta') or beta # weight for negative samples. 
        self.adapt_ce = self.pair.get('adapt_ce') or adapt_ce 
        self.adapt_sample = self.pair.get('adapt_sample') or adapt_sample 

        self.sampling = self.pair.get('sampling') or sampling
        self.p_margin = self.pair.get('p_margin') or p_margin
        self.n_margin = self.pair.get('n_margin') or n_margin

        self.criterion = MultiLabelCrossEntropyLoss(topk_pos=topk_pos, temp=temp, adapt_ce=self.adapt_ce, eps=eps,
                                                   p_margin=self.p_margin, n_margin=self.n_margin) 


    def __set_num_classes__(self, num_classes):
        self.num_classes = num_classes

    def __repr__(self):
        return "{}(topk_pos={}, topk_neg={}, temp={}, crit={}), pair={}, no sampleno)".format(
            type(self).__name__, self.topk_pos, self.topk_neg, self.temp, self.criterion, self.pair)

    def get_classwise_mask(self, target):
        B = target.size(0)
        classwise_mask = target.expand(B, B).eq(target.expand(B, B).T)
        return classwise_mask
    

    def get_topk_neg(self, sim, pos_mask=None, topk_neg=None, labels=None, easyneg = True):

        sim_neg = sim.clone()
        B = sim_neg.size(0)

        neg_loss_type = self.neg_loss_type if self.neg_loss_type else self.loss_type
        #if neg_loss_type == 'unsupervised':
        if 'unsupervised' in neg_loss_type:
            pos_mask = self.get_classwise_mask(torch.arange(B)).to(labels.device) # 1 only if the same sample
        else:
            pos_mask = self.get_classwise_mask(labels)


        if self.neg_sample_type:
            if self.neg_sample_type == 'debug':
                breakpoint()
                
            if self.neg_sample_type == 'all':
                # mining negative samples from all classes except the same instance
                sim_neg[torch.eye(B)==1] = -np.inf
                sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]
            elif self.neg_sample_type == 'intra_class':
                # mining negative samples from the same class
                classwise_mask = self.get_classwise_mask(labels)
                sim_neg[~classwise_mask] = -np.inf
                sim_neg[torch.eye(B)==1] = -np.inf
                # sim_neg[pos_mask==1] = -np.inf
                sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]

            elif self.pos_sample_type == 'center':
                sim_neg[pos_mask==1] = np.nan
                sim_neg = sim_neg.nanmean(1, keepdim=True).repeat(1, topk_neg)

            elif self.neg_sample_type == 'inter_class':
                # mining negative samples from the different class only
                classwise_mask = self.get_classwise_mask(labels)
                sim_neg[classwise_mask] = -np.inf
                # sim_neg[pos_mask==1] = -np.inf #This line may be redundant
                sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]

            elif self.neg_sample_type == 'easy':
                
                sim_neg[pos_mask==1] = np.inf
                sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=False)[0]


        elif 'unsupervised_neg' in neg_loss_type and easyneg:
            sim_neg[pos_mask==1] = np.inf
            sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=False)[0]


        else:
            sim_neg[pos_mask==1] = -np.inf
            sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]


        return sim_neg
    

    def get_topk_pos(self, sim, topk_pos=None, labels=None, uncertainty=None):

        sim_pos = sim.clone()

        B = sim.size(0)

        pos_loss_type = self.pos_loss_type if self.pos_loss_type else self.loss_type
        if pos_loss_type == 'unsupervised':
            pos_mask = self.get_classwise_mask(torch.arange(B)).to(labels.device)
        else:
            pos_mask = self.get_classwise_mask(labels)

        inds = None

        if self.pos_sample_type:
            if self.pos_sample_type == 'easy': # high sim
                sim_pos[pos_mask==0] = -np.inf
                sim_pos, inds = torch.topk(sim_pos, topk_pos, dim=1, largest=True)

            elif self.pos_sample_type == 'no_grad':
                sim_pos[pos_mask==0] = np.inf
                sim_pos, inds = torch.topk(sim_pos, topk_pos, dim=1, largest=False)
                sim_pos.fill_(1)

            elif self.pos_sample_type == 'center':
                sim_pos[pos_mask==0] = np.nan
                sim_pos = sim_pos.nanmean(1, keepdim=True).repeat(1, self.topk_pos)

            elif self.pos_sample_type == 'conf_center':
                # breakpoint()
                sim_pos[pos_mask==0] = np.nan
                confs = 1 - uncertainty
                confs_ = confs.unsqueeze(0).repeat(confs.size(0), 1)
                confs_[pos_mask==0] = np.nan
                # confs_.fill_diagonal_(1)
                confs_classmean = confs_.nanmean(1, keepdim=True)
                confs_norm = confs_/confs_classmean
                sim_pos = (confs_norm*sim_pos).nanmean(1, keepdim=True).repeat(1, self.topk_pos)
                # sim_pos2 = sim_pos.nanmean(1, keepdim=True).repeat(1, self.topk_pos)
                
            elif self.pos_sample_type == 'hard_center':
                sim_pos[pos_mask==0] = np.inf
                sim_pos, inds = torch.topk(sim_pos, topk_pos, dim=1, largest=False)
                sim_pos[sim_pos==np.inf] = np.nan

                sim_pos = sim_pos.nanmean(1, keepdim=True).repeat(1, self.topk_pos)

            elif self.pos_sample_type == 'random':
                sim_pos[pos_mask==0] = np.nan
                random_pos = torch.rand_like(sim_pos)
                random_pos[pos_mask==0] = -np.inf
                random_pos_inds = torch.topk(random_pos, topk_pos, dim=1, largest=True)[1]
                sim_pos = sim_pos.gather(1, random_pos_inds)

            elif self.pos_sample_type == 'positive':
                sim_pos[pos_mask==0] = np.inf
                sim_pos[sim_pos < 0] = np.inf
                # breakpoint()
                sim_pos, inds = torch.topk(sim_pos, topk_pos, dim=1, largest=False)

        
        else:
            sim_pos[pos_mask==0] = np.inf
            sim_pos, inds = torch.topk(sim_pos, topk_pos, dim=1, largest=False)
            # breakpoint()


        return sim_pos




    def forward(self, old_feat, new_feat, new_feat_aug, target, reduction=True, pair=None, topk_pos=None, topk_neg=None, 
    uncertainty=None, class_ratio=None, level=None, progress=None, name="loss1"):
        if old_feat.dim() > 2:
            old_feat = old_feat.squeeze(-1).squeeze(-1)
        if new_feat.dim() > 2:
            new_feat = new_feat.squeeze(-1).squeeze(-1)
        if new_feat_aug.dim() > 2:
            new_feat_aug = new_feat_aug.squeeze(-1).squeeze(-1)


        B, C = new_feat.size()    
        #B, C = old_feat.size()

        old_feat_ = F.normalize(old_feat, p=2, dim=1)
        new_feat_ = F.normalize(new_feat, p=2, dim=1)
        new_feat_aug_ = F.normalize(new_feat_aug, p=2, dim=1)

        sims = {}
        all_pair_types = set(self.pair.pos.split(' ') + self.pair.neg.split(' '))

        # sims['oo'] = torch.mm(old_feat_, old_feat_.t()) if 'oo' in all_pair_types else None
        sims['oo'] = torch.mm(old_feat_, old_feat_.t()) 
        sims['no'] = torch.mm(new_feat_, old_feat_.t()) if 'no' in all_pair_types else None
        sims['on'] = torch.mm(old_feat_, new_feat_.t()) if 'on' in all_pair_types else None
        sims['nn'] = torch.mm(new_feat_, new_feat_.t()) if 'nn' in all_pair_types else None
        sims['na'] = torch.mm(new_feat_, new_feat_aug_.t()) if 'nn' in all_pair_types else None
        sims['nd'] = torch.mm(new_feat_, (new_feat_.detach().clone().t())) if 'nd' in all_pair_types else None
        sims['dn'] = torch.mm(new_feat_.detach().clone(), new_feat_.t()) if 'dn' in all_pair_types else None
        sims['dd'] = torch.mm(new_feat_.detach().clone(), (new_feat_.detach().clone().t())) if 'dd' in all_pair_types else None



        loss = 0.

        sim_poss, sim_negs = {}, {}
        ind_poss = {}

        if topk_pos is None:
            topk_pos = self.topk_pos

        if topk_neg is None:
            topk_neg = self.topk_neg


        unc = None
        # conf = self.get_rel_confidence(sims['nn'], sims['oo'], labels=target)
        # print(f"conf: {conf.min()}/{conf.max()}/{conf.mean()} at level{level}, prog{progress}")

        neg_loss_type = self.neg_loss_type if self.neg_loss_type else self.loss_type


        for pair_type in all_pair_types:

            sim_poss[pair_type] = self.get_topk_pos(sims[pair_type], topk_pos=topk_pos, labels=target)
            sim_negs[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, labels=target)


        if pair is None:
            pair = self.pair

        pair_poss, pair_negs = [], []
        # ind_pair_poss = []
        for pos_name in pair['pos'].split(' '):
            pair_poss.append(sim_poss[pos_name])
            # ind_pair_poss.append(ind_poss[pos_name])

        for neg_name in pair['neg'].split(' '):
            pair_negs.append(sim_negs[neg_name])

        pair_poss = torch.cat(pair_poss, 1) # B*P
        pair_negs = torch.cat(pair_negs, 1) # B*N

        # ind_pair_poss = torch.cat(ind_pair_poss, 1)
        #breakpoint()
        pair_poss_ = pair_poss.unsqueeze(2).repeat(1, 1, 1) # B*P*1
        pair_negs_ = pair_negs.unsqueeze(1).repeat(1, pair_poss_.shape[1] , 1)
        pair_all_ = torch.cat((pair_poss_, pair_negs_), 2) # B*P*(N+1)


        binary_zero_labels_ = torch.zeros_like(pair_all_)
        binary_zero_labels_[:, :, 0] = 1


        # class_ratio = target.bincount()[target]
        # if uncertainty is not None:
        #     uncertainty = uncertainty[ind_pair_poss].squeeze()
            # class_ratio = target.bincount()[target]
            # breakpoint()

        # breakpoint()

        loss = self.criterion(pair_all_.reshape(-1, pair_all_.size(2))/self.temp, binary_zero_labels_.reshape(-1, pair_all_.size(2)),
        reduction=reduction, beta=self.beta, uncertainty=unc, class_ratio=class_ratio, level=level, progress=progress, data_label= target)
        loss = loss.reshape(B, -1).mean(1)


        # pair_all = torch.cat((pair_poss, pair_negs), 1)
        # binary_zero_labels = torch.zeros_like(pair_all)
        # binary_zero_labels[:, :pair_poss.size(1)] = 1
        # loss2 = self.criterion(pair_all/self.temp, binary_zero_labels, reduction=reduction, beta=self.beta)
        # loss2 /= self.topk_pos

        return loss


class MetricLoss_djr(MetricLoss):
    def __init__(self, topk_pos=-1, topk_neg=-1, temp=1, eps=0., pair=None, loss_type=None, beta=None,
                pos_sample_type=None, neg_sample_type=None, adapt_ce=None, pos_loss_type=None, neg_loss_type=None,
                adapt_sample=None, sampling=None, p_margin=0, n_margin=0,threshold=1,
                **kwargs):
        super(MetricLoss_djr, self).__init__(topk_pos=topk_pos, topk_neg=topk_neg, temp=temp, eps=eps, pair=pair, loss_type=loss_type, beta=beta,
                pos_sample_type=pos_sample_type, neg_sample_type=neg_sample_type, adapt_ce=adapt_ce, pos_loss_type=pos_loss_type, neg_loss_type=neg_loss_type,
                adapt_sample=adapt_sample, sampling=None, p_margin=0, n_margin=0,threshold=1,
                **kwargs)

        self.djr = kwargs['djr']
        #print(self.djr)


    # def get_topk_neg(self, sim, pos_mask=None, topk_neg=None, labels=None, easyneg = True, crj = None):

    #     sim_neg = sim.clone()
    #     B = sim_neg.size(0)

    #     neg_loss_type = self.neg_loss_type if self.neg_loss_type else self.loss_type
    #     #if neg_loss_type == 'unsupervised':
    #     if 'unsupervised' in neg_loss_type:
    #         pos_mask = self.get_classwise_mask(torch.arange(B)).to(labels.device) # 1 only if the same sample
    #     else:
    #         pos_mask = self.get_classwise_mask(labels)


    #     if self.neg_sample_type:
    #         if self.neg_sample_type == 'debug':
    #             breakpoint()
                
    #         if self.neg_sample_type == 'all':
    #             # mining negative samples from all classes except the same instance
    #             sim_neg[torch.eye(B)==1] = -np.inf
    #             sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]
    #         elif self.neg_sample_type == 'intra_class':
    #             # mining negative samples from the same class
    #             classwise_mask = self.get_classwise_mask(labels)
    #             sim_neg[~classwise_mask] = -np.inf
    #             sim_neg[torch.eye(B)==1] = -np.inf
    #             # sim_neg[pos_mask==1] = -np.inf
    #             sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]

    #         elif self.neg_sample_type == 'center':
    #             sim_neg[pos_mask==1] = np.nan
    #             sim_neg = sim_neg.nanmean(1, keepdim=True).repeat(1, topk_neg)


    #         elif self.neg_sample_type == 'hard_center':
    #             sim_neg[pos_mask==1] = -np.inf
    #             sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]

    #             breakpoint()

    #         elif self.neg_sample_type == 'inter_class':
    #             # mining negative samples from the different class only
    #             classwise_mask = self.get_classwise_mask(labels)
    #             sim_neg[classwise_mask] = -np.inf
    #             # sim_neg[pos_mask==1] = -np.inf #This line may be redundant
    #             sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]

    #         elif self.neg_sample_type == 'easy':
                
    #             sim_neg[pos_mask==1] = np.inf
    #             sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=False)[0]

    #         elif self.neg_sample_type == 'random':
    #             # raise NotImplementedError
    #             # breakpoint()
    #             sim_neg[pos_mask==1] = np.nan
    #             random_neg = torch.rand_like(sim_neg)
    #             random_neg[pos_mask==1] = -np.inf
    #             random_neg_inds = torch.topk(random_neg, min(topk_neg, B), dim=1, largest=True)[1]
    #             sim_neg = sim_neg.gather(1, random_neg_inds)

    #         # elif self.neg_sample_type == 'pos_cond':
    #         #     breakpoint()
    #     elif 'unsupervised_neg' in neg_loss_type and easyneg:
    #         sim_neg[pos_mask==1] = np.inf
    #         sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=False)[0]


    #     else:
    #         # sim_neg[pos_mask==1] = -np.inf
    #         # weight_matrix = (crj[labels][:, labels]).detach()
    #         # sim_neg = sim_neg * weight_matrix
    #         # sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]

    #         sim_neg[pos_mask==1] = -np.inf
    #         weight_matrix = (crj[labels][:, labels]).detach()
    #         sim_neg, sim_indices = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)
    #         weight_ = torch.gather(weight_matrix, 1, sim_indices)
    #         if torch.rand(1).item() < 0.0001:
    #             print(weight_.mean(1).squeeze())
    #         sim_neg = sim_neg * weight_


    #     return sim_neg


    @torch.no_grad()
    def cal_crj(self, nj_div_nr, Ompjj_div_Omprr = None, phij_div_phir = None, num_unique_labels = None, progress = None):
        #result[r][j] = nj/nr
        result = 1
        if self.djr.bal.sample > 0:
            result *= nj_div_nr
        if Ompjj_div_Omprr != None and self.djr.bal.prob > 0:
            result *= Ompjj_div_Omprr
        if phij_div_phir != None and self.djr.bal.featnorm > 0:
            result *= phij_div_phir
        
        # breakpoint()
        result = self.rescale_crj_nondiagonal_only(result, num_unique_labels=num_unique_labels, progress=progress)
        return result



    @torch.no_grad()
    def rescale_crj_nondiagonal_only(self, crj_raw, num_unique_labels, progress=None):
        # B = len(crj_raw)
        # # Extract the diagonal elements
        # diagonal = torch.diagonal(crj_raw)
        # saved_diagonal = diagonal.detach().clone()

        # Normalize the off-diagonal elements
        tensor_rescaled = crj_raw.detach().clone()  # Create a copy of the original tensor
        # tensor_rescaled.diagonal().zero_()
        

        # if self.djr.bal.norm_scale_crit == 'topk_neg':
        #     s = self.topk_neg
        # elif self.djr.bal.norm_scale_crit == 'num_unique_labels':
        #     s = num_unique_labels - 1
        # elif self.djr.bal.norm_scale_crit == 'one':
        #     s = 1

        # if self.djr.bal.norm_mode == 'L2':
        #     tensor_rescaled = tensor_rescaled / tensor_rescaled.norm(dim = 1, keepdim = True)
        #     tensor_rescaled = tensor_rescaled * ((s)**0.5)
        # elif self.djr.bal.norm_mode == 'sum':
        #     tensor_rescaled = tensor_rescaled / tensor_rescaled.sum(dim = 1, keepdim = True)
        #     tensor_rescaled = tensor_rescaled * ((s))
        if self.djr.bal.norm_mode == 'sqrt':
            tensor_rescaled = (crj_raw**0.5)
        elif self.djr.bal.norm_mode == 'sqrt4':
            tensor_rescaled = (crj_raw**0.25)
        elif self.djr.bal.norm_mode == 'sqrt8':
            tensor_rescaled = (crj_raw**0.125)
        elif self.djr.bal.norm_mode == 'sqrt100':
            tensor_rescaled = (crj_raw**0.01)
        elif self.djr.bal.norm_mode == 'sq':
            tensor_rescaled = (crj_raw**2)
        elif self.djr.bal.norm_mode == 'clamp':
            tensor_rescaled = tensor_rescaled.clamp(0.5, 2) #tensor_rescaled.clamp(0.5, 2)
        elif self.djr.bal.norm_mode == 'interpolate':
            tensor_rescaled = tensor_rescaled * 0.5 + 1 * 0.5
        elif self.djr.bal.norm_mode == 'sqrt_prog':
            tensor_rescaled = crj_raw ** max(0, 0.5 - 5*progress)
        elif self.djr.bal.norm_mode == 'sqrt_prog_rev':
            tensor_rescaled = crj_raw ** min(0.5, 2*progress)
        elif self.djr.bal.norm_mode == 'rev_sqrt4':
            tensor_rescaled = (1/crj_raw)**0.25
        elif self.djr.bal.norm_mode == 'rev_sqrt':
            tensor_rescaled = (1/crj_raw)**0.5
        elif self.djr.bal.norm_mode == 'rev':
            tensor_rescaled = (1/crj_raw)
        elif self.djr.bal.norm_mode == 'softclamp':
            gamma = self.djr.bal.gamma
            tensor_rescaled =  ( crj_raw / (crj_raw + gamma)) * (1 + gamma)
        elif self.djr.bal.norm_mode == 'default':
            # breakpoint()
            tensor_rescaled = crj_raw
        else:
            print("Not specified bal_norm_mode. Use 'default' to raw crj value.")
            raise ValueError

        # elif self.djr.bal.norm_mode == 'sum_sqrt':
        #     tensor_rescaled = tensor_rescaled / tensor_rescaled.sum(dim = 1, keepdim = True)
        #     tensor_rescaled = tensor_rescaled * ((s))
        #     tensor_rescaled = tensor_rescaled**0.5

        # tensor_rescaled.diagonal().copy_(saved_diagonal)
        return tensor_rescaled

    def forward(self, old_feat, new_feat, target, reduction=True, pair=None, topk_pos=None, topk_neg=None, 
    uncertainty=None, class_ratio=None, level=None, progress=None, name="loss1", djr =False, class_major = None, class_minor = None, djr_dict = None):


        # with torch.no_grad():
        #     unique_labels= target.unique()
        #     self.unique_labels = unique_labels
        #     new_feat_norm = new_feat.norm(dim = 1,keepdim = True)
        #     avg_feat_norm = get_avg_data_per_class(new_feat_norm, target, self.num_classes, unique_labels = unique_labels)
            
            #phij_div_phir = cal_att_j_div_r(avg_feat_norm) 
            #crj = self.cal_crj(nj_div_nr, Ompjj_div_Omprr,phij_div_phir, num_unique_labels=len(self.unique_labels), progress=progress)
        if old_feat.dim() > 2:
            old_feat = old_feat.squeeze(-1).squeeze(-1)
            if new_feat.dim() > 2:
                new_feat = new_feat.squeeze(-1).squeeze(-1)
            if 'uniquecat' in self.loss_type:
                bin_count = torch.bincount(target)
                unique_classes = torch.unique(target)
                unique_classes = unique_classes[bin_count[unique_classes] == 1]
                is_unique = (target.unsqueeze(1) == unique_classes)
                is_unique = (is_unique.sum(dim=1))>0
                if 'put' in self.loss_type:
                    pass
                else:
                    old_feat = torch.cat((new_feat, old_feat[is_unique]), dim = 0)
                    concat_target = torch.cat((target, target[is_unique]), dim = 0)

            if 'concat' in self.loss_type:
                #new_feat = torch.cat((new_feat, old_feat), dim = 0)
                old_feat = torch.cat((new_feat, old_feat), dim = 0)
                concat_target = torch.cat((target, target), dim = 0)
                #target = torch.cat((target, target), dim = 0)



            B, C = new_feat.size()    
            #B, C = old_feat.size()

            old_feat_ = F.normalize(old_feat, p=2, dim=1)
            new_feat_ = F.normalize(new_feat, p=2, dim=1)

            sims = {}
            all_pair_types = set(self.pair.pos.split(' ') + self.pair.neg.split(' '))

            # sims['oo'] = torch.mm(old_feat_, old_feat_.t()) if 'oo' in all_pair_types else None
            sims['oo'] = torch.mm(old_feat_, old_feat_.t()) 
            sims['no'] = torch.mm(new_feat_, old_feat_.t()) if 'no' in all_pair_types else None
            sims['on'] = torch.mm(old_feat_, new_feat_.t()) if 'on' in all_pair_types else None
            sims['nn'] = torch.mm(new_feat_, new_feat_.t()) if 'nn' in all_pair_types else None
            sims['nd'] = torch.mm(new_feat_, (new_feat_.detach().clone().t())) if 'nd' in all_pair_types else None
            sims['dn'] = torch.mm(new_feat_.detach().clone(), new_feat_.t()) if 'dn' in all_pair_types else None
            sims['dd'] = torch.mm(new_feat_.detach().clone(), (new_feat_.detach().clone().t())) if 'dd' in all_pair_types else None


            if djr:
                with torch.no_grad():
                    p_zy = djr_dict['averaged_probabilities'].detach().clone()
                    for k in range(len(p_zy)):
                        p_zy[k][k] = 1 - p_zy[k][k]


                    unique_labels= target.unique()
                    new_feat_norm = (new_feat.detach().clone()).norm(dim = 1,keepdim = True)
                    avg_feat_norm = get_avg_data_per_class(new_feat_norm.detach().clone(), target.detach().clone(), self.num_classes, unique_labels = unique_labels)
                    avg_feat_norm = avg_feat_norm.t()
                    # major_feat_norm = {cl:avg_feat_norm[cl] for cl in class_major}
                    # minor_feat_norm = {cl:avg_feat_norm[cl] for cl in class_minor}
                    # feat_norm_stat = {}
                    # feat_norm_stat['major_self'] = major_feat_norm
                    # feat_norm_stat['minor_self'] = minor_feat_norm
                    # feat_norm_stat['major_self_minall'] = min_dict(major_feat_norm)
                    # feat_norm_stat['minor_self_minall'] = min_dict(minor_feat_norm)

                    
                    #feat_norm_ratio = avg_feat_norm / (avg_feat_norm.t())
                    #feat_norm_ratio_stat = get_major_minor_stat(class_major, class_minor, feat_norm_ratio)



                    avg_class_sim_per_sample = self.get_avg_class_sim_per_sample(sims['nn'].detach().clone(), target.detach().clone(), self.num_classes, unique_labels = unique_labels.detach().clone())
                    #breakpoint()
                    #avg_class_sim_per_class = get_avg_data_per_class(avg_class_sim_per_sample,  target, self.num_classes, unique_labels = unique_labels)
                    #feat_cos_stat = get_major_minor_stat(class_major, class_minor, avg_class_sim_per_class)

                    #avg_sims = get_avg_data_per_class(sims['nn'] , target, self.num_classes, unique_labels = unique_labels)
                    #breakpoint()

                    samplewise_deviation_bound_dict = {}
                    samplewise_deviation_bound_dict['indep_P_yz'] = []
                    samplewise_deviation_bound_dict['indep_sampleratio'] = []
                    samplewise_deviation_bound_dict['indep_phi_norm'] = []
                    samplewise_deviation_bound_dict['indep_Sr_Sj'] = []
                    samplewise_deviation_bound_dict['Sr'] = []
                    samplewise_deviation_bound_dict['Sj'] = []
                    samplewise_deviation_bound_dict['effective_Sj'] = []
                    samplewise_deviation_bound_dict['Sr-Sj'] = []
                    samplewise_deviation_bound_dict['Sr-effective_Sj'] = []
                    samplewise_deviation_bound_dict['samplewise_deviation_bound'] = []
                    samplewise_deviation_bound_dict['threshold'] = []
                    samplewise_deviation_bound_dict['Dx_<_threshold'] = []
                    samplewise_deviation_bound_dict['Dx_threshold_ratio'] = []
                    samplewise_deviation_bound_dict['effective_threshold'] = []
                    samplewise_deviation_bound_dict['Dx_<_effective_threshold'] = []
                    samplewise_deviation_bound_dict['Dx<1'] = []
                    samplewise_deviation_bound_dict['Dx<0.5'] = []
                    samplewise_deviation_bound_dict['Dx<0.1'] = []
                    samplewise_deviation_bound_dict['Dx<0.05'] = []
                    samplewise_deviation_bound_dict['Dx_effective_threshold_ratio'] = []

                    Dx_minorclasses = {x:[] for x in class_minor}
                    Dx_majorclasses = {x:[] for x in class_major}
                    
                    samplewise_deviation_bound_dict['minorclass_min'] = []
                    samplewise_deviation_bound_dict['minorclass_mean'] = []
                    #samplewise_deviation_bound_dict['minorclass_max'] = []
                    
                    samplewise_deviation_bound_dict['majorclass_min'] = []
                    samplewise_deviation_bound_dict['majorclass_mean'] = []
                    #samplewise_deviation_bound_dict['majorclass_max'] = []


                    def ratio_for_sample(sampledata, samplelabel):
                        data_at_label = sampledata[samplelabel]
                        summation = sampledata.sum()
                        result = data_at_label/(summation - data_at_label)
                        return result
                    for Dx_idx, Dx_label in enumerate(target):
                        sample_P_yz = p_zy[Dx_label].detach().clone()
                        samplewise_deviation_bound_dict['indep_P_yz'].append(ratio_for_sample(sample_P_yz, Dx_label))

                        sample_phi_norm = (avg_feat_norm.detach().clone()).squeeze()
                        samplewise_deviation_bound_dict['indep_phi_norm'].append(ratio_for_sample(sample_phi_norm, Dx_label))

                        sample_nr_nj = djr_dict['num_samples'].detach().clone().squeeze()
                        samplewise_deviation_bound_dict['indep_sampleratio'].append(ratio_for_sample(sample_nr_nj, Dx_label))
                                                
                        sample_Sr_Sj = avg_class_sim_per_sample[Dx_idx].detach().clone()
                        S_except_target_class = torch.cat((sample_Sr_Sj[:Dx_label], sample_Sr_Sj[Dx_label+1:]), dim = 0)
                        samplewise_deviation_bound_dict['Sr'].append(sample_Sr_Sj[Dx_label])
                        samplewise_deviation_bound_dict['Sj'].append(S_except_target_class.mean())
                        samplewise_deviation_bound_dict['effective_Sj'].append(S_except_target_class.sum()/(len(unique_labels) - 1))
                        samplewise_deviation_bound_dict['Sr-Sj'].append(sample_Sr_Sj[Dx_label] - S_except_target_class.mean())
                        samplewise_deviation_bound_dict['Sr-effective_Sj'].append(sample_Sr_Sj[Dx_label] - S_except_target_class.sum()/(len(unique_labels) - 1))
                        
                        samplewise_deviation_bound_dict['indep_Sr_Sj'].append(ratio_for_sample(sample_Sr_Sj, Dx_label))
                        
                        sample_Dx = sample_P_yz*sample_phi_norm*sample_nr_nj*sample_Sr_Sj
                        this_Dx = ratio_for_sample(sample_Dx, Dx_label)
                        samplewise_deviation_bound_dict['samplewise_deviation_bound'].append(this_Dx)
                        #breakpoint()
                        if Dx_label.item() in class_minor:
                            Dx_minorclasses[Dx_label.item()].append(this_Dx)
                        elif Dx_label.item() in class_major:
                            Dx_majorclasses[Dx_label.item()].append(this_Dx)


                        #measure the threshold of the deviation bound

                        for_upper_bound = sample_P_yz * sample_phi_norm * sample_nr_nj
                        target_class_data = for_upper_bound[Dx_label]
                        except_target_class_data = torch.cat((for_upper_bound[:Dx_label], for_upper_bound[Dx_label+1:]), dim = 0)
                        minimum_value_except_target_class = (1/except_target_class_data).min()
                        threshold = target_class_data * minimum_value_except_target_class /(self.num_classes - 1)
                        samplewise_deviation_bound_dict['threshold'].append(threshold)
                        samplewise_deviation_bound_dict['Dx_<_threshold'].append(1.0*(this_Dx < threshold))
                        samplewise_deviation_bound_dict['Dx<1'].append(1.0*(this_Dx < 1))
                        samplewise_deviation_bound_dict['Dx<0.5'].append(1.0*(this_Dx < 0.5))
                        samplewise_deviation_bound_dict['Dx<0.1'].append(1.0*(this_Dx < 0.1))
                        samplewise_deviation_bound_dict['Dx<0.05'].append(1.0*(this_Dx < 0.05))


                        samplewise_deviation_bound_dict['Dx_threshold_ratio'].append(this_Dx/threshold)
                        effective_threshold = target_class_data * minimum_value_except_target_class / (len(unique_labels) - 1)
                        samplewise_deviation_bound_dict['effective_threshold'].append(effective_threshold)
                        samplewise_deviation_bound_dict['Dx_<_effective_threshold'].append(1.0*(this_Dx < effective_threshold))
                        samplewise_deviation_bound_dict['Dx_effective_threshold_ratio'].append(this_Dx/effective_threshold)

                    for key in Dx_minorclasses.keys():
                        samplewise_deviation_bound_dict['minorclass_min'].append(min_dict(Dx_minorclasses[key]))
                        samplewise_deviation_bound_dict['minorclass_mean'].append(mean_dict(Dx_minorclasses[key]))
                        #samplewise_deviation_bound_dict['minorclass_max'].append(max_dict(samplewise_deviation_bound_dict['minorclasses'][key]))
                    

                    for key in Dx_majorclasses.keys():
                        samplewise_deviation_bound_dict['majorclass_min'].append(min_dict(Dx_majorclasses[key]))
                        samplewise_deviation_bound_dict['majorclass_mean'].append(mean_dict(Dx_majorclasses[key]))
                        #samplewise_deviation_bound_dict['majorclass_max'].append(max_dict(samplewise_deviation_bound_dict['majorclasses'][key]))

                    Dx_dict = {}
                    for key in samplewise_deviation_bound_dict.keys():
                        Dx_dict[key + "/min"] = min_dict(samplewise_deviation_bound_dict[key])
                        Dx_dict[key + "/mean"] = mean_dict(samplewise_deviation_bound_dict[key])
                #breakpoint()



            loss = 0.

            sim_poss, sim_negs = {}, {}
            ind_poss = {}

            if topk_pos is None:
                topk_pos = self.topk_pos

            if topk_neg is None:
                topk_neg = self.topk_neg


            unc = None
            # conf = self.get_rel_confidence(sims['nn'], sims['oo'], labels=target)
            # print(f"conf: {conf.min()}/{conf.max()}/{conf.mean()} at level{level}, prog{progress}")

            neg_loss_type = self.neg_loss_type if self.neg_loss_type else self.loss_type

            if 'cat' in self.loss_type:
                if pair is None:
                    pair = self.pair
                if sims['no']==None:
                    sims['no'] = torch.mm(new_feat_, old_feat_.t())

                unc = None
                for pair_type in pair['pos'].split(' '):
                    if 'put' in self.loss_type:
                        matrix_unique = torch.diag(is_unique)
                        modified = sims[pair_type]
                        modified[matrix_unique] = (sims['no'])[matrix_unique]
                        sim_poss[pair_type] = self.get_topk_pos(modified, topk_pos=topk_pos, labels=target)
                        
                        #sim_poss[pair_type][is_unique] = old_feat[is_unzique]
                    else:
                        sim_poss[pair_type] = self.get_topk_pos(sims[pair_type], topk_pos=topk_pos, labels=concat_target)
                        if 'nocut' in self.loss_type:
                            pass
                        else:
                            sim_poss[pair_type] = sim_poss[pair_type][:B]
                for pair_type in pair['neg'].split(' '):
                    if 'nocut' in self.loss_type:
                        sim_negs[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, labels=concat_target)
                    else:
                        sim_negs[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, labels=target)
            else:
                for pair_type in all_pair_types:
                    unc = self.get_uncertainty(sims[pair_type], labels=target)
                    #if neg_loss_type == 'unsupervised':
                    if 'unsupervised_neg' in neg_loss_type:
                        sim_poss[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, labels=target)
                        sim_negs[pair_type] = ((sim_poss[pair_type]).min(dim = 1, keepdim = True)[0]).detach().clone()
                        if 'unsupervised_neg1' in neg_loss_type:
                            pass
                        elif 'unsupervised_neg2' in neg_loss_type:
                            sim_negs[pair_type] *= 0
                        elif 'unsupervised_neg3' in neg_loss_type:
                            sim_negs[pair_type] = sim_negs[pair_type] * 0 - 1
                        elif 'unsupervised_neg4' in neg_loss_type:
                            sim_negs[pair_type] = (sim_negs[pair_type] * 0.9 + (- 1) * 0.1)
                        elif 'unsupervised_neg5' in neg_loss_type:
                            sim_negs[pair_type] = sim_negs[pair_type] * 0 + 1
                        elif 'unsupervised_negpos' in neg_loss_type:
                            sim_negs[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, labels=target, easyneg = False)
                            #print(((sim_poss[pair_type] - sim_negs[pair_type])**2).sum())
                        
                    else:
                        
                        sim_poss[pair_type] = self.get_topk_pos(sims[pair_type], topk_pos=topk_pos, labels=target)
                        #sim_negs[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, labels=target)
                        sim_negs[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, topk_pos=topk_pos, labels=target)
                    # , sim_pos=sim_poss[pair_type])
                    # sim_pos_p, ind_p = self.get_topk_pos(sims[pair_type], topk_pos=topk_pos, labels=target, uncertainty=uncertainty)
                    # sim_poss[pair_type] = sim_pos_p
                    # ind_poss[pair_type] = ind_p

            if pair is None:
                pair = self.pair

            pair_poss, pair_negs = [], []
            # ind_pair_poss = []
            for pos_name in pair['pos'].split(' '):
                pair_poss.append(sim_poss[pos_name])
                # ind_pair_poss.append(ind_poss[pos_name])

            for neg_name in pair['neg'].split(' '):
                pair_negs.append(sim_negs[neg_name])

            pair_poss = torch.cat(pair_poss, 1) # B*P
            pair_negs = torch.cat(pair_negs, 1) # B*N

            # ind_pair_poss = torch.cat(ind_pair_poss, 1)
            #breakpoint()
            pair_poss_ = pair_poss.unsqueeze(2).repeat(1, 1, 1) # B*P*1
            #pair_negs_ = pair_negs.unsqueeze(1).repeat(1, topk_pos, 1) # B*P*N
            #pair_negs_ = pair_negs.unsqueeze(1).repeat(1, topk_pos * len(pair['pos'].split(' ')) , 1)
            pair_negs_ = pair_negs.unsqueeze(1).repeat(1, pair_poss_.shape[1] , 1)
            pair_all_ = torch.cat((pair_poss_, pair_negs_), 2) # B*P*(N+1)


            binary_zero_labels_ = torch.zeros_like(pair_all_)
            binary_zero_labels_[:, :, 0] = 1


            # class_ratio = target.bincount()[target]
            # if uncertainty is not None:
            #     uncertainty = uncertainty[ind_pair_poss].squeeze()
                # class_ratio = target.bincount()[target]
                # breakpoint()

            # breakpoint()

            loss = self.criterion(pair_all_.reshape(-1, pair_all_.size(2))/self.temp, binary_zero_labels_.reshape(-1, pair_all_.size(2)),
            reduction=reduction, beta=self.beta, uncertainty=unc, class_ratio=class_ratio, level=level, progress=progress, data_label= target)
            loss = loss.reshape(B, -1).mean(1)


            # pair_all = torch.cat((pair_poss, pair_negs), 1)
            # binary_zero_labels = torch.zeros_like(pair_all)
            # binary_zero_labels[:, :pair_poss.size(1)] = 1
            # loss2 = self.criterion(pair_all/self.temp, binary_zero_labels, reduction=reduction, beta=self.beta)
            # loss2 /= self.topk_pos

            if djr:
                return loss, Dx_dict
                
            return loss
        # breakpoint()
        # if old_feat.dim() > 2:
        #     old_feat = old_feat.squeeze(-1).squeeze(-1)
        # if new_feat.dim() > 2:
        #     new_feat = new_feat.squeeze(-1).squeeze(-1)



        # B, C = new_feat.size()    
        # #B, C = old_feat.size()

        # old_feat_ = F.normalize(old_feat, p=2, dim=1)
        # new_feat_ = F.normalize(new_feat, p=2, dim=1)

        # sims = {}
        # all_pair_types = set(self.pair.pos.split(' ') + self.pair.neg.split(' '))

        # # sims['oo'] = torch.mm(old_feat_, old_feat_.t()) if 'oo' in all_pair_types else None
        # sims['oo'] = torch.mm(old_feat_, old_feat_.t()) 
        # sims['no'] = torch.mm(new_feat_, old_feat_.t()) if 'no' in all_pair_types else None
        # sims['on'] = torch.mm(old_feat_, new_feat_.t()) if 'on' in all_pair_types else None
        # sims['nn'] = torch.mm(new_feat_, new_feat_.t()) if 'nn' in all_pair_types else None
        # sims['nd'] = torch.mm(new_feat_, (new_feat_.detach().clone().t())) if 'nd' in all_pair_types else None
        # sims['dn'] = torch.mm(new_feat_.detach().clone(), new_feat_.t()) if 'dn' in all_pair_types else None
        # sims['dd'] = torch.mm(new_feat_.detach().clone(), (new_feat_.detach().clone().t())) if 'dd' in all_pair_types else None




        






        # loss = 0.

        # sim_poss, sim_negs = {}, {}
        # ind_poss = {}

        # if topk_pos is None:
        #     topk_pos = self.topk_pos

        # if topk_neg is None:
        #     topk_neg = self.topk_neg


        # unc = None
        # # conf = self.get_rel_confidence(sims['nn'], sims['oo'], labels=target)
        # # print(f"conf: {conf.min()}/{conf.max()}/{conf.mean()} at level{level}, prog{progress}")


        # neg_loss_type = self.neg_loss_type if self.neg_loss_type else self.loss_type

        # if 'cat' in self.loss_type:
        #     if pair is None:
        #         pair = self.pair
        #     if sims['no']==None:
        #         sims['no'] = torch.mm(new_feat_, old_feat_.t())

        #     unc = None
        #     for pair_type in pair['pos'].split(' '):
        #         if 'put' in self.loss_type:
        #             matrix_unique = torch.diag(is_unique)
        #             modified = sims[pair_type]
        #             modified[matrix_unique] = (sims['no'])[matrix_unique]
        #             sim_poss[pair_type] = self.get_topk_pos(modified, topk_pos=topk_pos, labels=target)
                    
        #             #sim_poss[pair_type][is_unique] = old_feat[is_unique]
        #         else:
        #             sim_poss[pair_type] = self.get_topk_pos(sims[pair_type], topk_pos=topk_pos, labels=concat_target)
        #             if 'nocut' in self.loss_type:
        #                 pass
        #             else:
        #                 sim_poss[pair_type] = sim_poss[pair_type][:B]
        #     for pair_type in pair['neg'].split(' '):
        #         if 'nocut' in self.loss_type:
        #             sim_negs[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, labels=concat_target)
        #         else:
        #             sim_negs[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, labels=target)
        # else:
        #     for pair_type in all_pair_types:
        #         unc = self.get_uncertainty(sims[pair_type], labels=target)
        #         #if neg_loss_type == 'unsupervised':
        #         if 'unsupervised_neg' in neg_loss_type:
        #             sim_poss[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, labels=target)
        #             sim_negs[pair_type] = ((sim_poss[pair_type]).min(dim = 1, keepdim = True)[0]).detach().clone()
        #             if 'unsupervised_neg1' in neg_loss_type:
        #                 pass
        #             elif 'unsupervised_neg2' in neg_loss_type:
        #                 sim_negs[pair_type] *= 0
        #             elif 'unsupervised_neg3' in neg_loss_type:
        #                 sim_negs[pair_type] = sim_negs[pair_type] * 0 - 1
        #             elif 'unsupervised_neg4' in neg_loss_type:
        #                 sim_negs[pair_type] = (sim_negs[pair_type] * 0.9 + (- 1) * 0.1)
        #             elif 'unsupervised_neg5' in neg_loss_type:
        #                 sim_negs[pair_type] = sim_negs[pair_type] * 0 + 1
        #             elif 'unsupervised_negpos' in neg_loss_type:
        #                 sim_negs[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, labels=target, easyneg = False)
        #                 #print(((sim_poss[pair_type] - sim_negs[pair_type])**2).sum())
                    
        #         else:
                    
        #             sim_poss[pair_type] = self.get_topk_pos(sims[pair_type], topk_pos=topk_pos, labels=target)
        #             sim_negs[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, labels=target)#, crj=crj)
        #         # , sim_pos=sim_poss[pair_type])
        #         # sim_pos_p, ind_p = self.get_topk_pos(sims[pair_type], topk_pos=topk_pos, labels=target, uncertainty=uncertainty)
        #         # sim_poss[pair_type] = sim_pos_p
        #         # ind_poss[pair_type] = ind_p

        # if pair is None:
        #     pair = self.pair

        # pair_poss, pair_negs = [], []
        # # ind_pair_poss = []
        # for pos_name in pair['pos'].split(' '):
        #     pair_poss.append(sim_poss[pos_name])
        #     # ind_pair_poss.append(ind_poss[pos_name])

        # for neg_name in pair['neg'].split(' '):
        #     pair_negs.append(sim_negs[neg_name])

        # pair_poss = torch.cat(pair_poss, 1) # B*P
        # pair_negs = torch.cat(pair_negs, 1) # B*N

        # # ind_pair_poss = torch.cat(ind_pair_poss, 1)
        # #breakpoint()
        # pair_poss_ = pair_poss.unsqueeze(2).repeat(1, 1, 1) # B*P*1
        # #pair_negs_ = pair_negs.unsqueeze(1).repeat(1, topk_pos, 1) # B*P*N
        # #pair_negs_ = pair_negs.unsqueeze(1).repeat(1, topk_pos * len(pair['pos'].split(' ')) , 1)
        # pair_negs_ = pair_negs.unsqueeze(1).repeat(1, pair_poss_.shape[1] , 1)
        # pair_all_ = torch.cat((pair_poss_, pair_negs_), 2) # B*P*(N+1)


        # binary_zero_labels_ = torch.zeros_like(pair_all_)
        # binary_zero_labels_[:, :, 0] = 1


        # # class_ratio = target.bincount()[target]
        # # if uncertainty is not None:
        # #     uncertainty = uncertainty[ind_pair_poss].squeeze()
        #     # class_ratio = target.bincount()[target]
        #     # breakpoint()

        # # breakpoint()

        # loss = self.criterion(pair_all_.reshape(-1, pair_all_.size(2))/self.temp, binary_zero_labels_.reshape(-1, pair_all_.size(2)),
        # reduction=reduction, beta=self.beta, uncertainty=unc, class_ratio=class_ratio, level=level, progress=progress, data_label= target)
        # loss = loss.reshape(B, -1).mean(1)


        # # pair_all = torch.cat((pair_poss, pair_negs), 1)
        # # binary_zero_labels = torch.zeros_like(pair_all)
        # # binary_zero_labels[:, :pair_poss.size(1)] = 1
        # # loss2 = self.criterion(pair_all/self.temp, binary_zero_labels, reduction=reduction, beta=self.beta)
        # # loss2 /= self.topk_pos
            
        # return loss


class MetricLossSubset(MetricLoss):


    def forward(self, old_feat, new_feat, target, reduction=True, pair=None, topk_pos=None, topk_neg=None, 
    uncertainty=None, class_ratio=None, level=None, progress=None, sampling_range=1., name="loss1"):
        if old_feat.dim() > 2:
            old_feat = old_feat.squeeze(-1).squeeze(-1)
        if new_feat.dim() > 2:
            new_feat = new_feat.squeeze(-1).squeeze(-1)

        labels = target


        B, C = new_feat.size()    
        #B, C = old_feat.size()

        old_feat_ = F.normalize(old_feat, p=2, dim=1)
        new_feat_ = F.normalize(new_feat, p=2, dim=1)

        sims = {}
        all_pair_types = set(self.pair.pos.split(' ') + self.pair.neg.split(' '))

        # sims['oo'] = torch.mm(old_feat_, old_feat_.t()) if 'oo' in all_pair_types else None
        sims['oo'] = torch.mm(old_feat_, old_feat_.t()) 
        sims['no'] = torch.mm(new_feat_, old_feat_.t()) if 'no' in all_pair_types else None
        sims['on'] = torch.mm(old_feat_, new_feat_.t()) if 'on' in all_pair_types else None
        sims['nn'] = torch.mm(new_feat_, new_feat_.t()) if 'nn' in all_pair_types else None
        sims['nd'] = torch.mm(new_feat_, (new_feat_.detach().clone().t())) if 'nd' in all_pair_types else None
        sims['dn'] = torch.mm(new_feat_.detach().clone(), new_feat_.t()) if 'dn' in all_pair_types else None
        sims['dd'] = torch.mm(new_feat_.detach().clone(), (new_feat_.detach().clone().t())) if 'dd' in all_pair_types else None


        # cut = False
        # if cut:
        if True:
            # sim_nn = sims['nn']
            P = int(B*sampling_range)
            gather_indices = (torch.arange(P).unsqueeze(0).repeat(B, 1) + torch.arange(B).unsqueeze(1)).to(new_feat).long()%B

            for sim_key in sims:
                if sims[sim_key] is not None:
                    sims[sim_key] = torch.gather(sims[sim_key], 1, gather_indices.to(sims[sim_key]).long())

            # gather_labels = torch.gather(labels, 1, gather_indices)
            # pos_mask = (labels[gather_indices] == labels.unsqueeze(1))
            # neg_mask = (labels[gather_indices] != labels.unsqueeze(1))

            


        loss = 0.

        sim_poss, sim_negs = {}, {}
        ind_poss = {}

        if topk_pos is None:
            topk_pos = self.topk_pos

        if topk_neg is None:
            topk_neg = self.topk_neg

        unc = None


        neg_loss_type = self.neg_loss_type if self.neg_loss_type else self.loss_type


        for pair_type in all_pair_types:
            # unc = self.get_uncertainty(sims[pair_type], labels=target)
            #if neg_loss_type == 'unsupervised':
            # sim_poss[pair_type] = self.get_topk_pos(sims[pair_type], topk_pos=topk_pos, labels=target)
            sim_poss[pair_type] = self.get_topk_pos(sims[pair_type], topk_pos=topk_pos, labels=target, gather_indices=gather_indices)
            sim_negs[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, labels=target, gather_indices=gather_indices)

        if pair is None:
            pair = self.pair

        pair_poss, pair_negs = [], []
        # ind_pair_poss = []
        for pos_name in pair['pos'].split(' '):
            pair_poss.append(sim_poss[pos_name])
            # ind_pair_poss.append(ind_poss[pos_name])

        for neg_name in pair['neg'].split(' '):
            pair_negs.append(sim_negs[neg_name])

        pair_poss = torch.cat(pair_poss, 1) # B*P
        pair_negs = torch.cat(pair_negs, 1) # B*N

        # ind_pair_poss = torch.cat(ind_pair_poss, 1)
        #breakpoint()
        pair_poss_ = pair_poss.unsqueeze(2).repeat(1, 1, 1) # B*P*1
        #pair_negs_ = pair_negs.unsqueeze(1).repeat(1, topk_pos, 1) # B*P*N
        #pair_negs_ = pair_negs.unsqueeze(1).repeat(1, topk_pos * len(pair['pos'].split(' ')) , 1)
        pair_negs_ = pair_negs.unsqueeze(1).repeat(1, pair_poss_.shape[1] , 1)
        pair_all_ = torch.cat((pair_poss_, pair_negs_), 2) # B*P*(N+1)


        binary_zero_labels_ = torch.zeros_like(pair_all_)
        binary_zero_labels_[:, :, 0] = 1


        loss = self.criterion(pair_all_.reshape(-1, pair_all_.size(2))/self.temp, binary_zero_labels_.reshape(-1, pair_all_.size(2)),
        reduction=reduction, beta=self.beta, uncertainty=unc, class_ratio=class_ratio, level=level, progress=progress, data_label= target)
        loss = loss.reshape(B, -1).mean(1)

            
        return loss


    def get_topk_neg(self, sim, pos_mask=None, topk_neg=None, labels=None, gather_indices=None, easyneg = True):

        sim_neg = sim.clone()
        B = sim_neg.size(0)

        neg_loss_type = self.neg_loss_type if self.neg_loss_type else self.loss_type
        #if neg_loss_type == 'unsupervised':
        if 'unsupervised' in neg_loss_type:
            # pos_mask = self.get_classwise_mask(torch.arange(B)).to(labels.device) # 1 only if the same sample
            pos_mask = (gather_indices == torch.arange(B).unsqueeze(1).to(gather_indices.device))
        else:
            # pos_mask = self.get_classwise_mask(labels)
            pos_mask = (labels[gather_indices] == labels.unsqueeze(1))


        if self.neg_sample_type:
                
            if self.neg_sample_type == 'all':
                # mining negative samples from all classes except the same instance
                sim_neg[torch.eye(B)==1] = -np.inf
                sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]

            elif self.pos_sample_type == 'center':
                sim_neg[pos_mask==1] = np.nan
                sim_neg = sim_neg.nanmean(1, keepdim=True).repeat(1, topk_neg)


            elif self.neg_sample_type == 'easy':
                
                sim_neg[pos_mask==1] = np.inf
                sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=False)[0]




        else:
            sim_neg[pos_mask==1] = -np.inf
            sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]


        return sim_neg
    

    def get_topk_pos(self, sim, topk_pos=None, labels=None, gather_indices=None, uncertainty=None):

        sim_pos = sim.clone()

        B = sim.size(0)

        pos_loss_type = self.pos_loss_type if self.pos_loss_type else self.loss_type
        if pos_loss_type == 'unsupervised':
            # pos_mask = self.get_classwise_mask(torch.arange(B)).to(labels.device)
            pos_mask = (gather_indices == torch.arange(B).unsqueeze(1).to(gather_indices.device))
        else:
            pos_mask = (labels[gather_indices] == labels.unsqueeze(1))
            # pos_mask = self.get_classwise_mask(labels)

        inds = None

        if self.pos_sample_type:
            if self.pos_sample_type == 'easy': # high sim
                sim_pos[pos_mask==0] = -np.inf
                sim_pos, inds = torch.topk(sim_pos, topk_pos, dim=1, largest=True)


            elif self.pos_sample_type == 'center':
                sim_pos[pos_mask==0] = np.nan
                sim_pos = sim_pos.nanmean(1, keepdim=True).repeat(1, self.topk_pos)


            elif self.pos_sample_type == 'random':
                sim_pos[pos_mask==0] = np.nan
                random_pos = torch.rand_like(sim_pos)
                random_pos[pos_mask==0] = -np.inf
                random_pos_inds = torch.topk(random_pos, topk_pos, dim=1, largest=True)[1]
                sim_pos = sim_pos.gather(1, random_pos_inds)

        
        else:
            sim_pos[pos_mask==0] = np.inf
            sim_pos, inds = torch.topk(sim_pos, topk_pos, dim=1, largest=False)
            # breakpoint()


        return sim_pos


class MetricLoss2(MetricLoss):
    def __init__(self, topk_pos=-1, topk_neg=-1, temp=1, eps=0., pair=None, loss_type=None, beta=None,
                 pos_sample_type=None, neg_sample_type=None, adapt_ce=None, pos_loss_type=None, neg_loss_type=None,
                 adapt_sample=None,
                 **kwargs):
        super(MetricLoss2, self).__init__(topk_pos=topk_pos, topk_neg=topk_neg, temp=temp, eps=eps, pair=pair, loss_type=loss_type, beta=beta,
                 pos_sample_type=pos_sample_type, neg_sample_type=neg_sample_type, adapt_ce=adapt_ce, pos_loss_type=pos_loss_type, neg_loss_type=neg_loss_type,
                 adapt_sample=adapt_sample,
                 **kwargs)
        
    def create_hook(self, val, opt_sim, mode):
        def myhook(grad):
            saved_val = val
            saved_opt_sim = opt_sim
            saved_mode = mode


            #Don't use "grad[:,0] = torch.where(saved_val[:,0]<0.9, grad[:,0] , 0)" for ml2.1, ml2.3(It is like using torch.where(saved_val[:,0]<1, grad[:,0] , 0)), and use it for ml 2.0, 2.2, 2.6
            if saved_mode in ['2.0', '2.2', '2.6']:
                grad[:,0] = torch.where(saved_val[:,0]<0.9, grad[:,0] , 0)
            
            elif saved_mode in ['2.4', '2.5']:
                #for ml 2.4, ml2.5 
                grad[:,0] = torch.where(saved_val[:,0]<0.99, grad[:,0] , 0)

            elif saved_mode in ['2.7', '2.8', '2.9']:
                for l in range(1, grad.shape[1]):
                    grad[:,0] = torch.where(saved_val[:,0]>saved_val[:,l], grad[:,0] , 0)

            elif saved_mode in ['2.1', '2.3', 'sus', 'sup', 'else']:
                pass                
                
            else:
                raise ValueError

            for l in range(1, grad.shape[1]):
                grad[:,l] = torch.where(saved_val[:,l] > saved_opt_sim, grad[:,l] , 0)
  
            return grad
        return myhook

    def forward(self, old_feat, new_feat, target, reduction=True, pair=None, topk_pos=None, topk_neg=None, 
    uncertainty=None, class_ratio=None, level=None, progress=None, name="loss1", mode = ""):
        if old_feat.dim() > 2:
            old_feat = old_feat.squeeze(-1).squeeze(-1)
        if new_feat.dim() > 2:
            new_feat = new_feat.squeeze(-1).squeeze(-1)
            
        B, C = old_feat.size()

        old_feat_ = F.normalize(old_feat, p=2, dim=1)
        new_feat_ = F.normalize(new_feat, p=2, dim=1)

        sims = {}

        all_pair_types = set(self.pair.pos.split(' ') + self.pair.neg.split(' '))

        sims['oo'] = torch.mm(old_feat_, old_feat_.t()) if 'oo' in all_pair_types else None
        sims['no'] = torch.mm(new_feat_, old_feat_.t()) if 'no' in all_pair_types else None
        sims['on'] = torch.mm(old_feat_, new_feat_.t()) if 'on' in all_pair_types else None
        sims['nn'] = torch.mm(new_feat_, new_feat_.t()) if 'nn' in all_pair_types else None
        sims['nd'] = torch.mm(new_feat_, (new_feat_.detach().clone().t())) if 'nd' in all_pair_types else None
        sims['dn'] = torch.mm(new_feat_.detach().clone(), new_feat_.t()) if 'dn' in all_pair_types else None
        sims['dd'] = torch.mm(new_feat_.detach().clone(), (new_feat_.detach().clone().t())) if 'dd' in all_pair_types else None


        loss = 0.

        all_sims = []
        sim_poss, sim_negs = {}, {}
        ind_poss = {}

        if topk_pos is None:
            topk_pos = self.topk_pos

        if topk_neg is None:
            topk_neg = self.topk_neg


        unc = None
        for pair_type in all_pair_types:
            unc = self.get_uncertainty(sims[pair_type], labels=target)

            all_sims.append(sims[pair_type].view(len(sims[pair_type]), -1))

            sim_poss[pair_type] = self.get_topk_pos(sims[pair_type], topk_pos=topk_pos, labels=target)
            sim_negs[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, labels=target)
            # , sim_pos=sim_poss[pair_type])
            # sim_pos_p, ind_p = self.get_topk_pos(sims[pair_type], topk_pos=topk_pos, labels=target, uncertainty=uncertainty)
            # sim_poss[pair_type] = sim_pos_p
            # ind_poss[pair_type] = ind_p

        if pair is None:
            pair = self.pair


        



        pair_poss, pair_negs = [], []
        # ind_pair_poss = []
        for pos_name in pair['pos'].split(' '):
            pair_poss.append(sim_poss[pos_name])
            # ind_pair_poss.append(ind_poss[pos_name])

        for neg_name in pair['neg'].split(' '):
            pair_negs.append(sim_negs[neg_name])

        pair_poss = torch.cat(pair_poss, 1) # B*P
        pair_negs = torch.cat(pair_negs, 1) # B*N

        # ind_pair_poss = torch.cat(ind_pair_poss, 1)

        pair_poss_ = pair_poss.unsqueeze(2).repeat(1, 1, 1) # B*P*1
        pair_negs_ = pair_negs.unsqueeze(1).repeat(1, topk_pos, 1) # B*P*N
        pair_all_ = torch.cat((pair_poss_, pair_negs_), 2) # B*P*(N+1)

        # breakpoint()

        binary_zero_labels_ = torch.zeros_like(pair_all_)
        binary_zero_labels_[:, :, 0] = 1


        # class_ratio = target.bincount()[target]
        # if uncertainty is not None:
        #     uncertainty = uncertainty[ind_pair_poss].squeeze()
            # class_ratio = target.bincount()[target]
            # breakpoint()

        pos_mean_sim = pair_poss_.reshape(B,-1).mean(1)
        neg_mean_sim = pair_negs_.reshape(B,-1).mean(1)

        contra_input = pair_all_.reshape(-1, pair_all_.size(2))
        loss = self.criterion(contra_input/self.temp, binary_zero_labels_.reshape(-1, pair_all_.size(2)),
        reduction=reduction, beta=self.beta, uncertainty=unc, class_ratio=class_ratio, level=level, progress=progress, data_label= target)
        loss = loss.reshape(B, -1).mean(1)

        
        #print(contra_input[0])

        # pair_all = torch.cat((pair_poss, pair_negs), 1)
        # binary_zero_labels = torch.zeros_like(pair_all)
        # binary_zero_labels[:, :pair_poss.size(1)] = 1
        # loss2 = self.criterion(pair_all/self.temp, binary_zero_labels, reduction=reduction, beta=self.beta)
        # loss2 /= self.topk_pos




        # Define for CIFAR 100. for another class, change the num_classes
        all_sims = torch.cat(all_sims, 1)
        num_classes = 100
        opt_sim  = -1/(num_classes-1)
        #for ml2.0, ml2.1, ml2.4
        if mode in ['2.0','2.1','2.4', '2.9']:
            dist_opt_sim = (opt_sim - all_sims)**2


        elif mode in ['2.2','2.3','2.5', '2.8', '2.6', '2.7', 'sus', 'sup']:
            dist_opt_sim = (pair_negs_ - opt_sim)**2
        elif mode in ['2.6', '2.7', 'sus', 'sup', 'else']: 
            dist_opt_sim = pair_negs_ * 0
        
        else:
            print("ERROR")
            print(mode)
            print(type(mode))
            raise ValueError

        dist_opt_sim = dist_opt_sim.mean(1)

        #for ml 2.6
        #dist_opt_sim *= 0
        #dist_opt_sim[all_sims > 0.9]
        contra_input.register_hook(self.create_hook(contra_input, opt_sim, mode))

        #
            
        return loss, dist_opt_sim, pos_mean_sim, neg_mean_sim



class MetricLoss_rel(MetricLoss):
    def __init__(self, topk_pos=-1, topk_neg=-1, temp=1, eps=0., pair=None, loss_type=None, beta=None,
                 pos_sample_type=None, neg_sample_type=None, adapt_ce=None, pos_loss_type=None, neg_loss_type=None,
                 adapt_sample=None,
                 **kwargs):
        super(MetricLoss_rel, self).__init__(topk_pos=topk_pos, topk_neg=topk_neg, temp=temp, eps=eps, pair=pair, loss_type=loss_type, beta=beta,
                 pos_sample_type=pos_sample_type, neg_sample_type=neg_sample_type, adapt_ce=adapt_ce, pos_loss_type=pos_loss_type, neg_loss_type=neg_loss_type,
                 adapt_sample=adapt_sample,
                 **kwargs)

    def get_topk_neg(self, criterion, sim, pos_mask=None, topk_neg=None, labels=None, mode = None):


        sim_neg = criterion.clone()
        #sim_neg = sim.clone()
        # if 
        B = sim_neg.size(0)

        neg_loss_type = self.neg_loss_type if self.neg_loss_type else self.loss_type
        if 'unsupervised' in neg_loss_type:
        #if neg_loss_type == 'unsupervised':
            if 'unsuprel' in mode:
                #print("YES in topkneg")
                pass
            # elif 'unsupinvrel' in mode:
            #     sim_pos = -(criterion.clone())
            else:
                sim_neg = sim.clone()
            pos_mask = self.get_classwise_mask(torch.arange(B)).to(labels.device) # 1 only if the same sample
        else:
            if 'supnegorg' in mode:
                sim_neg = sim.clone()
            pos_mask = self.get_classwise_mask(labels)


        if self.neg_sample_type:
            raise ValueError
            if self.neg_sample_type == 'debug':
                breakpoint()
                
            if self.neg_sample_type == 'all':
                # mining negative samples from all classes except the same instance
                sim_neg[torch.eye(B)==1] = -np.inf
                sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]
            elif self.neg_sample_type == 'intra_class':
                # mining negative samples from the same class
                classwise_mask = self.get_classwise_mask(labels)
                sim_neg[~classwise_mask] = -np.inf
                sim_neg[torch.eye(B)==1] = -np.inf
                # sim_neg[pos_mask==1] = -np.inf
                sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]

            elif self.pos_sample_type == 'center':
                sim_neg[pos_mask==1] = np.nan
                sim_neg = sim_neg.nanmean(1, keepdim=True).repeat(1, topk_neg)


            elif self.pos_sample_type == 'hard_center':
                sim_neg[pos_mask==1] = -np.inf
                sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]

                breakpoint()

            elif self.neg_sample_type == 'inter_class':
                # mining negative samples from the different class only
                classwise_mask = self.get_classwise_mask(labels)
                sim_neg[classwise_mask] = -np.inf
                # sim_neg[pos_mask==1] = -np.inf #This line may be redundant
                sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]

            elif self.neg_sample_type == 'easy':
                
                sim_neg[pos_mask==1] = np.inf
                sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=False)[0]

            elif self.neg_sample_type == 'random':
                # raise NotImplementedError
                # breakpoint()
                sim_neg[pos_mask==1] = np.nan
                random_neg = torch.rand_like(sim_neg)
                random_neg[pos_mask==1] = -np.inf
                random_neg_inds = torch.topk(random_neg, min(topk_neg, B), dim=1, largest=True)[1]
                sim_neg = sim_neg.gather(1, random_neg_inds)

            # elif self.neg_sample_type == 'pos_cond':
            #     breakpoint()

        elif 'unsupervised_neg' in neg_loss_type:
            sim_neg[pos_mask==1] = np.inf
            #sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)
            topk_values, topk_indices = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=False)
            selected_elements = torch.gather(sim, 1, topk_indices)

        else:
            sim_neg[pos_mask==1] = -np.inf
            #sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)
            topk_values, topk_indices = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)
            selected_elements = torch.gather(sim, 1, topk_indices)
            #selected_elements = sim[torch.arange(sim_neg.size(0)), topk_indices.squeeze(1)]
            #selected_elements = selected_elements.reshape(sim.shape)


        return selected_elements
    

    def get_topk_pos(self, criterion, sim, topk_pos=None, labels=None, uncertainty=None, mode = None):
        sim_pos = criterion.clone()
        #sim_pos = sim.clone()

        B = sim.size(0)

        pos_loss_type = self.pos_loss_type if self.pos_loss_type else self.loss_type
        if pos_loss_type == 'unsupervised':
            # if 'unsuprel' in mode:
            #     pass
            # elif 'unsupinvrel' in mode:
            #     sim_pos = -(criterion.clone())
            # else:
            #     sim_pos = sim.clone()
            pos_mask = self.get_classwise_mask(torch.arange(B)).to(labels.device)
        else:
            pos_mask = self.get_classwise_mask(labels)

        inds = None

        if self.pos_sample_type:
            raise ValueError

            if self.pos_sample_type == 'easy': # high sim
                sim_pos[pos_mask==0] = -np.inf
                sim_pos, inds = torch.topk(sim_pos, topk_pos, dim=1, largest=True)

            elif self.pos_sample_type == 'no_grad':
                sim_pos[pos_mask==0] = np.inf
                sim_pos, inds = torch.topk(sim_pos, topk_pos, dim=1, largest=False)
                sim_pos.fill_(1)

            elif self.pos_sample_type == 'center':
                sim_pos[pos_mask==0] = np.nan
                sim_pos = sim_pos.nanmean(1, keepdim=True).repeat(1, self.topk_pos)

            elif self.pos_sample_type == 'conf_center':
                # breakpoint()
                sim_pos[pos_mask==0] = np.nan
                confs = 1 - uncertainty
                confs_ = confs.unsqueeze(0).repeat(confs.size(0), 1)
                confs_[pos_mask==0] = np.nan
                # confs_.fill_diagonal_(1)
                confs_classmean = confs_.nanmean(1, keepdim=True)
                confs_norm = confs_/confs_classmean
                sim_pos = (confs_norm*sim_pos).nanmean(1, keepdim=True).repeat(1, self.topk_pos)
                # sim_pos2 = sim_pos.nanmean(1, keepdim=True).repeat(1, self.topk_pos)
                
            elif self.pos_sample_type == 'hard_center':
                sim_pos[pos_mask==0] = np.inf
                sim_pos, inds = torch.topk(sim_pos, topk_pos, dim=1, largest=False)
                sim_pos[sim_pos==np.inf] = np.nan

                sim_pos = sim_pos.nanmean(1, keepdim=True).repeat(1, self.topk_pos)

            elif self.pos_sample_type == 'random':
                sim_pos[pos_mask==0] = np.nan
                random_pos = torch.rand_like(sim_pos)
                random_pos[pos_mask==0] = -np.inf
                random_pos_inds = torch.topk(random_pos, topk_pos, dim=1, largest=True)[1]
                sim_pos = sim_pos.gather(1, random_pos_inds)

            elif self.pos_sample_type == 'positive':
                sim_pos[pos_mask==0] = np.inf
                sim_pos[sim_pos < 0] = np.inf
                # breakpoint()
                sim_pos, inds = torch.topk(sim_pos, topk_pos, dim=1, largest=False)

        
        else:
            sim_pos[pos_mask==0] = np.inf
            topk_values, topk_indices = torch.topk(sim_pos, topk_pos, dim=1, largest=False)
            selected_elements = torch.gather(sim, 1, topk_indices)
            #selected_elements = sim[torch.arange(sim_pos.size(0)), topk_indices.squeeze(1)]
            #selected_elements = selected_elements.reshape(sim_pos.size(0), -1)


        return selected_elements



    def forward(self, old_feat, new_feat, target, reduction=True, pair=None, topk_pos=None, topk_neg=None, 
    uncertainty=None, class_ratio=None, level=None, progress=None, name="loss1", mode = None):
        #print("metric_rel forward mode: ",mode)
        if old_feat.dim() > 2:
            old_feat = old_feat.squeeze(-1).squeeze(-1)
        if new_feat.dim() > 2:
            new_feat = new_feat.squeeze(-1).squeeze(-1)
            
        B, C = old_feat.size()

        old_feat_ = F.normalize(old_feat, p=2, dim=1)
        new_feat_ = F.normalize(new_feat, p=2, dim=1)

        sims = {}

        all_pair_types = set(self.pair.pos.split(' ') + self.pair.neg.split(' '))

        # sims['oo'] = torch.mm(old_feat_, old_feat_.t()) if 'oo' in all_pair_types else None
        sims['oo'] = torch.mm(old_feat_, old_feat_.t()) 
        sims['no'] = torch.mm(new_feat_, old_feat_.t()) if 'no' in all_pair_types else None
        sims['on'] = torch.mm(old_feat_, new_feat_.t()) if 'on' in all_pair_types else None
        sims['nn'] = torch.mm(new_feat_, new_feat_.t()) if 'nn' in all_pair_types else None
        sims['nd'] = torch.mm(new_feat_, (new_feat_.detach().clone().t())) if 'nd' in all_pair_types else None
        sims['dn'] = torch.mm(new_feat_.detach().clone(), new_feat_.t()) if 'dn' in all_pair_types else None
        sims['dd'] = torch.mm(new_feat_.detach().clone(), (new_feat_.detach().clone().t())) if 'dd' in all_pair_types else None


        loss = 0.

        sim_poss, sim_negs = {}, {}
        ind_poss = {}

        if topk_pos is None:
            topk_pos = self.topk_pos

        if topk_neg is None:
            topk_neg = self.topk_neg


        unc = None
        # conf = self.get_rel_confidence(sims['nn'], sims['oo'], labels=target)
        # print(f"conf: {conf.min()}/{conf.max()}/{conf.mean()} at level{level}, prog{progress}")

        for pair_type in all_pair_types:
            unc = self.get_uncertainty(sims[pair_type], labels=target)

            #rel = (sims[pair_type]/sims['oo'])
            # if 'exprel' in mode:
            #     rel = (sims[pair_type] + 1) * torch.exp(rel - 1)
            # else:
            #     is_initial = ((rel - 1)**2).mean() < 1e-10
            #     if is_initial:
            #         rel = sims[pair_type]
            # rel = rel.detach().clone()

            rel_diff = (sims[pair_type] - sims['oo'])
            is_initial = ((rel_diff - 0)**2).mean() < 1e-10
            if is_initial:
                rel_pos, rel_neg = sims[pair_type], sims[pair_type]
            else:
                rel_0 = rel_diff / (1 - sims['oo'])
                rel_1 = rel_diff / (sims['oo'] + 1)
                if 'v2_raw' in mode:
                    rel_pos, rel_neg = rel_diff, rel_diff
                elif 'v2_0' in mode:
                    rel_pos, rel_neg = rel_0, rel_0
                    if 'v2_0.5' in mode:
                        neg_loss_type = self.neg_loss_type if self.neg_loss_type else self.loss_type
                        if neg_loss_type == 'unsupervised':
                            rel_neg = rel_1
                elif 'v2_1' in mode:
                    rel_pos, rel_neg = rel_0, rel_1
                    if 'v2_1.5' in mode:
                        neg_loss_type = self.neg_loss_type if self.neg_loss_type else self.loss_type
                        if neg_loss_type == 'unsupervised':
                            rel_neg = rel_0

                elif 'v2_2' in mode:
                    rel_pos, rel_neg = rel_1, rel_0
                    if 'v2_2.5' in mode:
                        neg_loss_type = self.neg_loss_type if self.neg_loss_type else self.loss_type
                        if neg_loss_type == 'unsupervised':
                            rel_neg = rel_1
                elif 'v2_3' in mode:
                    rel_pos, rel_neg = rel_1, rel_1
                    if 'v2_3.5' in mode:
                        neg_loss_type = self.neg_loss_type if self.neg_loss_type else self.loss_type
                        if neg_loss_type == 'unsupervised':
                            rel_neg = rel_0

            rel_pos, rel_neg = rel_pos.detach().clone(), rel_neg.detach().clone()


            sim_poss[pair_type] = self.get_topk_pos(rel_pos, sims[pair_type], topk_pos=topk_pos, labels=target, mode = mode)
            sim_negs[pair_type] = self.get_topk_neg(rel_neg, sims[pair_type], topk_neg=topk_neg, labels=target, mode = mode)
            # , sim_pos=sim_poss[pair_type])
            # sim_pos_p, ind_p = self.get_topk_pos(sims[pair_type], topk_pos=topk_pos, labels=target, uncertainty=uncertainty)
            # sim_poss[pair_type] = sim_pos_p
            # ind_poss[pair_type] = ind_p

        if pair is None:
            pair = self.pair

        pair_poss, pair_negs = [], []
        # ind_pair_poss = []
        for pos_name in pair['pos'].split(' '):
            pair_poss.append(sim_poss[pos_name])
            # ind_pair_poss.append(ind_poss[pos_name])

        for neg_name in pair['neg'].split(' '):
            pair_negs.append(sim_negs[neg_name])

        pair_poss = torch.cat(pair_poss, 1) # B*P
        pair_negs = torch.cat(pair_negs, 1) # B*N

        # ind_pair_poss = torch.cat(ind_pair_poss, 1)

        pair_poss_ = pair_poss.unsqueeze(2).repeat(1, 1, 1) # B*P*1
        pair_negs_ = pair_negs.unsqueeze(1).repeat(1, topk_pos, 1) # B*P*N
        pair_all_ = torch.cat((pair_poss_, pair_negs_), 2) # B*P*(N+1)

        # breakpoint()

        binary_zero_labels_ = torch.zeros_like(pair_all_)
        binary_zero_labels_[:, :, 0] = 1


        # class_ratio = target.bincount()[target]
        # if uncertainty is not None:
        #     uncertainty = uncertainty[ind_pair_poss].squeeze()
            # class_ratio = target.bincount()[target]
            # breakpoint()

        # breakpoint()

        loss = self.criterion(pair_all_.reshape(-1, pair_all_.size(2))/self.temp, binary_zero_labels_.reshape(-1, pair_all_.size(2)),
        reduction=reduction, beta=self.beta, uncertainty=unc, class_ratio=class_ratio, level=level, progress=progress, data_label= target)
        loss = loss.reshape(B, -1).mean(1)


        # pair_all = torch.cat((pair_poss, pair_negs), 1)
        # binary_zero_labels = torch.zeros_like(pair_all)
        # binary_zero_labels[:, :pair_poss.size(1)] = 1
        # loss2 = self.criterion(pair_all/self.temp, binary_zero_labels, reduction=reduction, beta=self.beta)
        # loss2 /= self.topk_pos
            
        return loss







class TripletLoss(MetricLoss):
    def __init__(self, topk_pos=-1, topk_neg=-1, temp=1, eps=0., pair=None, loss_type=None, beta=None,
                 pos_sample_type=None, neg_sample_type=None, adapt_ce=None, pos_loss_type=None, neg_loss_type=None,
                 adapt_sample=None,
                 **kwargs):
        super(TripletLoss, self).__init__(topk_pos=topk_pos, topk_neg=topk_neg, temp=temp, eps=eps, pair=pair, loss_type=loss_type, beta=beta,
                 pos_sample_type=pos_sample_type, neg_sample_type=neg_sample_type, adapt_ce=adapt_ce, pos_loss_type=pos_loss_type, neg_loss_type=neg_loss_type,
                 adapt_sample=adapt_sample,
                 **kwargs)
        
    def repeat_reshape(self, in_x, in_y):
        x = in_x.repeat(1, len(in_x))
        x = x.reshape(in_x.shape[0], -1, in_x.shape[1])

        y = in_y.repeat(len(in_y), 1)
        y = y.reshape(in_y.shape[0], -1, in_y.shape[1])

        return x,y
    
    def diff_feats(self, in_x, in_y):
        x,y = self.repeat_reshape(in_x,in_y)
        diff = ((x-y)**2).sum(dim = -1)
        return diff

    def forward(self, old_feat, new_feat, target, reduction=True, pair=None, topk_pos=None, topk_neg=None, 
    uncertainty=None, class_ratio=None, level=None, progress=None, name="loss1", pos_weight = None, neg_weight = None, threshold = None):
        if old_feat.dim() > 2:
            old_feat = old_feat.squeeze(-1).squeeze(-1)
        if new_feat.dim() > 2:
            new_feat = new_feat.squeeze(-1).squeeze(-1)
            
        B, C = old_feat.size()

        old_feat_ = F.normalize(old_feat, p=2, dim=1)
        new_feat_ = F.normalize(new_feat, p=2, dim=1)

        sims = {}

        all_pair_types = set(self.pair.pos.split(' ') + self.pair.neg.split(' '))





        
        

        sims['oo'] = self.diff_feats(old_feat_, old_feat_) if 'oo' in all_pair_types else None
        sims['no'] = self.diff_feats(new_feat_, old_feat_) if 'no' in all_pair_types else None
        sims['on'] = self.diff_feats(old_feat_, new_feat_) if 'on' in all_pair_types else None
        sims['nn'] = self.diff_feats(new_feat_, new_feat_) if 'nn' in all_pair_types else None
        sims['nd'] = self.diff_feats(new_feat_, (new_feat_.detach().clone())) if 'nd' in all_pair_types else None
        sims['dn'] = self.diff_feats(new_feat_.detach().clone(), new_feat_) if 'dn' in all_pair_types else None
        sims['dd'] = self.diff_feats(new_feat_.detach().clone(), (new_feat_.detach().clone())) if 'dd' in all_pair_types else None


        loss = 0.

        sim_poss, sim_negs = {}, {}
        ind_poss = {}

        if topk_pos is None:
            topk_pos = self.topk_pos

        if topk_neg is None:
            topk_neg = self.topk_neg


        unc = None
        for pair_type in all_pair_types:
            unc = self.get_uncertainty(sims[pair_type], labels=target)

            sim_poss[pair_type] = -self.get_topk_pos(-sims[pair_type], topk_pos=topk_pos, labels=target)
            sim_negs[pair_type] = -self.get_topk_neg(-sims[pair_type], topk_neg=topk_neg, labels=target)
            # , sim_pos=sim_poss[pair_type])
            # sim_pos_p, ind_p = self.get_topk_pos(sims[pair_type], topk_pos=topk_pos, labels=target, uncertainty=uncertainty)
            # sim_poss[pair_type] = sim_pos_p
            # ind_poss[pair_type] = ind_p

        if pair is None:
            pair = self.pair

        pair_poss, pair_negs = {}, {}
        triplet_losses = []

        assert(len(pair['pos'].split(' ')) == len(pair['neg'].split(' ')))
        #1-by-1 matching

        # ind_pair_poss = []
        for pos_name, neg_name in zip(pair['pos'].split(' '), pair['neg'].split(' ')):
            pair_poss[pos_name] = (sim_poss[pos_name].mean(dim = 1))
            pair_negs[neg_name] = (sim_negs[neg_name].mean(dim = 1))
            # triplet_loss_ = pos_weight * pair_poss[pos_name] - neg_weight * pair_negs[neg_name] + threshold
            # triplet_loss_ = (torch.where(triplet_loss_ > 0, triplet_loss_, 0)).unsqueeze(dim = 1)
            triplet_loss_ = pos_weight * pair_poss[pos_name] - neg_weight * pair_negs[neg_name] 
            triplet_loss_ = (torch.where(triplet_loss_ > -threshold , triplet_loss_, 0)).unsqueeze(dim = 1)
            triplet_losses.append(triplet_loss_)

        #breakpoint()
        triplet_loss = (torch.cat(triplet_losses, 1)).mean(dim = 1)
        

        # pair_poss = torch.cat(pair_poss, 1) # B*P
        # pair_negs = torch.cat(pair_negs, 1) # B*N


        # ind_pair_poss = torch.cat(ind_pair_poss, 1)

        # pair_poss_ = pair_poss.unsqueeze(2).repeat(1, 1, 1) # B*P*1
        # pair_negs_ = pair_negs.unsqueeze(1).repeat(1, topk_pos, 1) # B*P*N
        # pair_all_ = torch.cat((pair_poss_, pair_negs_), 2) # B*P*(N+1)

        # # breakpoint()

        # binary_zero_labels_ = torch.zeros_like(pair_all_)
        # binary_zero_labels_[:, :, 0] = 1


        # # class_ratio = target.bincount()[target]
        # # if uncertainty is not None:
        # #     uncertainty = uncertainty[ind_pair_poss].squeeze()
        #     # class_ratio = target.bincount()[target]
        #     # breakpoint()

        # # breakpoint()

        # loss = self.criterion(pair_all_.reshape(-1, pair_all_.size(2))/self.temp, binary_zero_labels_.reshape(-1, pair_all_.size(2)),
        # reduction=reduction, beta=self.beta, uncertainty=unc, class_ratio=class_ratio, level=level, progress=progress, data_label= target)
        # loss = loss.reshape(B, -1).mean(1)


        # pair_all = torch.cat((pair_poss, pair_negs), 1)
        # binary_zero_labels = torch.zeros_like(pair_all)
        # binary_zero_labels[:, :pair_poss.size(1)] = 1
        # loss2 = self.criterion(pair_all/self.temp, binary_zero_labels, reduction=reduction, beta=self.beta)
        # loss2 /= self.topk_pos
            
        return triplet_loss, pair_poss, pair_negs





def KL_u_p_loss(outputs):
    # KL(u||p)
    num_classes = outputs.size(1)
    uniform_tensors = torch.ones(outputs.size())
    uniform_dists = torch.autograd.Variable(uniform_tensors / num_classes).cuda()
    instance_losses = F.kl_div(F.log_softmax(outputs, dim=1), uniform_dists, reduction='none').sum(dim=1)
    return instance_losses

'''
class MetricLoss2(nn.Module):

    def __init__(self, topk_pos=-1, topk_neg=-1, temp=1, eps=0., pairs=None, margin=0., **kwargs):
        super(MetricLoss, self).__init__()
        self.criterion = MultiLabelCrossEntropyLoss(topk_pos=topk_pos) 
        # if ce['name'] == 'multi_ce':
        #     self.criterion = MultiLabelCrossEntropyLoss(topk_pos=topk_pos, **ce) 
        # else:
        #     raise ValueError

        self.topk_pos = topk_pos
        self.topk_neg = topk_neg
        self.temp = temp
        self.margin = margin
        self.pairs = pairs
        self.sims_dict = {}
        # print("self.pairs:",self.pairs)

    def __repr__(self):
        return "{}(topk_pos={}, topk_neg={}, temp={}, crit={}), pairs={})".format(
            type(self).__name__, self.topk_pos, self.topk_neg, self.temp, self.criterion, self.pairs,)

    def get_classwise_mask(self, target):
        B = target.size(0)
        classwise_mask = target.expand(B, B).eq(target.expand(B, B).T)
        return classwise_mask

    def forward(self, old_feat, new_feat, target, name="loss1"):
        # if old_feat.dim() > 2:
        #     old_feat = old_feat.squeeze(-1).squeeze(-1)
        # if feat.dim() > 2:
        #     feat = feat.squeeze(-1).squeeze(-1)
        # if new_feat.dim() > 2:
        #     new_feat = new_feat.squeeze(-1).squeeze(-1)
            
        P, B, C = old_feat.size()
        classwise_mask = self.get_classwise_mask(target)

        old_feat_ = F.normalize(old_feat, p=2, dim=1)
        new_feat_ = F.normalize(new_feat, p=2, dim=1)

        sim_oo = torch.bmm(old_feat_, old_feat_.t()) # P,B,C x P,B,C => P,B,B
        sim_no = torch.bmm(new_feat_, old_feat_.t())
        sim_nn = torch.bmm(new_feat_, new_feat_.t())
        
        loss = 0.

        sim_poss, sim_negs = {}, {}

        sim_no_neg = sim_no.clone()
        sim_no_neg[classwise_mask==1] = -np.inf
        sim_no_neg = torch.topk(sim_no_neg, min(self.topk_neg, B), dim=1, largest=True)[0]
        sim_negs['no'] = sim_no_neg
        
        sim_nn_neg = sim_nn.clone()
        sim_nn_neg[classwise_mask==1] = -np.inf
        sim_nn_neg = torch.topk(sim_nn_neg, min(self.topk_neg, B), dim=1, largest=True)[0]
        sim_negs['nn'] = sim_nn_neg

        sim_oo_neg = sim_oo.clone()
        sim_oo_neg[classwise_mask==1] = -np.inf
        sim_oo_neg = torch.topk(sim_oo_neg, min(self.topk_neg, B), dim=1, largest=True)[0]
        sim_negs['oo'] = sim_oo_neg


        sim_no_pos = sim_no.clone()
        sim_no_pos[classwise_mask==0] = np.inf
        sim_no_pos = torch.topk(sim_no_pos, self.topk_pos, dim=1, largest=False)[0]
        sim_poss['no'] = sim_no_pos
        
        sim_nn_pos = sim_nn.clone()
        sim_nn_pos[classwise_mask==0] = np.inf
        sim_nn_pos = torch.topk(sim_nn_pos, self.topk_pos, dim=1, largest=False)[0]
        sim_poss['nn'] = sim_nn_pos

        sim_oo_pos = sim_nn.clone()
        sim_oo_pos[classwise_mask==0] = np.inf
        sim_oo_pos = torch.topk(sim_oo_pos, self.topk_pos, dim=1, largest=False)[0]
        sim_poss['oo'] = sim_oo_pos

        # for _, pair in self.pairs:
        if True:

            pair_poss, pair_negs = [], []
            # for pos_name in pair['pos'].split(' '):
            for pos_name in self.pairs['pos']:
                # print(pos_name)
                pair_poss.append(sim_poss[pos_name])

            for neg_name in self.pairs['neg']:
                # print(neg_name)
                pair_negs.append(sim_negs[neg_name])

            pair_poss = torch.cat(pair_poss, 1)
            pair_negs = torch.cat(pair_negs, 1)

            pair_all = torch.cat((pair_poss, pair_negs), 1)

            binary_zero_labels = torch.zeros_like(pair_all)
            binary_zero_labels[:, :pair_poss.size(1)] = 1
            
            loss += self.criterion(pair_all/self.temp, binary_zero_labels)
            # loss += pair['weight']*self.criterion(pair_all/self.temp, binary_zero_labels)
            
        return loss
'''


class IL_negsum():
    def __init__(self,device,mean=True,gap=0.5,abs_thres=True):
        self.device=device
        self.mean=mean
        self.gap=gap
        self.abs_thres=abs_thres
        self.top_num=1
    def __call__(self,outputs,labels):
        l=len(labels)
        sigmoid=torch.sigmoid(outputs)
        onehot=(torch.eye(10)[labels]).to(self.device)
        p_of_answer=(sigmoid*onehot).sum(axis=1)

        neg_losses=sigmoid*(1-onehot)
        val,idx=neg_losses.topk(self.top_num)
        print("sigmoid",sigmoid[0])
        #print(neg_losses[0])
        #print(onehot[0])
        '''
        extend_p_of_answer=(p_of_answer*self.gap).unsqueeze(dim=1).expand(outputs.shape)
        if self.abs_thres==True:
            bigger_than_answer=(sigmoid>(self.gap))*(1-onehot)
            print("sigmoid",sigmoid[0])
            print("bigger_than_answer",bigger_than_answer[0])
        else:
            bigger_than_answer=(sigmoid>(extend_p_of_answer))*(1-onehot)'''
        pos=(((1-p_of_answer)**2))*self.top_num
        neg=(((val**2)).sum(dim=1))##((bigger_than_answer.sum(dim=1)+1e-10))
        s=pos+neg
        
        s=s.sum()
        if self.mean==True:
            s/=l
        return s


class IL():
    def __init__(self,device,mean=True,gap=0.5,abs_thres=True):
        self.device=device
        self.mean=mean
        self.gap=gap
        self.abs_thres=abs_thres
    def __call__(self,outputs,labels):
        l=len(labels)
        sigmoid=torch.sigmoid(outputs)
        onehot=(torch.eye(10)[labels]).to(self.device)
        p_of_answer=(sigmoid*onehot).sum(axis=1)

        extend_p_of_answer=(p_of_answer*self.gap).unsqueeze(dim=1).expand(outputs.shape)
        if self.abs_thres==True:
            bigger_than_answer=(sigmoid>(self.gap))*(1-onehot)
            #print("sigmoid",sigmoid[0])
            #print("bigger_than_answer",bigger_than_answer[0])
        else:
            bigger_than_answer=(sigmoid>(extend_p_of_answer))*(1-onehot)
        pos=(((1-p_of_answer)**2))
        neg=(((sigmoid*bigger_than_answer)**2).sum(dim=1))/9##((bigger_than_answer.sum(dim=1)+1e-10))
        s=pos+neg
        
        s=s.sum()
        if self.mean==True:
            s/=l
        return s


class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]

        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(var.type()) - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


CE=nn.CrossEntropyLoss()




### LossManager
# from utils import AverageMeter
from utils.logging_utils import AverageMeter
import math

class LossManager():

    def __init__(self, args, weights, epoch=0):
        self.args = args
        self.loss_meters = {}
        self.init_weights = weights
        # self.update_weights(epoch)
        # print(self.weights)

        self.cls = AverageMeter("ClsLoss", ":.3f")
        self.data_time = AverageMeter('Data', ':3.1f')
        self.top1 = AverageMeter("Acc@1", ":.2f")
        self.top5 = AverageMeter("Acc@5", ":.2f")

        self.loss_meters.update({
            'cls': self.cls,
            'data_time': self.data_time,
            'top1': self.top1,
            'top5': self.top5,
        })
        self._check_integrity()


    def update_weights(self, epoch=0):
        weights = {}
        for key in self.init_weights:
            if isinstance(self.init_weights[key], float) or isinstance(self.init_weights[key], int):
                weights[key] = self.init_weights[key]
            else:
                key_weight = self.init_weights[key]

                if key_weight["progress"] == 'log':
                    weights[key] = key_weight["weight"] * self.progress(epoch)
                elif key_weight["progress"] == 'linear':
                    weights[key] = key_weight["weight"] * self.linear(epoch)
                elif key_weight["progress"] == 'linear_rev':
                    weights[key] = key_weight["weight"] * self.linear_rev(epoch)
                else:
                    weights[key] = key_weight["weight"] 
        self.weights = weights
        return
        # return weights


    def _check_integrity(self):
        for key in self.loss_meters:
            if key not in self.weights:
                self.weights[key] = 0.

    def progress(self, epoch):
        progress = (epoch+1) / self.args.global_epochs
        den = 1.0 + math.exp(-10 * progress)
        adapt_factor = min(2 / den - 1.0, 1.0)
        return adapt_factor

    def log(self, epoch):
        progress = (epoch+1) / self.args.global_epochs
        den = 1.0 + math.exp(-10 * progress)
        adapt_factor = min(2 / den - 1.0, 1.0)
        return adapt_factor

    def linear(self, epoch):
        progress = (epoch+1) / self.args.global_epochs
        return progress
    
    def exp(self, epoch):
        progress = (self.args.global_epochs - epoch - 1) / self.args.global_epochs
        den = 1.0 + math.exp(-10 * progress)
        adapt_factor = 1 - min(2 / den - 1.0, 1.0)
        return adapt_factor
    

    def weight_desc(self, epoch):
        desc = "Train[{}/{}]: ClsW: {:.2f}".format(
            epoch, self.args.global_epochs, self.weights['cls'],
        )
        return desc

    def update_loss_meters(self, losses, N):
        for key in losses:
            if key in self.loss_meters:
                if isinstance(losses[key], float) or isinstance(losses[key], int):
                    self.loss_meters[key].update(losses[key], N)
                else:
                    self.loss_meters[key].update(losses[key].item(), N)
        # self.weights[name].update(value)
        return
    
    def get_total_loss(self, losses):
        total_loss = 0.
        for key in losses:
            total_loss += losses[key] * self.weights[key]
        return total_loss
    
    def loss_desc(self, epoch):
        desc = "Train[{}/{}]: Data:{:3.1f} Cls = {:.3f}".format(
                epoch, self.args.global_epochs, self.data_time.avg, self.cls.avg,
            )
        return desc

    def wandb_desc(self, dataset_name):
        return {
            "Loss/{}/{}".format(dataset_name, "train"): self.cls.avg,
            "Acc/{}/{}".format(dataset_name, "train"): self.top1.avg,
            "Acc@5/{}/{}".format(dataset_name, "train"): self.top5.avg,
        }




def FedLC(label_distrib, logit, y, tau):
    cal_logit = torch.exp(logit- (tau* torch.pow(label_distrib, -1 / 4).unsqueeze(0).expand((logit.shape[0], -1))))
    #breakpoint()
    y_logit = torch.gather(cal_logit, dim=-1, index=y.unsqueeze(1))
    sum_y_logit = cal_logit.sum(dim=-1, keepdim=True)
    #loss = -torch.log(y_logit / (sum_y_logit - y_logit))
    loss = -torch.log(y_logit / (sum_y_logit))
    return loss.sum() / logit.shape[0]



class FedDecorrLoss(nn.Module):

    def __init__(self):
        super(FedDecorrLoss, self).__init__()
        self.eps = 1e-8

    def _off_diagonal(self, mat):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = mat.shape
        assert n == m
        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x):
        if len(x.shape) == 4:
            x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        N, C = x.shape
        if N == 1:
            return 0.0

        x = x - x.mean(dim=0, keepdim=True)
        x = x / torch.sqrt(self.eps + x.var(dim=0, keepdim=True))

        corr_mat = torch.matmul(x.t(), x)

        loss = (self._off_diagonal(corr_mat).pow(2)).mean()
        loss = loss / N

        return loss