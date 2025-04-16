# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import copy
import time
import sys
import math
from functools import partial

import torch
import torch.nn as nn
import random
import numpy as np
import os

from torch.nn.init import trunc_normal_
from models.build import ENCODER_REGISTRY
from scipy.optimize import linear_sum_assignment as linear_assignment
from torch.nn import functional as F
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


# Sinkhorn Knopp 
def sknopp(cZ, lamd=25, max_iters=100):
    with torch.no_grad():
        N_samples, N_centroids = cZ.shape # cZ is [N_samples, N_centroids]
        probs = F.softmax(cZ * lamd, dim=1).T # probs should be [N_centroids, N_samples]

        r = torch.ones((N_centroids, 1), device=probs.device) / N_centroids # desired row sum vector
        c = torch.ones((N_samples, 1), device=probs.device) / N_samples # desired col sum vector

        inv_N_centroids = 1. / N_centroids
        inv_N_samples = 1. / N_samples

        err = 1e3
        for it in range(max_iters):
            r = inv_N_centroids / (probs @ c)  # (N_centroids x N_samples) @ (N_samples, 1) = N_centroids x 1
            c_new = inv_N_samples / (r.T @ probs).T  # ((1, N_centroids) @ (N_centroids x N_samples)).t() = N_samples x 1
            if it % 10 == 0:
                err = torch.nansum(torch.abs(c / c_new - 1))
            c = c_new
            if (err < 1e-2):
                break

        # inplace calculations. 
        probs *= c.squeeze()
        probs = probs.T # [N_samples, N_centroids]
        probs *= r.squeeze()

        return probs * N_samples # Soft assignments


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)
    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

######### Progress bar #########
term_width = 150 
TOTAL_BAR_LENGTH = 30.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')
    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)
    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')
    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))
    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if return_attention:
            return x, attn
        else:
            return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x, return_all_patches=False):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if return_all_patches:
            return x
        else:
            return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                x, attn = blk(x, return_attention=True)
                x = self.norm(x)
                return x, attn

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output
def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@ENCODER_REGISTRY.register()
def vit_base_simgcd(args, num_classes, patch_size=16, **kwargs):
    pretrained = kwargs['pretrained']
    feat_dim = kwargs['feat_dim']
    mlp_out_dim = num_classes
    num_mlp_layers = kwargs['num_mlp_layers']
    grad_from_block = kwargs['grad_from_block']
    cancel_last_layer = kwargs['cancel_last_layer']
    classifier_at_proj = kwargs['classifier_at_proj']

    backbone = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        vitb16 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        vitb16_state_dict = vitb16.state_dict()
        print(backbone.load_state_dict(vitb16_state_dict))

    model = VisionTransformerWithProjection(args, base_vit=backbone, feat_dim=feat_dim, mlp_out_dim=mlp_out_dim,
                                            num_mlp_layers=num_mlp_layers, grad_from_block=grad_from_block, classifier_at_proj=classifier_at_proj)
    
    if args.client.pretrained_dir == 'gcd':
        pretrained_dir = f'./gcd_warmup_checkpoints/{args.dataset.name}/' + f'{args.trainer.num_clients}clients_'
        pretrained_dir = pretrained_dir + (f'dir{args.split.alpha}' if args.split.mode == 'dirichlet' else 'iid')
        pretrained_dir = os.path.join(pretrained_dir, 'GCD.pth')
        state_dicts = torch.load(pretrained_dir)
        
        # Remove proj_layer parameters from state_dict
        keys_to_delete = [key for key in state_dicts['model_state_dict'].keys() if 'proj_layer' in key]
        for key in keys_to_delete:
            del state_dicts['model_state_dict'][key]
        
        print(f'Loading state dict from {args.client.pretrained_dir}')
        print(model.load_state_dict(state_dicts['model_state_dict'], strict=False))

    # ## tmp for debugging
    # head_dict_pth = torch.load("/home2/geeho/SimGCD/head_state_dict.pth")
    # head_state_dict = head_dict_pth['head_state_dict']
    # model.proj_layer.load_state_dict(head_state_dict, strict=False)
    # print(f'Loaded head state dict from /home2/geeho/SimGCD/head_state_dict.pth!!!!!')

    if cancel_last_layer:
        model.proj_layer.last_layer = nn.Identity()

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    if not args.model.fft:
        for m in model.base_vit.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in model.base_vit.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= grad_from_block:
                    m.requires_grad = True
            # if args.model.tune_normalization_layer:
            #     pass
            #     if 'norm.weight' in name or 'norm.bias' in name:
            #         m.requires_grad = True
            # if args.model.tune_dino_head_norm:
            #     pass
            #     if 'parameterizations.weight.original1' in name:
            #         m.requires_grad = True

        # if args.model.freeze_classifier:
        #     model.freeze_classifier()


    return model




class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.in_dim = in_dim
        self.out_dim = out_dim
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        # self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        # tmp = nn.utils.weight_norm(self.last_layer)

        self._init_param_normalized_last_layer(nn.Linear(in_dim, out_dim, bias=False), norm_last_layer=norm_last_layer)

    def _init_param_normalized_last_layer(self, layer, norm_last_layer=True, freeze=False):
        self.last_layer = nn.utils.parametrizations.weight_norm(layer)
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        if norm_last_layer:
            # self.last_layer.weight_g.requires_grad = False
            self.last_layer.parametrizations.weight.original0.requires_grad = False
        if freeze:
            self.last_layer.parametrizations.weight.original1.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        logits = self.last_layer(x)
        return x_proj, logits
    

class DINOHead2(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.in_dim = in_dim
        self.out_dim = out_dim
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        # self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        # tmp = nn.utils.weight_norm(self.last_layer)

        self._init_param_normalized_last_layer(nn.Linear(bottleneck_dim, out_dim, bias=False), norm_last_layer=norm_last_layer)

    def _init_param_normalized_last_layer(self, layer, norm_last_layer=True, freeze=False):
        self.last_layer = nn.utils.parametrizations.weight_norm(copy.deepcopy(layer))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        if norm_last_layer:
            # self.last_layer.weight_g.requires_grad = False
            self.last_layer.parametrizations.weight.original0.requires_grad = False
        if freeze:
            self.last_layer.parametrizations.weight.original1.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        #x_proj = nn.functional.normalize(x_proj, dim=-1, p=2)
        x_norm = nn.functional.normalize(x_proj, dim=-1, p=2)
        logits = self.last_layer(x_norm)
        return x_proj, logits

class DINOHead3(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=768):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.in_dim = in_dim
        self.out_dim = out_dim
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        # self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        # tmp = nn.utils.weight_norm(self.last_layer)
        # self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        # self.last_layer.weight_g.data.fill_(1)
        # if norm_last_layer:
        #     self.last_layer.weight_g.requires_grad = False
        
        self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, 65536, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        if norm_last_layer:
            #self.last_layer.weight_g.requires_grad = False
            self.last_layer.parametrizations.weight.original0.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x, None


class DINOHead4(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256, len_seen_classes=None):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.in_dim = in_dim
        self.out_dim = out_dim
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        # self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        # tmp = nn.utils.weight_norm(self.last_layer)

        self._init_param_normalized_last_layer(nn.Linear(in_dim, out_dim, bias=False), norm_last_layer=norm_last_layer)

        self.local_prototypes = nn.Parameter(torch.randn(out_dim*4, in_dim))
        self.register_buffer('act_protos', torch.tensor(out_dim*4))
        self.act_protos_labelled = len_seen_classes
        self.local_prototypes.data[:len_seen_classes] = self.last_layer.parametrizations.weight.original1.data[:len_seen_classes].clone()
        self.global_prototypes = nn.Parameter(torch.randn(out_dim*4, in_dim))
        self.register_buffer('act_protos_global', torch.tensor(out_dim*4))

    def _init_param_normalized_last_layer(self, layer, norm_last_layer=True, freeze=False):
        self.last_layer = nn.utils.parametrizations.weight_norm(layer)
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        if norm_last_layer:
            # self.last_layer.weight_g.requires_grad = False
            self.last_layer.parametrizations.weight.original0.requires_grad = False
        if freeze:
            self.last_layer.parametrizations.weight.original1.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        logits_labelled = self.last_layer(x)
        logits_unlabelled = F.linear(x, F.normalize(self.local_prototypes, dim=-1, p=2))
        logits_global = F.linear(x, F.normalize(self.global_prototypes, dim=-1, p=2))
        # return x_proj, (logits_labelled[:, :self.act_protos_labelled], logits_unlabelled[:, :self.act_protos])
        return x_proj, (logits_labelled[:, :self.act_protos_labelled], logits_unlabelled[:, :self.act_protos.item()], logits_global[:, :self.act_protos_global.item()])

class VisionTransformerWithProjection(nn.Module):
    def __init__(self, args, base_vit, feat_dim, mlp_out_dim, num_mlp_layers, grad_from_block, N_local=None, classifier_at_proj=False):

        super().__init__()

        self.args = args
        self.base_vit = base_vit
        # with temp_seed(1):
        if classifier_at_proj:
            print(f'Using DINOHead2')
            self.proj_layer = DINOHead2(in_dim=feat_dim, out_dim=mlp_out_dim, nlayers=num_mlp_layers, bottleneck_dim=self.args.model.bottleneck_dim)
        elif args.model.no_classifier:
            print(f'No classifier, using DINOHead3')
            self.proj_layer = DINOHead3(in_dim=768, bottleneck_dim=self.args.model.bottleneck_dim, out_dim=65536, nlayers=3)
        elif args.model.cluster_head:
            print(f'Using DINOHead4')
            self.proj_layer = DINOHead4(in_dim=feat_dim, out_dim=mlp_out_dim, nlayers=num_mlp_layers, bottleneck_dim=self.args.model.bottleneck_dim,
                                        len_seen_classes=len(args.dataset.seen_classes))
        else:
            print(f'Using DINOHead1')
            self.proj_layer = DINOHead(in_dim=feat_dim,
                            out_dim=mlp_out_dim, nlayers=num_mlp_layers, bottleneck_dim=self.args.model.bottleneck_dim)
        if N_local is None:
            self.n_local_centroids = mlp_out_dim
        else:
            self.n_local_centroids = N_local
        self.local_labelled_centroids = None
        self.grad_from_block = grad_from_block
        # self.global_centroids = nn.Linear(feat_dim, len(self.args.dataset.unseen_classes), bias=False)
        # self.global_centroids.weight.data = copy.deepcopy(self.proj_layer.last_layer.parametrizations.weight.original1.data[len(self.args.dataset.seen_classes):])
        # self.global_centroids_all = nn.Linear(feat_dim, len(self.args.dataset.seen_classes) + len(self.args.dataset.unseen_classes), bias=False)
        # self.global_centroids_all.weight.data = copy.deepcopy(self.proj_layer.last_layer.parametrizations.weight.original1.data)
        self.local_centroids = None
        self.T = 0.1

    def forward(self, x, return_features=False, return_all=False, return_feats_only=False):

        features = self.base_vit(x)
        x_proj, logits = self.proj_layer(features)
        if isinstance(logits, tuple):
            logits_labelled, logits_clustered, logits_global = logits
            if return_features:
                features = nn.functional.normalize(features, dim=-1, p=2)
                return features, logits_labelled, logits_clustered, logits_global
            elif return_all:
                #features = nn.functional.normalize(features, dim=-1, p=2)
                return features, x_proj, logits_labelled, logits_clustered, logits_global
            elif return_feats_only:
                return features
            else:
                return x_proj, logits_labelled, logits_clustered, logits_global
        else:
            if return_features:
                features = nn.functional.normalize(features, dim=-1, p=2)
                return features, logits
            elif return_all:
                #features = nn.functional.normalize(features, dim=-1, p=2)
                return features, x_proj, logits
            else:
                return x_proj, logits

    def freeze_extractor(self):
        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        for m in self.base_vit.parameters():
            m.requires_grad = False
        for name, m in self.proj_layer.named_parameters():
            if 'last_layer' not in name:
                m.requires_grad = False

    def unfreeze_extractor(self):
        for name, m in self.base_vit.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= self.grad_from_block:
                    m.requires_grad = True
        
        for name, m in self.proj_layer.named_parameters():
            if 'last_layer' not in name:
                m.requires_grad = True

    def freeze_classifier(self):
        for name, m in self.proj_layer.named_parameters():
            if 'last_layer' in name:
                m.requires_grad = False
                print(f'Freezing {name}')
    
    def unfreeze_classifier(self):
        for name, m in self.proj_layer.named_parameters():
            if 'last_layer' in name:
                if 'original1' in name:
                    m.requires_grad = True
                    print(f'Unfreezing {name}')
                else:
                    m.requires_grad = False
                    print(f'Freezing {name}')

    

    # Global clustering (happens only on the server; see Genevay et al. for full details on the algorithm)
    def global_clustering(self, Z1, current_lr=0.01, nG=1., nL=1.):
        N = Z1.shape[0] # Z has dimensions [m_size * n_clients, D]

        if self.args.model.sinkhorn_normalize:
            Z1 = F.normalize(Z1, dim=1)
            self.global_centroids.weight.data.copy_(F.normalize(self.global_centroids.weight.data, dim=1))
        
        # Optimizer setup
        self.global_centroids.requires_grad = True
        optimizer = torch.optim.SGD(self.global_centroids.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-5)
        train_loss = 0.
        total_rounds = 500

        for round_idx in range(total_rounds):
            with torch.no_grad():
                # Cluster assignments from Sinkhorn Knopp
                SK_assigns = sknopp(self.global_centroids(Z1))

            # Zero grad
            optimizer.zero_grad()

            # Predicted cluster assignments [N, N_centroids] = local centroids [N, D] x global centroids [D, N_centroids]
            probs1 = F.softmax(self.global_centroids(F.normalize(Z1, dim=1)) / self.T, dim=1)

            # Match predicted assignments with SK assignments
            loss = - F.cosine_similarity(SK_assigns, probs1, dim=-1).mean()

            # Train
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.global_centroids.weight.copy_(F.normalize(self.global_centroids.weight.data.clone(), dim=1)) # Normalize centroids
                train_loss += loss.item()

            progress_bar(round_idx, total_rounds, 'Loss: %.3f' % (train_loss/(round_idx+1)))

    def global_clustering_all(self, Z1, current_lr=0.01, nG=1., nL=1.):
        N = Z1.shape[0] # Z has dimensions [m_size * n_clients, D]

        # Optimizer setup
        self.global_centroids_all.requires_grad = True
        optimizer = torch.optim.SGD(self.global_centroids_all.parameters(), lr=self.args.model.sinkhorn_lr, momentum=0.9, weight_decay=1e-4)
        train_loss = 0.
        total_rounds = 500

        for round_idx in range(total_rounds):
            with torch.no_grad():
                # Cluster assignments from Sinkhorn Knopp
                SK_assigns = sknopp(self.global_centroids_all(Z1))

            # Zero grad
            optimizer.zero_grad()

            # Predicted cluster assignments [N, N_centroids] = local centroids [N, D] x global centroids [D, N_centroids]
            probs1 = F.softmax(self.global_centroids_all(F.normalize(Z1, dim=1)) / self.T, dim=1)

            # Match predicted assignments with SK assignments
            loss = - F.cosine_similarity(SK_assigns, probs1, dim=-1).mean()

            # Train
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.global_centroids_all.weight.copy_(F.normalize(self.global_centroids_all.weight.data.clone(), dim=1)) # Normalize centroids
                train_loss += loss.item()

            progress_bar(round_idx, total_rounds, 'Loss: %.3f' % (train_loss/(round_idx+1)))


    def update_classifier_weights(self, global_centroids_all):
        self.proj_layer.last_layer.parametrizations.weight.original1.data.copy_(global_centroids_all)

    def update_novel_classifier_weights(self, global_centroids):
        ## Align global centroids using linear assignment
        self.proj_layer.last_layer.parametrizations.weight.original1.data[len(self.args.dataset.seen_classes):].copy_(global_centroids)

    def update_global_centroids_as_classifier_weights(self):
        self.global_centroids.weight.data.copy_(self.proj_layer.last_layer.parametrizations.weight.original1.data[len(self.args.dataset.seen_classes):])


class ClientGMM(nn.Module):
    def __init__(self, args, feat_dim, n_clusters):
        super().__init__()
        self.cluster_means = nn.Parameter(torch.randn(n_clusters, feat_dim))
        self.cluster_log_covariances = nn.Parameter(torch.randn(n_clusters, feat_dim))
