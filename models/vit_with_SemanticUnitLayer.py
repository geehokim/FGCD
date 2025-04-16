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
import math
from functools import partial, reduce
from operator import mul

import torch
import torch.nn as nn
import random
import numpy as np
from torch.nn.modules.utils import _pair

from torch.nn.init import trunc_normal_
from models.build import ENCODER_REGISTRY
from models.vision_transformer import VisionTransformer
from models.bert import BertAttention
from transformers import BertConfig

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



class VisionTransformerWithSU(VisionTransformer):
    """ Vision Transformer with semantic units"""
    def __init__(self, img_size=[224], patch_size=16, num_semantics=10, in_chans=3, num_classes=0, embed_dim=768,
                 depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__(img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs)

        attention_config = BertConfig.from_pretrained("bert-base-uncased")
        attention_config.hidden_size = embed_dim
        self.dynamic_unit_fusion = BertAttention(attention_config, num_semantics, patch_size)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def forward(self, x, return_all_patches=False):

        x = self.prepare_tokens(x)
        cls_tokens = x[:, :1, :]
        token_embdedings = x[:, 1:, :]

        query_tokens = self.dynamic_unit_fusion.query_tokens.expand(token_embdedings.size(0), -1, -1)

        # Apply dynamic unit fusion before transformer blocks
        attn_output, attn_weights = self.dynamic_unit_fusion(
            hidden_states=query_tokens,
            encoder_hidden_states=token_embdedings,
            output_attentions=True
        )
        x = torch.cat((cls_tokens,
                       attn_output,
                       token_embdedings
                       ), dim=1)


        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if return_all_patches:
            return x
        else:
            return x[:, 0]
    



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
def vit_base_semantic_units(args, num_classes, patch_size=16, **kwargs):
    pretrained = kwargs['pretrained']
    feat_dim = kwargs['feat_dim']
    mlp_out_dim = kwargs['mlp_out_dim']
    num_mlp_layers = kwargs['num_mlp_layers']
    grad_from_block = kwargs['grad_from_block']
    cancel_last_layer = kwargs['cancel_last_layer']

    backbone = VisionTransformerWithSU(
        patch_size=patch_size, embed_dim=768,  depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        vitb16 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        vitb16_state_dict = vitb16.state_dict()
        print(backbone.load_state_dict(vitb16_state_dict, strict=False))

    model = VisionTransformerWithProjection(base_vit=backbone, feat_dim=feat_dim, mlp_out_dim=mlp_out_dim,
                                            num_mlp_layers=num_mlp_layers)
    
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
            if 'query_tokens' in name:
                m.requires_grad = True
            if 'dynamic_unit_fusion' in name:
                m.requires_grad = True
            if args.model.tune_normalization_layer:
                pass
                if 'norm.weight' in name or 'norm.bias' in name:
                    m.requires_grad = True
            if args.model.tune_dino_head_norm:
                pass
                if 'parameterizations.weight.original1' in name:
                    m.requires_grad = True

            print(f'{name}: {m.requires_grad}')




    return model




class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        np.random.seed(1)
        super().__init__()
        nlayers = max(nlayers, 1)
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
        self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
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
        return x


class VisionTransformerWithLinear(nn.Module):

    def __init__(self, base_vit, num_classes=200):

        super().__init__()

        self.base_vit = base_vit
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x, return_features=False):

        features = self.base_vit(x)
        features = torch.nn.functional.normalize(features, dim=-1)
        logits = self.fc(features)

        if return_features:
            return logits, features
        else:
            return logits

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.fc.weight.data.clone()
        w = torch.nn.functional.normalize(w, dim=1, p=2)
        self.fc.weight.copy_(w)


class VisionTransformerWithProjection(nn.Module):
    def __init__(self, base_vit, feat_dim, mlp_out_dim, num_mlp_layers):

        super().__init__()

        self.base_vit = base_vit
        self.proj_layer = DINOHead(in_dim=feat_dim,
                               out_dim=mlp_out_dim, nlayers=num_mlp_layers)
    def forward(self, x, return_features=False):

        features = self.base_vit(x)
        projected_features = self.proj_layer(features)

        return projected_features, features