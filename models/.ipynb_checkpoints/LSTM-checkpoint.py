#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from models.build import ENCODER_REGISTRY


@ENCODER_REGISTRY.register()
class CharLSTM(nn.Module):
    def __init__(self, args, num_classes, **kwargs):
        super(CharLSTM, self).__init__()
        self.num_layers = 1
        self.embed = nn.Embedding(80, 8)
        self.lstm = nn.LSTM(8, 256, 2, batch_first=True)
        self.drop = nn.Dropout()
        self.out = nn.Linear(256, 80)

    def forward(self, x, no_relu=False):
        results = {}
        x = self.embed(x)
        x, hidden = self.lstm(x)
        results['layer0'] = x[:, -1, :]
        x = self.drop(x)
        results['feature'] = x[:, -1, :]
        # print(x.size())
        logit = self.out(x[:, -1, :])
        results['logit'] = logit

        return results