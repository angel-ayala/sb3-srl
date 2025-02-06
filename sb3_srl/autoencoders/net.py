#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:23:42 2025

@author: angel
"""

import torch as th
from torch import nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid], gain)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class MLP(nn.Module):
    def __init__(self, n_input, n_output, hidden_dim, num_layers=2):
        super(MLP, self).__init__()
        self.h_layers = nn.ModuleList([nn.Linear(n_input, hidden_dim)])
        for i in range(num_layers - 1):
            self.h_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.h_layers.append(nn.Linear(hidden_dim, n_output))
        self.num_layers = len(self.h_layers)

    def forward(self, obs):
        h = obs
        for h_layer in self.h_layers:
            h = th.relu(h_layer(h))

        return h


class Conv1dMLP(MLP):
    def __init__(self, state_shape, out_dim, hidden_dim, num_layers=2):
        super(Conv1dMLP, self).__init__(
            state_shape[-1], out_dim, hidden_dim, num_layers=num_layers)
        if len(state_shape) == 2:
            self.h_layers[0] = nn.Conv1d(state_shape[0], hidden_dim,
                                         kernel_size=state_shape[-1])

    def forward_h(self, obs):
        h = obs
        for hidden_layer in self.h_layers[:-1]:
            h = th.relu(hidden_layer(h))
            if isinstance(hidden_layer, nn.Conv1d):
                h = h.squeeze(2)
        return h

    def forward(self, obs):
        h = self.forward_h(obs)
        return self.h_layers[-1](h)


class VectorEncoder(Conv1dMLP):
    def __init__(self, 
                 vector_shape: tuple,
                 latent_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2):
        super(VectorEncoder, self).__init__(
            vector_shape, latent_dim, hidden_dim, num_layers=num_layers)
        self.feature_dim = latent_dim
        self.fc = nn.Linear(latent_dim, latent_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

    def forward(self, obs, detach=False):
        z = th.relu(super().forward(obs))
        if detach:
            z = z.detach()
        out = self.ln(self.fc(z))
        return th.tanh(out)

    def copy_weights_from(self, source):
        """Tie hidden layers"""
        # only tie hidden layers
        for i in range(self.num_layers):
            tie_weights(src=source.h_layers[i], trg=self.h_layers[i])


class VectorDecoder(MLP):
    def __init__(self, 
                 vector_shape: tuple,
                 latent_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2):
        super(VectorDecoder, self).__init__(
            latent_dim, vector_shape[-1], hidden_dim, num_layers=num_layers)
        if len(vector_shape) == 2:
            self.h_layers[-1] = nn.ConvTranspose1d(
                hidden_dim, vector_shape[0], kernel_size=vector_shape[-1])

    def forward(self, z):
        h = z
        for hidden_layer in self.h_layers[:-1]:
            h = th.relu(hidden_layer(h))
        last_layer = self.h_layers[-1]
        if isinstance(last_layer, nn.ConvTranspose1d):
            h = h.unsqueeze(2)
        out = last_layer(h)
        return out
