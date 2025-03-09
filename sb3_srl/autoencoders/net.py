#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:23:42 2025

@author: angel
"""
from typing import List

import torch as th
from torch import nn
import torch.nn.functional as F

from stable_baselines3.common.torch_layers import create_mlp


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


def renormalize(tensor, first_dim=1):
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = th.max(flat_tensor, first_dim, keepdim=True).values
    min = th.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min)/(max - min)

    return flat_tensor.view(*tensor.shape)


class MLP(nn.Module):
    def __init__(self, n_input, n_output, layers_dim=[256, 256]): #hidden_dim, num_layers=2):
        super(MLP, self).__init__()
        self.h_layers = nn.ModuleList([nn.Linear(n_input, layers_dim[0])])
        for i in range(len(layers_dim) - 1):
            self.h_layers.append(nn.Linear(layers_dim[i], layers_dim[i + 1]))
        self.h_layers.append(nn.Linear(layers_dim[-1], n_output))
        self.num_layers = len(self.h_layers)

    def forward(self, obs):
        h = obs
        for h_layer in self.h_layers:
            h = F.leaky_relu(h_layer(h))

        return h


class Conv1dMLP(MLP):
    def __init__(self, input_shape, n_output, layers_dim=[256, 256]):
        super(Conv1dMLP, self).__init__(input_shape[-1], n_output, layers_dim=layers_dim)
        if len(input_shape) == 2:
            self.h_layers[0] = nn.Conv1d(input_shape[0], layers_dim[0],
                                         kernel_size=input_shape[-1])

    def forward_h(self, obs):
        h = obs
        for hidden_layer in self.h_layers[:-1]:
            h = F.leaky_relu(hidden_layer(h))
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
                 layers_dim: List[int] = [256, 256]):
        super(VectorEncoder, self).__init__(
            vector_shape, latent_dim, layers_dim=layers_dim)
        self.feature_dim = latent_dim
        self.fc = nn.Linear(latent_dim, latent_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

    def forward(self, obs, detach=False):
        z = F.leaky_relu(super().forward(obs))
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
                 layers_dim: List[int] = [256, 256]):
        super(VectorDecoder, self).__init__(
            latent_dim, vector_shape[-1], layers_dim=layers_dim)
        if len(vector_shape) == 2:
            self.h_layers[-1] = nn.ConvTranspose1d(
                layers_dim[0], vector_shape[0], kernel_size=vector_shape[-1])
        self.fc = nn.Linear(latent_dim, latent_dim)

    def forward(self, z):
        h = self.fc(z)
        for hidden_layer in self.h_layers[:-1]:
            h = F.leaky_relu(hidden_layer(h))
        last_layer = self.h_layers[-1]
        if isinstance(last_layer, nn.ConvTranspose1d):
            h = h.unsqueeze(2)
        out = last_layer(h)
        return out


class ProjectionN(nn.Module):
    def __init__(self, latent_dim: int,
                 hidden_dim: int,
                 out_act: nn.Module = nn.Tanh()):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim),
            out_act)

    def forward(self, x):
        return self.projection(x)


class VectorSPRDecoder(nn.Module):
    """VectorSPRDecoder for reconstruction function."""
    def __init__(self,
                 action_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256]):
        super(VectorSPRDecoder, self).__init__()
        layers = create_mlp(latent_dim + action_shape[-1], latent_dim, layers_dim, nn.LeakyReLU, False, True)
        self.code = nn.Sequential(*layers)
        self.pred = nn.Linear(latent_dim, latent_dim)

    def transition(self, z, action):
        h_fc = self.code(th.cat([z, action], dim=1))
        h_fc = renormalize(th.relu(h_fc), 1)
        return th.tanh(h_fc)

    def predict(self, z_prj):
        h_fc = self.pred(z_prj)
        return h_fc

    def forward(self, z, action):
        code = self.transition(z, action)
        return self.predict(code)


class SimpleSPRDecoder(nn.Module):
    """SimpleSPRDecoder as representation learning function."""
    def __init__(self,
                 action_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256]):
        super(SimpleSPRDecoder, self).__init__()
        code_layers = create_mlp(latent_dim + action_shape[-1], latent_dim, layers_dim, nn.LeakyReLU, True, True)
        code_layers.insert(-1, nn.LayerNorm(latent_dim))
        self.code = nn.Sequential(*code_layers)
        proj_layers = create_mlp(latent_dim, latent_dim, layers_dim, nn.LeakyReLU, True, True)
        self.projection = nn.Sequential(*proj_layers)

    def forward_z_hat(self, z, action):
        return self.code(th.cat([z, action], dim=1))

    def forward_proj(self, code):
        return self.projection(code)

    def forward(self, z, action):
        code = self.forward_z_hat(z, action)
        proj = self.forward_proj(code)
        return proj
