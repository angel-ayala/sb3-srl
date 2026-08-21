#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 12:49:03 2026

@author: angel
"""
from torch import nn
import torch.distributions as D
import torch.nn.functional as F


def create_dist(mean, log_var):
    # std = th.exp(0.5 * log_var)
    std = F.softplus(log_var) + 1e-5
    base_dist = D.Normal(mean, std)
    # transforms_list = [D.transforms.TanhTransform(cache_size=1)]
    # tanh_dist = D.TransformedDistribution(base_dist, transforms_list)
    return D.Independent(base_dist, 1)


class MeanVarHead(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super(MeanVarHead, self).__init__()
        self.head = nn.Linear(input_dim, latent_dim * 2)
        self.mu_norm = nn.LayerNorm(latent_dim)
        self.var_norm = nn.LayerNorm(latent_dim)

    def forward(self, feats):
        mu, log_var = self.head(feats).chunk(2, dim=1)
        mu = self.mu_norm(mu)
        log_var = self.var_norm(log_var)
        return mu, log_var
