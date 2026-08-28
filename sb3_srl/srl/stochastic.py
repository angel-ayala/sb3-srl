#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 00:00:34 2026

@author: angel
"""

import torch as th
from torch import nn
import torch.distributions as D
import torch.nn.functional as F

from ..models import BaseFunction


def normal_independent_dist(mean, log_var):
    # std = th.exp(0.5 * log_var)
    std = F.softplus(log_var) + 1e-5
    base_dist = D.Normal(mean, std)
    # transforms_list = [D.transforms.TanhTransform(cache_size=1)]
    # tanh_dist = D.TransformedDistribution(base_dist, transforms_list)
    return D.Independent(base_dist, 1)


class NormalDistributionHead(BaseFunction):
    def __init__(self, latent_dim: int, pre_act: nn.Module = nn.LeakyReLU):
        super(NormalDistributionHead, self).__init__(latent_dim)
        self.head = nn.Linear(latent_dim, latent_dim * 2)
        self.mu_norm = nn.LayerNorm(latent_dim)
        self.var_norm = nn.LayerNorm(latent_dim)
        self.pre_act = pre_act()

    def forward_dist(self, mu, log_var):
        return normal_independent_dist(mu, log_var)

    def forward(self, feats):
        if self.pre_act is not None:
            z = self.pre_act(feats)
        else:
            z = feats
        mu, log_var = self.head(z).chunk(2, dim=1)
        mu = self.mu_norm(mu)
        log_var = self.var_norm(log_var)
        return mu, log_var


class StochasticRepresentation(NormalDistributionHead):
    def forward(self, obs_z):
        if isinstance(obs_z, tuple):
            obs_z = th.cat(obs_z, dim=1)
        mean, log_var = super().forward(obs_z)
        distribution = self.forward_dist(mean, log_var)
        return distribution  # return distribution object by default

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(output_dim={self.output_dim})"
        )


class StochasticWrapper(nn.Module):
    def __init__(self, model: BaseFunction, pre_act: nn.Module = nn.LeakyReLU):
        super().__init__()
        self.model = model
        self.head_model = StochasticRepresentation(model.output_dim, pre_act)

    def forward(self, *args, **kwargs) -> D:
        if self.model is None:
            raise NotImplementedError("No deterministic backbone was defined")
        obs_z = self.model(*args, **kwargs)
        dist = self.head_model(obs_z)
        return dist  # return distribution object by default

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model.__class__.__name__},"
            f"head={self.head_model.__class__.__name__})"
        )
