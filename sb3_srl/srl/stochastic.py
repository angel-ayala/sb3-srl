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
from ..models import BaseDecoder

def normal_independent_dist(mean, log_var):
    # std = th.exp(0.5 * log_var)
    std = F.softplus(log_var) + 1e-5
    base_dist = D.Normal(mean, std)
    # transforms_list = [D.transforms.TanhTransform(cache_size=1)]
    # tanh_dist = D.TransformedDistribution(base_dist, transforms_list)
    return D.Independent(base_dist, 1)


class NormalDistributionHead(BaseFunction):
    def __init__(self, latent_dim: int, pre_act: nn.Module = nn.LeakyReLU):
        super(NormalDistributionHead, self).__init__(latent_dim, latent_dim, False)
        self.pre_act = pre_act() if pre_act is not None else None
        self.head = nn.Linear(latent_dim, latent_dim * 2)
        self.mu_norm = nn.LayerNorm(latent_dim)
        self.var_norm = nn.LayerNorm(latent_dim)

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


class StochasticRepresentation(BaseFunction):

    def _instance_model(self, z_dim):
        return NormalDistributionHead(z_dim)

    def forward(self, obs_feats):
        if isinstance(obs_feats, tuple):
            if self.n_models > 1:
                mean, log_var = [], []
                for i, m in enumerate(self.models):
                    _mean, _log_var = m(obs_feats[i])
                    mean.append(_mean)
                    log_var.append(_log_var)
                mean = th.concat(mean, dim=1)
                log_var = th.concat(log_var, dim=1)

            else:
                mean, log_var = self.models[-1](th.cat(obs_feats, dim=1))

        else:
            mean, log_var = self.models[-1](obs_feats)

        distribution = self.models[-1].forward_dist(mean, log_var)
        return distribution  # return distribution object by default


class StochasticWrapper(nn.Module):
    def __init__(self, model: BaseFunction, pre_act: nn.Module = nn.LeakyReLU):
        super().__init__()
        self.model = model
        self.replaced_head = False
        prob_model = NormalDistributionHead(model.output_dim, pre_act)
        if isinstance(model, BaseDecoder):
            del self.model.projection
            self.model.projection = prob_model
            self.replaced_head = True
        else:
            self.prob_model = prob_model

    def forward(self, *args, **kwargs) -> D:
        if self.model is None:
            raise NotImplementedError("No deterministic backbone was defined")

        params = self.model(*args, **kwargs)
        if self.replaced_head:
            return self.model.projection.forward_dist(*params)

        params = self.prob_model(params)
        return self.prob_model.forward_dist(*params)  # return distribution object by default

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model.__class__.__name__},"
            f"head={NormalDistributionHead.__name__})\n"
            f"{super().__repr__()}"
        )
