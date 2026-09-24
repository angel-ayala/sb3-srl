#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 00:00:34 2026

@author: angel
"""
from typing import Optional
from enum import Enum
import math
import torch as th
from torch import nn
import torch.distributions as D
import torch.nn.functional as F

from ..models import BaseFunction
from ..models import BaseDecoder
from .representation import RepresentationLayer

class ScaleParameterization(str, Enum):
    STD = "std"
    LOG_VAR = "log_var"


class NormalDistributionHead(BaseFunction):
    def __init__(
        self,
        latent_dim: int,
        pre_act: nn.Module = nn.LeakyReLU,
        # Distribution parameterization
        scale_parameterization: ScaleParameterization = ScaleParameterization.STD,
        # Parameter normalization
        normalize_mean: bool = False,
        normalize_scale: bool = False,
        # Mean bounds
        mean_min: Optional[float] = None,
        mean_max: Optional[float] = None,
        # Scale bounds
        std_min: Optional[float] = None,
        std_max: Optional[float] = None,
    ):
        super().__init__(latent_dim, latent_dim, False)

        self.pre_act = pre_act() if pre_act is not None else None
        self.head = nn.Linear(latent_dim, latent_dim * 2)

        self.scale_parameterization = ScaleParameterization(
            scale_parameterization
        )

        self.mean_min = mean_min
        self.mean_max = mean_max
        self.std_min = std_min
        self.std_max = std_max

        self.mu_norm = (
            nn.LayerNorm(latent_dim)
            if normalize_mean else nn.Identity()
        )

        self.scale_norm = (
            nn.LayerNorm(latent_dim)
            if normalize_scale else nn.Identity()
        )

    def _bound_mean(self, mean):
        if self.mean_min is None or self.mean_max is None:
            return mean

        center = (self.mean_max + self.mean_min) / 2
        scale = (self.mean_max - self.mean_min) / 2

        return center + scale * th.tanh((mean - center) / scale)

    def _bound_std(self, std):
        # smooth bounding
        if self.std_max is not None:
            std = self.std_max - F.softplus(self.std_max - std)

        if self.std_min is not None:
            std = self.std_min + F.softplus(std - self.std_min)

        return std

    def _bound_log_var(self, log_var):
        if self.std_min is None or self.std_max is None:
            return log_var

        log_var_min = math.log(self.std_min ** 2)
        log_var_max = math.log(self.std_max ** 2)

        center = (log_var_max + log_var_min) / 2
        scale = (log_var_max - log_var_min) / 2

        return center + scale * th.tanh((log_var - center) / scale)

    def forward_dist(self, mean, scale):
        mean = self._bound_mean(mean)

        if self.scale_parameterization == ScaleParameterization.STD:
            std = self._bound_std(scale)

        elif self.scale_parameterization == ScaleParameterization.LOG_VAR:
            log_var = self._bound_log_var(scale)
            std = th.exp(0.5 * log_var)

        else:
            raise ValueError(
                f"Unknown scale parameterization: "
                f"{self.scale_parameterization}"
            )

        return D.Independent(D.Normal(mean, std), 1)

    def forward(self, feats):
        if self.pre_act is not None:
            feats = self.pre_act(feats)

        mean, scale = self.head(feats).chunk(2, dim=-1)

        mean = self.mu_norm(mean)
        scale = self.scale_norm(scale)

        return mean, scale


class NormalizedUnboundedDistribution(NormalDistributionHead):
    def __init__(self, z_dim, pre_act: nn.Module = nn.LeakyReLU):
        super().__init__(
            latent_dim=z_dim,
            pre_act=pre_act,
            scale_parameterization=ScaleParameterization.STD,
            normalize_mean=True,
            normalize_scale=True,
            std_min=1e-5
        )


class BoundedDistribution(NormalDistributionHead):
    def __init__(self, z_dim, pre_act: nn.Module = nn.LeakyReLU):
        super().__init__(
            latent_dim=z_dim,
            pre_act=pre_act,
            scale_parameterization=ScaleParameterization.STD,
            normalize_mean=False,
            normalize_scale=False,
            mean_min=-30.0,
            mean_max=30.0,
            std_min=0.1,
            std_max=10.0,
        )


class NormalizedBoundedDistribution(NormalDistributionHead):
    def __init__(self, z_dim, pre_act: nn.Module = nn.LeakyReLU):
        super().__init__(
            latent_dim=z_dim,
            pre_act=pre_act,
            scale_parameterization=ScaleParameterization.STD,
            normalize_mean=True,
            normalize_scale=True,
            mean_min=-30.0,
            mean_max=30.0,
            std_min=0.1,
            std_max=10.0,
        )


class LogVarBoundedDistribution(NormalDistributionHead):
    def __init__(self, z_dim, pre_act: nn.Module = nn.LeakyReLU):
        super().__init__(
            latent_dim=z_dim,
            pre_act=pre_act,
            scale_parameterization=ScaleParameterization.LOG_VAR,
            normalize_mean=False,
            normalize_scale=False,
            mean_min=-30.0,
            mean_max=30.0,
            std_min=0.1,
            std_max=10.0,
        )


class NormalizedLogVarBoundedDistribution(NormalDistributionHead):
    def __init__(self, z_dim, pre_act: nn.Module = nn.LeakyReLU):
        super().__init__(
            latent_dim=z_dim,
            pre_act=pre_act,
            scale_parameterization=ScaleParameterization.LOG_VAR,
            normalize_mean=True,
            normalize_scale=True,
            mean_min=-30.0,
            mean_max=30.0,
            std_min=0.1,
            std_max=10.0,
        )


STCH_HEADS = {
    "NormalizedUnbounded": NormalizedUnboundedDistribution,
    "Bounded": BoundedDistribution,
    "NormalizedBounded": NormalizedBoundedDistribution,
    "LogVarBounded": LogVarBoundedDistribution,
    "NormalizedLogVarBounded": NormalizedLogVarBoundedDistribution,
}


class StochasticRepresentation(RepresentationLayer):
    
    @staticmethod
    def instance_dist_head(model_name, params):
        if model_name is None:
            print("No head defined, using NormalizedUnboundedDistribution")
            return NormalizedUnboundedDistribution(**params)

        try:
            dist_head = STCH_HEADS[model_name]
        except KeyError:
            raise ValueError(
                f"Representation function '{model_name}' not registered. "
                f"Available: {list(STCH_HEADS)}"
            )
        return dist_head(**params)

    def _instance_model(self, z_dim):
        return self.instance_dist_head(self.rep_head, {'z_dim': z_dim})

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
    def __init__(self, model: BaseFunction, rep_head: str = None, pre_act: nn.Module = nn.LeakyReLU):
        super().__init__()
        self.model = model
        self.replaced_head = False
        prob_model = StochasticRepresentation.instance_dist_head(
            rep_head, {'z_dim': model.output_dim, 'pre_act': pre_act})
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
        head_model = self.model.projection if self.replaced_head else self.prob_model
        return (
            f"{self.__class__.__name__}("
            f"model={self.model.__class__.__name__},"
            f"head={head_model.__class__.__name__})\n"
            f"{super().__repr__()}"
        )
