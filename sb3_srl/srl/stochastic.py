#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 00:00:34 2026

@author: angel
"""
from typing import Optional
from dataclasses import dataclass
from enum import Enum
import math
import torch as th
from torch import nn
import torch.distributions as D
import torch.nn.functional as F

from ..models import BaseFunction
from ..models import BaseDecoder
from ..models.base import FunctionArguments
from .representation import RepresentationLayer
from .representation import RepresentationArguments

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

    def _bound_mean_softplus(self, mean):
        # soft upper bound
        if self.mean_max is not None:
            mean = self.mean_max - F.softplus(self.mean_max - mean)

        # soft lower bound
        if self.mean_min is not None:
            mean = self.mean_min + F.softplus(mean - self.mean_min)

        return mean

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

    def _bound_log_var_softplus(self, log_var):
        log_var_min = None
        log_var_max = None

        # soft upper bound
        if self.std_max is not None:
            log_var_max = math.log(self.std_max ** 2)
            log_var = log_var_max - F.softplus(log_var_max - log_var)

        # soft lower bound
        if self.std_min is not None:
            log_var_min = math.log(self.std_min ** 2)
            log_var = log_var_min + F.softplus(log_var - log_var_min)

        return log_var

    def forward_dist(self, mean, scale):
        mean = self._bound_mean_softplus(mean)

        if self.scale_parameterization == ScaleParameterization.STD:
            std = self._bound_std(scale)

        elif self.scale_parameterization == ScaleParameterization.LOG_VAR:
            log_var = self._bound_log_var_softplus(scale)
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


class StochasticHead:

    @staticmethod
    def create(model_name, z_dim, **params):
        if model_name is None:
            print("No head defined, using NormalizedUnboundedDistribution")
            return NormalizedUnboundedDistribution(z_dim=z_dim, **params)

        try:
            head = STCH_HEADS[model_name]
        except KeyError:
            raise ValueError(
                f"Representation function '{model_name}' not registered. "
                f"Available: {list(STCH_HEADS)}"
            )

        return head(z_dim=z_dim, **params)


@dataclass
class StochasticArguments(FunctionArguments):
    dist_head: str | None = None
    pre_act: nn.Module = nn.LeakyReLU


class StochasticWrapper(BaseFunction):
    def __init__(self, model, dist_head=None, pre_act=nn.LeakyReLU):
        self.dist_head = dist_head
        self.pre_act = pre_act
        super().__init__(model.input_dim, model.output_dim, True)
        self.function = model

    def _instance_model(self, args):
        return StochasticHead.create(
            args.dist_head, args.output_dim, pre_act=args.pre_act
        )

    def _function_args(self, input_dim, output_dim):
        return StochasticArguments(
            input_dim, output_dim, [], self.dist_head, self.pre_act
        )

    def forward(self, *args, **kwargs):
        z = self.function(*args, **kwargs)
        zs = th.split(z, self.output_dim, 1) if self.multi_output else (z,)
        params = [h(z) for h, z in zip(self.models, zs)]
        mean = th.cat([p[0] for p in params], 1)
        log_var = th.cat([p[1] for p in params], 1)
        return self.models[0].forward_dist(mean, log_var)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.function, name)
