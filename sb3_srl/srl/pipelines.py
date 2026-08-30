#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 19:46:05 2026

@author: angel
"""
from typing import Iterable

import torch as th
from torch import nn

from ..models import BaseFunction


class BaseRepresentation(BaseFunction):
    def __init__(self, latent_dim: int):
        super().__init__(latent_dim)
        models = []
        if isinstance(latent_dim, tuple):
            for z_dim in latent_dim:
                models.append(self._instance_model(z_dim))
        else:
            models.append(self._instance_model(latent_dim))

        self.n_models = len(models)
        self.models = nn.ModuleList(models)
    
    def _instance_model(self, z_dim):
        raise NotImplementedError

    def forward(self, obs_feats):
        if isinstance(obs_feats, tuple):
            if self.n_models > 1:
                z = th.cat(tuple(m(obs_feats[i])
                            for i, m in enumerate(self.models)), dim=1)
            else:
                z = self.models[-1](th.cat(obs_feats, dim=1))
            return z

        return self.models[-1](obs_feats)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(output_dim={self.output_dim})"
        )

class DeterministicRepresentation(BaseRepresentation):

    def _instance_model(self, z_dim):
        return nn.Sequential(nn.LayerNorm(z_dim), nn.Tanh())


class TransformationBranch(nn.Module):
    """
    One downstream processing path.

    Each Branch owns its own stages, therefore parameters are not shared
    with other branches.
    """

    def __init__(self, stages: Iterable[BaseFunction]):
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.output_dim = stages[-1].output_dim

    def add_function(self, stage: BaseFunction):
        self.stages.append(stage)
        self.output_dim = stage.output_dim

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x

    def __repr__(self):
        stages = "\n -> ".join([str(m) for m in self.stages])
        return (
            f"{self.__class__.__name__}(stages=[\n -> {stages}\n])"
        )


class StatePipeline(nn.Module):
    """
    Executes a configured collection of processing branches.
    """

    def __init__(self,
                 branches: dict[str, TransformationBranch],
                 configuration: dict,
                 is_stochastic: bool = False
                 ):
        super().__init__()

        self.branches = nn.ModuleDict(branches)
        self.configuration = configuration
        self.is_stochastic = is_stochastic

    @property
    def n_branches(self):
        return len(self.branches.keys())

    @property
    def branch_keys(self):
        return list(self.branches.keys())

    @property
    def latent_dim(self):
        return self.branches["representation"].output_dim

    def forward_branch(self, branch: str, obs_feats, deterministic=False, use_grad=True, use_distribution=False):
        try:
            out = self.branches[branch](obs_feats)

        except KeyError:
            print(f"No branch {branch} in pipeline")
            return None

        if self.is_stochastic and not use_distribution:
            if deterministic:
                z = out.mean
            else:
                z = out.rsample() if use_grad else out.sample()
            return th.tanh(z)

        return out

    def forward(self, x):
        return {
            name: self.forward_branch(name, x)
            for name, branch in self.branches.items()
        }

    def forward_representation(self, obs_feats, deterministic=False, use_grad=True, use_distribution=False):
        return self.forward_branch("representation", obs_feats, deterministic, use_grad, use_distribution)

    def forward_critic(self, obs_feats, deterministic=False, use_grad=True, use_distribution=False):
        return self.forward_branch("critic", obs_feats, deterministic, use_grad, use_distribution)

    def forward_z(self, obs_feats, deterministic=False, use_grad=True):
        if self.n_branches == 1:
            transform = self.forward_representation
        else:
            transform = self.forward_critic
        return transform(obs_feats, deterministic, use_grad)

    def __repr__(self) -> str:
        configurations = {
            name: branch
            for name, branch in self.branches.items()
        }
        return (
            f"{self.__class__.__name__}(configuration=[{configurations}])"
        )
