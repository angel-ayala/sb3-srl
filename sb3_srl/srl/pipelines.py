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


class TransformationBranch(nn.Module):
    """
    One downstream processing path.

    Each Branch owns its own stages, therefore parameters are not shared
    with other branches.
    """

    def __init__(self, stages: Iterable[BaseFunction]):
        super().__init__()
        self.stages = nn.ModuleList(stages)
    
    def add_function(self, stage: BaseFunction):
        self.stages.append(stage)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x

    def __repr__(self):
        stages = ">".join([m.__class__.__name__ for m in self.stages])
        return f"{self.__class__.__name__}(stages=[{stages}])"


class StateRepresentation(BaseFunction):
    def __init__(self, latent_dim: int):
        super().__init__()

        model = [nn.LayerNorm(latent_dim), nn.Tanh()]
        self.model = nn.Sequential(*model)
    
    def forward(self, obs_feats):
        feats = obs_feats
        if isinstance(feats, tuple):
            feats = th.cat(feats, dim=1)
        return self.model(feats)


class StatePipeline(nn.Module):
    """
    Executes a configured collection of processing branches.
    """

    def __init__(self,
                 branches: dict[str, TransformationBranch],
                 configuration: dict
                 ):
        super().__init__()
        # TODO: append representation layer to any defined branch

        self.branches = nn.ModuleDict(branches)
        self.configuration = configuration

    @property
    def n_branches(self):
        return len(self.branches.keys())

    @property
    def branch_keys(self):
        return list(self.branches.keys())

    def forward_branch(self, branch: str, x):
        try:
            out = self.branches[branch](x)
        except KeyError:
            print(f"No branch {branch} in pipeline")
            out = None
        return out

    def forward(self, x):
        return {
            name: branch(x)
            for name, branch in self.branches.items()
        }

    def forward_representation(self, obs_z, deterministic=False, use_grad=True):
        return self.forward_branch("representation", obs_z)

    def forward_critic(self, obs_z, deterministic=False, use_grad=True):
        return self.forward_branch("critic", obs_z)

    def forward_z(self, obs_z, deterministic=False, use_grad=True):
        if self.n_branches == 1:
            transform = self.forward_representation
        else:
            transform = self.forward_critic
        return transform(obs_z, deterministic, use_grad)

    def __repr__(self) -> str:
        configurations = {
            name: branch
            for name, branch in self.branches.items()
            }
        return (
            f"{self.__class__.__name__}(configuration=[{configurations}])"
        )
