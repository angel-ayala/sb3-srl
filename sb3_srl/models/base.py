#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 22:53:38 2026

@author: angel
"""

from __future__ import annotations

import torch as th
import torch.nn as nn


class BaseFunction(nn.Module):
    """
    Base type for reusable SRL function models.

    No optimization logic belongs here.
    """

    def __init__(self, input_dim: int | tuple[int, ...],
                 output_dim: int | tuple[int, ...],
                 auto_setup: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.multiple_output = isinstance(output_dim, tuple)
        if auto_setup:
            self.models, self.n_models = self.instance_models()
    
    def instance_models(self):
        models = []
        if isinstance(self.input_dim, tuple):
            for z_dim in self.input_dim:
                models.append(self._instance_model(z_dim))
        else:
            models.append(self._instance_model(self.output_dim))
        
        return nn.ModuleList(models), len(models)
    
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
            f"{self.__class__.__name__}(input_dim={self.input_dim}, output_dim={self.output_dim})\n"
            f"{super().__repr__()}"
        )
