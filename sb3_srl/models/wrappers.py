#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:09:52 2026

@author: angel
"""
# from typing import Optional
from dataclasses import dataclass

import torch as th
from torch import nn

from .base import BaseFunction
from .base import FunctionArguments
from .decoder import BaseDecoder

from .stochastic import StochasticHead
from .mamba3 import MambaBlock


class BaseWrapper(BaseFunction):
    def __init__(self, model: BaseFunction):
        assert model is not None
        super().__init__(model.input_dim, model.output_dim, True)
        self.function = model
        # remove projection head on decoder function if exists
        if isinstance(model, BaseDecoder) and hasattr(model, 'projection'):
            self.function.projection = nn.Identity()

    # exposes wrapped function attributes
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            has_attr = hasattr(self.function, name)
            if not has_attr:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'"
                )
            return getattr(self.function, name)

@dataclass
class StochasticArguments(FunctionArguments):
    dist_head: str | None = None
    pre_act: nn.Module = nn.LeakyReLU


class StochasticWrapper(BaseWrapper):
    def __init__(self, model, dist_head=None, pre_act=nn.LeakyReLU):
        self.dist_head = dist_head
        self.pre_act = pre_act
        super().__init__(model)

    def _function_args(self, input_dim, output_dim):
        return StochasticArguments(
            input_dim, output_dim, [], self.dist_head, self.pre_act
        )

    def _instance_model(self, args: StochasticArguments):
        return StochasticHead.create(
            args.dist_head, args.output_dim, pre_act=args.pre_act
        )

    def forward(self, *args, **kwargs):
        z = self.function(*args, **kwargs)
        zs = th.split(z, self.output_dim, 1) if self.multi_output else (z,)
        params = [h(z) for h, z in zip(self.models, zs)]
        mean = th.cat([p[0] for p in params], 1)
        log_var = th.cat([p[1] for p in params], 1)
        out = self.models[0].forward_dist(mean, log_var)
        return out


class MambaWrapper(BaseWrapper):
    
    def _function_args(self, input_dim, output_dim):
        return FunctionArguments(input_dim, output_dim, [])

    def _instance_model(self, args: FunctionArguments):
        dim = sum(self.input_dim) if self.multi_input else args.output_dim
        state_dim = max(1, dim // 4)
        head_dim = max(1, dim // 8)        
        return MambaBlock(
            dim,
            ssm_cfg={
                'd_state': state_dim,
                'expand': 2,
                'headdim': head_dim,
                'ngroups': 1,
                'rope_fraction': 0.5,
                'dt_min': 0.001,
                'dt_max': 0.1,
                'dt_init_floor': 1e-4,
                'A_floor': 1e-4,
                'is_mimo': True,
                'mimo_rank': 4
            }
        )

    def forward(self, *args, **kwargs):
        z = self.function(*args, **kwargs)
        sequence = th.stack((args[0], z), dim=1)
        out = super().forward(sequence)[:, -1]
        return out
