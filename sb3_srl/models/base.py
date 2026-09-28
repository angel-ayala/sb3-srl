#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 22:53:38 2026

@author: angel
"""

from __future__ import annotations
from typing import List

from dataclasses import dataclass
import torch as th
import torch.nn as nn


@dataclass
class FunctionArguments:
    input_dim: int
    output_dim: int
    layers_dim: List[int]


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
        if auto_setup:
            self.models, self.n_models = self.instance_models()

    @property
    def multi_input(self):
        return isinstance(self.input_dim, tuple)

    @property
    def multi_output(self):
        return isinstance(self.output_dim, tuple)

    def instance_models(self):
        models = []

        if self.multi_input:
            for i, input_dim in enumerate(self.input_dim):
                out = self.output_dim[i] if self.multi_output else self.output_dim
                args = self._function_args(input_dim, out)
                models.append(self._instance_model(args))

        elif self.multi_output:
            for i, output_dim in enumerate(self.output_dim):
                args = self._function_args(self.input_dim, output_dim)
                models.append(self._instance_model(args))

        else:
            args = self._function_args(self.input_dim, self.output_dim)
            models.append(self._instance_model(args))

        return nn.ModuleList(models), len(models)

    def _function_args(self, input_dim, output_dim):
        return FunctionArguments(
            input_dim=input_dim,
            output_dim=output_dim,
            layers_dim=self.layers_dim,
        )

    def _instance_model(self, args: FunctionArguments):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        if self.n_models > 1:
            return th.cat(tuple(model(self._model_args(i, *args, **kwargs))
                                for i, model in enumerate(self.models)),
                          dim=1)
        out = self.models[-1](self._model_args(0, *args, **kwargs))
        return out

    def _model_args(self, i, obs_feats):
        if isinstance(obs_feats, tuple):
            if self.n_models > 1:
                return obs_feats[i]

            return th.cat(obs_feats, dim=1)

        return obs_feats

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(input_dim={self.input_dim}, output_dim={self.output_dim})\n"
            f"{super().__repr__()}"
        )
