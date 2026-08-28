#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 19:46:05 2026

@author: angel
"""
from typing import Iterable

from torch import nn

from ..models import FunctionBase
from ..models import create_function_model


class TransformationBranch(nn.Module):
    """
    One downstream processing path.

    Each Branch owns its own stages, therefore parameters are not shared
    with other branches.
    """

    def __init__(self, stages: Iterable[FunctionBase]):
        super().__init__()
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x

    def __repr__(self):
        stages = ">".join([m.__class__.__name__ for m in self.stages])
        return f"{self.__class__.__name__}(stages=[{stages}])"


class StatePipeline(nn.Module):
    """
    Executes a configured collection of processing branches.
    """

    def __init__(self,
                 branches: dict[str, TransformationBranch],
                 configuration: dict
                 ):
        super().__init__()

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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(configuration=[{self.configuration}])"
        )


class StatePipelineFactory:

    @staticmethod
    def create(configuration: dict[str, list[tuple]]) -> StatePipeline:
        """
        A:XXXX -> Attention model
        F:XXXX -> Fusion model
        configuration:
            {
                "representation": ["A:CrossAtention", "F:MLP"],
            }

            {
                "representation": ["A:CrossAtention", "F:MLP"],
                "critic": ["A:CrossAtention", "F:MLP"],
            }
        """

        branches = {}

        for branch_name, models in configuration.items():
            stages = []

            for model_name, model_params in models:
                stage = create_function_model(model_name, model_params)
                stages.append(stage)

            branches[branch_name] = TransformationBranch(stages)

        return StatePipeline(branches, configuration)
