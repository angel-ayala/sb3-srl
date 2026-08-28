#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 22:53:38 2026

@author: angel
"""

from __future__ import annotations

import torch.nn as nn


class FunctionBase(nn.Module):
    """
    Base type for reusable SRL function models.

    No optimization logic belongs here.
    """

    pass


class EncoderBase(FunctionBase):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    def forward_feats(self, observation):
        raise NotImplementedError

    def forward_z(self, feats):
        raise NotImplementedError

    def forward(self, observation):
        return self.forward_z(self.forward_feats(observation))


class DecoderBase(FunctionBase):
    def forward(self, *args, **kwargs):
        raise NotImplementedError
