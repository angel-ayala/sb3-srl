#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 11:54:27 2026

@author: angel
"""

import torch as th
from torch import nn

from .base import BaseFunction


class BaseFusion(BaseFunction):
    def __init__(self, latent_dim):
        super(BaseFusion, self).__init__((latent_dim, latent_dim), latent_dim, True)
        if isinstance(self.input_dim, tuple):
            self.input_dim = sum(self.input_dim)

    def forward(self, z):

        zf = super().forward(z)
        
        if isinstance(zf, tuple):
            zf = th.cat(zf, dim=1)

        return self.fusion_layers(zf)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(input_dim={self.input_dim},"
            f"output_dim={self.output_dim})\n"
            f"{super().__repr__()}"
        )


class FusionMLP(BaseFusion):
    def __init__(self, latent_dim):
        super(FusionMLP, self).__init__(latent_dim)

        self.fusion_layers = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim, bias=True)
        )
    
    def _instance_model(self, z_dim):
        return nn.Sequential(nn.LayerNorm(z_dim),nn.Tanh())



class FusionConv1d(FusionMLP):
    def __init__(self, latent_dim):
        super(FusionConv1d, self).__init__(latent_dim)

        self.fusion_layers = nn.Sequential(
            nn.Unflatten(1, (self.input_dim, 1)),
            nn.Conv1d(
                in_channels=self.input_dim,
                out_channels=self.output_dim,
                kernel_size=1,
                groups=self.output_dim,
                bias=True
            ),
            nn.Flatten(start_dim=1),
        )


class FusionGated(FusionMLP):
    def __init__(self, latent_dim):
        super(FusionGated, self).__init__(latent_dim)
        self.fusion_layers = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim, bias=True),
            nn.Sigmoid()
        )

    def forward(self, z):
        if isinstance(z, tuple):
            z1, z2 = z
        else:
            z1, z2 = z.chunk(2, dim=1)

        g = super().forward(z)
        return g * z1 + (1 - g) * z2


class FusionFiLM(FusionMLP):
    def __init__(self, latent_dim):
        super(FusionFiLM, self).__init__(latent_dim)

        self.gamma = nn.Linear(self.output_dim, self.output_dim)
        self.beta = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, z):
        if isinstance(z, tuple):
            z_p, z_e = z
        else:
            z_p, z_e = z.chunk(2, dim=1)

        gamma = 1.0 + self.gamma(z_p)
        beta = self.beta(z_p)

        z_e_mod = gamma * z_e + beta
        return self.fusion_layers(th.cat([z_p, z_e_mod], dim=-1))


class CrossAttention(BaseFusion):
    def __init__(self, latent_dim):
        super(CrossAttention, self).__init__(latent_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=4,
            batch_first=True
        )

    def forward(self, z):
        # [batch, latent_dim]
        if isinstance(z, tuple):
            latent1, latent2 = z
        else:
            latent1, latent2 = z.chunk(2, dim=1)

        # latent1 attends to latent2
        latent1 = latent1.unsqueeze(1)  # [batch, 1, latent_dim]
        latent2 = latent2.unsqueeze(1)  # [batch, 1, latent_dim]

        fused, _ = self.attention(latent1, latent2, latent2)
        fused = fused.squeeze(1)  # Back to [batch, latent_dim]

        return fused
