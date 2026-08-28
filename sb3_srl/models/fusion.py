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
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim


class Identity(BaseFusion):
    def forward(self, x):
        return x


class Concatenate(BaseFusion):
    def forward(self, x):
        z = x
        if isinstance(z, tuple):
            z = th.cat(z, dim=1)
        return z


class FusionMLP(BaseFusion):
    def __init__(self, latent_dim):
        super(FusionMLP, self).__init__(latent_dim)
        self.fusion_layers = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim, bias=True)
        )
        self.activation = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Tanh()
        )

    def forward(self, z):
        if isinstance(z, tuple):
            zf = th.cat(z, dim=1)
        else:
            zf = z
        zf = self.fusion_layers(zf)
        return self.activation(zf)


class FusionConv1d(BaseFusion):
    def __init__(self, latent_dim):
        super(FusionConv1d, self).__init__(latent_dim)
        self.fusion_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=2 * latent_dim,
                out_channels=latent_dim,
                kernel_size=1,
                groups=latent_dim,
                bias=True
            ),
            nn.Flatten(start_dim=1),
        )
        self.activation = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Tanh()
        )

    def forward(self, z):
        if isinstance(z, tuple):
            zf = th.cat(z, dim=1)
        else:
            zf = z
        zf = zf.reshape(zf.shape[0], 2 * self.latent_dim, 1)
        zf = self.fusion_layers(zf)
        return self.activation(zf)


class FusionGated(BaseFusion):
    def __init__(self, latent_dim):
        super(FusionGated, self).__init__(latent_dim)
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()
        )
        self.activation = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Tanh()
        )

    def forward(self, z):
        if isinstance(z, tuple):
            z1, z2 = z
            z_concat = th.cat(z, dim=1)
        else:
            z1, z2 = z.chunk(2, dim=1)
            z_concat = z

        g = self.gate(z_concat)
        return self.activation(g * z1 + (1 - g) * z2)


class FusionFiLM(BaseFusion):
    def __init__(self, latent_dim):
        super(FusionFiLM, self).__init__(latent_dim)

        self.gamma = nn.Linear(latent_dim, latent_dim)
        self.beta = nn.Linear(latent_dim, latent_dim)

        self.activation = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh()
        )

    def forward(self, z):
        if isinstance(z, tuple):
            z_p, z_e = z
        else:
            z_p, z_e = z.chunk(2, dim=1)

        gamma = 1.0 + self.gamma(z_p)
        beta = self.beta(z_p)

        z_e_mod = gamma * z_e + beta
        return self.activation(th.cat([z_p, z_e_mod], dim=-1))


class CrossAttention(BaseFusion):
    def __init__(self, latent_dim):
        super(CrossAttention, self).__init__(latent_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=4,
            batch_first=True
        )
        self.activation = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Tanh()
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

        return self.activation(fused)
