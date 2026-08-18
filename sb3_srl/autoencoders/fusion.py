#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 11:54:27 2026

@author: angel
"""

import torch as th
from torch import nn

from .dist import create_dist
from .dist import MeanVarHead


def fusion_model(fusion_type, latent_dim, need_stochastic=False):
    fusion_name = None
    if fusion_type == 'mlp':
        fusion_name = "FusionMLP"
    elif fusion_type == 'conv1d':
        fusion_name = "FusionConv1d"
    elif fusion_type == 'gated':
        fusion_name = "FusionGated"
    elif fusion_type == 'film':
        fusion_name = "FusionFiLM"
    elif fusion_type == 'attention':
        fusion_name = "FusionCrossAttention"
    else:
        raise NotImplementedError(f"Fusion method ({fusion_type}) not found, "
                                  "try with: --fusion-mlp "
                                  "--fusion-conv1d "
                                  "--fusion-gated "
                                  "--fusion-film"
                                  "--fusion-attention")
    if need_stochastic:
        fusion_name += "Stochastic"

    return globals()[fusion_name](latent_dim)


class FusionMLP(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim, bias=True),
            nn.LayerNorm(latent_dim),
            nn.Tanh()
        )

    def forward(self, z):
        if isinstance(z, tuple):
            zf = th.cat(z, dim=1)
        else:
            zf = z
        zf = self.fusion(zf)
        return zf


class FusionMLPStochastic(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim, bias=True),
            # nn.LayerNorm(latent_dim),
            # nn.Tanh()
        )
        layers = [nn.LeakyReLU(),
                  MeanVarHead(latent_dim, latent_dim)]
        self.head_model = nn.Sequential(*layers)

    def forward(self, z):
        if isinstance(z, tuple):
            zf = th.cat(z, dim=1)
        else:
            zf = z
        zf = self.fusion(zf)
        mean, log_var = self.head_model(zf)
        dist = create_dist(mean, log_var)
        return dist  # return distribution object by default


class FusionConv1d(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.fusion = nn.Sequential(
            nn.Conv1d(
                in_channels=2 * latent_dim,
                out_channels=latent_dim,
                kernel_size=1,
                groups=latent_dim,
                bias=True
            ),
            nn.Flatten(start_dim=1),
            nn.LayerNorm(latent_dim),
            nn.Tanh()
        )

    def forward(self, z):
        if isinstance(z, tuple):
            zf = th.cat(z, dim=1)
        else:
            zf = z
        zf = zf.reshape(zf.shape[0], 2 * self.latent_dim, 1)
        zf = self.fusion(zf)
        return zf


class FusionConv1dStochastic(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.fusion = nn.Sequential(
            nn.Conv1d(
                in_channels=2 * latent_dim,
                out_channels=latent_dim,
                kernel_size=1,
                groups=latent_dim,
                bias=True
            ),
            nn.Flatten(start_dim=1),
            # nn.LayerNorm(latent_dim),
            # nn.Tanh()
        )
        layers = [nn.LeakyReLU(),
                  MeanVarHead(latent_dim, latent_dim)]
        self.head_model = nn.Sequential(*layers)

    def forward(self, z):
        if isinstance(z, tuple):
            zf = th.cat(z, dim=1)
        else:
            zf = z
        zf = zf.reshape(zf.shape[0], 2 * self.latent_dim, 1)
        zf = self.fusion(zf)
        mean, log_var = self.head_model(zf)
        dist = create_dist(mean, log_var)
        return dist  # return distribution object by default


class FusionGated(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Tanh()
        )

    def forward(self, z):
        z1, z2 = z.chunk(2, dim=1)
        g = self.gate(z)
        return self.fusion(g * z1 + (1 - g) * z2)


class FusionFiLM(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.gamma = nn.Linear(latent_dim, latent_dim)
        self.beta = nn.Linear(latent_dim, latent_dim)

        self.fusion = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh()
        )

    def forward(self, z):
        z_p, z_e = z.chunk(2, dim=1)

        gamma = 1.0 + self.gamma(z_p)
        beta = self.beta(z_p)

        z_e_mod = gamma * z_e + beta
        zf = self.fusion(th.cat([z_p, z_e_mod], dim=-1))
        return  zf


class FusionCrossAttention(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=8,
            batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(latent_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        latent1, latent2 = z.chunk(2, dim=1)
        # [batch, latent_dim] → [batch, 1, latent_dim]
        latent1 = latent1.unsqueeze(1)
        # [batch, latent_dim] → [batch, latent_dim, 1]
        latent2 = latent2.unsqueeze(1)

        # latent1 attends to latent2
        fused, _ = self.attention(latent1, latent2, latent2)
        
        return self.fusion(fused.transpose(1, 2))  # [batch, latent_dim, L]
