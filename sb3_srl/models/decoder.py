#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 22:33:13 2026

@author: angel
"""
from typing import List, Optional

from stable_baselines3.common.torch_layers import create_mlp
import torch as th
from torch import nn
import torch.nn.functional as F

from .base import BaseFunction
from .encoder import PixelEncoder


class BaseDecoder(BaseFunction):
    def __init__(self, output_dim: int):
        super(BaseDecoder, self).__init__(output_dim)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class VectorDecoder(BaseDecoder):
    def __init__(self,
                 state_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256, 256]):
        super(VectorDecoder, self).__init__(state_shape[-1])
        layers = create_mlp(latent_dim, state_shape[-1], layers_dim,
                            nn.LeakyReLU, False, True)
        layers.insert(0, nn.Linear(latent_dim, latent_dim))

        if len(state_shape) == 2:
            layers.insert(-1, nn.ConvTranspose1d(layers_dim[0], state_shape[0],
                                                 kernel_size=state_shape[-1]))
            layers.insert(-1, nn.Unflatten(2, (1, layers_dim[-1])))
        self.projection = nn.Sequential(*layers)

    def forward(self, z):
        return self.projection(z)


class SPRDecoder(BaseDecoder):
    """VectorSPRDecoder for reconstruction function."""
    def __init__(self,
                 action_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256]):
        super(SPRDecoder, self).__init__(latent_dim)
        layers = create_mlp(latent_dim + action_shape[-1], latent_dim, layers_dim, nn.LeakyReLU, False, True)
        self.code = nn.Sequential(*layers)
        self.projection = nn.Linear(latent_dim, latent_dim)

    def transition(self, z, action):
        h_fc = self.code(th.cat([z, action], dim=1))
        return th.tanh(h_fc)

    def predict(self, z_prj):
        h_fc = self.projection(z_prj)
        return h_fc

    def forward(self, z, action):
        code = self.transition(z, action)
        return self.predict(code)


class SimpleSPRDecoder(BaseDecoder):
    """SimpleSPRDecoder as representation learning function."""

    def __init__(self,
                 action_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256]):
        super(SimpleSPRDecoder, self).__init__(latent_dim)
        code_layers = create_mlp(latent_dim + action_shape[-1], latent_dim, layers_dim, nn.LeakyReLU, True, True)
        code_layers.insert(-1, nn.LayerNorm(latent_dim))
        self.transition = nn.Sequential(*code_layers)
        proj_layers = create_mlp(latent_dim, latent_dim, layers_dim, nn.LeakyReLU, True, True)
        self.projection = nn.Sequential(*proj_layers)
        self.action_dim = action_shape[-1]
        self.hot_encode_action = False

    def forward_z_hat(self, z, action):
        if self.hot_encode_action:
            hot_action = th.zeros((action.shape[0], self.action_dim))
            if z.get_device() >= 0:
                hot_action = hot_action.to(device=z.get_device())
            hot_action[th.arange(hot_action.size(0)).unsqueeze(1), action] = 1
            action = hot_action

        return self.transition(th.cat([z, action], dim=1))

    def forward_proj(self, code):
        return self.projection(code)

    def forward(self, z, action):
        code = self.forward_z_hat(z, action)
        proj = self.forward_proj(code)
        return proj


class PixelDecoder(BaseDecoder):
    def __init__(self, state_shape: tuple,
                 latent_dim: int,
                 layers_filter: List[int] = [32, 32]):
        super(PixelDecoder, self).__init__(PixelEncoder.OUT_DIM[self.num_layers])
        self.num_layers = len(layers_filter)
        self.num_filters = layers_filter[0]

        self.fc = nn.Linear(
            latent_dim, self.num_filters * self.output_dim * self.output_dim
        )

        self.deconvs = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.deconvs.extend([
                nn.ConvTranspose2d(layers_filter[i], layers_filter[i + 1], 3, stride=1)
            ])
        self.deconvs.extend([
            nn.ConvTranspose2d(
                layers_filter[-1], state_shape[0], 3, stride=2, output_padding=1
            )
        ])

    def forward(self, h):
        h = F.leaky_relu(self.fc(h))
        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)

        for i in range(len(self.deconvs) - 1):
            deconv = F.leaky_relu(self.deconvs[i](deconv))
        obs = self.deconvs[-1](deconv)

        return obs


class ProprioceptiveSPRDecoder(BaseDecoder):
    """ProprioceptiveSPRDecoder as representation learning function."""

    def __init__(self,
                 action_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256],
                 with_fusion: bool = False):
        super(ProprioceptiveSPRDecoder, self).__init__(latent_dim)
        code_layers = create_mlp(latent_dim + action_shape[-1], latent_dim, layers_dim, nn.LeakyReLU, True, True)
        code_layers.insert(-1, nn.LayerNorm(latent_dim))
        self.proprio_trans = nn.Sequential(*code_layers)
        out_latent = latent_dim
        self.dual_transition = not with_fusion
        if self.dual_transition: # no fusion performed
            code_layers = create_mlp(latent_dim + action_shape[-1], latent_dim, layers_dim, nn.LeakyReLU, True, True)
            code_layers.insert(-1, nn.LayerNorm(latent_dim))
            self.extero_trans = nn.Sequential(*code_layers)
            out_latent = 2 * latent_dim
            self.output_dim = out_latent
        proj_layers = create_mlp(out_latent, out_latent, layers_dim, nn.LeakyReLU, True, True)
        self.projection = nn.Sequential(*proj_layers)

    def forward_z_hat(self, z, action):
        if self.dual_transition:
            proprio_z, extero_z = z.chunk(2, dim=1)
            proprio_z_hat = self.proprio_trans(th.cat([proprio_z, action], dim=1))
            extero_z_hat = self.extero_trans(th.cat([extero_z, action], dim=1))
            return proprio_z_hat, extero_z_hat
        else:
            return self.proprio_trans(th.cat([z, action], dim=1))

    def forward_proj(self, code):
        if isinstance(code, tuple):
            code = th.cat(code, dim=1)
        return self.projection(code)

    def forward(self, z, action):
        code = self.forward_z_hat(z, action)
        proj = self.forward_proj(code)
        return proj


class GuidedSPRDecoder(SimpleSPRDecoder):
    """SimpleSPRDecoder as representation learning function."""
    def __init__(self,
                 action_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256],
                 pixel_dim: Optional[int] = None):
        super(GuidedSPRDecoder, self).__init__(action_shape, latent_dim, layers_dim)
        self.latent_dim = latent_dim
        self.pixel_dim = pixel_dim
        # Linear acceleration belief
        layers = create_mlp(latent_dim, 3,
                            layers_dim, nn.LeakyReLU, False, True)
        self.accel_proj = nn.Sequential(*layers)
        # Home distance, orientation, and elevation diff belief
        layers = create_mlp(latent_dim, 3,
                            layers_dim, nn.LeakyReLU, False, True)
        self.home_proj = nn.Sequential(*layers)
        # UAV pose belief
        if pixel_dim is not None:
            layers = create_mlp(pixel_dim, 7,
                                layers_dim, nn.LeakyReLU, False, True)
            self.pose_proj = nn.Sequential(*layers)

    def forward(self, z, action):
        # forward transition
        z1_hat = self.forward_z_hat(z, action)
        # forward aux projections
        # expects z_stack with shape (B, proprio_dim+extero_dim(+pixel_dim)*)
        z1_proprio_hat, z1_extero_hat = z1_hat[:, :self.latent_dim * 2].chunk(2, dim=1)# values inference
        accel = self.accel_proj(z1_proprio_hat)
        home = self.home_proj(z1_extero_hat)
        pose = None
        if self.pixel_dim is not None:
            pose = self.pose_proj(z1_hat[:, -self.pixel_dim:])  # pose inference
        # forward latent projection
        z1_hat = self.forward_proj(z1_hat)
        return z1_hat, (accel, home, pose)
