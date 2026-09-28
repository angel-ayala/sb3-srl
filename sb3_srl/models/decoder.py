#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 22:33:13 2026

@author: angel
"""
from typing import List, Optional

from dataclasses import dataclass
from stable_baselines3.common.torch_layers import create_mlp
import torch as th
from torch import nn
import torch.nn.functional as F

from .base import BaseFunction
from .encoder import PixelEncoder
from .base import FunctionArguments


@dataclass
class DecoderArguments(FunctionArguments):
    action_shape: Optional[tuple | int]

    @property
    def latent_dim(self) -> int | tuple[int, ...]:
        return self.input_dim


class BaseDecoder(BaseFunction):
    def __init__(self,
                 latent_dim: tuple | int,
                 output_dim: tuple | int,
                 action_shape: Optional[tuple | int] = None,
                 layers_dim: List[int] = [256, 256],
                 auto_setup: bool = True):
        self.layers_dim = layers_dim
        self.action_shape = action_shape

        super(BaseDecoder, self).__init__(latent_dim, output_dim, auto_setup)

    def _function_args(self, input_dim, output_dim):
        return DecoderArguments(
            input_dim=input_dim,
            output_dim=output_dim,
            layers_dim=self.layers_dim,
            action_shape=self.action_shape,
        )

    def _model_args(self, i, z, action=None):
        if isinstance(z, tuple):
            if self.n_models > 1:
                z = z[i]
            else:
                z = th.cat(z, dim=1)

        elif self.multi_input:
            z = z.chunk(len(self.input_dim), dim=1)[i]

        if action is not None:
            z = th.cat((z, action), dim=1)

        return z


class VectorDecoder(BaseDecoder):
    def __init__(self,
                 state_shape: tuple | int,
                 latent_dim: tuple | int,
                 action_shape: Optional[tuple | int] = None,
                 layers_dim: List[int] = [256],
                 auto_setup: bool = True):
        out_dim = state_shape[-1] if isinstance(state_shape, tuple) else state_shape
        super(VectorDecoder, self).__init__(
            latent_dim=latent_dim,
            output_dim=out_dim,
            action_shape=action_shape,
            layers_dim=layers_dim
        )

    def _instance_model(self, args: DecoderArguments):
        layers = create_mlp(args.latent_dim, args.output_dim,
                            args.layers_dim, nn.LeakyReLU, False, True)
        layers.insert(0, nn.Linear(args.latent_dim, args.latent_dim))

        if isinstance(args.output_dim, tuple) and len(args.output_dim) == 2:
            layers.insert(-1, nn.ConvTranspose1d(
                args.layers_dim[0], args.output_dim[0],
                kernel_size=args.output_dim[-1]))
            layers.insert(-1, nn.Unflatten(2, (1, args.layers_dim[-1])))
        return nn.Sequential(*layers)


class SPRDecoder(BaseDecoder):
    """VectorSPRDecoder for reconstruction function."""
    def __init__(self,
                 action_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256]):
        super(SPRDecoder, self).__init__(
            latent_dim=latent_dim,
            output_dim=latent_dim,
            action_shape=action_shape,
            layers_dim=layers_dim
        )
        self.projection = nn.Linear(latent_dim, latent_dim)

    def _instance_model(self, args: DecoderArguments):
        layers = create_mlp(args.latent_dim + args.action_shape[-1],
                            args.latent_dim, args.layers_dim,
                            nn.LeakyReLU, True, True)
        return nn.Sequential(*layers)

    def transition(self, z, action):
        return super().forward(z, action)

    def predict(self, z_prj):
        h_fc = self.projection(z_prj)
        return h_fc

    def forward(self, z, action):
        code = self.transition(z, action)
        return self.predict(code)


class SimpleSPRDecoder(BaseDecoder):
    """SimpleSPRDecoder as representation learning function."""

    def __init__(self,
                 state_shape: tuple | int,
                 latent_dim: tuple | int,
                 action_shape: Optional[tuple | int] = None,
                 layers_dim: List[int] = [256],
                 auto_setup: bool = True):
        self.hot_encode_action = False
        super(SimpleSPRDecoder, self).__init__(
            latent_dim=latent_dim,
            output_dim=latent_dim,
            action_shape=action_shape,
            layers_dim=layers_dim
        )
        if auto_setup:
            proj_dim = sum(latent_dim) if self.multi_input else latent_dim
            self.projection = self._instance_projection(
                proj_dim, proj_dim, layers_dim)

    def _instance_model(self, args: DecoderArguments):
        code_layers = create_mlp(args.latent_dim + args.action_shape[-1],
                                 args.latent_dim, args.layers_dim,
                                 nn.LeakyReLU, True, True)
        code_layers.insert(-1, nn.LayerNorm(args.latent_dim))
        return nn.Sequential(*code_layers)

    def _instance_projection(self, input_dim: int,
                             output_dim: int,
                             layers_dim: List[int] = [256]):
        proj_layers = create_mlp(input_dim, output_dim, layers_dim,
                                 nn.LeakyReLU, True, True)
        return nn.Sequential(*proj_layers)

    def preprocess_action(self, action):
        if self.hot_encode_action:
            hot_action = th.zeros((action.shape[0], self.action_shape[-1]))
            hot_action[th.arange(hot_action.size(0)).unsqueeze(1), action] = 1
            return hot_action.to(device=action.device)

        return action

    def forward_transition(self, z, action):
        return super().forward(z, self.preprocess_action(action))

    def forward(self, z, action):
        z = self.forward_transition(z, action)
        return self.projection(z)


class PixelDecoder(BaseDecoder):
    def __init__(self, state_shape: tuple,
                 latent_dim: int,
                 layers_filter: List[int] = [32, 32]):
        super(PixelDecoder, self).__init__(
            state_shape=state_shape,
            latent_dim=PixelEncoder.OUT_DIM[self.num_layers],
            auto_setup=False)
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
