#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 21:59:56 2026

@author: angel
"""
from typing import List, Optional

from dataclasses import dataclass
from stable_baselines3.common.torch_layers import create_mlp
import torch as th
from torch import nn

from .base import BaseFunction
from .base import FunctionArguments


@dataclass
class EncoderArguments(FunctionArguments):
    state_shape: tuple

    @property
    def feature_dim(self) -> int | tuple[int, ...]:
        return self.output_dim


class BaseEncoder(BaseFunction):
    def __init__(self,
                 state_shape: tuple,
                 input_dim: int | tuple[int, ...],
                 feature_dim: int | tuple[int, ...],
                 layers_dim: List[int] = [256, 256],
                 auto_setup: bool = True):
        self.state_shape = state_shape
        self.layers_dim = layers_dim
        super(BaseEncoder, self).__init__(input_dim, feature_dim, auto_setup)

    @property
    def feature_dim(self) -> int | tuple[int, ...]:
        return self.output_dim

    def _function_args(self, input_dim, output_dim):
        return EncoderArguments(
            input_dim=input_dim,
            state_shape=self.state_shape,
            output_dim=output_dim,
            layers_dim=self.layers_dim
        )


class VectorEncoder(BaseEncoder):
    def __init__(self,
                 state_shape: tuple,
                 feature_dim: int | tuple[int, ...],
                 layers_dim: List[int] = [256, 256],
                 auto_setup: bool = True):
        in_dim = state_shape[-1] if isinstance(state_shape, tuple) else state_shape
        super(VectorEncoder, self).__init__(
            state_shape=state_shape,
            input_dim=in_dim,
            feature_dim=feature_dim,
            layers_dim=layers_dim
        )

    def _instance_model(self, args: EncoderArguments):
        feats = create_mlp(args.input_dim, args.feature_dim,
                           args.layers_dim, nn.LeakyReLU, False, True)
        if isinstance(args.input_dim, tuple) and len(args.input_dim) == 2:
            feats[0] = nn.Conv1d(args.input_dim[0], args.layers_dim[0],
                                  kernel_size=args.input_dim[-1])
            feats.insert(1, nn.Flatten(start_dim=1))
        return nn.Sequential(*feats)


class SimpleSPREncoder(VectorEncoder):
    def __init__(self,
                 state_shape: tuple,
                 feature_dim: int,
                 hidden_dim: int,
                 out_act: nn.Module = nn.Tanh()):
        self.activation = out_act
        super(SimpleSPREncoder, self).__init__(
            state_shape=state_shape,
            input_dim=state_shape[-1],
            feature_dim=feature_dim)

    def _instance_model(self, args: EncoderArguments):
        head = [
            nn.Linear(args.feature_dim, args.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(args.hidden_dim, args.feature_dim),
            self.activation
        ]
        return nn.Sequential(super()._instance_model(args), *head)


class NatureCNNEncoder(BaseEncoder):
    """
    CNN from DQN Nature paper:
    """

    def __init__(
        self,
        state_shape: tuple,
        feature_dim: int = 512,
        normalized_image: bool = False) -> None:
        super(NatureCNNEncoder, self).__init__(
            state_shape=state_shape,
            input_dim=state_shape,
            feature_dim=feature_dim,
            auto_setup=False)
        # We assume CxHxW images (channels first)
        n_input_channels = state_shape[0]
        self.feats_model = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(3136, feature_dim)
        )
        self.normalized_image = normalized_image

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if not self.normalized_image:
            observations = observations.float() / 255.
        feats = self.feats_model(observations.float())
        return feats


class PixelEncoder(BaseEncoder):
    """Convolutional encoder of pixels observations."""
    OUT_DIM = {2: 39, 4: 35, 6: 31}

    def __init__(self,
                 state_shape: tuple,
                 feature_dim: int,
                 layers_filter: List[int] = [32, 32]):
        super(PixelEncoder, self).__init__(
            state_shape=state_shape,
            input_dim=state_shape,
            feature_dim=feature_dim,
            auto_setup=False)
        assert len(state_shape) == 3
        num_layers = len(layers_filter)
        feats_layers = [nn.Conv2d(state_shape[0], layers_filter[0], 3, stride=2)]
        for i in range(num_layers - 1):
            feats_layers.extend([
                nn.LeakyRelu(),
                nn.Conv2d(layers_filter[i], layers_filter[i + 1], 3, stride=1)])
        self.feats_model = nn.Sequential(*feats_layers)

        out_dim = self.OUT_DIM[num_layers]
        self.feature_dim = (layers_filter[-1], out_dim, out_dim)
        head_layers = [
            nn.LeakyRelu(),
            nn.Linear(layers_filter[-1] * out_dim * out_dim, feature_dim),
            nn.Linear(feature_dim, feature_dim),
            ]
        self.head_model = nn.Sequential(*head_layers)

    def forward(self, obs):
        feats = self.feats_model(obs.float() / 255.)
        return self.head_model(feats.view(feats.size(0), -1))


class AdPuEncoder(VectorEncoder):
    def __init__(self,
                 state_shape: tuple,
                 feature_dim: int,
                 layers_dim: List[int] = [256, 256],
                 prop_mask: list[bool] = [True, True, True, True, True, True,  # imu, gyro
                                          False, False, False, False, False, False,  # gps_pos, gps_vel
                                          False, False, False, False, False, False,  # target-sensing
                                          True, True, True, True],  # motors
                 pixel_shape: Optional[tuple] = None,
                 pixel_dim: Optional[int] = None):
        assert state_shape[-1] == len(prop_mask), f"Invalid proprioceptive mask's, length ({len(prop_mask)}) != observation length ({state_shape[-1]})."
        self.prop_mask = prop_mask
        self.exte_mask = [not m for m in self.prop_mask]
        self.pixel_shape = pixel_shape
        self.pixel_dim = pixel_dim
        proprio_input = sum(self.prop_mask)  # = 3 imu + 3 gyro + 4 motors
        extero_input = len(self.prop_mask) - proprio_input
        # split observation into proprioceptive and exteroceptive
        input_shape = (proprio_input, extero_input)
        output_shape = (feature_dim, feature_dim)

        # super(AdPuEncoder, self).__init__(
        BaseEncoder.__init__(self,
            state_shape=state_shape,
            input_dim=input_shape,
            feature_dim=output_shape,
            layers_dim=layers_dim,
            auto_setup=True)

    def instance_models(self):
        assert self.multi_input and self.multi_output
        # Proprioceptive observation
        proprio = self._instance_model(
            self._function_args(self.input_dim[0], self.feature_dim[0]))
        # Exteroceptive observation
        extero = self._instance_model(
            self._function_args(self.input_dim[1], self.feature_dim[1]))
        return nn.ModuleList([proprio, extero]), 2

    @property
    def proprio(self) -> nn.Module:
        return self.models[0]

    @property
    def extero(self) -> nn.Module:
        return self.models[1]

    def prop_observation(self, observation):
        if isinstance(observation, dict):
            observation = observation['vector']
        if len(observation.shape) == 3:
            observation = observation[:, -1].squeeze(1)
        return observation[:, self.prop_mask]

    def exte_observation(self, observation):
        if isinstance(observation, dict):
            observation = observation['vector']
        if len(observation.shape) == 3:
            observation = observation[:, -1].squeeze(1)
        return observation[:, self.exte_mask]

    @staticmethod
    def split_observation_mask(observation, prop_mask):
        return (observation[:, prop_mask],
                observation[:, [not m for m in prop_mask]])

    def split_observation(self, observation):
        return self.prop_observation(observation), self.exte_observation(observation)

    def forward(self, obs):
        # forward features
        obs_prop, obs_exte = self.split_observation(obs)
        feats_proprio = self.proprio(obs_prop)
        feats_extero = self.extero(obs_exte)
        return feats_proprio, feats_extero
