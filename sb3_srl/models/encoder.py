#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 21:59:56 2026

@author: angel
"""
from typing import List, Optional

from stable_baselines3.common.torch_layers import create_mlp
import torch as th
from torch import nn

from .base import BaseFunction


class BaseEncoder(BaseFunction):
    def __init__(self, feature_dim: int, latent_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

    def forward_feats(self, observation):
        raise NotImplementedError

    def forward_z(self, feats):
        raise NotImplementedError

    def forward(self, observation):
        return self.forward_z(self.forward_feats(observation))


class VectorEncoder(BaseEncoder):
    def __init__(self,
                 state_shape: tuple,
                 feature_dim: int,
                 latent_dim: int,
                 layers_dim: List[int] = [256, 256]):
        super(VectorEncoder, self).__init__(feature_dim, latent_dim)
        feats = create_mlp(state_shape[-1], feature_dim, layers_dim,
                            nn.LeakyReLU, False, True)
        if len(state_shape) == 2:
            feats[0] = nn.Conv1d(state_shape[0], layers_dim[0],
                                  kernel_size=state_shape[-1])
            feats.insert(1, nn.Flatten(start_dim=1))
        self.feats_model = nn.Sequential(*feats)
        
        head = [nn.LeakyReLU(), nn.Linear(feature_dim, latent_dim, bias=True)]
        self.head_model = nn.Sequential(*head)

    def forward_feats(self, obs):
        return self.feats_model(obs)

    def forward_z(self, feats):
        return self.head_model(feats)

    def forward(self, obs):
        feats = self.forward_feats(obs)
        return self.forward_z(feats)


class SimpleSPREncoder(BaseEncoder):
    def __init__(self,
                 base_encoder: BaseEncoder,
                 feature_dim: int,
                 latent_dim: int,
                 hidden_dim: int,
                 out_act: nn.Module = nn.Tanh()):
        super(SimpleSPREncoder, self).__init__(feature_dim, latent_dim)
        self.feats_model = base_encoder
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim),
            out_act
        )

    def forward_feats(self, obs):
        return self.feats_model(obs)

    def forward_z(self, feats):
        return self.projection(feats)

    def forward(self, obs):
        feats = self.forward_feats(obs)
        return self.forward_z(feats)


class NatureCNNEncoder(BaseEncoder):
    """
    CNN from DQN Nature paper:
    """

    def __init__(
        self,
        state_shape: tuple,
        feature_dim: int = 512,
        latent_dim: int = 256,
        normalized_image: bool = False) -> None:
        super(NatureCNNEncoder, self).__init__(feature_dim, latent_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = state_shape[0]
        # self.features_dim = features_dim
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
        self.head_model = nn.Sequential(nn.LeakyReLU(),
                                        nn.Linear(feature_dim, latent_dim),
                                        nn.LeakyReLU(),
                                        nn.Linear(latent_dim, latent_dim),
                                        nn.LayerNorm(latent_dim),
                                        nn.Tanh())

    def forward_feats(self, observations: th.Tensor) -> th.Tensor:
        if not self.normalized_image:
            observations = observations.float() / 255.
        return self.feats_model(observations.float())

    def forward_z(self, feats: th.Tensor) -> th.Tensor:
        return self.head_model(feats)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        feats = self.forward_feats(observations)
        return self.forward_z(feats)


class PixelEncoder(BaseEncoder):
    """Convolutional encoder of pixels observations."""
    OUT_DIM = {2: 39, 4: 35, 6: 31}

    def __init__(self,
                 state_shape: tuple,
                 feature_dim: int,
                 latent_dim: int,
                 layers_filter: List[int] = [32, 32]):
        super(PixelEncoder, self).__init__(feature_dim, latent_dim)
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
            nn.Linear(layers_filter[-1] * out_dim * out_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh
            ]
        self.head_model = nn.Sequential(*head_layers)

    def forward_feats(self, obs):
        return self.feats_model(obs.float() / 255.)

    def forward_z(self, feats):
        return self.head_model(feats.view(feats.size(0), -1))

    def forward(self, obs):
        feats = self.forward_feats(obs)
        return self.forward_z(feats)


class AdPuEncoder(BaseEncoder):
    def __init__(self,
                 state_shape: tuple,
                 feature_dim: int,
                 latent_dim: int,
                 layers_dim: List[int] = [256, 256],
                 prop_mask: list[bool] = [True, True, True, True, True, True,  # imu, gyro
                                          False, False, False, False, False, False,  # gps_pos, gps_vel
                                          False, False, False, False, False, False,  # target-sensing
                                          True, True, True, True],  # motors
                 pixel_shape: Optional[tuple] = None,
                 pixel_dim: Optional[int] = None):
        assert state_shape[-1] == len(prop_mask), f"Invalid proprioceptive mask's, length ({len(prop_mask)}) != observation length ({state_shape[-1]})."
        super(AdPuEncoder, self).__init__(feature_dim, latent_dim)
        proprio_input = sum(prop_mask)  # = 3 imu + 3 gyro + 4 motors
        extero_input = len(prop_mask) - proprio_input
        self.prop_mask = prop_mask
        self.exte_mask = [not m for m in self.prop_mask]
        self.pixel_dim = pixel_dim
        self.latent_dim = 0

        # Proprioceptive observation
        self.proprio = VectorEncoder((proprio_input, ), feature_dim, latent_dim, layers_dim)
        self.latent_dim += self.proprio.latent_dim
        # Exteroceptive observation
        self.extero = VectorEncoder((extero_input, ), feature_dim, latent_dim, layers_dim)
        self.latent_dim += self.extero.latent_dim
        # Pixel-based observation
        is_pixel = pixel_shape is not None
        if is_pixel:
            if self.pixel_dim is None:
                self.pixel_dim = latent_dim
            self.pixel = NatureCNNEncoder(pixel_shape, feature_dim, latent_dim=self.pixel_dim)
            self.latent_dim += self.pixel_dim

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
        # expecting (IMU, Gyro, GPS, Vel, TargetSensors, Motors) order
        return self.prop_observation(observation), self.exte_observation(observation)

    # def forward_quaternion(self, euler):
    #     return matrix_to_quaternion(euler_angles_to_matrix(euler, convention='XYZ'))

    def forward_feats(self, obs):
        # forward features
        obs_prop, obs_exte = self.split_observation(obs)
        feats_proprio = self.proprio.forward_feats(obs_prop)
        feats_extero = self.extero.forward_feats(obs_exte)
        return feats_proprio, feats_extero

    def forward_z(self, feats):
        # forward heads
        feats_proprio, feats_extero = feats
        z_proprio = self.proprio.forward_z(feats_proprio)
        z_extero = self.extero.forward_z(feats_extero)
        return z_proprio, z_extero

    def forward(self, obs):
        feats = self.forward_feats(obs)
        z_proprio, z_extero = self.forward_z(feats)
        # if hasattr(self, 'pixel'):
        #     z_pixel = self.pixel(obs['pixel'])
        #     z_extero = th.cat((z_extero, z_pixel), dim=1)

        return z_proprio, z_extero
