#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:23:42 2025

@author: angel
"""
from typing import List, Optional

import torch as th
from torch import nn
import torch.nn.functional as F
# from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix, matrix_to_quaternion

from stable_baselines3.common.torch_layers import create_mlp


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid], gain)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class VectorEncoder(nn.Module):
    def __init__(self,
                 state_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256, 256]):
        super(VectorEncoder, self).__init__()
        self.latent_dim = latent_dim
        layers = create_mlp(state_shape[-1], latent_dim, layers_dim,
                            nn.LeakyReLU, False, True)
        if len(state_shape) == 2:
            layers[0] = nn.Conv1d(state_shape[0], layers_dim[0],
                                  kernel_size=state_shape[-1])
            layers.insert(1, nn.Flatten(start_dim=1))
        self.feats_model = nn.Sequential(*layers)

        layers = [nn.LeakyReLU(),
                  nn.Linear(latent_dim, latent_dim),
                  nn.LayerNorm(latent_dim),
                  nn.Tanh()]
        self.head_model = nn.Sequential(*layers)

    def forward_feats(self, obs):
        return self.feats_model(obs)

    def forward_z(self, feats):
        return self.head_model(feats)

    def forward(self, obs):
        feats = self.forward_feats(obs)
        return self.forward_z(feats)


class VectorDecoder(nn.Module):
    def __init__(self,
                 state_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256, 256]):
        super(VectorDecoder, self).__init__()
        layers = create_mlp(latent_dim, state_shape[-1], layers_dim,
                            nn.LeakyReLU, False, True)
        layers.insert(0, nn.Linear(latent_dim, latent_dim))

        if len(state_shape) == 2:
            layers.insert(-1, nn.ConvTranspose1d(layers_dim[0], state_shape[0],
                                                 kernel_size=state_shape[-1]))
            layers.insert(-1, nn.Unflatten(2, (1, layers_dim[-1])))
        self.head_model = nn.Sequential(*layers)

    def forward(self, z):
        return self.head_model(z)


class ProjectionN(nn.Module):
    def __init__(self, latent_dim: int,
                 hidden_dim: int,
                 out_act: nn.Module = nn.Tanh()):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim),
            out_act)

    def forward(self, x):
        return self.projection(x)


class SPRDecoder(nn.Module):
    """VectorSPRDecoder for reconstruction function."""
    def __init__(self,
                 action_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256]):
        super(SPRDecoder, self).__init__()
        layers = create_mlp(latent_dim + action_shape[-1], latent_dim, layers_dim, nn.LeakyReLU, False, True)
        self.code = nn.Sequential(*layers)
        self.pred = nn.Linear(latent_dim, latent_dim)

    def transition(self, z, action):
        h_fc = self.code(th.cat([z, action], dim=1))
        return th.tanh(h_fc)

    def predict(self, z_prj):
        h_fc = self.pred(z_prj)
        return h_fc

    def forward(self, z, action):
        code = self.transition(z, action)
        return self.predict(code)


class SimpleSPRDecoder(nn.Module):
    """SimpleSPRDecoder as representation learning function."""

    def __init__(self,
                 action_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256]):
        super(SimpleSPRDecoder, self).__init__()
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
            hot_action[th.arange(hot_action.size(0)).unsqueeze(1), action] = 1
            action = hot_action

        return self.transition(th.cat([z, action], dim=1))

    def forward_proj(self, code):
        return self.projection(code)

    def forward(self, z, action):
        code = self.forward_z_hat(z, action)
        proj = self.forward_proj(code)
        return proj


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    OUT_DIM = {2: 39, 4: 35, 6: 31}

    def __init__(self,
                 state_shape: tuple,
                 latent_dim: int,
                 layers_filter: List[int] = [32, 32]):
        super().__init__()
        assert len(state_shape) == 3
        num_layers = len(layers_filter)
        self.latent_dim = latent_dim
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


class PixelDecoder(nn.Module):
    def __init__(self, state_shape: tuple,
                 latent_dim: int,
                 layers_filter: List[int] = [32, 32]):
        super().__init__()
        self.num_layers = len(layers_filter)
        self.num_filters = layers_filter[0]
        self.out_dim = PixelEncoder.OUT_DIM[self.num_layers]

        self.fc = nn.Linear(
            latent_dim, self.num_filters * self.out_dim * self.out_dim
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


class SimpleMuMoEncoder(nn.Module):
    def __init__(self, vector_encoder: nn.Module,
                 pixel_encoder: nn.Module,
                 hidden_dim: int = 256):
        super(SimpleMuMoEncoder, self).__init__()
        # self.feature_dim = vector_encoder.feature_dim + pixel_encoder.feature_dim
        self.feature_dim = pixel_encoder.feature_dim
        self.vector = vector_encoder
        self.pixel = pixel_encoder

    def forward_feats(self, obs):
        feats_vector = self.vector.forward_feats(obs['vector'])
        feats_pixel = self.pixel.forward_feats(obs['pixel'])
        return {'vector': feats_vector, 'pixel': feats_pixel}

    def forward_z(self, obs):
        z_vector = self.vector.forward_z(obs['vector'])
        z_pixel = self.pixel.forward_z(obs['pixel'])
        return {'vector': z_vector, 'pixel': z_pixel}

    def forward(self, obs, detach=False):
        feats = self.forward_feats(obs)
        if detach:
            feats['vector'] = feats['vector'].detach()
            feats['pixel'] = feats['pixel'].detach()
        return self.forward_z(feats)


class NatureCNNEncoder(nn.Module):
    """
    CNN from DQN Nature paper:
    """

    def __init__(
        self,
        state_shape: tuple,
        latent_dim: int = 256,
        features_dim: int = 512,
        normalized_image: bool = False) -> None:
        super().__init__()
        # We assume CxHxW images (channels first)
        n_input_channels = state_shape[0]
        self.features_dim = features_dim
        self.latent_dim = latent_dim
        self.feats_model = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(3136, features_dim)
        )
        self.normalized_image = normalized_image
        self.head_model = nn.Sequential(nn.LeakyReLU(),
                                        nn.Linear(features_dim, latent_dim),
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


class ProprioceptiveEncoder(nn.Module):
    def __init__(self,
                 vector_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256, 256],
                 pixel_shape: Optional[tuple] = None,
                 pixel_dim: Optional[int] = None):
        assert vector_shape[-1] == 22, "Observation state insufficient length, (IMU, Gyro, GPS, Vel, TargetSensors, Motors)"
        super(ProprioceptiveEncoder, self).__init__()
        proprio_input = 10  # = 3 imu + 3 gyro + 4 motors
        extero_input = 22 - proprio_input
        home_input = latent_dim
        self.pixel_dim = pixel_dim
        self.latent_dim = 0

        # Proprioceptive observation
        self.proprio = VectorEncoder((proprio_input, ), latent_dim, layers_dim)
        self.latent_dim += self.proprio.latent_dim
        # Exteroceptive observation
        self.extero = VectorEncoder((extero_input, ), latent_dim, layers_dim)
        self.latent_dim += self.extero.latent_dim
        # Pixel-based observation
        is_pixel = pixel_shape is not None
        if is_pixel:
            if self.pixel_dim is None:
                self.pixel_dim = latent_dim
            self.pixel = NatureCNNEncoder(pixel_shape, latent_dim=self.pixel_dim, features_dim=512)
            self.latent_dim += self.pixel_dim

    def prop_observation(self, observation):
        if isinstance(observation, dict):
            observation = observation['vector']
        if len(observation.shape) == 3:
            observation = observation[:, -1].squeeze(1)
        return th.cat((observation[:, :6], observation[:, -4:]), dim=1)

    def exte_observation(self, observation):
        if isinstance(observation, dict):
            observation = observation['vector']
        if len(observation.shape) == 3:
            exterioceptive = observation[:, -1, 6:18].squeeze(1)
        else:
            exterioceptive = observation[:, 6:18]
        return exterioceptive

    def split_observation(self, observation):
        # expecting (IMU, Gyro, GPS, Vel, TargetSensors, Motors) order
        return self.prop_observation(observation), self.exte_observation(observation)

    # def forward_quaternion(self, euler):
    #     return matrix_to_quaternion(euler_angles_to_matrix(euler, convention='XYZ'))

    def forward(self, obs):
        # forward features
        obs_prop = self.prop_observation(obs)
        obs_exte = self.exte_observation(obs)
        z_proprio = self.proprio(obs_prop)
        z_extero = self.extero(obs_exte)
        z_stack = th.cat((z_proprio, z_extero), dim=1)
        if hasattr(self, 'pixel'):
            z_pixel = self.pixel(obs['pixel'])
            z_stack = th.cat((z_stack, z_pixel), dim=1)

        return z_stack


class ProprioceptiveSPRDecoder(nn.Module):
    """ProprioceptiveSPRDecoder as representation learning function."""

    def __init__(self,
                 action_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256]):
        super(ProprioceptiveSPRDecoder, self).__init__()
        code_layers = create_mlp(latent_dim + action_shape[-1], latent_dim, layers_dim, nn.LeakyReLU, True, True)
        code_layers.insert(-1, nn.LayerNorm(latent_dim))
        self.proprio_trans = nn.Sequential(*code_layers)
        code_layers = create_mlp(latent_dim + action_shape[-1], latent_dim, layers_dim, nn.LeakyReLU, True, True)
        code_layers.insert(-1, nn.LayerNorm(latent_dim))
        self.extero_trans = nn.Sequential(*code_layers)
        out_latent = 2 * latent_dim
        proj_layers = create_mlp(out_latent, out_latent, layers_dim, nn.LeakyReLU, True, True)
        self.projection = nn.Sequential(*proj_layers)

    def forward_z_hat(self, z, action):
        proprio_z, extero_z = z.chunk(2, dim=1)
        proprio_z_hat = self.proprio_trans(th.cat([proprio_z, action], dim=1))
        extero_z_hat = self.extero_trans(th.cat([extero_z, action], dim=1))
        return th.cat([proprio_z_hat, extero_z_hat], dim=1)

    def forward_proj(self, code):
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
                 feature_dim: int,
                 layers_dim: List[int] = [256],
                 pixel_dim: Optional[int] = None):
        super(GuidedSPRDecoder, self).__init__(action_shape, feature_dim, layers_dim)
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
