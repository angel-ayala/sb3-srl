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
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix, matrix_to_quaternion

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


def renormalize(tensor, first_dim=1):
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = th.max(flat_tensor, first_dim, keepdim=True).values
    min = th.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min)/(max - min)

    return flat_tensor.view(*tensor.shape)


class MLP(nn.Module):
    def __init__(self, n_input, n_output, layers_dim=[256, 256]): #hidden_dim, num_layers=2):
        super(MLP, self).__init__()
        self.h_layers = nn.ModuleList([nn.Linear(n_input, layers_dim[0])])
        for i in range(len(layers_dim) - 1):
            self.h_layers.append(nn.Linear(layers_dim[i], layers_dim[i + 1]))
        self.h_layers.append(nn.Linear(layers_dim[-1], n_output))
        self.num_layers = len(self.h_layers)

    def forward(self, obs):
        h = obs
        for h_layer in self.h_layers:
            h = F.leaky_relu(h_layer(h))

        return h


class Conv1dMLP(MLP):
    def __init__(self, input_shape, n_output, layers_dim=[256, 256]):
        super(Conv1dMLP, self).__init__(input_shape[-1], n_output, layers_dim=layers_dim)
        if len(input_shape) == 2:
            self.h_layers[0] = nn.Conv1d(input_shape[0], layers_dim[0],
                                         kernel_size=input_shape[-1])

    def forward_h(self, obs):
        h = obs
        for hidden_layer in self.h_layers[:-1]:
            h = F.leaky_relu(hidden_layer(h))
            if isinstance(hidden_layer, nn.Conv1d):
                h = h.squeeze(2)
        return h

    def forward(self, obs):
        h = self.forward_h(obs)
        return self.h_layers[-1](h)


class VectorEncoder(Conv1dMLP):
    def __init__(self,
                 state_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256, 256]):
        super(VectorEncoder, self).__init__(
            state_shape, latent_dim, layers_dim=layers_dim)
        self.feature_dim = latent_dim
        self.fc = nn.Linear(latent_dim, latent_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

    def forward_feats(self, obs):
        return super().forward(obs)

    def forward_z(self, feats):
        out = self.ln(self.fc(feats))
        return th.tanh(out)

    def forward(self, obs, detach=False):
        feats = F.leaky_relu(self.forward_feats(obs))
        if detach:
            feats = feats.detach()
        return self.forward_z(feats)

    def copy_weights_from(self, source):
        """Tie hidden layers"""
        # only tie hidden layers
        for i in range(self.num_layers):
            tie_weights(src=source.h_layers[i], trg=self.h_layers[i])


class VectorDecoder(MLP):
    def __init__(self,
                 state_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256, 256]):
        super(VectorDecoder, self).__init__(
            latent_dim, state_shape[-1], layers_dim=layers_dim)
        if len(state_shape) == 2:
            self.h_layers[-1] = nn.ConvTranspose1d(
                layers_dim[0], state_shape[0], kernel_size=state_shape[-1])
        self.fc = nn.Linear(latent_dim, latent_dim)

    def forward(self, z):
        h = self.fc(z)
        for hidden_layer in self.h_layers[:-1]:
            h = F.leaky_relu(hidden_layer(h))
        last_layer = self.h_layers[-1]
        if isinstance(last_layer, nn.ConvTranspose1d):
            h = h.unsqueeze(2)
        out = last_layer(h)
        return out


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

    def forward_z_hat(self, z, action):
        return self.transition(th.cat([z, action], dim=1))

    def forward_proj(self, code):
        return self.projection(code)

    def forward(self, z, action):
        code = self.forward_z_hat(z, action)
        proj = self.forward_proj(code)
        return proj


OUT_DIM = {2: 39, 4: 35, 6: 31}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, state_shape: tuple,
                 latent_dim: int,
                 layers_filter: List[int] = [32, 32]):
        super().__init__()
        assert len(state_shape) == 3
        self.feature_dim = latent_dim
        self.num_layers = len(layers_filter)
        self.convs = nn.ModuleList(
            [nn.Conv2d(state_shape[0], layers_filter[0], 3, stride=2)]
        )
        for i in range(self.num_layers - 1):
            self.convs.append(nn.Conv2d(layers_filter[i], layers_filter[i + 1], 3, stride=1))

        out_dim = OUT_DIM[self.num_layers]
        self.fc = nn.Linear(layers_filter[-1] * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

    def forward_feats(self, obs):
        obs = obs.float() / 255.  # normalize
        conv = F.leaky_relu(self.convs[0](obs.float()))
        for i in range(1, self.num_layers):
            conv = self.convs[i](conv)
            if i + 1 < self.num_layers:
                conv = F.leaky_relu(conv)
        # last layer linear
        return conv

    def forward_z(self, feats):
        feats = feats.view(feats.size(0), -1)
        z = self.ln(self.fc(feats))
        return th.tanh(z)

    def forward(self, obs, detach=False):
        feats = F.leaky_relu(self.forward_feats(obs))
        if detach:
            feats = feats.detach()
        return self.forward_z(feats)

    def copy_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])


class PixelDecoder(nn.Module):
    def __init__(self, state_shape: tuple,
                 latent_dim: int,
                 layers_filter: List[int] = [32, 32]):
        super().__init__()
        self.num_layers = len(layers_filter)
        self.num_filters = layers_filter[0]
        self.out_dim = OUT_DIM[self.num_layers]

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


class NatureCNN(nn.Module):
    """
    CNN from DQN Nature paper:
    """

    def __init__(
        self,
        pixel_shape: tuple,
        features_dim: int = 512,
        output_dim: int = 256,
        normalized_image: bool = False) -> None:
        super().__init__()
        # We assume CxHxW images (channels first)
        n_input_channels = pixel_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        self.normalized_image = normalized_image
        # self.features_dim = features_dim
        self.linear = nn.Sequential(nn.Linear(3136, features_dim),
                                    nn.LeakyReLU(),
                                    nn.Linear(features_dim, output_dim))
        self.fc_feats = nn.Sequential(nn.Linear(output_dim, output_dim),
                                      nn.LayerNorm(output_dim),
                                      nn.Tanh())

    def forward_feats(self, observations: th.Tensor) -> th.Tensor:
        if not self.normalized_image:
            observations = observations.float() / 255.
        return self.linear(self.cnn(observations))

    def forward_z(self, feats: th.Tensor) -> th.Tensor:
        return self.fc_feats(feats)

    def forward(self, observations: th.Tensor, detach: bool = False) -> th.Tensor:
        feats = F.leaky_relu(self.forward_feats(observations))
        if detach:
            feats = feats.deatch()
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
        self.feature_dim = 0

        # Proprioceptive observation
        self.proprio = VectorEncoder((proprio_input, ), latent_dim, layers_dim)
        self.feature_dim += self.proprio.feature_dim
        # Exteroceptive observation
        self.extero = VectorEncoder((extero_input, ), latent_dim, layers_dim)
        self.feature_dim += self.extero.feature_dim
        # Pixel-based observation
        is_pixel = pixel_shape is not None
        if is_pixel:
            if self.pixel_dim is None:
                self.pixel_dim = latent_dim
            self.pixel = NatureCNN(pixel_shape, features_dim=512, output_dim=self.pixel_dim)  # 50)
            self.feature_dim += self.pixel_dim


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

    
    def forward_quaternion(self, euler):
        return matrix_to_quaternion(euler_angles_to_matrix(euler, convention='XYZ'))

    def forward(self, obs, detach=False):
        # forward features
        obs_prop = self.prop_observation(obs)
        obs_exte = self.exte_observation(obs)
        z_proprio = self.proprio(obs_prop, detach)
        z_extero = self.extero(obs_exte, detach)
        z_stack = th.cat((z_proprio, z_extero), dim=1)
        if hasattr(self, 'pixel'):
            z_pixel = self.pixel(obs['pixel'], detach)
            z_stack = th.cat((z_stack, z_pixel), dim=1)

        return z_stack


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
