#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 12:09:51 2025

@author: angel
"""
from typing import List, Optional

import torch as th
from torch import nn
import torch.distributions as D
import torch.nn.functional as F

from .models import RepresentationModel
from .net import PixelEncoder
from .net import ProprioceptiveEncoder
from .net import ProprioceptiveSPRDecoder
from .net import SimpleSPRDecoder
from .net import VectorEncoder


def create_dist(mean, log_var):
    # std = th.exp(0.5 * log_var)
    std = F.softplus(log_var) + 1e-5
    base_dist = D.Normal(mean, std)
    # transforms_list = [D.transforms.TanhTransform(cache_size=1)]
    # tanh_dist = D.TransformedDistribution(base_dist, transforms_list)
    return D.Independent(base_dist, 1)


class MeanVarHead(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super(MeanVarHead, self).__init__()
        self.head = nn.Linear(input_dim, latent_dim * 2)
        self.mu_norm = nn.LayerNorm(latent_dim)
        self.var_norm = nn.LayerNorm(latent_dim)

    def forward(self, feats):
        mu, log_var = self.head(feats).chunk(2, dim=1)
        mu = self.mu_norm(mu)
        log_var = self.var_norm(log_var)
        return mu, log_var


class VectorEncoderStochastic(VectorEncoder):
    def __init__(self,
                 state_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256, 256]):
        super(VectorEncoderStochastic, self).__init__(
            state_shape, latent_dim, layers_dim=layers_dim)
        layers = [nn.LeakyReLU(),
                  MeanVarHead(latent_dim, latent_dim)]
        self.head_model = nn.Sequential(*layers)

    def forward(self, obs):
        mean, log_var = super().forward(obs)
        dist = create_dist(mean, log_var)
        return dist  # return distribution object by default


class ISPRDecoderStochastic(SimpleSPRDecoder):
    def __init__(self,
                 action_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256]):
        super(ISPRDecoderStochastic, self).__init__(
            action_shape, latent_dim, layers_dim=layers_dim)
        self.projection = MeanVarHead(latent_dim, latent_dim)

    def forward_proj(self, z_code):
        mean, log_var = self.projection(z_code)
        dist = create_dist(mean, log_var)
        return dist  # return distribution object by default


class PixelEncoderStochastic(PixelEncoder):
    def __init__(self,
                 state_shape: tuple,
                 latent_dim: int,
                 layers_filter: List[int] = [32, 32]):
        super(PixelEncoderStochastic, self).__init__(
            state_shape, latent_dim, layers_filter=layers_filter)
        layers = [nn.LeakyReLU(),
                  MeanVarHead(self.feature_dim, latent_dim)]
        self.head_model = nn.Sequential(*layers)

    def forward(self, obs, detach=False):
        mean, log_var = super().forward(obs)
        dist = create_dist(mean, log_var)
        return dist  # return distribution object by default


class RepresentationModelStochastic(RepresentationModel):
    def __init__(self,
                 model_type: str,
                 action_shape: tuple,
                 state_shape: tuple,
                 latent_dim: int = 50,
                 layers_dim: List[int] = [256, 256],
                 layers_filter: List[int] = [32, 32],
                 encoder_only: bool = False,
                 decoder_lambda: float = 1e-6,
                 joint_optimization: bool = False,
                 introspection_lambda: float = 0,
                 is_pixels: bool = False,
                 is_multimodal: bool = False):
        super(RepresentationModelStochastic, self).__init__(
            model_type=model_type,
            action_shape=action_shape,
            state_shape=state_shape,
            latent_dim=latent_dim,
            layers_dim=layers_dim,
            layers_filter=layers_filter,
            encoder_only=encoder_only,
            decoder_lambda=decoder_lambda,
            joint_optimization=joint_optimization,
            introspection_lambda=introspection_lambda,
            is_pixels=is_pixels,
            is_multimodal=is_multimodal)
        self.type += "Stochastic"

    def _setup_encoder(self):
        enc_args = self.args.copy()
        del enc_args['action_shape']
        if self.is_multimodal:
            raise ValueError(f"{self.type}Model is not Multimodal ready!")
        elif self.is_pixels:
            del enc_args['layers_dim']
            self.encoder = PixelEncoderStochastic(**enc_args)
        else:
            del enc_args['layers_filter']
            self.encoder = VectorEncoderStochastic(**enc_args)

    def forward_z(self, observation, deterministic=False, use_grad=False):
        dist = super().forward_z(observation)
        if deterministic:
            z = dist.mean
        else:
            z = dist.rsample() if use_grad else dist.sample()
        return th.tanh(z)

    def target_forward_z(self, observation, deterministic=False, use_grad=False):
        dist = super().target_forward_z(observation)
        if deterministic:
            z = dist.mean
        else:
            z = dist.rsample() if use_grad else dist.sample()
        return th.tanh(z)


class InfoSPRStochasticModel(RepresentationModelStochastic):
    def __init__(self, *args, **kwargs):
        super(InfoSPRStochasticModel, self).__init__(
            'InfoSPR', *args, **kwargs)

    def _setup_decoder(self):
        dec_args = self.args.copy()
        del dec_args['state_shape']
        del dec_args['layers_filter']
        if self.is_pixels:
            dec_args['layers_dim'] = [dec_args['layers_dim'][-1]] * (len(dec_args['layers_dim']) - 1)
        self.decoder = ISPRDecoderStochastic(**dec_args)

    def set_stopper(self, patience, threshold=0.):
        # not required
        pass

    def compute_representation_loss(self, observations, actions, next_observations):
        # Encode observations
        obs_z = self.encoder(observations).rsample()
        obs_z1_hat = self.decoder(obs_z, actions)
        obs_z1 = self.encoder_target(next_observations)
        # compare next_latent with transition
        kl_loss = D.kl.kl_divergence(obs_z1, obs_z1_hat).mean()
        self.log("kl_loss", kl_loss.item())
        return kl_loss


class ProprioceptiveEncoderStochastic(ProprioceptiveEncoder):
    def __init__(self,
                 vector_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256, 256],
                 pixel_shape: Optional[tuple] = None,
                 pixel_dim: Optional[int] = None):
        assert vector_shape[-1] == 22, "Observation state insufficient length, (IMU, Gyro, GPS, Vel, TargetSensors, Motors)"
        super(ProprioceptiveEncoderStochastic, self).__init__(
            vector_shape, latent_dim, layers_dim=layers_dim,
            pixel_shape=pixel_shape, pixel_dim=pixel_dim)
        # remove deterministic encoder layers
        del self.proprio.head_model
        del self.extero.head_model

        layers = [nn.LeakyReLU(),
                  MeanVarHead(self.latent_dim, self.latent_dim)]
        self.head_model = nn.Sequential(*layers)

    def forward(self, obs):
        # forward features
        obs_prop = self.prop_observation(obs)
        obs_exte = self.exte_observation(obs)
        z_proprio = self.proprio.forward_feats(obs_prop)
        z_extero = self.extero.forward_feats(obs_exte)
        z_stack = th.cat((z_proprio, z_extero), dim=1)
        if hasattr(self, 'pixel'):
            z_pixel = self.pixel(obs['pixel'])
            z_stack = th.cat((z_stack, z_pixel), dim=1)

        mean, log_var = self.head_model(z_stack)
        dist = create_dist(mean, log_var)
        return dist  # return distribution object by default


class ProprioceptiveDecoderStochastic(ProprioceptiveSPRDecoder):
    """ProprioceptiveSPRDecoder as representation learning function."""

    def __init__(self,
                 action_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256]):
        super(ProprioceptiveDecoderStochastic, self).__init__(
            action_shape, latent_dim, layers_dim=layers_dim)
        out_latent = 2 * latent_dim
        self.projection = MeanVarHead(out_latent, out_latent)

    def forward_proj(self, z_code):
        mean, log_var = self.projection(z_code)
        dist = create_dist(mean, log_var)
        return dist  # return distribution object by default
    
    
class ProprioceptiveStochasticModel(RepresentationModelStochastic):
    def __init__(self, *args, **kwargs):
        super(ProprioceptiveStochasticModel, self).__init__('Proprioception', *args, **kwargs)
        assert not self.is_pixels or self.is_multimodal, "ProprioceptiveStochasticModel is not Pixel-based ready."
        self.home_pos = th.FloatTensor([0., 0., 0.3])
        self.set_scaler((-1, 1))

    def fit_observation(self, observation_space):
        obs_space = observation_space['vector'] if self.is_multimodal else observation_space
        super().fit_observation(obs_space)

    def preprocess(self, observations):
        obs = observations['vector'] if self.is_multimodal else observations
        return super().preprocess(obs)

    def _setup_encoder(self):
        state_shape = self.args['state_shape']
        pixel_shape = None
        pixel_dim = None
        # if self.is_multimodal:
        #     state_shape = self.args['state_shape'][0]
        #     pixel_shape = self.args['state_shape'][1]
        #     pixel_dim = 50
        #     self.augment_model = AutoAugment()
        self.encoder = ProprioceptiveEncoderStochastic(
            state_shape, self.args['latent_dim'],
            layers_dim=self.args['layers_dim'],
            pixel_shape=pixel_shape,
            pixel_dim=pixel_dim)
        print(self.encoder)

    def _setup_decoder(self):
        dec_args = self.args.copy()
        del dec_args['state_shape']
        del dec_args['layers_filter']
        # dec_args['latent_dim'] = self.encoder.latent_dim
        self.decoder = ProprioceptiveDecoderStochastic(**dec_args)
        print(self.decoder)

    def set_stopper(self, patience, threshold=0.):
        # not required
        pass

    def compute_representation_loss(self, observations, actions, next_observations):
        # Encode observations
        obs_z = self.encoder(observations).rsample()
        obs_z1_hat = self.decoder(obs_z, actions)
        obs_z1 = self.encoder_target(next_observations)
        # compare next_latent with transition
        kl_loss = D.kl.kl_divergence(obs_z1, obs_z1_hat).mean()
        self.log("kl_loss", kl_loss.item())
        return kl_loss  # *2.
