#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 12:09:51 2025

@author: angel
"""
from typing import List

import torch as th
from torch import nn
import torch.distributions as D

from .models import RepresentationModel
from .net import PixelEncoder
from .net import SimpleSPRDecoder
from .net import VectorEncoder


def create_dist(mean, log_var):
    std = th.exp(0.5 * log_var)
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

    def forward(self, obs, detach=False):
        mean, log_var = super().forward(obs)
        dist = create_dist(mean, log_var)
        return dist  # return distribution object by default


class StochasticISPRDecoder(SimpleSPRDecoder):
    def __init__(self,
                 action_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256]):
        super(StochasticISPRDecoder, self).__init__(
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

    def forward_z(self, observation, deterministic=False):
        dist = super().forward_z(observation)
        if deterministic:
            return dist.mean
        else:
            return dist.sample()

    def target_forward_z(self, observation, deterministic=False):
        return super().target_forward_z(observation).mean


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
        self.decoder = StochasticISPRDecoder(**dec_args)

    def set_stopper(self, patience, threshold=0.):
        # not required
        pass

    def compute_representation_loss(self, observations, actions, next_observations):
        # Encode observations
        obs_z = self.encoder(observations).rsample()
        obs_z1_hat = self.decoder(obs_z, actions)
        obs_z1 = self.encoder_target(next_observations)
        # compare next_latent with transition
        loss = D.kl.kl_divergence(obs_z1, obs_z1_hat).mean()
        # L2 over Z
        latent_loss = obs_z1_hat.rsample()
        latent_loss = latent_loss.pow(2).mean()
        self.log("l2_loss", latent_loss.item())
        # loss = kl_loss + latent_loss * self.decoder_lambda
        self.log("rep_loss", loss.item())
        return loss  # *2.
