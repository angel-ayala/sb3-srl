#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:23:42 2025

@author: angel
"""
import copy
import torch as th
from torch.nn import functional as F

from .net import weight_init
from .net import VectorEncoder
from .net import VectorDecoder
from .net import VectorSPREncoder
from .net import VectorSPRDecoder
from .utils import obs_reconstruction_loss
from .utils import latent_l2_loss
from .utils import obs2target_diff


class AEModel:
    LOG_FREQ = 1000

    def __init__(self,
                 ae_type: str):
        self.type = ae_type
        self.encoder = None
        self.decoder = None
        self.n_calls = 0
        self.preprocess = None

    def enc_optimizer(self, encoder_lr, optim_class=th.optim.Adam,
                      **optim_kwargs):
        self.encoder_optim = optim_class(self.encoder.parameters(),
                                         lr=encoder_lr, **optim_kwargs)
    def dec_optimizer(self, decoder_lr, optim_class=th.optim.Adam,
                      **optim_kwargs):
        self.decoder_optim = optim_class(self.decoder.parameters(),
                                         lr=decoder_lr, **optim_kwargs)

    def adam_optimizer(self, encoder_lr, decoder_lr):
        self.enc_optimizer(encoder_lr)
        self.dec_optimizer(decoder_lr)

    def apply(self, function):
        self.encoder.apply(function)
        self.decoder.apply(function)

    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)
        if hasattr(self, 'encoder_target'):
            self.encoder_target.to(device)

    def encoder_optim_zero_grad(self):
        self.encoder_optim.zero_grad()

    def encoder_optim_step(self):
        self.encoder_optim.step()

    def decoder_optim_zero_grad(self):
        self.decoder_optim.zero_grad()

    def decoder_optim_step(self):
        self.decoder_optim.step()

    def encode_obs(self, observation, detach=False):
        return self.encoder(observation, detach=detach)

    def decode_latent(self, observation_z):
        return self.decoder(observation_z)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the model in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.encoder.train(mode)
        self.decoder.train(mode)

    def compute_representation_loss(self, observation, action, next_observation):
        raise NotImplementedError()

    def make_target(self):
        self.encoder_target = copy.deepcopy(self.encoder)
        self.encoder_target.train(False)

    def update_representation(self, loss):
        self.encoder_optim_zero_grad()
        self.decoder_optim_zero_grad()
        loss.backward()
        self.encoder_optim_step()
        self.decoder_optim_step()

    def __repr__(self):
        out_str = f"{self.type}Model:\n"
        for e in self.encoder:
            out_str += str(e) + '\n'
        out_str += '\n'
        for d in self.decoder:
            out_str += str(d) + '\n'
        return out_str


class VectorModel(AEModel):
    def __init__(self,
                 vector_shape: tuple,
                 latent_dim: int = 50,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 encoder_only: bool = False,
                 decoder_latent_lambda: float = 1e-6):
        super(VectorModel, self).__init__('Vector')
        self.encoder = VectorEncoder(vector_shape, latent_dim, hidden_dim,
                                     num_layers)
        if not encoder_only:
            self.decoder = VectorDecoder(vector_shape, latent_dim, hidden_dim,
                                         num_layers)
        self.decoder_latent_lambda = decoder_latent_lambda
        self.apply(weight_init)
        self.make_target()

    def compute_representation_loss(self, observations, actions, next_observations):
        # Compute reconstruction loss
        obs_z = self.encoder(observations)
        rec_obs = self.decoder(obs_z)
        # reconstruct normalized observation
        obs_norm = th.FloatTensor(self.preprocess(observations.cpu()))
        rec_loss = obs_reconstruction_loss(rec_obs, obs_norm.to(rec_obs.device))
        # add L2 penalty on latent representation
        latent_loss = latent_l2_loss(obs_z)
        loss = rec_loss + latent_loss * self.decoder_latent_lambda
        return loss, latent_loss


class VectorSPRModel(AEModel):
    def __init__(self,
                 vector_shape: tuple,
                 action_shape: tuple,
                 latent_dim: int = 50,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 encoder_only: bool = False,
                 decoder_latent_lambda: float = 1e-6):
        super(VectorSPRModel, self).__init__('VectorSPR')
        self.encoder = VectorSPREncoder(vector_shape, latent_dim, hidden_dim,
                                        num_layers)
        if not encoder_only:
            self.decoder = VectorSPRDecoder(action_shape, latent_dim,
                                            hidden_dim, num_layers)
        self.decoder_latent_lambda = decoder_latent_lambda
        self.apply(weight_init)
        self.make_target()

    def forward_y_hat(self, observation, action):
        z_t = self.encoder(observation)
        z_hat = self.decoder.transition(z_t, action)
        g0_out = self.encoder.project(z_hat)
        y_hat = self.decoder.predict(g0_out)
        return y_hat

    def compute_regression_loss(self, y_curl, y_hat):
        """Compute Similarity loss function.

        based on:
            - https://arxiv.org/pdf/2007.05929
            - https://arxiv.org/pdf/2006.07733
        """
        # https://github.com/mila-iqia/spr/blob/release/src/models.py
        f_x1 = F.normalize(y_curl.float(), p=2., dim=-1, eps=1e-3)
        f_x2 = F.normalize(y_hat.float(), p=2., dim=-1, eps=1e-3)
        # Gradients of normalized L2 loss and cosine similiarity are proportional.
        # See: https://stats.stackexchange.com/a/146279
        loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1).mean(0)
        return loss

    def compute_representation_loss(self, observations, actions, next_observations):
        # Compute reconstruction loss
        y_hat = self.forward_y_hat(observations, actions)
        with th.no_grad():
            z_t1 = self.encoder_target(next_observations)
            y_curl = self.encoder_target.project(z_t1)
        loss = self.compute_regression_loss(y_curl, y_hat)
        return loss, None


class VectorTargetDistModel(VectorModel):
    def __init__(self,
                 vector_shape: tuple,
                 latent_dim: int = 50,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 encoder_only: bool = False,
                 decoder_latent_lambda: float = 1e-6):
        super(VectorTargetDistModel, self).__init__(
            vector_shape,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            encoder_only=encoder_only,
            decoder_latent_lambda=decoder_latent_lambda)
        self.type = 'VectorTargetDist'

    def compute_representation_loss(self, observations, actions, next_observations):
        # Compute reconstruction loss
        obs_z = self.encoder(observations)
        rec_obs = self.decoder(obs_z)
        # reconstruct target distance
        obs_norm = observations.cpu().clone()  # clone to allows inplace modification
        obs_dist, obs_ori = obs2target_diff(observations)
        obs_dist_norm = obs_dist / 10.  # normalize to a maximum distance
        obs_dist_norm[obs_dist_norm > 1.] = 1.
        obs_dist_norm[obs_dist_norm < -1.] = -1.
        obs_norm = th.FloatTensor(self.preprocess(obs_norm))
        obs_norm[:, 12] = obs_ori
        obs_norm[:, 13:] = obs_dist_norm
        rec_loss = obs_reconstruction_loss(rec_obs, obs_norm.to(rec_obs.device))
        # add L2 penalty on latent representation
        latent_loss = latent_l2_loss(obs_z)
        loss = rec_loss + latent_loss * self.decoder_latent_lambda
        return loss, latent_loss
