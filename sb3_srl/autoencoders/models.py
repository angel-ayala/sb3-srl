#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:23:42 2025

@author: angel
"""
import copy
import torch as th
from torch.nn import functional as F
from sklearn.preprocessing import MinMaxScaler

from .net import weight_init
from .net import VectorEncoder
from .net import VectorDecoder
from .net import VectorSPRDecoder
from .net import AdvantageDecoder
from .utils import obs_reconstruction_loss
from .utils import latent_l2_loss
from .utils import obs2target_dist
from .utils import info_nce_loss


class EarlyStopper:
    def __init__(self, patience=4500, min_delta=0.):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.stop = False

    def __call__(self, validation_loss):
        if self.stop:
            return True
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                print('EarlyStopper')
                self.stop = True
        return self.stop


class AEModel:
    LOG_FREQ = 1000

    def __init__(self,
                 ae_type: str):
        self.type = ae_type
        self.encoder = None
        self.decoder = None
        self.scaler = None
        self.stop = None

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

    def set_scaler(self, feature_range=(-1, 1)):
        """Transform features by scaling each feature to a given range.

        Instanciate the sklearn.preprocessing.MinMaxScaler.
        feature_range : tuple (min, max), default=(-1, 1)
            Desired range of transformed data.
        """
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.scaler_ok = False

    def fit_scaler(self, samples):
        self.scaler.fit(samples)
        self.scaler_ok = True

    def preprocess(self, observations):
        if self.scaler is not None:
            if not self.scaler_ok:
                print('Warning! preprocess called without fit the scaler')
            return self.scaler.transform(observations)
        else:
            return observations

    def set_stopper(self, patience, threshold=0.):
        self.stop = EarlyStopper(patience, threshold)

    def update_stopper(self, loss):
        if self.stop is not None:
            self.stop(loss)

    @property
    def must_update(self):
        return not self.stop.stop

    def make_target(self):
        self.encoder_target = copy.deepcopy(self.encoder)
        self.encoder_target.train(False)

    def update_representation(self, loss):
        if self.must_update:
            self.encoder_optim_zero_grad()
            self.decoder_optim_zero_grad()
        loss.backward()
        if self.must_update:
            self.encoder_optim_step()
            self.decoder_optim_step()

    def __repr__(self):
        out_str = f"{self.type}Model:\n"
        out_str += str(self.encoder)
        out_str += '\n'
        out_str += str(self.decoder)
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
        self.set_scaler((-1, 1))
        self.make_target()

    def compute_representation_loss(self, observations, actions, next_observations):
        # Compute reconstruction loss
        obs_z = self.encoder(observations)
        rec_obs = self.decoder(obs_z)
        # reconstruct normalized observation
        obs_norm = th.FloatTensor(self.preprocess(observations.cpu()))
        rec_loss = obs_reconstruction_loss(rec_obs, obs_norm.to(rec_obs.device))
        self.update_stopper(rec_loss)
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
        self.encoder = VectorEncoder(vector_shape, latent_dim, hidden_dim,
                                     num_layers)
        if not encoder_only:
            self.decoder = VectorSPRDecoder(action_shape, latent_dim,
                                            hidden_dim, num_layers)
            self.encoder.projection = copy.deepcopy(self.decoder.projection)
        self.decoder_latent_lambda = decoder_latent_lambda
        self.apply(weight_init)
        self.make_target()

    def fit_scaler(self, values=None):
        # not required
        pass

    def project_target(self, z):
        h_fc = self.encoder_target.projection(z)
        return th.tanh(h_fc)

    def project_online(self, z):
        h_fc = self.encoder.projection(z)
        return th.tanh(h_fc)

    def forward_y_hat(self, observation, action):
        z_t = self.encoder(observation)
        z_hat = self.decoder.transition(z_t, action)
        g0_out = self.project_online(z_hat)
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
            y_curl = self.project_target(z_t1)
        # loss = self.compute_regression_loss(y_curl, y_hat)
        contrastive = info_nce_loss(y_curl, y_hat)
        # L2 over Z
        # return loss, None
        latent_loss = latent_l2_loss(z_t1)
        self.update_stopper(latent_loss)
        loss = contrastive + latent_loss * self.decoder_latent_lambda
        return loss, latent_loss


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
        obs_dist, obs_ori = obs2target_dist(observations)
        obs_dist_norm = obs_dist.abs() / 10.  # normalize to a maximum distance
        obs_dist_norm = 2 * (obs_dist_norm - 0.5)
        obs_dist_norm[obs_dist_norm > 1.] = 1.
        obs_dist_norm[obs_dist_norm < -1.] = -1.
        obs_norm = th.FloatTensor(self.preprocess(obs_norm))
        obs_norm[:, 12] = obs_ori
        obs_norm[:, 13:] = obs_dist_norm
        rec_loss = obs_reconstruction_loss(rec_obs, obs_norm.to(rec_obs.device))
        self.update_stopper(rec_loss)
        # add L2 penalty on latent representation
        latent_loss = latent_l2_loss(obs_z)
        loss = rec_loss + latent_loss * self.decoder_latent_lambda
        return loss, latent_loss


class AdvantageModel(AEModel):
    def __init__(self,
                 vector_shape: tuple,
                 action_shape: tuple,
                 latent_dim: int = 50,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 encoder_only: bool = False,
                 decoder_latent_lambda: float = 1e-6):
        super(AdvantageModel, self).__init__('Advantage')
        self.encoder = VectorEncoder(vector_shape, latent_dim, hidden_dim,
                                     num_layers)
        if not encoder_only:
            self.decoder = AdvantageDecoder(vector_shape, action_shape, latent_dim, hidden_dim,
                                            num_layers)
        self.decoder_latent_lambda = decoder_latent_lambda
        self.apply(weight_init)
        self.make_target()

    def fit_scaler(self, values=None):
        # not required
        pass

    def compute_probability_of_success(self, Q_sa, V_s):
        """
        Computes the probability of success loss for optimization.

        Args:
            Q_sa (tuple): (Q1, Q2) from critics.
            V_s (tuple): (Q1_target, Q2_target) from target critics.

        Returns:
            torch.Tensor: Loss to maximize probability of success.
        """
        # Compute probability of success values
        ratio = V_s.squeeze() / (Q_sa + 1e-6)  # Avoid division by zero
        prob_success = 1 - th.sigmoid(ratio * 10.)
        return prob_success

    def compute_success_loss(self, observations, actions, current_q_values, target_q_values):
        success_prob = self.compute_probability_of_success(current_q_values, target_q_values)
        with th.no_grad():
            obs_z = self.encoder(observations)
        success_hat = self.decoder.forward_prob(obs_z, actions).squeeze()
        return F.mse_loss(success_prob, success_hat)

    def compute_representation_loss(self, observations, actions, next_observations):
        # Compute reconstruction loss
        obs_z = self.encoder(observations)
        adv_code = self.decoder(obs_z, actions)
        obs_z1 = self.encoder_target(next_observations)
        contrastive = info_nce_loss(obs_z1, adv_code)
        # obs_code = self.decoder.forward_proj(obs_z1)
        # contrastive = info_nce_loss(obs_code, adv_code)
        latent_loss = latent_l2_loss(obs_z1)
        self.update_stopper(latent_loss)
        loss = contrastive + latent_loss * self.decoder_latent_lambda
        return loss, latent_loss

        # obs_z1 = self.encoder(next_observations)

        # reconstruct normalized observation
        # obs_norm = th.FloatTensor(self.preprocess(observations.cpu()))
        # rec_loss = obs_reconstruction_loss(rec_obs, obs_norm.to(rec_obs.device))
        # # add L2 penalty on latent representation
        # latent_loss = latent_l2_loss(obs_z)
        # loss = rec_loss + latent_loss * self.decoder_latent_lambda
        # return loss, latent_loss
