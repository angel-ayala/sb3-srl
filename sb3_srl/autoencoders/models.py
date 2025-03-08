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

from sb3_srl.introspection import IntrospectionBelief
from .net import ProjectionN
from .net import SimpleSPRDecoder
from .net import VectorEncoder
from .net import VectorDecoder
from .net import VectorSPRDecoder
from .net import weight_init
from .utils import compute_mutual_information
from .utils import EarlyStopper
from .utils import info_nce_loss
from .utils import latent_l2_loss
from .utils import obs_reconstruction_loss
from .utils import obs2target_dist


class AEModel:
    LOG_FREQ = 1000

    def __init__(self,
                 ae_type: str):
        self.type = ae_type
        self.encoder = None
        self.decoder = None
        self.scaler = None
        self.stop = None
        self.joint_optimize = False
        self._log_fn = None

    def set_logger(self, logger_function, tag_prefix=''):
        # Expects a SB3 logger from algorithm
        self._log_fn = logger_function
        self._tag_prefix = tag_prefix

    def log(self, tag, value):
        tag = self._tag_prefix + tag
        if self._log_fn is None:
            print(f"{tag}: {value}")
        else:
            self._log_fn.record(tag, value)

    def log_mi(self, observation_z, q_min):
        # Mutual Information to assess latent features' impact
        mi = compute_mutual_information(observation_z, q_min)
        self.log("mutual_information_zq", mi.mean())
        return mi

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

    def encode(self, observation, detach=False):
        return self.encoder(observation, detach=detach)

    def encode_target(self, observation, detach=False):
        return self.encoder_target(observation, detach=detach)

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
        if self.stop is None:
            return True
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
        obs_z = self.encode(observations)
        rec_obs = self.decoder(obs_z)
        # reconstruct normalized observation
        obs_norm = th.FloatTensor(self.preprocess(observations.cpu()))
        rec_loss = obs_reconstruction_loss(rec_obs, obs_norm.to(rec_obs.device))
        self.update_stopper(rec_loss)
        # add L2 penalty on latent representation
        latent_loss = latent_l2_loss(obs_z)
        loss = rec_loss + latent_loss * self.decoder_latent_lambda
        self.log("mse_loss", rec_loss.item())
        self.log("l2_loss", latent_loss.item())
        self.log("ae_loss", loss.item())
        return loss


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
        self.joint_optimize = True
        self.encoder = VectorEncoder(vector_shape, latent_dim, hidden_dim,
                                     num_layers)
        self.encoder.projection = ProjectionN(latent_dim, hidden_dim)

        self.decoder = VectorSPRDecoder(action_shape, latent_dim,
                                        hidden_dim, num_layers)
        self.decoder_latent_lambda = decoder_latent_lambda
        self.apply(weight_init)
        self.make_target()

    def fit_scaler(self, values=None):
        # not required
        pass

    def set_stopper(self, patience, threshold=0.):
        # not required
        pass

    def forward_y_hat(self, observation, action):
        z_t = self.encode(observation)
        z_hat = self.decoder.transition(z_t, action)
        g0_out = self.encoder.projection(z_hat)
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
            y_curl = self.encode_target(next_observations)
        loss = self.compute_regression_loss(y_curl, y_hat)
        # L2 over Z?
        self.log("rep_loss", loss.item())
        return 2. * loss  # according to https://arxiv.org/pdf/2007.05929


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
        self.log("mse_loss", rec_loss.item())
        self.log("l2_loss", latent_loss.item())
        self.log("ae_loss", loss.item())
        return loss


class VectorSPRIModel(AEModel):
    def __init__(self,
                 vector_shape: tuple,
                 action_shape: tuple,
                 latent_dim: int = 50,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 encoder_only: bool = False,
                 decoder_latent_lambda: float = 1e-6):
        super(VectorSPRIModel, self).__init__('VectorSPRI')
        self.joint_optimize = True
        self.encoder = VectorEncoder(vector_shape, latent_dim, hidden_dim,
                                     num_layers)
        self.decoder = SimpleSPRDecoder(vector_shape, action_shape, latent_dim, hidden_dim,
                                        num_layers)
        self.decoder_latent_lambda = decoder_latent_lambda
        self.apply(weight_init)
        self.make_target()

    def fit_scaler(self, values=None):
        # not required
        pass

    def compute_representation_loss(self, observations, actions, next_observations):
        # Encode observations
        obs_z = self.encode(observations)
        obs_z1_hat = self.decoder(obs_z, actions)
        obs_z1 = self.encode_target(next_observations)
        # compare next_latent with transition
        contrastive = info_nce_loss(obs_z1, obs_z1_hat)
        # L2 over Z
        latent_loss = latent_l2_loss(obs_z1)
        self.update_stopper(latent_loss)
        loss = contrastive + latent_loss * self.decoder_latent_lambda
        self.log("info_nce_loss", contrastive.item())
        self.log("l2_loss", latent_loss.item())
        self.log("rep_loss", loss.item())
        return loss  # *2.


class VectorSPRI2Model(VectorSPRIModel, IntrospectionBelief):
    def __init__(self,
                 vector_shape: tuple,
                 action_shape: tuple,
                 latent_dim: int = 50,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 encoder_only: bool = False,
                 decoder_latent_lambda: float = 1e-6,
                 introsp_latent_lambda: float = 1e-3):
        IntrospectionBelief.__init__(
            self,
            action_shape,
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            latent_lambda=introsp_latent_lambda)
        VectorSPRIModel.__init__(
            self,
            vector_shape,
            action_shape,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            encoder_only=encoder_only,
            decoder_latent_lambda=decoder_latent_lambda)
        self.type = 'VectorSPRI2'

    def enc_optimizer(self, encoder_lr, optim_class=th.optim.Adam,
                      **optim_kwargs):
        VectorSPRIModel.enc_optimizer(self, encoder_lr, optim_class, **optim_kwargs)
        IntrospectionBelief.prob_optimizer(self, encoder_lr, optim_class, **optim_kwargs)

    def set_stopper(self, patience, threshold=0.):
        self.prob_stop = EarlyStopper(patience, threshold, models=[self.probability])

    @property
    def must_update_prob(self):
        return not self.prob_stop.stop

    def apply(self, function):
        VectorSPRIModel.apply(self, function)
        IntrospectionBelief.apply(self, function)

    def to(self, device):
        VectorSPRIModel.to(self, device)
        IntrospectionBelief.to(self, device)

    def update_representation(self, loss):
        if self.must_update_prob:
            self.prob_zero_grad()
        VectorSPRIModel.update_representation(self, loss)
        if self.must_update_prob:
            self.prob_step()

    def compute_success_loss(self, observations_z, actions, current_q_values, next_v_values, dones):
        # compute actual probabilities from actor and critic functions
        success_prob = self.compute_Ps(current_q_values, next_v_values, dones)
        self.log("probability_success", success_prob.mean().item())
        # infer the probabilities with a MLP model with NLL
        success_hat = self.infer_Ps(observations_z, actions)
        success_loss = self.compute_nll_loss(success_prob, success_hat)
        self.log("probability_loss", success_loss.mean().item())
        self.prob_stop(success_loss)
        # ponderate aiming to increase the probability success values
        p_loss = success_loss * self.introspection_lambda * self.must_update_prob + (success_prob.mean() - 1.)
        return p_loss  # *= 2.
