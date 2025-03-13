#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:23:42 2025

@author: angel
"""
from typing import List

import copy
import torch as th
import torchvision
from torch.nn import functional as F
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3.common.logger import Image as ImageLogger

from sb3_srl.introspection import IntrospectionBelief
from sb3_srl.utils import EarlyStopper

from .net import PixelDecoder
from .net import PixelEncoder
from .net import ProjectionN
from .net import SimpleSPRDecoder
from .net import SPRDecoder
from .net import VectorDecoder
from .net import VectorEncoder
from .net import weight_init
from .utils import compute_mutual_information
from .utils import info_nce_loss
from .utils import latent_l2_loss
from .utils import obs2target_dist
from .utils import preprocess_pixel_obs


class AEModel:
    LOG_FREQ = 1000

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
                 introspection_lambda: float = 0):
        self.type = model_type
        self.args = {'action_shape': action_shape,
                     'state_shape': state_shape,
                     'latent_dim': latent_dim,
                     'layers_dim': layers_dim,
                     'layers_filter': layers_filter}
        self.encoder = None
        self.decoder = None
        self.scaler = None
        self.stop = None
        self._log_fn = None
        self.joint_optimization = joint_optimization
        self.decoder_lambda = decoder_lambda
        self.introspection_lambda = introspection_lambda
        self.encoder_only = encoder_only
        self._setup_models()

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

    def _setup_models(self):
        if self.introspection_lambda != 0:
            IntrospectionBelief.__init__(self, self.args['action_shape'],
                                         self.args['latent_dim'],
                                         self.args['layers_dim'][0],
                                         self.introspection_lambda)
            self.decoder.probability = self.probability
        self.apply(weight_init)
        self._setup_target()

    @property
    def is_introspection(self):
        return hasattr(self, 'probability')

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

    def forward_z(self, observation, detach=False):
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
        if self.scaler is not None:
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

    def _setup_target(self):
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

    def compute_success_loss(self, observations_z, actions, current_q_values, next_v_values, dones):
        # infer the probabilities with a MLP model with NLL
        success_hat = self.decoder.probability(observations_z, actions)
        p_loss, success_prob, success_loss = IntrospectionBelief.success_loss(self, success_hat, current_q_values, next_v_values, dones)
        self.log("probability_success", success_prob.item())
        self.log("probability_loss", success_loss.item())
        return p_loss

    def __repr__(self):
        out_str = f"{self.type}Model:\n"
        out_str += str(self.encoder)
        out_str += '\n'
        out_str += str(self.decoder)
        return out_str


class VectorModel(AEModel):
    def __init__(self, *args, **kwargs):
        super(VectorModel, self).__init__('Vector', *args, **kwargs)
        self.set_scaler((-1, 1))

    def _setup_models(self):
        del self.args['layers_filter']
        enc_args = self.args.copy()
        del enc_args['action_shape']
        self.encoder = VectorEncoder(**enc_args)
        if not self.encoder_only:
            self.decoder = VectorDecoder(**enc_args)
        super(VectorModel, self)._setup_models()

    def preprocess_reconstruction(self, observations):
        # reconstruct normalized observation
        return th.FloatTensor(self.preprocess(observations.cpu()))

    def compute_representation_loss(self, observations, actions, next_observations):
        # Compute reconstruction loss
        obs_z = self.encoder(observations)
        rec_obs = self.decoder(obs_z)
        # MSE loss reconstruction
        obs_norm = self.preprocess_reconstruction(observations)
        rec_loss = F.mse_loss(rec_obs, obs_norm.to(rec_obs.device))
        self.update_stopper(rec_loss)
        # add L2 penalty on latent representation
        latent_loss = latent_l2_loss(obs_z)
        loss = rec_loss + latent_loss * self.decoder_lambda
        self.log("l2_loss", latent_loss.item())
        self.log("rep_loss", loss.item())
        return loss


class VectorSPRModel(AEModel):
    def __init__(self, *args, **kwargs):
        super(VectorSPRModel, self).__init__('VectorSPR', *args, **kwargs)

    def _setup_models(self):
        del self.args['layers_filter']
        enc_args = self.args.copy()
        del enc_args['action_shape']
        self.encoder = VectorEncoder(**enc_args)
        self.encoder.projection = ProjectionN(enc_args['latent_dim'], enc_args['layers_dim'][0])

        if not self.encoder_only:
            dec_args = self.args.copy()
            del dec_args['state_shape']
            self.decoder = SPRDecoder(**dec_args)
        super(VectorSPRModel, self)._setup_models()

    def set_stopper(self, patience, threshold=0.):
        # not required
        pass

    def forward_y_hat(self, observation, action):
        z_t = self.encoder(observation)
        z_hat = self.decoder.transition(z_t, action)
        g0_out = self.encoder.projection(z_hat)
        y_hat = self.decoder.predict(g0_out)
        return y_hat

    def forward_y_curl(self, next_observations):
        with th.no_grad():
            z_curl = self.encoder_target(next_observations)
            y_curl = self.encoder_target.projection(z_curl)
        return y_curl

    def compute_similarity_loss(self, y_curl, y_hat):
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
        y_curl = self.forward_y_curl(next_observations)
        loss = self.compute_similarity_loss(y_curl, y_hat)
        # L2 over Z?
        self.log("rep_loss", loss.item())
        return 2. * loss  # according to https://arxiv.org/pdf/2007.05929


class VectorTargetDistModel(VectorModel):
    def __init__(self, *args, **kwargs):
        super(VectorTargetDistModel, self).__init__(*args, **kwargs)
        self.type = 'VectorTargetDist'

    def preprocess_reconstruction(self, observations):
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
        return obs_norm


class VectorSPRIModel(AEModel):
    def __init__(self, *args, **kwargs):
        super(VectorSPRIModel, self).__init__('VectorSPRI', *args, **kwargs)

    def _setup_models(self):
        del self.args['layers_filter']
        enc_args = self.args.copy()
        del enc_args['action_shape']
        self.encoder = VectorEncoder(**enc_args)

        if not self.encoder_only:
            dec_args = self.args.copy()
            del dec_args['state_shape']
            self.decoder = SimpleSPRDecoder(**dec_args)
        super(VectorSPRIModel, self)._setup_models()

    def compute_representation_loss(self, observations, actions, next_observations):
        # Encode observations
        obs_z = self.encoder(observations)
        obs_z1_hat = self.decoder(obs_z, actions)
        obs_z1 = self.encoder_target(next_observations)
        # compare next_latent with transition
        contrastive = info_nce_loss(obs_z1, obs_z1_hat)
        # L2 over Z
        latent_loss = latent_l2_loss(obs_z1)
        self.update_stopper(latent_loss)
        loss = contrastive + latent_loss * self.decoder_lambda
        self.log("l2_loss", latent_loss.item())
        self.log("rep_loss", loss.item())
        return loss  # *2.


class VectorSPRI2Model(VectorSPRIModel, IntrospectionBelief):
    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 latent_dim: int = 50,
                 layers_dim: List[int] = [256, 256],
                 encoder_only: bool = False,
                 decoder_lambda: float = 1e-6,
                 introsp_lambda: float = 1e-3):
        IntrospectionBelief.__init__(
            self,
            action_shape,
            input_dim=latent_dim,
            hidden_dim=layers_dim[0],
            latent_lambda=introsp_lambda)
        VectorSPRIModel.__init__(
            self,
            state_shape,
            action_shape,
            latent_dim=latent_dim,
            layers_dim=layers_dim,
            encoder_only=encoder_only,
            decoder_lambda=decoder_lambda)
        self.type = 'VectorSPRI2'

    def set_stopper(self, patience, threshold=0.):
        return IntrospectionBelief.set_stopper(self, patience, threshold)

    def enc_optimizer(self, encoder_lr, optim_class=th.optim.Adam,
                      **optim_kwargs):
        VectorSPRIModel.enc_optimizer(self, encoder_lr, optim_class, **optim_kwargs)
        IntrospectionBelief.prob_optimizer(self, encoder_lr, optim_class, **optim_kwargs)

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
        # infer the probabilities with a MLP model with NLL
        success_hat = self.infer_Ps(observations_z, actions)
        p_loss, success_prob, success_loss = IntrospectionBelief.success_loss(
            self, success_hat, current_q_values, next_v_values, dones)
        self.log("probability_success", success_prob.item())
        self.log("probability_loss", success_loss.item())
        return p_loss


class PixelModel(AEModel):
    def __init__(self, *args, **kwargs):
        super(PixelModel, self).__init__('Pixel', *args, **kwargs)
        self.call = 0

    def _setup_models(self):
        del self.args['layers_dim']
        enc_args = self.args.copy()
        del enc_args['action_shape']
        self.encoder = PixelEncoder(**enc_args)
        if not self.encoder_only:
            self.decoder = PixelDecoder(**enc_args)
        super(PixelModel, self)._setup_models()

    def compute_representation_loss(self, observations, actions, next_observations):
        self.call += 1
        # Compute reconstruction loss
        obs_z = self.encoder(observations)
        rec_obs = self.decoder(obs_z)
        # MSE loss reconstruction
        obs_norm = preprocess_pixel_obs(observations.float(), bits=5)
        rec_loss = F.mse_loss(rec_obs, obs_norm.to(rec_obs.device))
        self.update_stopper(rec_loss)
        # add L2 penalty on latent representation
        latent_loss = latent_l2_loss(obs_z)
        loss = rec_loss + latent_loss * self.decoder_lambda
        self.log("l2_loss", latent_loss.item())
        self.log("rep_loss", loss.item())
        if self.LOG_FREQ % self.call == 0:
            # n_stack = obs_shape[1] // 3
            # obs_log = obs.reshape((obs_shape[0] * n_stack, 3) + obs_shape[-2:])
            img_grid = torchvision.utils.make_grid(rec_obs[-3:], nrow=3, value_range=(-.5, .5), normalize=True)
            img = ImageLogger(img_grid, 'CHW')
            self.log("pixel_reconstruction", img)
        return loss


class PixelSPRModel(VectorSPRModel):
    def __init__(self, *args, **kwargs):
       AEModel.__init__(self, 'PixelSPR', *args, **kwargs)

    def _setup_models(self):
        enc_args = self.args.copy()
        del enc_args['action_shape']
        del enc_args['layers_dim']
        self.encoder = PixelEncoder(**enc_args)
        self.encoder.projection = ProjectionN(self.args['latent_dim'], self.args['layers_dim'][0])

        if not self.encoder_only:
            dec_args = self.args.copy()
            del dec_args['state_shape']
            del dec_args['layers_filter']
            self.decoder = SPRDecoder(**dec_args)
        AEModel._setup_models(self)


class PixelSPRIModel(VectorSPRIModel):
    def __init__(self, *args, **kwargs):
       AEModel.__init__(self, 'PixelSPRI', *args, **kwargs)

    def _setup_models(self):
        enc_args = self.args.copy()
        del enc_args['action_shape']
        del enc_args['layers_dim']
        self.encoder = PixelEncoder(**enc_args)

        if not self.encoder_only:
            dec_args = self.args.copy()
            del dec_args['state_shape']
            del dec_args['layers_filter']
            self.decoder = SimpleSPRDecoder(**dec_args)
        AEModel._setup_models(self)
