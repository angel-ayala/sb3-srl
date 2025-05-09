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
from torchvision.transforms import AutoAugment
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3.common.logger import Image as ImageLogger

from sb3_srl.introspection import IntrospectionBelief
from sb3_srl.utils import EarlyStopper

from .net import GuidedSPRDecoder
from .net import NatureCNNEncoder
from .net import PixelDecoder
from .net import ProjectionN
from .net import ProprioceptiveEncoder
from .net import SimpleSPRDecoder
from .net import SPRDecoder
from .net import VectorDecoder
from .net import VectorEncoder
from .net import weight_init
from .utils import compute_mutual_information
from .utils import dist2orientation
from .utils import info_nce_loss
from .utils import latent_l2_loss
from .utils import obs2target_dist
from .utils import preprocess_pixel_obs


class RepresentationModel:
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
                 introspection_lambda: float = 0,
                 is_pixels: bool = False,
                 is_multimodal: bool = False):
        self.encoder = None
        self.decoder = None
        self.scaler = None
        self.stop = None
        self._log_fn = None
        self.type = model_type
        self.args = {'action_shape': action_shape,
                     'state_shape': state_shape,
                     'latent_dim': latent_dim,
                     'layers_dim': layers_dim,
                     'layers_filter': layers_filter}
        self.is_pixels = is_pixels
        self.is_multimodal = is_multimodal
        self.joint_optimization = joint_optimization
        self.decoder_lambda = decoder_lambda
        self.introspection_lambda = introspection_lambda
        self.encoder_only = encoder_only
        self.device = 'cpu'
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
        if isinstance(observation_z, dict):
            observation_z = observation_z['pixel']
        mi = compute_mutual_information(observation_z, q_min)
        self.log("mutual_information_zq", mi.mean())
        return mi

    def _setup_encoder(self):
        enc_args = self.args.copy()
        del enc_args['action_shape']
        if self.is_multimodal:
            raise ValueError(f"{self.type}Model is not Multimodal ready!")
        elif self.is_pixels:
            del enc_args['layers_dim']
            del enc_args['layers_filter']
            enc_args['features_dim'] = 512
            enc_args['normalized_image'] = False
            self.encoder = NatureCNNEncoder(**enc_args)
        else:
            del enc_args['layers_filter']
            self.encoder = VectorEncoder(**enc_args)

    def _setup_decoder(self):
        raise NotImplementedError()

    def _setup_models(self):
        self._setup_encoder()
        if not self.encoder_only:
            self._setup_decoder()
            self._introspection()
        self.apply(weight_init)
        self._setup_target()

    def _introspection(self):
        if self.introspection_lambda != 0:
            IntrospectionBelief.__init__(self, self.args['action_shape'],
                                         self.args['latent_dim'],
                                         self.args['layers_dim'][0],
                                         self.introspection_lambda)
            self.decoder.probability = self.probability

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
        if self.decoder is not None:
            self.decoder.apply(function)

    def to(self, device):
        self.device = device
        self.encoder.to(device)
        if self.decoder is not None:
            self.decoder.to(device)
        if hasattr(self, 'encoder_target'):
            self.encoder_target.to(device)

    def encoder_optim_zero_grad(self):
        self.encoder_optim.zero_grad()

    def encoder_optim_step(self):
        self.encoder_optim.step()

    def decoder_optim_zero_grad(self):
        if self.decoder is not None:
            self.decoder_optim.zero_grad()

    def decoder_optim_step(self):
        if self.decoder is not None:
            self.decoder_optim.step()

    def forward_z(self, observation, deterministic=False):
        return self.encoder(observation)  # always deterministic

    def target_forward_z(self, observation, deterministic=False):
        return self.encoder_target(observation)  # always deterministic

    def decode_latent(self, observation_z):
        return self.decoder(observation_z)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the model in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.encoder.train(mode)
        if self.decoder is not None:
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

    def fit_observation(self, observation_space):
        if self.scaler is not None:
            self.scaler.fit([observation_space.low, observation_space.high])
            self.scaler_ok = True

    def preprocess(self, observations):
        obs_out = observations.cpu()
        if self.scaler is not None:
            if not self.scaler_ok:
                print('Warning! preprocess called without fit the scaler')
            obs_out = self.scaler.transform(obs_out)
        return th.FloatTensor(obs_out).to(self.device)

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
        if isinstance(observations_z, dict):
            observations_z = observations_z['pixel']
        success_hat = self.decoder.probability(observations_z, actions)
        p_loss, success_prob, success_loss = IntrospectionBelief.success_loss(self, success_hat, current_q_values, next_v_values, dones)
        self.log("probability_success", success_prob.item())
        self.log("probability_loss", success_loss.item())
        return p_loss

    def __repr__(self):
        if self.is_pixels:
            out_str = "Pixel"
        else:
            out_str = "Vector"
        out_str += f"{self.type}Model:\n"
        out_str += str(self.encoder)
        out_str += '\n'
        out_str += str(self.decoder)
        return out_str


class ReconstructionModel(RepresentationModel):
    def __init__(self, *args, **kwargs):
        super(ReconstructionModel, self).__init__('Reconstruction', *args, **kwargs)
        if not self.is_pixels:
            self.set_scaler((-1, 1))
        self._n_calls = 0

    def _setup_decoder(self):
        dec_args = self.args.copy()
        del dec_args['action_shape']
        if self.is_pixels:
            del dec_args['layers_dim']
            self.decoder = PixelDecoder(**dec_args)
        else:
            del dec_args['layers_filter']
            self.decoder = VectorDecoder(**dec_args)

    def preprocess_reconstruction(self, observations):
        # reconstruct normalized observation
        if self.is_pixels:
            obs = preprocess_pixel_obs(observations.float(), bits=5)
        else:
            obs = self.preprocess(observations)
        return obs

    def compute_representation_loss(self, observations, actions, next_observations):
        # Compute reconstruction loss
        obs_z = self.encoder(observations)
        rec_obs = self.decoder(obs_z)
        # MSE loss reconstruction
        obs_norm = self.preprocess_reconstruction(observations)
        rec_loss = F.mse_loss(rec_obs, obs_norm)
        self.update_stopper(rec_loss)
        # add L2 penalty on latent representation
        latent_loss = latent_l2_loss(obs_z)
        loss = rec_loss + latent_loss * self.decoder_lambda
        self.log("l2_loss", latent_loss.item())
        self.log("rep_loss", loss.item())
        self._n_calls += 1
        if self._n_calls % self.LOG_FREQ == 0 and self.is_pixels:
            obs_log = rec_obs[-3:]
            if obs_log.shape[1] > 3:
                n_stack = obs_log.shape[1] // 3
                obs_log = obs_log.reshape((obs_log.shape[0] * n_stack, 3) + obs_log.shape[-2:])
            img_grid = torchvision.utils.make_grid(obs_log, nrow=3, value_range=(-.5, .5), normalize=True)
            img = ImageLogger(img_grid, 'CHW')
            self.log("pixel_reconstruction", img)
        return loss


class SelfPredictiveModel(RepresentationModel):
    def __init__(self, *args, **kwargs):
        super(SelfPredictiveModel, self).__init__('SPR', *args, **kwargs)

    def _setup_encoder(self):
        super(SelfPredictiveModel, self)._setup_encoder()
        self.encoder.projection = ProjectionN(
            self.args['latent_dim'], self.args['layers_dim'][0])

    def _setup_decoder(self):
        dec_args = self.args.copy()
        del dec_args['state_shape']
        del dec_args['layers_filter']
        if self.is_pixels:
            dec_args['layers_dim'] = [dec_args['layers_dim'][-1]] * (len(dec_args['layers_dim']) - 1)
        self.decoder = SPRDecoder(**dec_args)

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


class ReconstructionDistModel(ReconstructionModel):
    def __init__(self, *args, **kwargs):
        super(ReconstructionDistModel, self).__init__(*args, **kwargs)
        self.type = 'ReconstructionDist'

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
        # TODO: idea for pixel observation use a segmented mask
        return obs_norm


class InfoSPRModel(RepresentationModel):
    def __init__(self, *args, **kwargs):
        super(InfoSPRModel, self).__init__('InfoSPR', *args, **kwargs)

    def _setup_decoder(self):
        dec_args = self.args.copy()
        del dec_args['state_shape']
        del dec_args['layers_filter']
        if self.is_pixels:
            dec_args['layers_dim'] = [dec_args['layers_dim'][-1]] * (len(dec_args['layers_dim']) - 1)
        self.decoder = SimpleSPRDecoder(**dec_args)

    def set_stopper(self, patience, threshold=0.):
        # not required
        pass

    def compute_representation_loss(self, observations, actions, next_observations):
        # Encode observations
        obs_z = self.encoder(observations)
        obs_z1_hat = self.decoder(obs_z, actions)
        obs_z1 = self.encoder_target(next_observations)
        # compare next_latent with transition
        contrastive = info_nce_loss(obs_z1, obs_z1_hat)
        # L2 over Z
        latent_loss = latent_l2_loss(obs_z1)
        self.log("l2_loss", latent_loss.item())
        self.update_stopper(latent_loss)
        loss = contrastive + latent_loss * self.decoder_lambda
        self.log("rep_loss", loss.item())
        return loss  # *2.


class IntrospectiveInfoSPR(InfoSPRModel, IntrospectionBelief):
    def __init__(self, action_shape: tuple,
                 state_shape: tuple,
                 latent_dim: int = 50,
                 layers_dim: List[int] = [256, 256],
                 layers_filter: List[int] = [32, 32],
                 encoder_only: bool = False,
                 decoder_lambda: float = 1e-6,
                 joint_optimization: bool = False,
                 introspection_lambda: float = 0,
                 is_pixel: bool = False,
                 is_multimodal: bool = False):
        IntrospectionBelief.__init__(
            self,
            action_shape,
            input_dim=latent_dim,
            hidden_dim=layers_dim[0],
            latent_lambda=introspection_lambda)
        InfoSPRModel.__init__(
            self,
            action_shape,
            state_shape,
            latent_dim=latent_dim,
            layers_dim=layers_dim,
            layers_filter=layers_filter,
            encoder_only=encoder_only,
            decoder_lambda=decoder_lambda,
            joint_optimization=joint_optimization,
            introspection_lambda=0.,
            is_pixel=is_pixel,
            is_multimodal=is_multimodal)
        self.type = 'IntrospectiveInfoSPR'

    @property
    def is_introspection(self):
        return True

    def set_stopper(self, patience, threshold=0.):
        return IntrospectionBelief.set_stopper(self, patience, threshold)

    def enc_optimizer(self, encoder_lr, optim_class=th.optim.Adam,
                      **optim_kwargs):
        InfoSPRModel.enc_optimizer(self, encoder_lr, optim_class, **optim_kwargs)
        IntrospectionBelief.prob_optimizer(self, encoder_lr, optim_class, **optim_kwargs)

    def apply(self, function):
        InfoSPRModel.apply(self, function)
        IntrospectionBelief.apply(self, function)

    def to(self, device):
        InfoSPRModel.to(self, device)
        IntrospectionBelief.to(self, device)

    def update_representation(self, loss):
        if self.must_update_prob:
            self.prob_zero_grad()
        InfoSPRModel.update_representation(self, loss)
        if self.must_update_prob:
            self.prob_step()

    def compute_success_loss(self, observations_z, actions, current_q_values, next_v_values, dones):
        # infer the probabilities with a MLP model with NLL
        if isinstance(observations_z, dict):
            observations_z = observations_z['pixel']
        success_hat = self.infer_Ps(observations_z, actions)
        p_loss, success_prob, success_loss = IntrospectionBelief.success_loss(
            self, success_hat, current_q_values, next_v_values, dones)
        self.log("probability_success", success_prob.item())
        self.log("probability_loss", success_loss.item())
        return p_loss


class ProprioceptiveModel(RepresentationModel):
    def __init__(self, *args, **kwargs):
        super(ProprioceptiveModel, self).__init__('Proprioception', *args, **kwargs)
        assert not self.is_pixels or self.is_multimodal, "ProprioceptionModel is not Pixel-based ready."
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
        if self.is_multimodal:
            state_shape = self.args['state_shape'][0]
            pixel_shape = self.args['state_shape'][1]
            pixel_dim = 50
            self.augment_model = AutoAugment()
        self.encoder = ProprioceptiveEncoder(
            state_shape, self.args['latent_dim'],
            layers_dim=self.args['layers_dim'],
            pixel_shape=pixel_shape,
            pixel_dim=pixel_dim)
        print(self.encoder)

    def _setup_decoder(self):
        dec_args = self.args.copy()
        del dec_args['state_shape']
        del dec_args['layers_filter']
        dec_args['latent_dim'] = self.encoder.latent_dim
        self.decoder = SimpleSPRDecoder(**dec_args)
        # if self.is_multimodal:
            # dec_args['pixel_dim'] = self.encoder.pixel_dim
        # self.decoder = GuidedSPRDecoder(**dec_args)
        print(self.decoder)

    # def to(self, device):
    #     super().to(device)
    #     self.home_pos = self.home_pos.to(device)
    #     if self.is_multimodal:
    #         self.augment_model = self.augment_model.to(device)

    # def dec_optimizer(self, decoder_lr, optim_class=th.optim.Adam,
    #                   **optim_kwargs):
    #     contrastive_parameters = (list(self.decoder.transition.parameters()) +
    #                               list(self.decoder.projection.parameters()))
    #     self.decoder_optim = optim_class(contrastive_parameters,
    #                                      lr=decoder_lr, **optim_kwargs)

    #     aux_parameters = (list(self.decoder.accel_proj.parameters()) +
    #                       list(self.decoder.home_proj.parameters()))
    #     if self.is_multimodal:
    #            aux_parameters += list(self.decoder.pose_proj.parameters())
    #     self.dec_proj_optim = optim_class(aux_parameters,
    #                                      lr=decoder_lr, **optim_kwargs)

    # def decoder_optim_zero_grad(self):
    #     self.decoder_optim.zero_grad()
    #     self.dec_proj_optim.zero_grad()

    # def decoder_optim_step(self):
    #     self.decoder_optim.step()
    #     self.dec_proj_optim.step()

    def forward_z(self, observation, deterministic=False):
        obs_z = self.encoder(observation)
        if self.is_multimodal and not isinstance(obs_z, dict):
            obs_z = {'pixel': obs_z}
        return obs_z

    def target_forward_z(self, observation, deterministic=False):
        obs_z = self.encoder_target(observation)
        if self.is_multimodal and not isinstance(obs_z, dict):
            obs_z = {'pixel': obs_z}
        return obs_z

    def set_stopper(self, patience, threshold=0.):
        # not required
        pass

    def compute_representation_loss(self, observations, actions, next_observations):
        # Encode observations
        obs_z = self.encoder(observations)
        obs_z1_hat = self.decoder(obs_z, actions)
        obs_z1 = self.encoder_target(next_observations)
        # compare next_latent with transition
        contrastive = info_nce_loss(obs_z1, obs_z1_hat)
        # L2 over Z
        latent_loss = latent_l2_loss(obs_z1)
        self.log("l2_loss", latent_loss.item())
        self.update_stopper(latent_loss)
        loss = contrastive + latent_loss * self.decoder_lambda
        self.log("rep_loss", loss.item())
        return loss  # *2.
        
    def __compute_representation_loss(self, observations, actions, next_observations):
        if self.is_multimodal:
            # augment pixel observation
            for i in range(0, observations['pixel'].shape[1], 3):
                observations['pixel'][:, i:i+3] = self.augment_model(
                    observations['pixel'][:, i:i+3])
        loss = 0
        # Compare next_latent with curr_latent
        obs_z = self.encoder(observations)
        obs_z1_hat, (acc_hat, home_hat, pose_hat) = self.decoder(obs_z, actions)
        obs_z1 = self.encoder_target(next_observations)
        transition = info_nce_loss(obs_z1, obs_z1_hat)
        loss += transition

        # split observation in proprioceptive an exteroceptive
        proprio, extero = self.encoder.split_observation(observations)
        _, extero_norm = self.encoder.split_observation(self.preprocess(observations))
        proprio_t1, extero_t1 = self.encoder.split_observation(next_observations)
        proprio_t1_norm, extero_t1_norm = self.encoder.split_observation(self.preprocess(next_observations))
        # linear accelerations
        # acce = th.cat((extero[:, 3:6], proprio[:, 5:6]), dim=1)
        acc = extero_t1_norm[:, 6:9] - extero_norm[:, 6:9]
        # normalize accelerations values
        acc_hat = F.normalize(acc_hat, p=2, dim=1)
        acc = F.normalize(acc, p=2, dim=1)
        # acc_loss = F.huber_loss(acc, acc_hat)
        acc_loss = F.mse_loss(acc, acc_hat)
        loss += acc_loss * 1e-4
        self.log("rep_acc_loss", acc_loss.item())

        # distance difference to home inference
        distance_t = extero[:, :3] - self.home_pos
        distance_t1 = extero_t1[:, :3] - self.home_pos
        delta_dist = (th.linalg.norm(distance_t1, dim=1, keepdims=True)
                      - th.linalg.norm(distance_t, dim=1, keepdims=True))
        delta_high = (distance_t1[:, -1] - distance_t[:, -1]).unsqueeze(1)
        orientation = dist2orientation(distance_t)
        orientation_t1 = dist2orientation(distance_t1)
        _pi = round(th.pi, 6)
        delta_theta = (orientation_t1 - orientation + _pi) % (2 * _pi) - _pi
        # delta_theta = th.round(delta_theta, decimals=6)#.unsqueeze(1)
        # home = th.cat((delta_dist, delta_high, delta_theta), dim=1)
        home = th.cat((delta_dist, delta_high), dim=1)
        # home_loss = F.huber_loss(home, home_hat)
        home_distance = F.mse_loss(home, home_hat[:, :-1])
        home_orientation = F.mse_loss(delta_theta, home_hat[:, -1])
        # home_loss = F.mse_loss(home, home_hat)
        home_loss = home_distance + home_orientation
        loss += home_loss * 1e-5
        self.log("rep_home_loss", home_loss.item())

        if self.is_multimodal:
            # pose inference
            position = extero_t1_norm[:, :3]
            orientation = self.encoder.forward_quaternion(proprio_t1[:, :3])
            position_hat, orientation_hat = pose_hat[:, :3], pose_hat[:, -4:]
            # normalize orientation values
            orientation_hat = F.normalize(orientation_hat, p=2, dim=1)
            orientation = F.normalize(orientation, p=2, dim=1)
            # mse loss
            position_loss = F.mse_loss(position, position_hat)
            orientation_loss = F.mse_loss(orientation, orientation_hat)
            pose_loss = position_loss + orientation_loss
            loss += pose_loss * self.decoder_lambda
            self.log("rep_pose_position", position_loss.item())
            self.log("rep_pose_orientation", orientation_loss.item())
            self.log("rep_pose_loss", pose_loss.item())

        self.log("rep_loss", loss.item())
        return loss  # *2.
