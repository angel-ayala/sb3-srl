#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 12:09:51 2025

@author: angel
"""
from typing import List, Optional

import copy
import torch as th
from torch import nn
import torch.distributions as D
import torch.nn.functional as F
import torchvision

from stable_baselines3.common.logger import Image as ImageLogger
from stable_baselines3.common.utils import polyak_update

from .dist import create_dist
from .dist import MeanVarHead
from .fusion import fusion_model
from .models import RepresentationModel
from .net import PixelEncoder
from .net import PixelDecoder
from .net import ProprioceptiveEncoder
from .net import ProprioceptiveSPRDecoder
from .net import SimpleSPRDecoder
from .net import VectorEncoder
from .net import VectorDecoder
from .utils import latent_l2_loss
from .utils import preprocess_pixel_obs


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
                 is_multimodal: bool = False,
                 prop_mask: List[bool] = []):
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
            is_multimodal=is_multimodal,
            prop_mask=prop_mask)
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


class ReconstructionStochasticModel(RepresentationModelStochastic):
    def __init__(self, *args, **kwargs):
        super(ReconstructionStochasticModel, self).__init__(
            'Reconstruction', *args, **kwargs)
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

    def set_stopper(self, patience, threshold=0.):
        # not required
        pass

    def preprocess_reconstruction(self, observations):
        # reconstruct normalized observation
        if self.is_pixels:
            obs = preprocess_pixel_obs(observations.float(), bits=5)
        else:
            obs = self.preprocess(observations)
        return obs

    def compute_representation_loss(self, observations, actions, next_observations):
        # Compute reconstruction loss
        obs_z = self.encoder(observations).rsample()
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
                 prop_mask: list[bool] = [True, True, True, True, True, True,  # imu, gyro
                                          False, False, False, False, False, False,  # gps_pos, gps_vel
                                          False, False, False, False, False, False,  # target-sensing
                                          True, True, True, True],  # motors
                 pixel_shape: Optional[tuple] = None,
                 pixel_dim: Optional[int] = None):
        super(ProprioceptiveEncoderStochastic, self).__init__(
            vector_shape, latent_dim, layers_dim=layers_dim, prop_mask=prop_mask,
            pixel_shape=pixel_shape, pixel_dim=pixel_dim)
        # remove deterministic encoder layers
        del self.proprio.head_model
        del self.extero.head_model

        self.proprio.head_model = nn.Sequential(
            nn.LeakyReLU(),
            MeanVarHead(latent_dim, latent_dim))
        self.extero.head_model = nn.Sequential(
            nn.LeakyReLU(),
            MeanVarHead(latent_dim, latent_dim))

    def forward(self, obs):
        # forward features
        obs_prop = self.prop_observation(obs)
        obs_exte = self.exte_observation(obs)
        z_proprio = self.proprio.forward_feats(obs_prop)
        z_extero = self.extero.forward_feats(obs_exte)

        # if hasattr(self, 'pixel'):
        #     z_pixel = self.pixel(obs['pixel'])
        #     z_stack = th.cat((z_stack, z_pixel), dim=1)
        mu1, log_var1 = self.proprio.forward_z(z_proprio)
        mu2, log_var2 = self.extero.forward_z(z_extero)

        mu_stack = th.cat((mu1, mu2), dim=1)
        log_var_stack = th.cat((log_var1, log_var2), dim=1)
        dist = create_dist(mu_stack, log_var_stack)
        return dist  # return distribution object by default


class ProprioceptiveDecoderStochastic(ProprioceptiveSPRDecoder):
    """ProprioceptiveSPRDecoder as representation learning function."""

    def __init__(self,
                 action_shape: tuple,
                 latent_dim: int,
                 layers_dim: List[int] = [256],
                 with_fusion: bool = False,
                 prop_mask: list[bool] = [True, True, True, True, True, True,  # imu, gyro
                                          False, False, False, False, False, False,  # gps_pos, gps_vel
                                          False, False, False, False, False, False,  # target-sensing
                                          True, True, True, True],  # motors
                 ):
        super(ProprioceptiveDecoderStochastic, self).__init__(
            action_shape, latent_dim, layers_dim=layers_dim, prop_mask=prop_mask,
            with_fusion=with_fusion)
        if not with_fusion:
            out_latent = 2 * latent_dim
        else:
            out_latent = latent_dim
        self.projection = MeanVarHead(out_latent, out_latent)

    def forward_proj(self, z_code):
        mean, log_var = self.projection(z_code)
        dist = create_dist(mean, log_var)
        return dist  # return distribution object by default
    
    
class ProprioceptiveStochasticModel(RepresentationModelStochastic):
    def __init__(self, *args, **kwargs):
        super(ProprioceptiveStochasticModel, self).__init__('Proprioception', *args, **kwargs)
        assert not self.is_pixels or self.is_multimodal, "ProprioceptiveStochasticModel is not Pixel-based ready."

    def fit_observation(self, observation_space):
        obs_space = observation_space['vector'] if self.is_multimodal else observation_space
        super().fit_observation(obs_space)

    def preprocess(self, observations):
        obs = observations['vector'] if self.is_multimodal else observations
        return super().preprocess(obs)

    def _setup_encoder(self):
        enc_args = self.args.copy()
        del enc_args['state_shape']
        del enc_args['action_shape']
        del enc_args['layers_filter']
        enc_args['vector_shape'] = self.args['state_shape']
        enc_args['pixel_shape'] = None
        enc_args['pixel_dim'] = None

        # if self.is_multimodal:
        #     enc_args['vector_shape'] = self.args['state_shape'][0]
        #     enc_args['pixel_shape'] = self.args['state_shape'][1]
        #     enc_args['pixel_dim'] = 50
        #     self.augment_model = AutoAugment()
        self.encoder = ProprioceptiveEncoderStochastic(**enc_args)
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


class ProprioceptiveFusionStochasticModel(ProprioceptiveStochasticModel):
    def __init__(self, *args, **kwargs):
        self.fusion_type = kwargs.get('fusion', None)
        self.late_fusion = kwargs.get('late_fusion', False)
        _kwargs = kwargs.copy()
        del _kwargs['fusion']
        del _kwargs['late_fusion']

        RepresentationModelStochastic.__init__(self, "ProprioceptionFusion", *args, **_kwargs)
        assert not self.is_pixels or self.is_multimodal, "ProprioceptionFusionStochasticModel is not Pixel-based ready."
        # self.home_pos = th.FloatTensor([0., 0., 0.3])
        # self.set_scaler((-1, 1))
        # super(ProprioceptiveFusionStochasticModel, self).__init__(*args, **_kwargs)
        # self.type = "ProprioceptionFusionStochastic"

    def _setup_encoder(self):
        super()._setup_encoder()
        if self.fusion_type is not None:
            self.encoder.latent_dim = self.args['latent_dim']
        self._setup_fusion()

    def _setup_decoder(self):
        dec_args = self.args.copy()
        del dec_args['state_shape']
        del dec_args['layers_filter']

        if self.fusion_type is not None:
            dec_args['with_fusion'] = True

        self.decoder = ProprioceptiveDecoderStochastic(**dec_args)
        print(self.decoder)
    
    def _setup_fusion(self):
        if self.fusion_type is None:
            return

        self.fusion_r = fusion_model(self.fusion_type, self.args['latent_dim'], True)
        self.fusion_r_target = copy.deepcopy(self.fusion_r)
        self.fusion_r_target.train(False)
        print("Representation fusion:", self.fusion_r)
        
        if self.late_fusion:
            self.fusion_q = fusion_model(self.fusion_type, self.args['latent_dim'], True)
            self.fusion_q_target = copy.deepcopy(self.fusion_q)
            self.fusion_q_target.train(False)
            print("Critic fusion:", self.fusion_q)

    def to(self, device):
        super().to(device)
        if self.fusion_type is not None:
            self.fusion_r = self.fusion_r.to(device)
            self.fusion_r_target = self.fusion_r_target.to(device)

            if self.late_fusion:
                self.fusion_q = self.fusion_q.to(device)
                self.fusion_q_target = self.fusion_q_target.to(device)

    def enc_optimizer(self, encoder_lr, optim_class=th.optim.Adam,
                      **optim_kwargs):
        if self.fusion_type is not None and not self.late_fusion:
            enc_parameters = (list(self.encoder.parameters()) +
                              list(self.fusion_r.parameters()))
        else:
            enc_parameters = self.encoder.parameters()

        self.encoder_optim = optim_class(enc_parameters,
                                         lr=encoder_lr, **optim_kwargs)

    def dec_optimizer(self, decoder_lr, optim_class=th.optim.Adam,
                      **optim_kwargs):
        if self.fusion_type is not None and self.late_fusion:
            dec_parameters = (list(self.decoder.parameters()) +
                              list(self.fusion_r.parameters()))
        else:
            dec_parameters = self.decoder.parameters()
        self.decoder_optim = optim_class(dec_parameters,
                                         lr=decoder_lr, **optim_kwargs)
    
    def fuse_optimizer(self, fusion_lr, optim_class=th.optim.Adam,
                      **optim_kwargs):
        if self.fusion_type is not None and self.late_fusion:
            print(f"Late sensor fusion Q lr: {fusion_lr}")
            self.fusion_optim = optim_class(self.fusion_q.parameters(),
                                             lr=fusion_lr, **optim_kwargs)
        

    def update_encoder_target(self, tau):
        super().update_encoder_target(tau)
        if self.fusion_type is not None:
            polyak_update(self.fusion_r.parameters(),
                          self.fusion_r_target.parameters(),
                          tau)
    
    def update_fusion_target(self, tau):
        if self.fusion_type is not None:
            polyak_update(self.fusion_q.parameters(),
                          self.fusion_q_target.parameters(),
                          tau)

    def fuse_optim_zero_grad(self):
        if self.fusion_type is not None and self.late_fusion:
            self.fusion_optim.zero_grad()

    def fuse_optim_step(self):
        if self.fusion_type is not None and self.late_fusion:
            self.fusion_optim.step()
    
    def forward_z(self, observation, deterministic=False, use_grad=True):
        # obs_z = self.encoder(observation)  # always deterministic
        # if self.is_multimodal and not isinstance(obs_z, dict):
        #     obs_z = {'pixel': obs_z}
        obs_z = super().forward_z(observation, deterministic, use_grad)
        if self.fusion_type is not None:
            if self.late_fusion:
                obs_z = self.fusion_q(obs_z)
            else:
                obs_z = self.fusion_r(obs_z)
        
        if deterministic:
            z = obs_z.mean
        else:
            z = obs_z.rsample() if use_grad else obs_z.sample()
        return th.tanh(z)

    def target_forward_z(self, observation, deterministic=False, use_grad=True):
        # obs_z = self.encoder_target(observation)  # always deterministic
        # if self.is_multimodal and not isinstance(obs_z, dict):
        #     obs_z = {'pixel': obs_z}
        obs_z = super().target_forward_z(observation, deterministic, use_grad)
        if self.fusion_type is not None:
            if self.late_fusion:
                obs_z = self.fusion_q_target(obs_z)
            else:
                obs_z = self.fusion_r_target(obs_z)
        
        if deterministic:
            z = obs_z.mean
        else:
            z = obs_z.rsample() if use_grad else obs_z.sample()
        return th.tanh(z)

    def set_training_mode(self, mode: bool) -> None:
        super().set_training_mode(mode)
        if self.fusion_type is not None:
            self.fusion_r.train(mode)
            if self.late_fusion:
                self.fusion_q.train(mode)

    def set_stopper(self, patience, threshold=0.):
        # not required
        pass
    
    def update_representation(self, loss):
        self.fuse_optim_zero_grad()
        super().update_representation(loss)
        self.fuse_optim_step()

    def compute_representation_loss(self, observations, actions, next_observations):
        # Encode observations
        # obs_z = super().forward_z(observations, False, True)
        obs_z = self.encoder(observations).rsample()
        if self.fusion_type is not None:
            obs_z = self.fusion_r(obs_z).rsample()
        obs_z1_hat = self.decoder(obs_z, actions)
        obs_z1 = self.encoder_target(next_observations)
        # obs_z1 = super().target_forward_z(next_observations, False, False)
        if self.fusion_type is not None:
            obs_z1 = self.fusion_r_target(obs_z1.rsample())
        # compare next_latent with transition
        kl_loss = D.kl.kl_divergence(obs_z1, obs_z1_hat).mean()
        self.log("kl_loss", kl_loss.item())
        return kl_loss  # *2.

    def __repr__(self):
        out_str = super().__repr__()
        if self.fusion_type is not None and self.late_fusion:
            out_str += "\n"
            out_str += str(self.fusion_q)
        return out_str

    def __str__(self):
        out_str = super().__str__()
        return out_str + f"+({self.fusion_r.__class__.__name__}, late: {self.late_fusion})"
