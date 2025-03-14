#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 16:16:29 2025

@author: angel
"""

import torch as th
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.utils import get_parameters_by_name
from stable_baselines3.common.utils import polyak_update
from sb3_srl.autoencoders import instance_autoencoder


class SRLPolicy:
    def __init__(self, ae_config: list, encoder_tau: float = 0.999):
        self.encoder_tau = encoder_tau
        self.is_pixel = False
        self.is_multimodal = False
        self.make_autencoder(ae_config)

    @staticmethod
    def get_features_dim(ae_config):
        ae_type, ae_params = ae_config[0]
        return ae_params['latent_dim'] * len(ae_config)

    def make_autencoder(self, ae_config):
        ae_type, ae_params = ae_config[0]
        self.rep_model = instance_autoencoder(ae_type, ae_params)
        self.rep_model.adam_optimizer(ae_params['encoder_lr'],
                                      ae_params['decoder_lr'])
        self.rep_model.set_stopper(ae_params['encoder_steps'])
        self.rep_model.fit_scaler([self.observation_space.low,
                                   self.observation_space.high])
        self.is_pixel = self.observation_space.high == 255

        if len(ae_config) > 2:
            self.is_multimodal = True
            self.is_pixel = True
            ae_type, ae_params = ae_config[1]
            ae_params['encoder_only'] = True  # add encoder function only
            self.encoder_rgb = instance_autoencoder(ae_type, ae_params)
            self.encoder_rgb.enc_optimizer(ae_params['encoder_lr'])

        # self.features_dim = ae_params['latent_dim'] * len(ae_config)
    
    def forward_z(self, observation: PyTorchObs) -> th.Tensor:
        obs_z = self.rep_model.forward_z(observation)
        # if self.is_multimodal:
        #     z_rgb = self.encoder_rgb.forward_z(observation)
        #     obs_z = th.cat((obs_z, z_rgb))
        return obs_z

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        with th.no_grad():
            obs_z = self.forward_z(observation)
        return self.actor(obs_z)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.rep_model.set_training_mode(mode)
        self.training = mode

    def logger_append(self, logger, tag_prefix=''):
        self.rep_model.set_logger(logger, tag_prefix)


class SRLAlgorithm:

    def _create_aliases(self) -> None:
        self.enc_obs = self.policy.rep_model.encoder
        self.enc_obs_target = self.policy.rep_model.encoder_target

    def _setup_model(self) -> None:
        self.policy.rep_model.to(self.device)
        # Running mean and running var
        self.encoder_batch_norm_stats = get_parameters_by_name(self.enc_obs, ["running_"])
        self.encoder_batch_norm_stats_target = get_parameters_by_name(self.enc_obs_target, ["running_"])

    def update_encoder_target(self):
        polyak_update(self.enc_obs.parameters(), self.enc_obs_target.parameters(), self.policy.encoder_tau)
        polyak_update(self.encoder_batch_norm_stats, self.encoder_batch_norm_stats_target, 1.0)

    def _excluded_save_params(self) -> list[str]:
        return ["enc_obs", "enc_obs_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy.rep_model.encoder"]
        state_dicts += ["policy.rep_model.encoder_optim"]
        state_dicts += ["policy.rep_model.decoder"]
        state_dicts += ["policy.rep_model.decoder_optim"]
        if hasattr(self.policy, "rep_model.probability"):
            state_dicts += ["policy.rep_model.probability"]
            state_dicts += ["policy.rep_model.probability_optim"]
            
        return state_dicts, []
