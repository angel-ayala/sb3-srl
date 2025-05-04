#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 16:16:29 2025

@author: angel
"""
from typing import Tuple
import torch as th
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.utils import get_parameters_by_name
from stable_baselines3.common.utils import polyak_update
from sb3_srl.autoencoders import instance_autoencoder


class SRLPolicy:
    def __init__(self, ae_config: Tuple[str, dict], encoder_tau: float = 0.999):
        self.encoder_tau = encoder_tau
        self.ae_config = ae_config

    @property
    def is_multimodal(self):
        return self.rep_model.is_multimodal

    @property
    def latent_dim(self):
        return self.rep_model.encoder.latent_dim #ae_params['latent_dim'] #+ ae_params['latent_dim'] * ae_params['is_multimodal']

    def _build(self, lr_schedule=None):
        ae_type, ae_params = self.ae_config
        self.rep_model = instance_autoencoder(ae_type, ae_params)
        self.rep_model.adam_optimizer(ae_params['encoder_lr'],
                                      ae_params['decoder_lr'])
        self.rep_model.set_stopper(ae_params['encoder_steps'])
        self.rep_model.fit_observation(self.observation_space)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        with th.no_grad():
            obs_z = self.rep_model.forward_z(observation, deterministic)
        return self.actor._predict(obs_z, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.rep_model.set_training_mode(mode)
        self.training = mode

    def logger_append(self, logger, tag_prefix=''):
        self.rep_model.set_logger(logger, tag_prefix)


class SRLAlgorithm:

    def _create_aliases(self) -> None:
        self.forward_z = self.policy.rep_model.forward_z
        self.target_forward_z = self.policy.rep_model.target_forward_z

    def _setup_model(self) -> None:
        self.policy.rep_model.to(self.device)
        # Running mean and running var
        self.encoder_batch_norm_stats = get_parameters_by_name(self.policy.rep_model.encoder, ["running_"])
        self.encoder_batch_norm_stats_target = get_parameters_by_name(self.policy.rep_model.encoder_target, ["running_"])

    def update_encoder_target(self):
        polyak_update(self.policy.rep_model.encoder.parameters(), self.policy.rep_model.encoder_target.parameters(), self.policy.encoder_tau)
        polyak_update(self.encoder_batch_norm_stats, self.encoder_batch_norm_stats_target, 1.0)

    def _excluded_save_params(self) -> list[str]:
        return ["forward_z", "target_forward_z"]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy.rep_model.encoder"]
        state_dicts += ["policy.rep_model.encoder_optim"]
        if hasattr(self.policy.rep_model, 'encoder_pixel_optim'):
            state_dicts += ["policy.rep_model.encoder_pixel_optim"]
        state_dicts += ["policy.rep_model.decoder"]
        state_dicts += ["policy.rep_model.decoder_optim"]
        if hasattr(self.policy.rep_model, 'probability_optim'):
            state_dicts += ["policy.rep_model.probability"]
            state_dicts += ["policy.rep_model.probability_optim"]

        return state_dicts, []
