#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:05:01 2025

@author: angel
"""
from typing import Optional

import copy
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update

from stable_baselines3.td3.policies import Actor
from stable_baselines3.td3.policies import TD3Policy

from sb3_srl.td3 import TD3
from sb3_srl.autoencoders import instance_autoencoder


class SRLTD3Policy(TD3Policy):
    def __init__(self, *args,
                 ae_type: str = 'Vector', ae_params : dict = {},
                 **kwargs):
        self.features_dim = ae_params['latent_dim']
        super(SRLTD3Policy, self).__init__(*args, **kwargs)

        self.make_autencoder(ae_type, ae_params)
        self.encoder_target = copy.deepcopy(self.ae_model.encoder)
        self.encoder_target.train(False)

    def make_autencoder(self, ae_type, ae_params):
        self.ae_model = instance_autoencoder(ae_type, ae_params)
        self.ae_model.adam_optimizer(ae_params['encoder_lr'],
                                     ae_params['decoder_lr'])

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs["features_dim"] = self.features_dim
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs["features_dim"] = self.features_dim
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        with th.no_grad():
            obs_z = self.ae_model.encoder(observation)
        return self.actor(obs_z)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.ae_model.set_training_mode(mode)
        self.training = mode


class SRLTD3(TD3):
    def __init__(self, *args, **kwargs):
        super(SRLTD3, self).__init__(*args, **kwargs)

    def _create_aliases(self) -> None:
        super()._create_aliases()
        self.encoder = self.policy.ae_model.encoder
        self.decoder = self.policy.ae_model.decoder
        self.encoder_target = self.policy.encoder_target

    def _setup_model(self) -> None:
        super()._setup_model()
        self.policy.ae_model.to(self.device)
        # Running mean and running var
        self.encoder_batch_norm_stats = get_parameters_by_name(self.encoder, ["running_"])
        self.encoder_batch_norm_stats_target = get_parameters_by_name(self.encoder_target, ["running_"])

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        ae_losses, l2_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                obs = self.encoder(replay_data.observations)
                next_obs = self.encoder_target(replay_data.next_observations)
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(next_obs) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(next_obs, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(obs, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute reconstruction loss
            rep_loss, [rec_loss, latent_loss] = self.policy.ae_model.compute_representation_loss(
                replay_data.observations, replay_data.actions, replay_data.next_observations)
            ae_losses.append(rec_loss.item())
            l2_losses.append(latent_loss.item())
            self.policy.ae_model.update_representation(rep_loss)

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                obs = self.encoder(replay_data.observations).detach()
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(obs, self.actor(obs)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.encoder.parameters(), self.encoder_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)
                polyak_update(self.encoder_batch_norm_stats, self.encoder_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/ae_loss", np.mean(ae_losses))
        self.logger.record("train/l2_loss", np.mean(l2_losses))
