#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:05:01 2025

@author: angel
"""
from typing import Optional

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.utils import polyak_update

from stable_baselines3 import TD3
from stable_baselines3.td3.policies import Actor
from stable_baselines3.td3.policies import TD3Policy

from sb3_srl.srl import SRLPolicy
from sb3_srl.srl import SRLAlgorithm
from sb3_srl.utils import DictFlattenExtractor


class SRLTD3Policy(TD3Policy, SRLPolicy):
    def __init__(self, *args,
                 ae_config: dict = {},
                 encoder_tau: float = 0.999, **kwargs):
        self.features_dim = SRLPolicy.get_features_dim(ae_config)
        kwargs['features_extractor_class'] = DictFlattenExtractor
        TD3Policy.__init__(self, *args, **kwargs)
        SRLPolicy.__init__(self, ae_config, encoder_tau)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs["features_dim"] = self.features_dim
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs["features_dim"] = self.features_dim
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return SRLPolicy._predict(self, observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        return SRLPolicy.set_training_mode(self, mode)


class SRLTD3(TD3, SRLAlgorithm):
    def __init__(self, *args, **kwargs):
        TD3.__init__(self, *args, **kwargs)
        # SRLAlgorithm.__init__(self, *args, **kwargs)

    def _create_aliases(self) -> None:
        TD3._create_aliases(self)
        SRLAlgorithm._create_aliases(self)

    def _setup_model(self) -> None:
        TD3._setup_model(self)
        SRLAlgorithm._setup_model(self)

    def _excluded_save_params(self) -> list[str]:
        return TD3._excluded_save_params(self) + \
            SRLAlgorithm._excluded_save_params(self)

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts1, extra1 = TD3._get_torch_save_params(self)
        state_dicts2, extra2 = SRLAlgorithm._get_torch_save_params(self)
        state_dicts = state_dicts1 + state_dicts2
        extra = extra1 + extra2
        return state_dicts, extra

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        self.policy.logger_append(self.logger, 'train/')

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        p_values, p_losses = [], []
        adv_values = []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                obs_z = self.target_forward_z(replay_data.observations)
                next_obs_z = self.target_forward_z(replay_data.next_observations)
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(next_obs_z) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(next_obs_z, next_actions), dim=1)
                next_v_values, _ = th.max(next_q_values, dim=1, keepdim=True)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            if self.policy.rep_model.joint_optimization:
                obs_z = self.forward_z(replay_data.observations)
            current_q_values = self.critic(obs_z, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())
            v_values = th.max(th.cat(current_q_values, dim=1), dim=1)[0].detach()
            Q_min = th.min(th.cat(current_q_values, dim=1), dim=1)[0].detach()
            adv = Q_min - v_values.squeeze()  # No division
            adv_values.append(adv.mean().item())

            # Compute reconstruction loss
            rep_loss = self.policy.rep_model.compute_representation_loss(
                replay_data.observations, replay_data.actions, replay_data.next_observations)
            if self.policy.rep_model.is_introspection:
                rep_loss += self.policy.rep_model.compute_success_loss(
                    obs_z, replay_data.actions, Q_min,
                    next_v_values, replay_data.dones)
            if self.policy.is_multimodal:
                rep_loss += self.policy.rep_model.compute_modal_loss(replay_data.observations)

            if self.policy.rep_model.joint_optimization:
                # Optimize the critics and representation
                self.critic.optimizer.zero_grad()
                self.policy.rep_model.update_representation(critic_loss + rep_loss)
                self.critic.optimizer.step()
            else:
                self.critic.optimizer.zero_grad()
                critic_loss.backward() # Optimize the critics first
                self.critic.optimizer.step()
                self.policy.rep_model.update_representation(rep_loss)

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # update representations
                self.update_encoder_target()
                _obs_z = self.target_forward_z(replay_data.observations)
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(_obs_z, self.actor(_obs_z)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

            if self._n_updates % 1000 == 0:
                self.policy.rep_model.log_mi(obs_z, Q_min)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

        self.logger.record("train/advantage_values", np.mean(adv_values))
