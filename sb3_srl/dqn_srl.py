#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:05:01 2025

@author: angel
"""
from typing import Any

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.utils import polyak_update

from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import QNetwork
from stable_baselines3.dqn.policies import DQNPolicy

from sb3_srl.srl import SRLPolicy
from sb3_srl.srl import SRLAlgorithm
from sb3_srl.utils import DictFlattenExtractor


class SRLDQNPolicy(DQNPolicy, SRLPolicy):
    def __init__(self, *args,
                 ae_config: dict = {},
                 encoder_tau: float = 0.999, **kwargs):
        kwargs['features_extractor_class'] = DictFlattenExtractor
        SRLPolicy.__init__(self, ae_config, encoder_tau)
        DQNPolicy.__init__(self, *args, **kwargs)

    def _build(self, lr_schedule):
        SRLPolicy._build(self, lr_schedule)
        DQNPolicy._build(self, lr_schedule)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = DQNPolicy._get_constructor_parameters(self)
        data.update(SRLPolicy._get_constructor_parameters(self))
        return data

    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        net_args["features_dim"] = self.latent_dim
        return QNetwork(**net_args).to(self.device)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        obs_z = SRLPolicy._predict(self, observation, deterministic)
        return DQNPolicy._predict(self, obs_z, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        DQNPolicy.set_training_mode(self, mode)
        SRLPolicy.set_training_mode(self, mode)


class SRLDQN(DQN, SRLAlgorithm):
    def __init__(self, *args, **kwargs):
        DQN.__init__(self, *args, **kwargs)
        # SRLAlgorithm.__init__(self, *args, **kwargs)

    def _create_aliases(self) -> None:
        DQN._create_aliases(self)
        SRLAlgorithm._create_aliases(self)

    def _setup_model(self) -> None:
        DQN._setup_model(self)
        SRLAlgorithm._setup_model(self)

    def _excluded_save_params(self) -> list[str]:
        return DQN._excluded_save_params(self) + \
            SRLAlgorithm._excluded_save_params(self)

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts1, extra1 = DQN._get_torch_save_params(self)
        state_dicts2, extra2 = SRLAlgorithm._get_torch_save_params(self)
        state_dicts = state_dicts1 + state_dicts2
        extra = extra1 + extra2
        return state_dicts, extra

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            self.update_encoder_target()  # update encoder function

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        self.policy.logger_append(self.logger, 'train/')
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        adv_values = []
        for _ in range(gradient_steps):
            # Increase update counter
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = self.gamma

            with th.no_grad():
                # encode observation
                obs_z = self.target_forward_z(replay_data.observations)
                next_obs_z = self.target_forward_z(replay_data.next_observations)
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(next_obs_z)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates
            if self.policy.rep_model.joint_optimization:
                obs_z = self.forward_z(replay_data.observations, use_grad=True)
            current_q_values = self.q_net(obs_z)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Compute reconstruction loss
            rep_loss = self.policy.rep_model.compute_representation_loss(
                replay_data.observations, replay_data.actions, replay_data.next_observations)
            # if self.policy.rep_model.is_introspection:
            #     rep_loss += self.policy.rep_model.compute_success_loss(
            #         obs_z, replay_data.actions, Q_min,
            #         next_v_values, replay_data.dones)
            # if self.policy.is_multimodal:
            #     rep_loss += self.policy.rep_model.compute_modal_loss(replay_data.observations)

            if self.policy.rep_model.joint_optimization:
                # Optimize the critics and representation
                self.policy.optimizer.zero_grad()
                self.policy.rep_model.update_representation(loss + rep_loss)
                # Clip gradient norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            else:
                self.policy.optimizer.zero_grad()
                loss.backward() # Optimize the critics first
                # Clip gradient norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                self.policy.rep_model.update_representation(rep_loss)

            if self._n_updates % 1000 == 0:
                Q_min = th.min(current_q_values, dim=1)[0].detach()
                self.policy.rep_model.log_mi(obs_z, Q_min)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
