#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 22:27:14 2025

@author: angel
"""
from typing import Optional

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import Actor
from stable_baselines3.sac.policies import SACPolicy

from sb3_srl.autoencoders import instance_autoencoder
from sb3_srl.autoencoders.utils import compute_mutual_information


class SRLSACPolicy(SACPolicy):
    def __init__(self, *args,
                 ae_type: str = 'Vector', ae_params: dict = {},
                 encoder_tau: float = 0.999, **kwargs):
        self.features_dim = ae_params['latent_dim']
        super(SRLSACPolicy, self).__init__(*args, **kwargs)
        self.make_autencoder(ae_type, ae_params)
        self.encoder_tau = encoder_tau

    def make_autencoder(self, ae_type, ae_params):
        self.ae_model = instance_autoencoder(ae_type, ae_params)
        self.ae_model.adam_optimizer(ae_params['encoder_lr'],
                                     ae_params['decoder_lr'])
        self.ae_model.set_stopper(ae_params['encoder_steps'])
        self.ae_model.fit_scaler([self.observation_space.low,
                                  self.observation_space.high])

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

class SRLSAC(SAC):
    def __init__(self, *args, **kwargs):
        super(SRLSAC, self).__init__(*args, **kwargs)

    def _create_aliases(self) -> None:
        super()._create_aliases()
        self.encoder = self.policy.ae_model.encoder
        self.decoder = self.policy.ae_model.decoder
        self.encoder_target = self.policy.ae_model.encoder_target

    def _setup_model(self) -> None:
        super()._setup_model()
        self.policy.ae_model.to(self.device)
        # Running mean and running var
        self.encoder_batch_norm_stats = get_parameters_by_name(self.encoder, ["running_"])
        self.encoder_batch_norm_stats_target = get_parameters_by_name(self.encoder_target, ["running_"])

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        ae_losses, l2_losses = [], []
        mi_min_values = []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                obs_z = self.encoder_target(replay_data.observations)
                next_obs_z = self.encoder_target(replay_data.next_observations)
                if "SPR" in self.policy.ae_model.type:
                    obs_z = self.encoder.project(obs_z)
                    next_obs_z = self.encoder_target.project(next_obs_z)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(obs_z)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(next_obs_z)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(next_obs_z, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(obs_z, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute reconstruction loss
            rep_loss, latent_loss = self.policy.ae_model.compute_representation_loss(
                replay_data.observations, replay_data.actions, replay_data.next_observations)
            if latent_loss is not None:
                l2_losses.append(latent_loss.item())
            ae_losses.append(rep_loss.item())
            self.policy.ae_model.update_representation(rep_loss)

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(obs_z, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.encoder.parameters(), self.encoder_target.parameters(), self.policy.encoder_tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
                polyak_update(self.encoder_batch_norm_stats, self.encoder_batch_norm_stats_target, 1.0)

            if gradient_step % 1000 == 0:
                # Mutual Information to assess latent features' impact
                q_min, _ = th.min(th.cat(current_q_values, dim=1), dim=1)
                mi_min = compute_mutual_information(obs_z, q_min.detach())
                mi_min_values.append(mi_min)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

        self.logger.record("train/ae_loss", np.mean(ae_losses))
        if len(l2_losses) > 0:
            self.logger.record("train/l2_loss", np.mean(l2_losses))
        if len(mi_min_values) > 0:
            self.logger.record("train/mutual_information_min", np.mean(mi_min_values))

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params(
            ) + ["encoder", "decoder", "encoder_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts, _ = super()._get_torch_save_params()
        state_dicts += ["policy.ae_model.encoder"]
        state_dicts += ["policy.ae_model.encoder_optim"]
        state_dicts += ["policy.ae_model.decoder"]
        state_dicts += ["policy.ae_model.decoder_optim"]
        return state_dicts, []
