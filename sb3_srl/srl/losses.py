#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:33:26 2026

@author: angel
"""

from __future__ import annotations
from typing import Any, Optional

from stable_baselines3.common.logger import Image as ImageLogger
import torch as th
from torch import nn
import torchvision
import torch.distributions as D
from torch.nn import functional as F

from .utils import info_nce_loss
from .utils import latent_l2_loss
from .utils import obs2target_dist
from .utils import preprocess_pixel_obs

from .representation import RepresentationLoss


class ReconstructionLoss(RepresentationLoss):

    def preprocess_reconstruction(self, observations):
        # reconstruct normalized observation
        if self.encoder.is_pixels:
            obs = preprocess_pixel_obs(observations.float(), bits=5)
        return obs

    def compute_loss(self, observations, actions, next_observations):
        # Compute reconstruction loss
        obs_z = self.model.forward_representation(observations)
        rec_obs = self.model.decode_latent(obs_z)
        # MSE loss reconstruction
        obs_norm = self.preprocess_reconstruction(observations)
        rec_loss = F.mse_loss(rec_obs, obs_norm)
        # self.update_stopper(rec_loss)
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


class ReconstructionDistLoss(ReconstructionLoss):

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


class SelfPredictiveLoss(RepresentationLoss):

    def forward_y_hat(self, observation, action):
        z_t = self.model.forward_representation(observation)
        z_hat = self.decoder.transition(z_t, action)
        g0_out = self.encoder.projection(z_hat)
        y_hat = self.model.decode_latent(g0_out)
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

    def compute_loss(self, observations, actions, next_observations):
        # Compute reconstruction loss
        y_hat = self.forward_y_hat(observations, actions)
        y_curl = self.forward_y_curl(next_observations)
        loss = self.compute_similarity_loss(y_curl, y_hat)
        # L2 over Z?
        self.log("rep_loss", loss.item())
        return 2. * loss  # according to https://arxiv.org/pdf/2007.05929


class InfoSPRLoss(RepresentationLoss):

    def compute_loss(self, observations, actions, next_observations):
        # Encode observations
        obs_z = self.model.forward_representation(observations)
        obs_z1_hat = self.model.decode_latent(obs_z, actions)
        obs_z1 = self.model.forward_representation(next_observations, use_target=True, use_distribution=True)
        # compare next_latent with transition
        if self.model.is_stochastic:
            # Kullback-Leibler
            srl_loss = D.kl.kl_divergence(obs_z1, obs_z1_hat).mean()
            self.log("kl_loss", srl_loss.item())
        else:
            # contrastive loss
            srl_loss = info_nce_loss(obs_z1, obs_z1_hat)
            self.log("info_nce_loss", srl_loss.item())
        # L2 over Z
        latent_loss = latent_l2_loss(obs_z)
        self.log("z_l2", latent_loss.item())
        return srl_loss

class ALMLoss(RepresentationLoss):
    def __init__(
        self,
        aux_type: str = 'l2',
        aux_optim: str = 'ema',
        aux_coef: float = 10.0,
        disable_reward: bool = True,
        freeze_critic: bool = True,
        disable_svg: bool = True,
        seq_len: int = 1,
        encoder_lr: float = 1e-3,
        encoder_tau: float = 0.999,
        decoder_lr: Optional[float] = None,
        decoder_lambda: Optional[float] = None,
        optimizer_class: th.optim.Optimizer = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            encoder_lr=encoder_lr,
            encoder_tau=encoder_tau,
            decoder_lr=decoder_lr,
            decoder_lambda=decoder_lambda,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.aux_type = aux_type
        self.aux_optim = aux_optim
        self.aux_coef = aux_coef
        self.disable_reward = disable_reward
        self.freeze_critic = freeze_critic
        self.disable_svg = disable_svg
        self.seq_len = seq_len

        self.reward_model: Optional[nn.Module] = None
        self.actor_model: Optional[nn.Module] = None

    # def attach(self, model: "RepresentationModel") -> None:
    #     super().attach(model)
    #     # Assume model has reward and actor attached
    #     self.reward_model = getattr(model, 'reward', None)
    #     self.actor_model = getattr(model, 'actor', None)

    def compute_loss(
        self,
        observations,
        actions,
        next_observations,
        std=None,
    ) -> th.Tensor:
        """
        Compute ALM loss over sequence.
        """
        metrics = {}

        z_dist = self.encoder(observations)
        z_batch = z_dist.rsample()
        self._check_collapse(z_batch.detach(), metrics)

        log = True
        alm_loss = 0.0

        if self.disable_reward:
            aux_loss, _ = self._aux_loss(
                z_batch, actions[0], next_observations[0], log, metrics
            )
            alm_loss = self.aux_coef * aux_loss.mean()
        else:
            for t in range(self.seq_len):
                if t > 0:
                    log = False

                aux_loss, z_next_prior_batch = self._aux_loss(
                    z_batch, actions[t], next_observations[t], log, metrics
                )
                reward_loss = self._alm_reward_loss(
                    z_batch, actions[t], log, metrics
                )
                alm_loss += self.aux_coef * aux_loss - reward_loss
                z_batch = z_next_prior_batch

            alm_loss = alm_loss.mean()

        # Actor loss
        if self.freeze_critic:
            actor_loss = self.actor_model(z_batch, std, detach_qz=True, detach_action=False)
        else:
            actor_loss = self.actor_model(z_batch, std, detach_qz=False, detach_action=True)

        alm_loss += actor_loss

        for key, val in metrics.items():
            self.log(f"alm/{key}", val)

        return alm_loss

    def _check_collapse(self, z_batch, metrics):
        from torch.linalg import matrix_rank, cond

        rank3 = matrix_rank(z_batch, atol=1e-3, rtol=1e-3)
        rank2 = matrix_rank(z_batch, atol=1e-2, rtol=1e-2)
        rank1 = matrix_rank(z_batch, atol=1e-1, rtol=1e-1)
        condition = cond(z_batch)
        metrics["rank-3"] = rank3.item()
        metrics["rank-2"] = rank2.item()
        metrics["rank-1"] = rank1.item()
        metrics["cond"] = condition.item()

    def _aux_loss(self, z_batch, action_batch, next_state_batch, log, metrics):
        if "op" in self.aux_type:
            next_state_pred = self.pipeline(z_batch, action_batch)
            if self.aux_type == "op-l2":
                distance = ((next_state_pred.rsample() - next_state_batch) ** 2).sum(-1, keepdim=True)
            else:  # op-kl
                distance = -next_state_pred.log_prob(next_state_batch).unsqueeze(-1)
            if log:
                metrics[self.aux_type] = distance.mean().item()
            return distance, None

        z_next_prior_dist = self.pipeline(z_batch, action_batch)

        if self.aux_optim == "ema":
            with th.no_grad():
                z_next_dist = self.encoder_target(next_state_batch)
        elif self.aux_optim == "detach":
            with th.no_grad():
                z_next_dist = self.encoder(next_state_batch)
        else:  # online
            z_next_dist = self.encoder(next_state_batch)

        if self.aux_type == "l2":
            distance = ((z_next_dist.rsample() - z_next_prior_dist.rsample()) ** 2).sum(-1, keepdim=True)
            if log:
                metrics["l2"] = distance.mean().item()
        elif self.aux_type == "fkl":
            distance = D.kl.kl_divergence(z_next_dist, z_next_prior_dist).unsqueeze(-1)
            if log:
                metrics["fkl"] = distance.mean().item()
                metrics["prior_entropy"] = z_next_prior_dist.entropy().mean().item()
                metrics["posterior_entropy"] = z_next_dist.entropy().mean().item()
        else:  # rkl
            distance = D.kl.kl_divergence(z_next_prior_dist, z_next_dist).unsqueeze(-1)
            if log:
                metrics["rkl"] = distance.mean().item()
                metrics["prior_entropy"] = z_next_prior_dist.entropy().mean().item()
                metrics["posterior_entropy"] = z_next_dist.entropy().mean().item()

        return distance, z_next_prior_dist.rsample()

    def _alm_reward_loss(self, z_batch, action_batch, log, metrics):
        reward = self.reward_model(z_batch, action_batch)
        if log:
            metrics["alm_reward_batch"] = reward.mean().item()
        return reward



SRL_LOSS = {
    "Reconstruction": ReconstructionLoss,
    "ReconstructionDist": ReconstructionDistLoss,
    "SelfPredictive": SelfPredictiveLoss,
    "InfoSPR": InfoSPRLoss
}


def create_loss(name: str, params: dict) -> RepresentationLoss:
    try:
        loss_class = SRL_LOSS[name]
    except KeyError:
        raise ValueError(
            f"Loss '{name}' not registered. "
            f"Available: {list(SRL_LOSS)}"
        )

    return loss_class(**params)
