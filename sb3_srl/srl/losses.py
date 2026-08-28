#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:33:26 2026

@author: angel
"""

from __future__ import annotations

from stable_baselines3.common.logger import Image as ImageLogger
import torch as th
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
        self.log("l2_loss", latent_loss.item())
        # self.update_stopper(latent_loss)
        # loss = contrastive #+ latent_loss * self.decoder_lambda
        # self.log("rep_loss", loss.item())
        return srl_loss  # *2.


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
