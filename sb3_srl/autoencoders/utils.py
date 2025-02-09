#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:33:38 2025

@author: angel
"""
import torch as th
from torch.nn import functional as F
from sklearn.feature_selection import mutual_info_regression


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == th.float32
    if bits < 8:
        obs = th.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + th.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


def destack(obs_stack, len_hist=3, is_rgb=False):
    orig_shape = obs_stack.shape
    if is_rgb:
        n_stack = (orig_shape[1] // len_hist) * orig_shape[0]
        obs_destack = obs_stack.reshape((n_stack, 3) + orig_shape[-2:])
    else:
        obs_destack = obs_stack.reshape(
            (orig_shape[0] * len_hist, orig_shape[-1]))
    return obs_destack, orig_shape


def obs_reconstruction_loss(true_obs, rec_obs):
    len_stack = true_obs.shape[1]
    if len(true_obs.shape) == 3:
        # de-stack
        true_obs, _ = destack(true_obs, len_hist=len_stack, is_rgb=False)

    if len(true_obs.shape) == 4:
        # preprocess images to be in [-0.5, 0.5] range
        true_obs = preprocess_obs(true_obs)
        # de-stack
        true_obs, _ = destack(true_obs, len_hist=len_stack // 3, is_rgb=True)

    output_obs = rec_obs.reshape(true_obs.shape)

    return F.mse_loss(output_obs, true_obs, reduction='none').mean(1)


def latent_l2_loss(latent_value):
    # add L2 penalty on latent representation
    # see https://arxiv.org/pdf/1903.12436.pdf
    latent_value = 0.5 * latent_value.pow(2).sum(1)
    return latent_value

def compute_mutual_information(latents, q_values):
    mi = mutual_info_regression(latents.cpu().numpy(), q_values.cpu().numpy().reshape(-1))
    return mi.mean()
