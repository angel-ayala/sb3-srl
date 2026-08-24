#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 11:54:27 2026

@author: angel
"""

import copy
import torch as th
from torch import nn
import torch.distributions as D
from stable_baselines3.common.utils import polyak_update

from .dist import create_dist
from .dist import MeanVarHead
from .models import ProprioceptiveModel
from .net import ProprioceptiveSPRDecoder
from .utils import info_nce_loss
from .utils import latent_l2_loss
from .stochastic import ProprioceptiveStochasticModel
from .stochastic import RepresentationModelStochastic
from .stochastic import ProprioceptiveDecoderStochastic


def fusion_model(fusion_type, latent_dim, need_stochastic=False):
    fusion_name = None
    if fusion_type == 'mlp':
        fusion_name = "FusionMLP"
    elif fusion_type == 'conv1d':
        fusion_name = "FusionConv1d"
    elif fusion_type == 'gated':
        fusion_name = "FusionGated"
    elif fusion_type == 'film':
        fusion_name = "FusionFiLM"
    elif fusion_type == 'crossatt':
        fusion_name = "CrossAttention"
    else:
        raise NotImplementedError(f"Fusion method ({fusion_type}) not found, "
                                  "try with: --fusion-mlp "
                                  "--fusion-conv1d "
                                  "--fusion-gated "
                                  "--fusion-film"
                                  "--fusion-attention")
    if need_stochastic:
        fusion_name += "Stochastic"

    return globals()[fusion_name](latent_dim)


class FusionMLP(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fusion_layers = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim, bias=True)
        )
        self.activation = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Tanh()
        )

    def forward(self, z):
        if isinstance(z, tuple):
            zf = th.cat(z, dim=1)
        else:
            zf = z
        zf = self.fusion_layers(zf)
        return self.activation(zf)


class FusionMLPStochastic(FusionMLP):
    def __init__(self, latent_dim):
        super().__init__(latent_dim)
        self.activation = nn.Sequential(
            nn.LeakyReLU(),
            MeanVarHead(latent_dim, latent_dim)
        )

    def forward(self, z):
        mean, log_var = super().forward(z)
        dist = create_dist(mean, log_var)
        return dist  # return distribution object by default


class FusionConv1d(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.fusion_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=2 * latent_dim,
                out_channels=latent_dim,
                kernel_size=1,
                groups=latent_dim,
                bias=True
            ),
            nn.Flatten(start_dim=1),
        )
        self.activation = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Tanh()
        )

    def forward(self, z):
        if isinstance(z, tuple):
            zf = th.cat(z, dim=1)
        else:
            zf = z
        zf = zf.reshape(zf.shape[0], 2 * self.latent_dim, 1)
        zf = self.fusion_layers(zf)
        return self.activation(zf)


class FusionConv1dStochastic(FusionConv1d):
    def __init__(self, latent_dim):
        super().__init__(latent_dim)
        self.activation = nn.Sequential(
            nn.LeakyReLU(),
            MeanVarHead(latent_dim, latent_dim)
        )

    def forward(self, z):
        mean, log_var = super().forward(z)
        dist = create_dist(mean, log_var)
        return dist  # return distribution object by default


class FusionGated(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()
        )
        self.activation = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Tanh()
        )

    def forward(self, z):
        if isinstance(z, tuple):
            z1, z2 = z
            z_concat = th.cat(z, dim=1)
        else:
            z1, z2 = z.chunk(2, dim=1)
            z_concat = z
        
        g = self.gate(z_concat)
        return self.activation(g * z1 + (1 - g) * z2)


class FusionGatedStochastic(FusionGated):
    def __init__(self, latent_dim):
        super().__init__(latent_dim)
        self.activation = nn.Sequential(
            nn.LeakyReLU(),
            MeanVarHead(latent_dim, latent_dim)
        )

    def forward(self, z):
        mean, log_var = super().forward(z)
        dist = create_dist(mean, log_var)
        return dist  # return distribution object by default


class FusionFiLM(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.gamma = nn.Linear(latent_dim, latent_dim)
        self.beta = nn.Linear(latent_dim, latent_dim)

        self.activation = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh()
        )

    def forward(self, z):
        if isinstance(z, tuple):
            z_p, z_e = z
        else:
            z_p, z_e = z.chunk(2, dim=1)

        gamma = 1.0 + self.gamma(z_p)
        beta = self.beta(z_p)

        z_e_mod = gamma * z_e + beta
        return self.activation(th.cat([z_p, z_e_mod], dim=-1))


class FusionFiLMStochastic(FusionFiLM):
    def __init__(self, latent_dim):
        super().__init__(latent_dim)
        self.activation = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.LeakyReLU(),
            MeanVarHead(latent_dim, latent_dim)
        )

    def forward(self, z):
        mean, log_var = super().forward(z)
        dist = create_dist(mean, log_var)
        return dist  # return distribution object by default


class CrossAttention(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=4,
            batch_first=True
        )
        self.activation = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Tanh()
        )

    def forward(self, z):
        # [batch, latent_dim]
        if isinstance(z, tuple):
            latent1, latent2 = z
        else:
            latent1, latent2 = z.chunk(2, dim=1)

        # latent1 attends to latent2
        latent1 = latent1.unsqueeze(1)  # [batch, 1, latent_dim]
        latent2 = latent2.unsqueeze(1)  # [batch, 1, latent_dim]
        
        fused, _ = self.attention(latent1, latent2, latent2)
        fused = fused.squeeze(1)  # Back to [batch, latent_dim]
        
        return self.activation(fused)

class CrossAttentionStochastic(CrossAttention):
    def __init__(self, latent_dim):
        super().__init__(latent_dim)
        self.activation = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LeakyReLU(),
            MeanVarHead(latent_dim, latent_dim)
        )

    def forward(self, z):
        mean, log_var = super().forward(z)
        dist = create_dist(mean, log_var)
        return dist  # return distribution object by default


class RepresentationFusionModel:
    def __init__(self, *args, **kwargs):
        self.fusion_type = kwargs.get('fusion', None)
        self.late_fusion = kwargs.get('late_fusion', False)
        
        assert self.fusion_type is not None, "Fusion type cannot be None"

    def _setup_fusion(self, use_stochastic=False):
        self.encoder.latent_dim = self.args['latent_dim']

        self.fusion_r = fusion_model(self.fusion_type, self.args['latent_dim'], use_stochastic)
        self.fusion_r_target = copy.deepcopy(self.fusion_r)
        self.fusion_r_target.train(False)
        print("Representation fusion:", self.fusion_r)

        if self.late_fusion:
            self.fusion_q = fusion_model(self.fusion_type, self.args['latent_dim'], use_stochastic)
            self.fusion_q_target = copy.deepcopy(self.fusion_q)
            self.fusion_q_target.train(False)
            print("Critic fusion:", self.fusion_q)

    def to(self, device):
        self.fusion_r = self.fusion_r.to(device)
        self.fusion_r_target = self.fusion_r_target.to(device)

        if self.late_fusion:
            self.fusion_q = self.fusion_q.to(device)
            self.fusion_q_target = self.fusion_q_target.to(device)

    def enc_optimizer(self, encoder_lr, optim_class=th.optim.Adam,
                      **optim_kwargs):
        if not self.late_fusion:
            enc_parameters = (list(self.encoder.parameters()) +
                              list(self.fusion_r.parameters()))
        else:
            enc_parameters = self.encoder.parameters()

        self.encoder_optim = optim_class(enc_parameters,
                                         lr=encoder_lr, **optim_kwargs)

    def dec_optimizer(self, decoder_lr, optim_class=th.optim.Adam,
                      **optim_kwargs):
        if self.late_fusion:
            dec_parameters = (list(self.decoder.parameters()) +
                              list(self.fusion_r.parameters()))
        else:
            dec_parameters = self.decoder.parameters()
        self.decoder_optim = optim_class(dec_parameters,
                                         lr=decoder_lr, **optim_kwargs)

    def fuse_optimizer(self, fusion_lr, optim_class=th.optim.Adam,
                      **optim_kwargs):
        if self.late_fusion:
            print(f"Late sensor fusion Q lr: {fusion_lr}")
            self.fusion_optim = optim_class(self.fusion_q.parameters(),
                                            lr=fusion_lr, **optim_kwargs)

    def update_encoder_target(self, tau):
        polyak_update(self.fusion_r.parameters(),
                      self.fusion_r_target.parameters(),
                      tau)
        if self.late_fusion:
            polyak_update(self.fusion_q.parameters(),
                          self.fusion_q_target.parameters(),
                          tau)

    def fuse_optim_zero_grad(self):
        if self.late_fusion:
            self.fusion_optim.zero_grad()

    def fuse_optim_step(self):
        if self.late_fusion:
            self.fusion_optim.step()

    def forward_fusion(self, obs_z):
        if self.late_fusion:
            obs_z = self.fusion_q(obs_z)
        else:
            obs_z = self.fusion_r(obs_z)
        return obs_z

    def target_forward_fusion(self, obs_z):
        if self.late_fusion:
            obs_z = self.fusion_q_target(obs_z)
        else:
            obs_z = self.fusion_r_target(obs_z)
        return obs_z

    def set_training_mode(self, mode: bool) -> None:
        self.fusion_r.train(mode)
        if self.late_fusion:
            self.fusion_q.train(mode)

    def __repr__(self):
        out_str = "\nFusionRep"
        out_str += str(self.fusion_r)
        if self.late_fusion:
            out_str += "\nFusionQ"
            out_str += str(self.fusion_q)
        return out_str

    def __str__(self):
        return f"+({self.fusion_r.__class__.__name__}, late: {self.late_fusion})"


class ProprioceptiveFusionModel(ProprioceptiveModel, RepresentationFusionModel):
    def __init__(self, *args, **kwargs):
        RepresentationFusionModel.__init__(self, *args, **kwargs)
        _kwargs = kwargs.copy()
        del _kwargs['fusion']
        del _kwargs['late_fusion']
        ProprioceptiveModel.__init__(self, *args, **_kwargs)

    def _setup_encoder(self):
        ProprioceptiveModel._setup_encoder(self)
        RepresentationFusionModel._setup_fusion(self)

    def _setup_decoder(self):
        dec_args = self.args.copy()
        del dec_args['state_shape']
        del dec_args['layers_filter']

        dec_args['with_fusion'] = True

        self.decoder = ProprioceptiveSPRDecoder(**dec_args)
        print(self.decoder)

    def to(self, device):
        ProprioceptiveModel.to(self, device)
        RepresentationFusionModel.to(self, device)

    def enc_optimizer(self, encoder_lr, optim_class=th.optim.Adam,
                      **optim_kwargs):
        RepresentationFusionModel.enc_optimizer(
            self, encoder_lr, optim_class=optim_class, **optim_kwargs)

    def dec_optimizer(self, decoder_lr, optim_class=th.optim.Adam,
                      **optim_kwargs):
        RepresentationFusionModel.dec_optimizer(
            self, decoder_lr, optim_class=optim_class, **optim_kwargs)

    def update_encoder_target(self, tau):
        ProprioceptiveModel.update_encoder_target(self, tau)
        RepresentationFusionModel.update_encoder_target(self, tau)

    def forward_z(self, observation, deterministic=False, use_grad=True):
        obs_z = ProprioceptiveModel.forward_z(self, observation)
        return RepresentationFusionModel.forward_fusion(self, obs_z)

    def target_forward_z(self, observation, deterministic=False, use_grad=True):
        obs_z = ProprioceptiveModel.target_forward_z(self, observation)
        return RepresentationFusionModel.target_forward_fusion(self, obs_z)

    def set_training_mode(self, mode: bool) -> None:
        ProprioceptiveModel.set_training_mode(self, mode)
        RepresentationFusionModel.set_training_mode(self, mode)

    def set_stopper(self, patience, threshold=0.):
        # not required
        pass

    def update_representation(self, loss):
        RepresentationFusionModel.fuse_optim_zero_grad(self)
        ProprioceptiveModel.update_representation(self, loss)
        RepresentationFusionModel.fuse_optim_step(self)

    def compute_representation_loss(self, observations, actions, next_observations):
        # Encode observations
        obs_z = self.fusion_r(self.encoder(observations))
        obs_z1_hat = self.decoder(obs_z, actions)
        obs_z1 = self.fusion_r_target(self.encoder_target(next_observations))
        # compare next_latent with transition
        contrastive = info_nce_loss(obs_z1, obs_z1_hat)
        # L2 over Z
        latent_loss = latent_l2_loss(obs_z1)
        self.log("l2_loss", latent_loss.item())
        self.update_stopper(latent_loss)
        loss = contrastive  # + latent_loss * self.decoder_lambda
        self.log("rep_loss", loss.item())
        return loss  # *2.

    def __repr__(self):
        out_str = ProprioceptiveModel.__repr__(self)
        out_str += RepresentationFusionModel.__repr__(self)
        return out_str

    def __str__(self):
        out_str = ProprioceptiveModel.__str__(self)
        out_str += RepresentationFusionModel.__str__(self)
        return out_str


class ProprioceptiveFusionStochasticModel(ProprioceptiveStochasticModel, RepresentationFusionModel):
    def __init__(self, *args, **kwargs):
        RepresentationFusionModel.__init__(self, *args, **kwargs)
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
        ProprioceptiveStochasticModel._setup_encoder(self)
        # remove distribution output layer
        del self.encoder.proprio.head_model
        del self.encoder.extero.head_model

        RepresentationFusionModel._setup_fusion(self, True)

    def _setup_decoder(self):
        dec_args = self.args.copy()
        del dec_args['state_shape']
        del dec_args['layers_filter']

        dec_args['with_fusion'] = True

        self.decoder = ProprioceptiveDecoderStochastic(**dec_args)
        print(self.decoder)

    def to(self, device):
        ProprioceptiveStochasticModel.to(self, device)
        RepresentationFusionModel.to(self, device)

    def enc_optimizer(self, encoder_lr, optim_class=th.optim.Adam,
                      **optim_kwargs):
        RepresentationFusionModel.enc_optimizer(
            self, encoder_lr, optim_class=optim_class, **optim_kwargs)

    def dec_optimizer(self, decoder_lr, optim_class=th.optim.Adam,
                      **optim_kwargs):
        RepresentationFusionModel.dec_optimizer(
            self, decoder_lr, optim_class=optim_class, **optim_kwargs)

    def update_encoder_target(self, tau):
        ProprioceptiveStochasticModel.update_encoder_target(self, tau)
        RepresentationFusionModel.update_encoder_target(self, tau)

    def set_training_mode(self, mode: bool) -> None:
        ProprioceptiveStochasticModel.set_training_mode(self, mode)
        RepresentationFusionModel.set_training_mode(self, mode)

    def set_stopper(self, patience, threshold=0.):
        # not required
        pass

    def update_representation(self, loss):
        RepresentationFusionModel.fuse_optim_zero_grad(self)
        ProprioceptiveStochasticModel.update_representation(self, loss)
        RepresentationFusionModel.fuse_optim_step(self)

    def forward_z(self, observation, deterministic=False, use_grad=True):
        obs_z = self.encoder.forward_feats(observation)  # always deterministic
        obs_z = RepresentationFusionModel.forward_fusion(self, obs_z)

        if deterministic:
            z = obs_z.mean
        else:
            z = obs_z.rsample() if use_grad else obs_z.sample()
        return th.tanh(z)

    def target_forward_z(self, observation, deterministic=False, use_grad=True):
        obs_z = self.encoder_target.forward_feats(observation)  # always deterministic
        obs_z = RepresentationFusionModel.target_forward_fusion(self, obs_z)

        if deterministic:
            z = obs_z.mean
        else:
            z = obs_z.rsample() if use_grad else obs_z.sample()
        return th.tanh(z)

    def compute_representation_loss(self, observations, actions, next_observations):
        # Encode observations
        obs_z = self.encoder.forward_feats(observations)
        obs_z = self.fusion_r(obs_z).rsample()
        obs_z1_hat = self.decoder(obs_z, actions)
        obs_z1 = self.encoder_target.forward_feats(next_observations)
        obs_z1 = self.fusion_r_target(obs_z1)
        # compare next_latent with transition
        kl_loss = D.kl.kl_divergence(obs_z1, obs_z1_hat).mean()
        self.log("kl_loss", kl_loss.item())
        return kl_loss  # *2.

    def __repr__(self):
        out_str = ProprioceptiveStochasticModel.__repr__(self)
        out_str += RepresentationFusionModel.__repr__(self)
        return out_str

    def __str__(self):
        out_str = ProprioceptiveStochasticModel.__str__(self)
        out_str += RepresentationFusionModel.__str__(self)
        return out_str
