#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 20:27:34 2026

@author: angel
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

import copy
import torch as th
from torch import nn
from stable_baselines3.common.utils import polyak_update

from ..models.encoder import BaseEncoder
from ..models.decoder import BaseDecoder
from .pipelines import StatePipeline
from .utils import compute_mutual_information


class RepresentationLoss:
    """
    Base class for representation-learning objectives.

    Owns:
        - references to RepresentationModel components
        - optimizer groups
        - gradient propagation
        - optimization steps

    Subclasses only need to implement compute_loss().
    """

    def __init__(
        self,
        encoder_lr: float = 1e-3,
        encoder_tau: float = 0.999,
        # encoder_steps: int = 9000,  # used for early stopping TODO
        decoder_lr: Optional[float] = None,
        decoder_lambda: Optional[float] = None,
        # decoder_weight_decay: Optional[float] = None, # custom adam param
        optimizer_class: th.optim.Optimizer = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.model: Optional["RepresentationModel"] = None

        self.encoder_lr = encoder_lr
        self.encoder_tau = encoder_tau
        self.decoder_lr = decoder_lr
        self.decoder_lambda = decoder_lambda
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}

        self.optimizers: dict[str, th.optim.Optimizer] = {}

        self._modules: dict[str, nn.Module] = {}
        self._target_modules: dict[str, nn.Module] = {}
        self._parameter_groups: dict[str, list[nn.Parameter]] = {}

    def log(self, tag, value):
        self.model.log(tag, value)

    # ------------------------------------------------------------------
    # Attachment
    # ------------------------------------------------------------------

    def attach(self, model: "RepresentationModel") -> None:
        """
        Attach the loss to a RepresentationModel.

        The loss obtains access to the model's FunctionModels but does
        not construct the architecture itself.
        """
        self.model = model

        self._modules = {
            "encoder": model.encoder,
            "pipeline": model.pipeline,
        }

        if model.decoder is not None:
            self._modules["decoder"] = model.decoder

        self._build_optimizers()

    # ------------------------------------------------------------------
    # Optimizer configuration
    # ------------------------------------------------------------------

    def _build_optimizers(self) -> None:
        self.optimizers.clear()
        self._parameter_groups.clear()

        encoder_params = [self._modules["encoder"]]

        if self.pipeline.n_branches == 1:
            encoder_params.append(self._modules["pipeline"].branches["representation"])

        self.add_parameter_group(
            name="encoder",
            modules=encoder_params,
            lr=self.encoder_lr,
        )

        if self.pipeline.n_branches > 1:
            self.add_parameter_group(
                name="downstream",
                modules=[self._modules["pipeline"].branches["critic"]],
                lr=self.encoder_lr,
            )

        if "decoder" in self._modules:
            decoder_params = [self._modules["decoder"]]
            if self.pipeline.n_branches > 1:
                decoder_params.append(self._modules["pipeline"].branches["representation"])
            self.add_parameter_group(
                name="decoder",
                modules=decoder_params,
                lr=self.decoder_lr,
            )

    def add_parameter_group(
        self,
        name: str,
        modules: Iterable[nn.Module],
        lr: float,
        optimizer_class=None,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Register parameters under an optimization group.

        A group may contain multiple FunctionModels, e.g.:

            encoder + attention
            decoder + fusion
            pipeline
        """
        if name in self.optimizers:
            raise ValueError(
                f"Optimizer group '{name}' already exists."
            )

        parameters = []

        for module in modules:
            parameters.extend(module.parameters())

        # Remove duplicate parameter references.
        parameters = list(dict.fromkeys(parameters))

        if not parameters:
            return

        optimizer_class = optimizer_class or self.optimizer_class
        kwargs = optimizer_kwargs or self.optimizer_kwargs

        self._parameter_groups[name] = parameters

        self.optimizers[name] = optimizer_class(
            parameters,
            lr=lr,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def zero_grad(self) -> None:
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

    def backward(self, loss: th.Tensor) -> None:
        loss.backward()

    def step(self) -> None:
        for optimizer in self.optimizers.values():
            optimizer.step()

    def optimize(
        self,
        loss: th.Tensor,
        update: bool = True,
    ) -> th.Tensor:
        """
        Execute one representation optimization step.
        """
        if update:
            self.zero_grad()

        self.backward(loss)

        if update:
            self.step()

        return loss

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        observations,
        actions,
        next_observations,
    ) -> th.Tensor:
        raise NotImplementedError

    def compute_mi(self, observation_z, q_min):
        # Mutual Information to assess latent features' impact
        if isinstance(observation_z, dict):
            observation_z = observation_z['pixel']
        return compute_mutual_information(observation_z, q_min)

    # ------------------------------------------------------------------
    # Model access
    # ------------------------------------------------------------------

    @property
    def encoder(self) -> BaseEncoder:
        return self._modules["encoder"]

    @property
    def encoder_target(self) -> BaseEncoder:
        return self.model.encoder_target

    @property
    def pipeline(self) -> StatePipeline:
        return self._modules["pipeline"]

    @property
    def decoder(self) -> Optional[BaseDecoder]:
        return self._modules.get("decoder")

    # ------------------------------------------------------------------
    # Device / mode
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> None:
        for module in self._modules.values():
            module.train(mode)

    def to(self, device) -> None:
        for module in self._modules.values():
            module.to(device)
        for module in self._target_modules.values():
            module.to(device)

    def __repr__(self) -> str:
        groups = ", ".join(self._parameter_groups.keys())
        return (
            f"{self.__class__.__name__}("
            f"optimizer_groups=[{groups}])"
        )


class RepresentationModel:
    LOG_FREQ = 1000

    def __init__(self,
                 model_type: str,
                 encoder: BaseEncoder,
                 loss: RepresentationLoss,
                 pipeline: StatePipeline,
                 decoder: Optional[BaseDecoder],
                 joint_optimization: bool = False):
        self._log_fn = None
        self.type = model_type
        self.joint_optimization = joint_optimization
        # self.decoder_lambda = None

        self.encoder = encoder
        self.decoder = decoder
        self.pipeline = pipeline
        self.encoder_only = self.decoder is None
        self.device = 'cpu'
        loss.attach(self)
        self.loss: RepresentationLoss = loss

    @property
    def latent_dim(self) -> int:
        return self.encoder.latent_dim

    @property
    def encoder_optim(self):
        return self.loss.optimizers["encoder"]

    @property
    def decoder_optim(self):
        return self.loss.optimizers["decoder"]

    @property
    def downstream_optim(self):
        return self.loss.optimizers["dowstream"]

    def set_logger(self, logger_function, tag_prefix=''):
        # Expects a SB3 logger from algorithm
        self._log_fn = logger_function
        self._tag_prefix = tag_prefix

    def log(self, tag, value):
        tag = self._tag_prefix + tag
        if self._log_fn is None:
            print(f"{tag}: {value}")
        else:
            self._log_fn.record(tag, value)

    def log_mi(self, observation_z, q_min):
        # Mutual Information to assess latent features' impact
        if isinstance(observation_z, dict):
            observation_z = observation_z['pixel']
        mi = self.loss.compute_mi(observation_z, q_min)
        self.log("mutual_information_zq", mi.mean())
        return mi

    # ------------------------------------------------------------------
    # Target model / EMA update
    # ------------------------------------------------------------------

    def create_target(self) -> None:
        self.encoder_target = copy.deepcopy(self.encoder)
        self.encoder_target.train(False)

        self.pipeline_target = copy.deepcopy(self.pipeline)
        self.pipeline_target.train(False)

    def update_target(self) -> None:
        polyak_update(self.encoder.parameters(),
                      self.encoder_target.parameters(),
                      self.loss.encoder_tau)

        polyak_update(self.pipeline.parameters(),
                      self.pipeline_target.parameters(),
                      self.loss.encoder_tau)

    def to(self, device):
        self.loss.to(device)

    def set_training_mode(self, mode: bool) -> None:
        self.loss.train(mode)

    def forward_representation(self, observation, deterministic=False, use_grad=True, use_target=False):
        if use_target:
            obs_z = self.encoder_target(observation)  # always deterministic
            transform = self.pipeline_target.forward_representation
        else:
            obs_z = self.encoder(observation)  # always deterministic
            transform = self.pipeline.forward_representation
        return transform(obs_z, deterministic, use_grad)

    def forward_z(self, observation, deterministic=False, use_grad=True):
        obs_z = self.encoder(observation)
        return self.pipeline.forward_z(obs_z, deterministic, use_grad)

    def target_forward_z(self, observation, deterministic=False, use_grad=True):
        obs_z = self.encoder_target(observation)
        return self.pipeline_target.forward_z(obs_z, deterministic, use_grad)

    def decode_latent(self, obs_z, action=None):
        if action is not None:
            return self.decoder(obs_z, action)
        else:
            return self.decoder(obs_z)

    def __repr__(self):
        out_str = f"{self.type}Model:\n"
        out_str += str(self.encoder)
        out_str += '\n'
        out_str += str(self.pipeline)
        out_str += '\n'
        out_str += str(self.decoder)
        return out_str

    def __str__(self):
        out_str = f"{self.type}Model"
        out_str += f"({self.encoder.__class__.__name__}"
        out_str += f"+{self.pipeline}"
        if not self.encoder_only:
            out_str += f"+{self.decoder.__class__.__name__})"
        else:
            out_str += ")"
        return out_str
