#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:11:27 2026

@author: angel
"""

from __future__ import annotations

from typing import Any

import torch as th
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.utils import (
    get_parameters_by_name,
    polyak_update,
)

from ..models import BaseFunction
from ..models import create_encoder
from ..models import create_decoder
from ..models import create_function_model
from .representation import RepresentationModel
from .losses import create_loss
from .pipelines import StatePipeline, TransformationBranch, DeterministicRepresentation
from .stochastic import StochasticRepresentation, StochasticWrapper


class StatePipelineFactory:

    @staticmethod
    def create_branch(models: list[tuple[str, BaseFunction]], input_dim: int,
                      is_stochastic: bool = False
                      ) -> TransformationBranch:
        """
        A:XXXX -> Attention model
        F:XXXX -> Fusion model
        models:[
            ("A:CrossAtention", {params}),
            ("F:MLP", {params})
        ]
        """
        stages = []
        representation_dim = input_dim

        for model_name, model_params in models:
            stage = create_function_model(model_name, model_params)
            stages.append(stage)
            representation_dim = stage.output_dim
        
        if is_stochastic:
            out_fn = StochasticRepresentation(representation_dim)
        else:
            out_fn = DeterministicRepresentation(representation_dim)

        stages.append(out_fn)

        return TransformationBranch(stages)

    @classmethod
    def create(cls, configuration: dict[str, list[tuple]],
               input_dim: int,
               is_stochastic: bool = False) -> StatePipeline:
        """
        "representation" -> Representation model related
        "critic" -> Downstream task
        configuration:
            {
                "representation": ["A:CrossAtention", "F:MLP"],
            }

            {
                "representation": ["A:CrossAtention", "F:MLP"],
                "critic": ["A:CrossAtention", "F:MLP"],
            }
        """

        branches = {}

        for branch_name, models in configuration.items():
            branches[branch_name] = cls.create_branch(
                models, input_dim, is_stochastic)

        return StatePipeline(branches, configuration, is_stochastic)


class RepresentationFactory:

    @staticmethod
    def create_encoder(config):
        name, params = config
        return create_encoder(name, params)

    @staticmethod
    def create_decoder(config, is_stochastic=False):
        name, params = config
        decoder = create_decoder(name, params)
        print('decoder', decoder)

        if is_stochastic:
            decoder = StochasticWrapper(decoder)
            print('decoderStochastic', decoder)

        return decoder

    @staticmethod
    def create_loss(config):
        name, params = config
        return create_loss(name, params)

    @classmethod
    def create(cls, srl_config: dict):
        model_config = srl_config["config"]

        encoder = cls.create_encoder(model_config["encoder"])
        print('encoder', encoder)

        loss = cls.create_loss(model_config["loss"])
        print('loss', loss)
        pipeline = StatePipelineFactory.create(
            model_config["pipeline"],
            encoder.latent_dim,
            model_config["is_stochastic"]
        )
        print('pipeline', pipeline)

        decoder_config = model_config.get("decoder")
        if decoder_config is not None:
            # fusion layer present in pipeline
            if pipeline.latent_dim != encoder.latent_dim:
                decoder_config[1]["with_fusion"] = True
                decoder_config[1]["latent_dim"] = pipeline.latent_dim
            decoder = cls.create_decoder(
                decoder_config, model_config["is_stochastic"])

        model = RepresentationModel(
            model_type=srl_config["model"],
            encoder=encoder,
            loss=loss,
            pipeline=pipeline,
            decoder=decoder,
            joint_optimization=model_config["joint_optimization"],
        )

        model.create_target()

        # objective/pipeline selection will be added here
        return model


# srl/policy.py
class SRLPolicy:
    """
    SRL middleware between an SB3 Policy and RepresentationModel.

    Responsibilities:
        - construct RepresentationModel
        - expose its representation for policy inference
        - manage representation training/evaluation mode
        - expose SRL-related policy configuration
    """

    def __init__(
        self,
        srl_config: dict,
    ):
        self.srl_config = srl_config
        self.rep_model = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_srl(self) -> None:
        """
        Construct the RepresentationModel.

        Called from the concrete SB3 policy _build().
        """
        self.rep_model = RepresentationFactory.create(self.srl_config)

        self.rep_model.to(self.device)

        # Environment-dependent initialization.
        # self.rep_model.fit_observation(
        #     self.observation_space
        # )

        self.rep_model.set_training_mode(self.training)

    # ------------------------------------------------------------------
    # Representation properties
    # ------------------------------------------------------------------

    @property
    def is_multimodal(self) -> bool:
        return self.rep_model.is_multimodal

    @property
    def srl_joint_optimization(self) -> bool:
        return self.rep_model.joint_optimization

    # ------------------------------------------------------------------
    # Inference / Loss
    # ------------------------------------------------------------------

    def srl_forward(
        self,
        observation: PyTorchObs,
        deterministic: bool = False,
    ) -> th.Tensor:
        """
        Representation inference used by the SB3 policy.
        """
        return self.rep_model.forward_z(
            observation,
            deterministic=deterministic,
            use_grad=False,
        )

    def _predict_srl(
        self,
        observation: PyTorchObs,
        deterministic: bool = False,
    ) -> th.Tensor:
        with th.no_grad():
            return self.srl_forward(
                observation,
                deterministic,
            )

    def compute_srl_loss(self, observations, actions, next_observations):
        return self.rep_model.loss.compute_loss(
            observations, actions, next_observations
        )

    def update_srl(self, loss):
        return self.rep_model.loss.optimize(loss)

    # ------------------------------------------------------------------
    # Training / device
    # ------------------------------------------------------------------

    def set_srl_training_mode(self, mode: bool) -> None:
        self.rep_model.set_training_mode(mode)

    def srl_to(self, device) -> None:
        self.rep_model.to(device)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def logger_append(
        self,
        logger,
        tag_prefix: str = "",
    ) -> None:
        self.rep_model.set_logger(
            logger,
            tag_prefix,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _get_srl_constructor_parameters(self) -> dict[str, Any]:
        return {
            "srl_config": self.srl_config,
        }


# srl/srl_algorithm.py
class SRLAlgorithm:
    """
    SRL middleware between an SB3 Algorithm and RepresentationModel.

    Responsibilities:
        - connect the algorithm to RepresentationModel
        - expose representation/target inference
        - maintain target statistics
        - participate in serialization
        - provide the SRL part of the training loop

    Concrete algorithms implement the actual train() method.
    """

    # ------------------------------------------------------------------
    # Construction / bridge
    # ------------------------------------------------------------------

    def _create_srl_aliases(self) -> None:
        """
        Create aliases used by the concrete SB3 algorithm.
        """
        self.forward_z = self.policy.rep_model.forward_z
        self.target_forward_z = self.policy.rep_model.target_forward_z

    def _setup_srl(self) -> None:
        """
        Finalize SRL initialization after SB3 has created the policy.
        """
        self.policy.rep_model.to(self.device)

        encoder = self.policy.rep_model.encoder
        encoder_target = self.policy.rep_model.encoder_target

        if encoder_target is None:
            print("No target encoder")
            return

        self.encoder_batch_norm_stats = get_parameters_by_name(
            encoder,
            ["running_"],
        )

        self.encoder_batch_norm_stats_target = get_parameters_by_name(
            encoder_target,
            ["running_"],
        )

    # ------------------------------------------------------------------
    # Target update
    # ------------------------------------------------------------------

    def update_srl_target(self) -> None:
        self.policy.rep_model.update_target()

        # BatchNorm statistics should not be Polyak averaged.
        polyak_update(
            self.encoder_batch_norm_stats,
            self.encoder_batch_norm_stats_target,
            1.0,
        )

    # ------------------------------------------------------------------
    # Representation training
    # ------------------------------------------------------------------

    def train_representation(self, observations, actions, next_observations
                             ) -> th.Tensor:
        """
        Execute one representation-learning step.

        The RepresentationModel/RepresentationLoss owns the
        representation optimization.
        """
        return self.policy.rep_model.train_step(
            observations,
            actions,
            next_observations,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _excluded_srl_save_params(self) -> list[str]:
        return [
            "forward_z",
            "target_forward_z",
        ]

    def _get_srl_torch_save_params(self) -> tuple[list[str], list[str]]:
        """
        Return torch objects managed by the RepresentationModel.

        Prefer delegating the actual list to RepresentationModel so
        this bridge does not depend on particular encoder/decoder names.
        """
        state_dicts = ["policy.rep_model.encoder"]
        state_dicts += ["policy.rep_model.encoder_optim"]

        state_dicts += ["policy.rep_model.pipeline"]
        if self.policy.rep_model.pipeline.n_branches > 1:
            state_dicts += ["policy.rep_model.downstream_optim"]

        if not self.policy.rep_model.encoder_only:
            state_dicts += ["policy.rep_model.decoder"]
            state_dicts += ["policy.rep_model.decoder_optim"]

        return state_dicts, []
