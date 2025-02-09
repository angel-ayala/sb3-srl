#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:18:39 2025

@author: angel
"""
from .models import VectorModel
from .models import VectorSPRModel
from .models import VectorTargetDistModel


def instance_autoencoder(ae_type: str, ae_params: dict):
    if ae_type == 'Vector':
        ae_model = VectorModel(ae_params['vector_shape'],
                               ae_params['latent_dim'],
                               ae_params['hidden_dim'],
                               ae_params['num_layers'],
                               ae_params['encoder_only'],
                               ae_params['decoder_latent_lambda'])
    if ae_type == 'VectorSPR':
        ae_model = VectorSPRModel(ae_params['vector_shape'],
                                  ae_params['action_shape'],
                                  ae_params['latent_dim'],
                                  ae_params['hidden_dim'],
                                  ae_params['num_layers'],
                                  ae_params['encoder_only'],
                                  ae_params['decoder_latent_lambda'])
    if ae_type == 'VectorTargetDist':
        ae_model = VectorTargetDistModel(ae_params['vector_shape'],
                                         ae_params['latent_dim'],
                                         ae_params['hidden_dim'],
                                         ae_params['num_layers'],
                                         ae_params['encoder_only'],
                                         ae_params['decoder_latent_lambda'])
    # if ae_type == 'VectorDifference':
    #     ae_model = VectorDifferenceModel(ae_params)
    return ae_model
