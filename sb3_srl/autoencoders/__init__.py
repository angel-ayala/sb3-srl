#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:18:39 2025

@author: angel
"""
from .models import VectorModel
from .models import VectorSPRModel
from .models import VectorTargetDistModel
from .models import VectorSPRIModel


def instance_autoencoder(ae_type: str, ae_params: dict):
    ae_params_ = ae_params.copy()
    del ae_params_['encoder_steps']
    del ae_params_['encoder_lr']
    del ae_params_['decoder_lr']
    del ae_params_['decoder_weight_decay']
    ae_class = globals()[ae_type + 'Model']
    ae_model = ae_class(**ae_params_)
    return ae_model
