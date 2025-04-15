#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:18:39 2025

@author: angel
"""
from .models import RepresentationModel
from .models import ReconstructionModel
from .models import ReconstructionDistModel
from .models import SelfPredictiveModel
from .models import InfoSPRModel
from .models import IntrospectiveInfoSPR
from .models import MuMoAESPRModel
from .models import ProprioceptiveModel


def instance_autoencoder(ae_type: str, ae_params: dict) -> RepresentationModel:
    ae_params_ = ae_params.copy()
    del ae_params_['encoder_steps']
    del ae_params_['encoder_lr']
    del ae_params_['decoder_lr']
    del ae_params_['decoder_weight_decay']
    ae_class = globals()[ae_type + 'Model']
    ae_model = ae_class(**ae_params_)
    return ae_model
