#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:21:28 2026

@author: angel
"""

from .base import BaseFunction

from .encoder import (
    BaseEncoder,
    VectorEncoder,
    NatureCNNEncoder,
    AdPuEncoder,
    SimpleSPREncoder
)
from .decoder import (
    BaseDecoder,
    VectorDecoder,
    PixelDecoder,
    ProprioceptiveSPRDecoder,
    SPRDecoder,
    SimpleSPRDecoder
)
from .fusion import (
    FusionMLP,
    FusionConv1d,
    FusionGated,
    FusionFiLM,
    CrossAttention
)


ENCODERS = {
    "Vector": VectorEncoder,
    "NatureCNN": NatureCNNEncoder,
    "AdPu": AdPuEncoder,
    "SimpleSPR": SimpleSPREncoder,
}


DECODERS = {
    "Vector": VectorDecoder,
    "Pixel": PixelDecoder,
    "ProprioceptiveSPR": ProprioceptiveSPRDecoder,
    "SPR": SPRDecoder,
    "SimpleSPR": SimpleSPRDecoder,
}


FUSION = {
    "mlp": FusionMLP,
    "conv1d": FusionConv1d,
    "gated": FusionGated,
    "film": FusionFiLM,
    "att": CrossAttention,
}


def create_encoder(name: str, params: dict) -> BaseEncoder:
    try:
        encoder_class = ENCODERS[name]
    except KeyError:
        raise ValueError(
            f"Encoder '{name}' not registered. "
            f"Available: {list(ENCODERS)}"
        )

    return encoder_class(**params)


def create_decoder(name: str, params: dict) -> BaseDecoder:
    try:
        decoder_class = DECODERS[name]
    except KeyError:
        raise ValueError(
            f"Decoder '{name}' not registered. "
            f"Available: {list(DECODERS)}"
        )

    return decoder_class(**params)


def create_function_model(name: str, params: dict) -> BaseFunction:
    assert ":" in name, f"Bad function model format: {name}"
    model_name = name.lower().split(":")
    model_type = model_name[0]
    model_name = model_name[1]
    # if model_type == "a":
    #     return ATTENTION[model_name]
    if model_type == "f":
        try:
            fusion_class = FUSION[model_name]
        except KeyError:
            raise ValueError(
                f"Fusion '{model_name}' not registered. "
                f"Available: {list(FUSION)}"
            )
        return fusion_class(**params)
    raise NotImplementedError(f"Model type {name} not found!")
