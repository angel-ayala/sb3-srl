#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 11:22:57 2025

@author: angel
"""


class EarlyStopper:
    def __init__(self, patience=4500, min_delta=0., models=[]):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.stop = False
        self.models = models

    def __call__(self, validation_loss):
        if self.stop:
            for m in self.models:
                m.requires_grad_(False)
            return True
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                print('EarlyStopper')
                self.stop = True
        return self.stop
