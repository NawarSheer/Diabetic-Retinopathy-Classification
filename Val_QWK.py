#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 22:48:05 2019

@author: visionlab
"""
from keras.callbacks import Callback
from Kappa_Skl import kappa
import numpy as np

class Val_QWK(Callback):
    
    def __init__(self, validation_generator, validation_steps):
        self.validation_generator = validation_generator
        self.validation_steps = validation_steps
    

    def on_epoch_end(self, epoch, logs=None):
        y_pred_all=[]
        y_test_all=[]
        for batch_index in range(self.validation_steps):
                features, y_true = next(self.validation_generator)            
                y_pred = np.asarray(self.model.predict(features))
                y_pred = np.clip(np.round(y_pred),
                             np.min(y_true), np.max(y_true)).astype(int)
        #            y_test_all=np.append(y_test_all,np.argmax(y_true, axis=-1))
        #            y_pred_all=np.append(y_pred_all,np.argmax(y_pred, axis=-1))
                y_test_all=np.append(y_test_all,np.round(y_true).astype(int))
                y_pred_all=np.append(y_pred_all,np.round(y_pred).astype(int))
                
        val_kappa = kappa(y_test_all,y_pred_all,weights='quadratic')
        
        # Add custom metrics to the logs, so that we can use them with
        # EarlyStop and csvLogger callbacks
        logs["val_kappa"] = val_kappa
        print("val_kappa: {} ".format(val_kappa))
  

        

    