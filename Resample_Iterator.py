#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://github.com/keras-team/keras/issues/2595
@author: visionlab
"""
import numpy as np
import keras.backend as K
from keras.callbacks import Callback

balance_ratio = 0.95
final_balance_weights = np.array([1, 2, 2, 2, 2], dtype=float)



class Resample_Iterator(Callback):
    def __init__(self, balance_weights):
        self.balance_weights = balance_weights
        self.count = 0
    def on_epoch_begin(self, epoch, logs=None):
        alpha= balance_ratio ** self.count

        K.set_value(self.balance_weights, K.get_value(self.balance_weights)*alpha \
        + final_balance_weights * (1 - alpha))
        self.count += 1
#        try:
        print(K.get_value(self.balance_weights))
        logs["alpha"] = alpha
        print("alpha: {} ".format(alpha))
        
#        except:
#            print("could print the weights")
#        logging.info("epoch %s, balance_weights = %s" % (epoch, K.get_value(self.balance_weights),))


