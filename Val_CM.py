# -*- coding: utf-8 -*-
"""
Validation Confusion matrix for each epoch 
@author: Nawar
"""
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.callbacks import Callback

class Val_CM(Callback):
    
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
            
        c = confusion_matrix(y_test_all, y_pred_all)
        print('Confusion matrix:\n', c)
            
             