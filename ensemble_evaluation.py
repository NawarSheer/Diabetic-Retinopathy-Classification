#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 19:50:21 2020

@author: visionlab
"""

from keras.models import load_model
from keras.models import Model
from keras import layers, Input
import efficientnet.keras as efn
from keras.layers import Dense
from tqdm import tqdm
import pandas as pd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from Kappa_Skl import kappa
import itertools
import seaborn as sns

def plot_cm(cm, y_true, y_pred, figsize=(10,10)):

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax, annot_kws={"size":15})
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    

    
def create_model(path):
    
    base_model = efn.EfficientNetB4(weights=None,
                        include_top=False,
                        input_shape=(512, 512, 3), pooling='avg')
##    base_model = InceptionV3(weights=None , include_top=False, input_tensor=None, input_shape=(299, 299, 3), pooling='avg') 
    x = base_model.output
    x = Dense(1, activation="relu")(x)
    model = Model(inputs = base_model.input, outputs = x)
    model.summary()
    model.load_weights(path)
    return model

def ensembleModels(models, model_input):
    # collect outputs of models in a list
    yModels=[model(model_input) for model in models] 
    # averaging outputs
    yAvg=layers.average(yModels) 
    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')  
       
    return modelEns




path1 = './Experiments/PH_1/EX_56/EX_56.hdf5'
path2 = './Experiments/PH_1/EX_57/EX_57.hdf5'

model1 = create_model(path1)
model2 = create_model(path2)
model2.name = "Model_2"     
models = [model1, model2]
    
model_input = Input(shape=models[0].input_shape[1:]) # c*h*w
modelEns = ensembleModels(models, model_input)
modelEns.summary()


test_df = pd.read_csv('./labels/messidor_data_dr.csv')

predicted = []

for i, name in tqdm(enumerate(test_df['image_id_dr'])):

    path = os.path.join('/data1/visionlab/data/Messidor_2_original/IMAGES_Teamo_O_512', name)
    image = cv2.imread(path)   
    image = image[...,::-1]
#    image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
    y_test = modelEns.predict((image[np.newaxis]))#/255
    coef = [0.5, 1.5, 2.5, 3.5]
    for i, pred in enumerate(y_test):
        if pred < coef[0]:
            y_test[i] = 0
        elif pred >= coef[0] and pred < coef[1]:
            y_test[i] = 1
        elif pred >= coef[1] and pred < coef[2]:
            y_test[i] = 2
        elif pred >= coef[2] and pred < coef[3]:
            y_test[i] = 3
        else:
            y_test[i] = 4
    predicted.append(int(y_test))

y_true = test_df['adjudicated_dr_grade'] 


print('Confusion Matrix')
cm=confusion_matrix(y_true, predicted)
print(cm)

plot_cm(cm, y_true, predicted)

print('Classification Report')
print(classification_report(y_true, predicted, target_names=['0', '1', '2', '3', '4']))



print(' Test Quadratic_wieghted_kappa')
print(kappa(y_true, predicted, weights='quadratic'))