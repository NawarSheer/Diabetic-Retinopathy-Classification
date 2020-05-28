#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 01:59:12 2020

@author: visionlab
"""
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization,Dense, Dropout, Flatten, LeakyReLU, Lambda
from custom_pooling import RMSPooling2D
from keras.optimizers import SGD, Adam
from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler, TensorBoard
from datetime import datetime as dt
from keras.initializers import Constant,Orthogonal
from keras import regularizers
from Kappa_Skl import kappa
import keras.backend as K
import matplotlib.pyplot as plt
import itertools
from Resample_Iterator import Resample_Iterator 
from Val_QWK import Val_QWK
from Val_CM import Val_CM
import tensorflow as tf
from keras.models import load_model
import random
from keras.applications.inception_resnet_v2 import preprocess_input
import wandb
import os
from wandb.keras import WandbCallback
from ImageDataAugmentor.image_data_augmentor import *
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)


wandb.init(project="diabetic-retinopathy")

def seed_everything(seed=42):
    
    print('Random seeds initialized')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)



def Maxout1(x):
    return tf.contrib.layers.maxout(x, 512)

def Maxout2(x):
    return tf.contrib.layers.maxout(x, 512)


def get_experiment_id():
    time_str = dt.now().strftime('%Y-%m-%d-%H-%M-%S')
    experiment_id = 'base_{}'.format(time_str)
    return experiment_id

def scheduler(epoch):
    if epoch == 150:
        K.set_value(model.optimizer.lr, 0.00014339)
    return K.get_value(model.optimizer.lr)


def strong_aug(p=1):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)


AUGMENTATIONS = strong_aug(p=0.9)


seed_everything()

model = Sequential()

"""128 Layers"""
model.add(Conv2D(32, (5, 5), padding="same", strides=(2, 2),  activation='linear', input_shape=(256, 256, 3), data_format="channels_last", 
                 name='conv2d_1',kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization(name='batch_normalization_1'))
model.add(LeakyReLU(alpha=0.01,name='leaky_re_lu_1'))


model.add(Conv2D(32, (3, 3),padding="same", activation='linear',
                 name='conv2d_2',kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization(name='batch_normalization_2'))
model.add(LeakyReLU(alpha=0.01,name='leaky_re_lu_2'))

model.add(MaxPooling2D(pool_size=3, strides=(2, 2),name='max_pooling2d_1'))


model.add(Conv2D(64, (5, 5),padding="same", strides=(2, 2), activation='linear',
                 name='conv2d_3',kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization(name='batch_normalization_3'))
model.add(LeakyReLU(alpha=0.01,name='leaky_re_lu_3'))

model.add(Conv2D(64, (3, 3),padding="same", activation='linear',
                 name='conv2d_4',kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization(name='batch_normalization_4'))
model.add(LeakyReLU(alpha=0.01,name='leaky_re_lu_4'))


model.add(Conv2D(64, (3, 3),padding="same", activation='linear',
                 name='conv2d_5',kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization(name='batch_normalization_5'))
model.add(LeakyReLU(alpha=0.01,name='leaky_re_lu_5'))


model.add(MaxPooling2D(pool_size=3, strides=(2, 2),name='max_pooling2d_2'))
#model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3),padding="same", activation='linear', 
                 name='conv2d_6',kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization(name='batch_normalization_6'))
model.add(LeakyReLU(alpha=0.01,name='leaky_re_lu_6'))


model.add(Conv2D(128, (3, 3),padding="same", activation='linear', 
                 name='conv2d_7',kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization(name='batch_normalization_7'))
model.add(LeakyReLU(alpha=0.01,name='leaky_re_lu_7'))


model.add(Conv2D(128, (3, 3),padding="same", activation='linear',
                 name='conv2d_8',kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization(name='batch_normalization_8'))
model.add(LeakyReLU(alpha=0.01,name='leaky_re_lu_8'))


"""256 Layers"""

model.add(MaxPooling2D(pool_size=3, strides=(2, 2),name='max_pooling2d_3'))
#model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3),padding="same", activation='linear', kernel_initializer=Orthogonal(gain=1.0),
                 name='conv2d_9',bias_initializer=Constant(value=0.05),kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization(name='batch_normalization_11'))
model.add(LeakyReLU(alpha=0.01,name='leaky_re_lu_12'))


model.add(Conv2D(256, (3, 3),padding="same", activation='linear', kernel_initializer=Orthogonal(gain=1.0),
                 name='conv2d_10',bias_initializer=Constant(value=0.05),kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization(name='batch_normalization_12'))
model.add(LeakyReLU(alpha=0.01,name='leaky_re_lu_13'))


model.add(Conv2D(256, (3, 3),padding="same", activation='linear', kernel_initializer=Orthogonal(gain=1.0),
                 name='conv2d_11',bias_initializer=Constant(value=0.05),kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization(name='batch_normalization_13'))
model.add(LeakyReLU(alpha=0.01,name='leaky_re_lu_14'))

"""512 Layers"""
#model.add(MaxPooling2D(pool_size=3, strides=(2, 2),name='max_pooling2d_4'))
#
#model.add(Conv2D(512, (3, 3),padding="same",  activation='linear', kernel_initializer=Orthogonal(gain=1.0),
#                 name='conv2d_12',bias_initializer=Constant(value=0.05),kernel_regularizer=regularizers.l2(0.0005)))
#model.add(BatchNormalization(name='batch_normalization_14'))
#model.add(LeakyReLU(alpha=0.01,name='leaky_re_lu_15'))
#
#model.add(Conv2D(512, (3, 3),padding="same", activation='linear', kernel_initializer=Orthogonal(gain=1.0),
#                 name='conv2d_13',bias_initializer=Constant(value=0.05),kernel_regularizer=regularizers.l2(0.0005)))
#model.add(BatchNormalization(name='batch_normalization_15'))
#model.add(LeakyReLU(alpha=0.01,name='leaky_re_lu_16'))




model.add(RMSPooling2D(pool_size=3,strides=(3, 3),name='rms_pooling2d_1'))


model.add(Dropout(0.5))

model.add(Flatten(name='flatten_1'))

model.add(Dense(1024, activation='linear', kernel_initializer=Orthogonal(gain=1.0),
                 name='dense_1',kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization(name='batch_normalization_9'))
model.add(LeakyReLU(alpha=0.01,name='leaky_re_lu_9'))




model.add(Lambda(Maxout1, name='lambda_1')) #instead of FeaturePoolLayer

model.add(Dropout(0.5))




model.add(Dense(1024, activation='linear', kernel_initializer=Orthogonal(gain=1.0),
                name='dense_2',kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization(name='batch_normalization_10'))
model.add(LeakyReLU(alpha=0.01,name='leaky_re_lu_10'))

model.add(Lambda(Maxout2, name='lambda_2'))#instead of FeaturePoolLayer

model.add(Dense(1, activation='relu', kernel_initializer=Orthogonal(gain=1.0),
                name='dense_3',kernel_regularizer=regularizers.l2(0.0005)))



pre_256_model = load_model('./Experiments/PH_1/EX_40/EX_40.hdf5', custom_objects={'RMSPooling2D': RMSPooling2D,'Lambda':Lambda,'tf':tf})
# copy weights from old model to new one

for layer in model.layers:
    try:
        layer.set_weights( pre_256_model.get_layer(name=layer.name).get_weights())
        print("Succesfully transfered weights for layer {}".format(layer.name))
    except:
        print("Could not transfer weights for layer {}".format(layer.name))
        


#model.add_loss( sample_loss(y_true, y_pred, Homotopy ) )
adam = Adam(lr=0.0014339, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=1e-6, amsgrad=True)
model.compile(loss='mse', optimizer=adam, metrics=['mae', 'acc'])

model.summary()


#weight checking
#a = pre_128_model.layers[0].get_weights()[0]
#b = model.layers[0].get_weights()[0]
#if np.array_equal(a, b):
#            print('equal')
#else:
#    print('not equal')
experiment_id = get_experiment_id()


train_df=pd.read_csv(("/data1/visionlab/Thesis/labels/EyePACS_2015_new_train.csv"), dtype={'image': str, 'level': float})
val_df=pd.read_csv(("/data1/visionlab/Thesis/labels/EyePACS_2015_new_val.csv"), dtype={'image': str, 'level': float})




train_datagen = ImageDataAugmentor(augment=AUGMENTATIONS,
        preprocess_input=None, rescale=1./255.)
val_datagen = ImageDataGenerator(rescale=1./255.)


def append_ext(fn):
    return fn+".tiff"

#def append_ext1(fn):
#    return fn+".png"

train_df["image"]=train_df["image"].apply(append_ext)
val_df["image"]=val_df["image"].apply(append_ext)


train_generator = train_datagen.flow_from_dataframe(
dataframe=train_df,
directory='/data1/visionlab/data/EyePACS_2015/256-EyePACS-all',
x_col="image",
y_col="level",
has_ext=False,                                      
batch_size=32,
seed=42,
shuffle=True,
class_mode="other",
target_size=(256,256))



valid_generator = val_datagen.flow_from_dataframe(
dataframe=val_df,
directory='/data1/visionlab/data/EyePACS_2015/256-EyePACS-all',
x_col="image",
y_col="level",
has_ext=False,                                      
batch_size=32,
seed=42,
shuffle=True,
class_mode="other",
target_size=(256,256))





STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size


#
#balance_weights = K.variable(class_weight.compute_class_weight(
#               'balanced',
#                np.unique(train_df.level), 
#                train_df.level))



callbacks = [
#    Resample_Iterator(balance_weights),
    Val_QWK(valid_generator, STEP_SIZE_VALID),
    ModelCheckpoint(experiment_id + "-val_kappa_checkpoint.hdf5", monitor='val_kappa', verbose=1, save_best_only=True, mode='max'),
#    EarlyStopping(monitor='val_kappa', patience=200, min_delta=0.001, verbose=1, restore_best_weights=True, mode='max',baseline=None),
    Val_CM(valid_generator, STEP_SIZE_VALID),
    LearningRateScheduler(scheduler, verbose=1),
#    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='min', min_delta=0.01, cooldown=0, min_lr=0),
    TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True),        
    WandbCallback(),
]




history=model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
#                    class_weight=[balance_weights],
                    callbacks=callbacks,
                    workers=8,
                    use_multiprocessing=False,
                    epochs=200  
)


#model.save(os.path.join(wandb.run.dir, "model.h5"))


# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#plt.savefig('model accuracy.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#plt.savefig('model loss.png')

# summarize history for val_kappa
plt.plot(history.history['val_kappa'])
plt.title('Validation_Kappa')
plt.ylabel('val_kappa')
plt.xlabel('epoch')
plt.legend([ 'validation'], loc='upper left')
plt.show()
#plt.savefig('model validation_kappa.png')