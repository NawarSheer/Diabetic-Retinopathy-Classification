# -*- coding: utf-8 -*-
"""
A combined method of FractionalMaxPooling and Team o_O solutions 
@author: Nawar
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
#from Resample_Iterator import resampling 
from Val_CM import Val_CM 
from Val_QWK import Val_QWK
import wandb
import os
import random
from wandb.keras import WandbCallback
import tensorflow as tf
from Fractional_MAXPOOL import FractionalPooling2D
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
        K.set_value(model.optimizer.lr, 0.0001)
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


#,padding="same"    , strides=(2, 2)

model.add(Conv2D(32, (5, 5),padding="same",strides=(2, 2), activation='linear', input_shape=(128, 128, 3), data_format="channels_last", kernel_initializer=Orthogonal(gain=1.0),
                 bias_initializer=Constant(value=0.05),kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))


model.add(Conv2D(32, (3, 3),padding="same", activation='linear', kernel_initializer=Orthogonal(gain=1.0),
                 bias_initializer=Constant(value=0.05),kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))

model.add(FractionalPooling2D(pooling_ratio=(1, 1.8, 1.8, 1),pseudo_random = True,overlap=True))


model.add(Conv2D(64, (5, 5),padding="same", strides=(2, 2), activation='linear', kernel_initializer=Orthogonal(gain=1.0),
                 bias_initializer=Constant(value=0.05),kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))

model.add(Conv2D(64, (3, 3),padding="same", activation='linear', kernel_initializer=Orthogonal(gain=1.0),
                 bias_initializer=Constant(value=0.05),kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))


model.add(Conv2D(64, (3, 3),padding="same", activation='linear', kernel_initializer=Orthogonal(gain=1.0),
                 bias_initializer=Constant(value=0.05),kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))


model.add(FractionalPooling2D(pooling_ratio=(1, 1.8, 1.8, 1),pseudo_random = True,overlap=True))
#model.add(Dropout(0.25))


model.add(Conv2D(128, (3, 3),padding="same", activation='linear', kernel_initializer=Orthogonal(gain=1.0),
                 bias_initializer=Constant(value=0.05),kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))

model.add(Conv2D(128, (3, 3),padding="same", activation='linear', kernel_initializer=Orthogonal(gain=1.0),
                 bias_initializer=Constant(value=0.05),kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))


model.add(Conv2D(128, (3, 3),padding="same", activation='linear', kernel_initializer=Orthogonal(gain=1.0),
                 bias_initializer=Constant(value=0.05),kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))




model.add(RMSPooling2D(pool_size=3,strides=(3, 3)))


model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(1024, activation='linear', kernel_initializer=Orthogonal(gain=1.0),
                 bias_initializer=Constant(value=0.05),kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))

model.add(Lambda(Maxout1)) #instead of FeaturePoolLayer



model.add(Dropout(0.5))

#model.add(Flatten())


model.add(Dense(1024, activation='linear', kernel_initializer=Orthogonal(gain=1.0),
                 bias_initializer=Constant(value=0.05),kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))

model.add(Lambda(Maxout2))#instead of FeaturePoolLayer


model.add(Dense(1, activation='relu', kernel_initializer=Orthogonal(gain=1.0),
                 bias_initializer=Constant(value=0.05),kernel_regularizer=regularizers.l2(0.0005)))

#sgd = SGD(lr=0.003, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=1e-6, amsgrad=True)
model.compile(loss='mse', optimizer=adam, metrics=['mae', 'acc'])

model.summary()


experiment_id = get_experiment_id()


train_df=pd.read_csv(("/data1/visionlab/Thesis/labels/EyePACS_2015_new_train.csv"), dtype={'image': str, 'level': float})
val_df=pd.read_csv(("/data1/visionlab/Thesis/labels/EyePACS_2015_new_val.csv"), dtype={'image': str, 'level': float})


#shuffling the data before splitting
#train_df= train_df1.sample(frac=1).reset_index(drop=True)

train_datagen = ImageDataAugmentor(augment=AUGMENTATIONS,
        preprocess_input=None, rescale=1./255.)
val_datagen = ImageDataGenerator(rescale=1./255.)

def append_ext(fn):
    return fn+".tiff"

train_df["image"]=train_df["image"].apply(append_ext)
val_df["image"]=val_df["image"].apply(append_ext)


train_generator = train_datagen.flow_from_dataframe(
dataframe=train_df,
directory='/data1/visionlab/data/EyePACS_2015/128_EyePACS_all',
x_col="image",
y_col="level",
has_ext=False,                                      
batch_size=32,
seed=42,
shuffle=True,
class_mode="other",
target_size=(128,128))
#save_to_dir='./train_aug',




valid_generator = val_datagen.flow_from_dataframe(
dataframe=val_df,
directory='/data1/visionlab/data/EyePACS_2015/128_EyePACS_all',
x_col="image",
y_col="level",
has_ext=False,                                      
batch_size=32,
seed=42,
shuffle=True,
class_mode="other",
target_size=(128,128))
#save_to_dir='./val_aug',






STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

#balance_weights = K.variable(class_weight.compute_class_weight(
#               'balanced',
#                np.unique(train_df.level), 
#                train_df.level))

#balance_weights = K.variable(1.3609453700116234,  14.378223495702006, 
#                            6.637566137566138, 40.235967926689575, 
#                            49.612994350282484)





callbacks = [
#    Resample_Iterator(balance_weights),
    Val_QWK(valid_generator, STEP_SIZE_VALID),
#    EarlyStopping(monitor='val_kappa', patience=45, min_delta=0.001, verbose=1, restore_best_weights=True, mode='max'),
    ModelCheckpoint(experiment_id + "-val_kappa_checkpoint.hdf5", monitor='val_kappa', verbose=1, save_best_only=True, mode='max'),
    Val_CM(valid_generator, STEP_SIZE_VALID),
    LearningRateScheduler(scheduler, verbose=1),
    TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True),
    WandbCallback(),        
]


history=model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
#                    class_weight=[balance_weights],
                    workers=8,
                    use_multiprocessing=False,
                    callbacks=callbacks,
                     epochs=200  
)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.show()

# summarize history for mean_absolute _error
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model mean_absolute_error')
plt.ylabel('mean_absolute_error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.show()


# summarize history for val_kappa
plt.plot(history.history['val_kappa'])
plt.title('Validation_Kappa')
plt.ylabel('val_kappa')
plt.xlabel('epoch')
plt.legend([ 'validation'], loc='best')
plt.show()