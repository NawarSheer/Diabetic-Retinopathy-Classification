# -*- coding: utf-8 -*-
"""
credits for using albumentation with keras : https://github.com/mjkvaak/ImageDataAugmentor
@author: Nawar
"""
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD, Adam
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from datetime import datetime as dt
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler, TensorBoard, ReduceLROnPlateau
import keras.backend as K
from Resample_Iterator import Resample_Iterator 
from Val_CM import Val_CM 
from Val_QWK import Val_QWK
import numpy as np 
import matplotlib.pyplot as plt
#from keras.regularizers import l2
#from custom_pooling import RMSPooling2D
import wandb
import os
from wandb.keras import WandbCallback
from ImageDataAugmentor.image_data_augmentor import *
from keras.applications.inception_resnet_v2 import preprocess_input
import tensorflow as tf
import random

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)



# for reproducibility
def seed_everything(seed=42):
    
    print('Random seeds initialized')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
#    ia.seed(seed)

seed_everything()


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




wandb.init(project="diabetic-retinopathy")


#def scheduler(epoch):
#    if epoch == 23:
#        K.set_value(model.optimizer.lr, 0.0003)
#        
#    return K.get_value(model.optimizer.lr)




def get_experiment_id():
    time_str = dt.now().strftime('%Y-%m-%d-%H-%M-%S')
    experiment_id = 'base_{}'.format(time_str)
    return experiment_id

def create_model():
    
    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=None, input_shape=(256, 256, 3), pooling='avg') 
    x = base_model.output
    x = Dense(1, activation="relu")(x)
    model = Model(inputs = base_model.input, outputs = x)
    return model


model = create_model()
#model.summary()
#
#sgd = SGD(lr=0.003, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='mse', optimizer=sgd, metrics=['mae', 'acc'])
#AUGMENTATIONS = strong_aug(p=0.9)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=1e-6, amsgrad=True)
model.compile(loss='mse', optimizer=adam, metrics=['mae', 'acc'])

experiment_id = get_experiment_id()

train_df=pd.read_csv(("/data1/visionlab/Thesis/labels/EyePACS_2015_new_train.csv"), dtype={'image': str, 'level': float})
val_df=pd.read_csv(("/data1/visionlab/Thesis/labels/EyePACS_2015_new_val.csv"), dtype={'image': str, 'level': float})



##shuffling the data before splitting
#train_df= train_df.sample(frac=1).reset_index(drop=True)
#
#val_sp = np.random.rand(len(train_df)) < 0.23
#val_df = train_df[val_sp]
#train_df = train_df[~val_sp]

#brightness_range=[1.5,1.7], horizontal_flip=True, vertical_flip=True, shear_range=15, zoom_range=[0.85,0.85],rescale=1./255
train_datagen = ImageDataAugmentor(augment=AUGMENTATIONS,
        preprocess_input=None)
val_datagen = ImageDataGenerator()


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



balance_weights = K.variable(class_weight.compute_class_weight(
               'balanced',
                np.unique(train_df.level), 
                train_df.level))



callbacks = [
    Resample_Iterator(balance_weights),
    Val_QWK(valid_generator, STEP_SIZE_VALID),
    ModelCheckpoint(experiment_id + "-val_kappa_checkpoint.hdf5", monitor='val_kappa', verbose=1, save_best_only=True, mode='max'),
#    EarlyStopping(monitor='val_kappa', patience=200, min_delta=0.001, verbose=1, restore_best_weights=True, mode='max',baseline=None),
    Val_CM(valid_generator, STEP_SIZE_VALID),
#    LearningRateScheduler(scheduler, verbose=1),
#    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='min', min_delta=0.01, cooldown=0, min_lr=0),
    TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True),        
    WandbCallback(),
]




history=model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    class_weight=[balance_weights],
                    callbacks=callbacks,
                    workers=8,
                    use_multiprocessing=False,
                    epochs=100  
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