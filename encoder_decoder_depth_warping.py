# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 13:15:50 2017

@author: DELL1
"""

import numpy as np

from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Flatten, concatenate, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import add, merge, multiply
from keras.layers import RepeatVector, Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model, load_model
from keras import optimizers
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

import sys
sys.path.insert(0,"G:/Johns Hopkins University/Projects/Depth estimation/Scripts/Spatial Transformer Layer")
#from SpatialTransformer import *
from BilinearSampler import *
    
prefix = 'G:/Johns Hopkins University/Projects/Depth estimation/'
depth_data = np.load(prefix + "original_depth_data.npy")
synthesis_depth_data = np.load(prefix + "synthesis_depth_data.npy")
translation_data = np.load(prefix + "translation_data.npy")

depth_data = np.reshape(depth_data, (-1, 640, 480, 1))
synthesis_depth_data = np.reshape(synthesis_depth_data, (-1, 640, 480, 1))
translation_data = np.reshape(translation_data, (-1, 3))

input_depth_map = Input(shape=(640, 480, 1))
input_translation = Input(shape=(3,))

##TODO: We need to add a smoothness loss to ensure the flow generated is smooth
## Also, we might want to add a mask also to rule out these black invalid regions in transformed depth map

y = RepeatVector(640 * 480)(input_translation)
y = Reshape((640, 480, 3))(y)
x = concatenate([input_depth_map, y], axis = -1)

x = Conv2D(16, (3, 3), padding='same', activation='relu') (input_depth_map)
x = concatenate([x, y], axis=-1)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D((4, 4)) (x)
x = Conv2D(64, (3, 3), padding='same', activation='relu') (x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D((4, 4)) (x)
x = Conv2D(256, (3, 3), padding='same', activation='relu') (x)
x = BatchNormalization(axis=-1)(x)



x_appearance_flow = UpSampling2D((4, 4)) (x)
x_appearance_flow = Conv2D(64, (3, 3), padding='same', activation='relu') (x_appearance_flow)
x_appearance_flow = BatchNormalization(axis=-1)(x_appearance_flow)
x_appearance_flow = UpSampling2D((4, 4)) (x_appearance_flow)
x_appearance_flow = Conv2D(16, (3, 3), padding='same', activation='relu') (x_appearance_flow)
x_appearance_flow = BatchNormalization(axis=-1)(x_appearance_flow)
appearance_flow = Conv2D(2, (3, 3), padding='same', activation='tanh', name='appearance_flow') (x_appearance_flow)


x_depth_change = UpSampling2D((4, 4)) (x)
x_depth_change = Conv2D(64, (3, 3), padding='same', activation='relu') (x_depth_change)
x_depth_change = BatchNormalization(axis=-1)(x_depth_change)
x_depth_change = UpSampling2D((4, 4)) (x_depth_change)
x_depth_change = Conv2D(16, (3, 3), padding='same', activation='relu') (x_depth_change)
x_depth_change = BatchNormalization(axis=-1)(x_depth_change)
depth_change = Conv2D(1, (3, 3), padding='same', activation='linear', name='depth_change') (x_depth_change)

compensated_depth_map = add([input_depth_map, depth_change])
warped_depth_map = BilinearSamplerLayer()([compensated_depth_map, appearance_flow])

model = Model([input_depth_map, input_translation], warped_depth_map)
model.summary()

sgd = optimizers.SGD(lr=1.0e-3, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.load_weights(prefix + "Net/encoder-decoder-10-0.70840.hdf5")

filepath = prefix + "Net/encoder-decoder-{epoch:02d}-{val_loss:.5f}.hdf5"
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=False)
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, min_delta=0.0001, mode='auto')


history = model.fit([depth_data, translation_data], synthesis_depth_data, batch_size=3, 
                          epochs=300, verbose=1, callbacks=[earlyStopping, checkpointer, reducelr], validation_split=0.05, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)

#model.save_weights(prefix + 'Net/model.h5')