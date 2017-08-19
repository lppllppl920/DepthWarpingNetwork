# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:37:34 2017

@author: DELL1
"""

import numpy as np
import cv2

from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Flatten, concatenate, Dropout, add
from keras.layers.normalization import BatchNormalization
from keras.layers import merge, multiply
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

def visualize_depth_map(depth_map_test, title):
    
    min_value, max_value, min_loc, max_loc = cv2.minMaxLoc(depth_map_test)
    depth_map_visualize = (depth_map_test - min_value) / (max_value - min_value) * 255
    depth_map_visualize = np.asarray(depth_map_visualize, dtype = 'uint8')
    cv2.imshow(title, depth_map_visualize)
    cv2.waitKey()
    
    
prefix = 'G:/Johns Hopkins University/Projects/Depth estimation/'
model = load_model(prefix + "Net/encoder-decoder-04-0.46025.hdf5")
depth_data = np.load(prefix + "original_depth_data.npy")
synthesis_depth_data = np.load(prefix + "synthesis_depth_data.npy")
translation_data = np.load(prefix + "translation_data.npy")

depth_data = np.reshape(depth_data, (-1, 640, 480, 1))
synthesis_depth_data = np.reshape(synthesis_depth_data, (-1, 640, 480, 1))
translation_data = np.reshape(translation_data, (-1, 3))


input_depth_map = Input(shape=(640, 480, 1))
input_translation = Input(shape=(3,))

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

warped_depth_data = model.predict([depth_data[:10], translation_data[:10]], batch_size=5)



visualize_depth_map(warped_depth_data[0], "warped depth")
visualize_depth_map(depth_data[0], "original depth")
visualize_depth_map(synthesis_depth_data[0], "gt warped depth")
