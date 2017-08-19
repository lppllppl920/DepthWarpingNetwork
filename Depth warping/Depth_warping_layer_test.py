# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 09:50:29 2017

@author: DELL1
"""


import cv2 
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
sys.path.insert(0,"G:/Johns Hopkins University/Projects/Depth estimation/Scripts/Depth Warping Layer")
from DepthWarpingLayer import *

def visualize_depth_map(depth_map_test, title):
    min_value, max_value, min_loc, max_loc = cv2.minMaxLoc(depth_map_test)
    depth_map_visualize = (depth_map_test - min_value) / (max_value - min_value) * 255
    depth_map_visualize = np.asarray(depth_map_visualize, dtype = 'uint8')
    cv2.imshow(title, depth_map_visualize)
    cv2.waitKey(100)


cv2.destroyAllWindows()    
## Test data reading
prefix = 'G:/Johns Hopkins University/Projects/Depth estimation/'
depth_data = np.load(prefix + "original_depth_data_affine.npy")
depth_data = np.reshape(depth_data, (-1, 640, 480, 1))
depth_data = depth_data[:50]

synthesis_depth_data = np.load(prefix + "synthesis_depth_data_affine.npy")
synthesis_depth_data = np.reshape(synthesis_depth_data, (-1, 640, 480, 1))
synthesis_depth_data = synthesis_depth_data[:50]

rotation_data = np.load(prefix + "affine_data_r.npy")
rotation_data = np.array(rotation_data, dtype='float32')

translation_data = np.load(prefix + "affine_data_t.npy")
translation_data = np.array(translation_data, dtype='float32')

input_depth_map_1 = Input(shape=(640, 480, 1))
input_depth_map_2 = Input(shape=(640, 480, 1))
input_translation_vector = Input(shape=(3, 1))
input_rotation_matrix = Input(shape=(3, 3))

## intrinsic matrix
P = np.zeros((3, 3), dtype = 'float32')
fx_rgb = 5.1885790117450188e+02
fy_rgb = 5.1946961112127485e+02
cx_rgb = 3.2558244941119034e+02
cy_rgb = 2.5373616633400465e+02
P[0,0] = fx_rgb
P[0,2] = cx_rgb
P[1,1] = fy_rgb
P[1,2] = cy_rgb
P[2,2] = 1.0

warped_depth_map = DepthWarpingLayer(P)([input_depth_map_1, input_depth_map_2, input_translation_vector, input_rotation_matrix])

model = Model([input_depth_map_1, input_depth_map_2, input_translation_vector, input_rotation_matrix], warped_depth_map)
model.summary()

sgd = optimizers.SGD(lr=1.0e-3, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

warped_depth_data = model.predict([synthesis_depth_data, depth_data, translation_data, rotation_data], batch_size=5)