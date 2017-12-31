# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 09:50:29 2017

@author: DELL1
"""


import cv2 
import numpy as np

from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Flatten, concatenate, Dropout, Lambda
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

def mask_invalid_element(x):
    x1, x2 = x
    return K.switch(x1 > 0, x2, 0)
def mask_invalid_element_output_shape(input_shape):
    shape1, shape2 = input_shape
    return shape1

def mask_invalid_element_argument(x, mask):
    return K.switch(mask > 0, x, 0)
def mask_invalid_element_argument_output_shape(input_shape):
    return input_shape   
    
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

input_translation_vector_inverse = Input(shape=(3, 1))
input_rotation_matrix_inverse = Input(shape=(3, 3))


rotation_data_I = []
translation_data_I = []
for i in range(rotation_data.shape[0]):
    rotation_data_I.append(np.transpose(rotation_data[i]))
    translation_data_I.append(np.matmul(-np.transpose(rotation_data[i]), translation_data[i]))

rotation_data_I = np.array(rotation_data_I)
translation_data_I = np.array(translation_data_I)
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

img_mask = np.ones((640, 480, 1), dtype = 'float32')
warped_depth_map_1 = DepthWarpingLayer(P)([input_depth_map_1, input_depth_map_2, input_translation_vector, input_rotation_matrix])
warped_depth_map_2 = DepthWarpingLayer(P)([input_depth_map_2, input_depth_map_1,
    input_translation_vector_inverse, input_rotation_matrix_inverse])
#model = Model([input_depth_map_1, input_depth_map_2, input_translation_vector, input_rotation_matrix], warped_depth_map)
#model.summary()
#
#sgd = optimizers.SGD(lr=1.0e-3, momentum=0.9, nesterov=True)
#model.compile(loss='mean_squared_error', optimizer=sgd)
#
#warped_depth_data = model.predict([synthesis_depth_data, depth_data, translation_data, rotation_data], batch_size=5)
#visualize_depth_map(warped_depth_data[0], 'Depth warping layer result')

#masked_input_depth_map_2 = Lambda(mask_invalid_element, output_shape=mask_invalid_element_output_shape)([warped_depth_map_1, input_depth_map_2])
model = Model([input_depth_map_1, input_depth_map_2, input_translation_vector, input_rotation_matrix,
               input_translation_vector_inverse, input_rotation_matrix_inverse], [warped_depth_map_1, warped_depth_map_2])
model.summary()

sgd = optimizers.SGD(lr=1.0e-3, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

masked_depth_data = model.predict([synthesis_depth_data, depth_data, translation_data, rotation_data, translation_data_I, rotation_data_I], batch_size=5)

i = 0
visualize_depth_map(masked_depth_data[0][i], 'Depth warping layer result 1')
visualize_depth_map(masked_depth_data[1][i], 'Depth warping layer result 2')
visualize_depth_map(synthesis_depth_data[i], 'synthesis depth data')
visualize_depth_map(depth_data[i], 'original depth data')
