# -*- coding: utf-8 -*-
"""
Created on Wed Oct 04 13:38:51 2017

@author: DELL1
"""

import cv2 
import numpy as np
import yaml
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Flatten, concatenate, Dropout, Lambda, add
from keras.layers.normalization import BatchNormalization
from keras.layers import add, merge, multiply
from keras.layers import RepeatVector, Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model, load_model
from keras import optimizers
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.models import model_from_json

prefix = "G:/Johns Hopkins University/Projects/Depth estimation/Scripts/"
import sys
sys.path.insert(0, prefix + "Depth Warping Layer")
from DepthWarpingLayer import *
from DepthWarpingLayerWithSpecularityWarping import *

sys.path.insert(0, prefix + "Argument Masking Layer")
from ArgumentMaskingLayer import *

sys.path.insert(0, prefix + "Specularity Masking Layer")
from SpecularityMaskingLayer import *

sys.path.insert(0, prefix + "Union Masking Layer")
from UnionMaskingLayer import *

sys.path.insert(0, prefix + "Callbacks")
from Callback_alternative_updating import *

## Function for visualizing depth image
def visualize_depth_map(depth_map_test, title):
    min_value, max_value, min_loc, max_loc = cv2.minMaxLoc(depth_map_test)
    depth_map_visualize = (depth_map_test - min_value) / (max_value - min_value) * 255
    depth_map_visualize = np.asarray(depth_map_visualize, dtype = 'uint8')
    cv2.imshow(title, depth_map_visualize)
    cv2.waitKey(100)

## Suppose the valid depth value should be larger than 0.01 and smaller than -0.01
## Clipping the mask region value to zero
def clip_close_to_zero(x):
    return K.switch(K.abs(x) < 0.01, 0, x)
def clip_close_to_zero_output_shape(input_shape):
    return input_shape
    
## Masking out invalid region for warped depth map
def mask_invalid_element(x):
    x1, x2 = x
    return K.switch(K.abs(x1) < 0.01, 0, x2)
def mask_invalid_element_output_shape(input_shapes):
    shape1, shape2 = input_shapes
    return shape1 

## Un-normalize the depth value to real one in order to apply depth warping algorithm
## The parameters for this log normalization still needs to be adjusted.
def depth_unlog(x, scale_factor):
    return K.switch(x > 1.0e-5, (K.exp(x * scale_factor) - 1.0) / 4.0, 0)
def depth_unlog_output_shape(input_shape):
    return input_shape
    
def depth_log(x, scale_factor):
    return K.switch(x > 1.0e-5, K.log(4.0 * x + 1.0) / scale_factor, 0)
def depth_log_output_shape(input_shape):
    return input_shape

def mean_squared_difference(x, weight):
    x1, x2 = x
    valid_sum =  K.greater(x1, 1.0e-3).sum()
    mse = weight * K.sqrt(K.sum(K.square(x1 - x2)) / valid_sum)
    return mse
def mean_squared_difference_output_shape(input_shapes):
    input_shape1, input_shape2 = input_shapes
    return (input_shape1[0], 1)
   
def sparse_mean_squared_difference(x, weight):
    x1, x2 = x
    valid_sum = K.greater(x1, 1.0e-3).sum()
    return weight * K.sum(K.exp(K.square((x1 - K.sum(x1) / valid_sum) / 0.2)) * K.square(x1 - x2)) / valid_sum
def sparse_mean_squared_difference_output_shape(input_shapes):
    input_shape1, input_shape2 = input_shapes
    return (input_shape1[0], 1) 
    
def image_difference(x):
    x1, x2 = x
    return x1 - x2
def image_difference_output_shape(input_shape):
    return input_shape[0]

def customized_loss(y_true, y_pred):
    return y_pred - y_true
    
    
prefix = 'G:/Johns Hopkins University/Projects/Sinus Navigation/Data/'
prefix_seq = prefix + 'seq01/'

get_custom_objects().update({"customized_loss":customized_loss})
depth_encoder_model = load_model(prefix_seq +"sv_3layer_sigmoid_log_depth_encoder_weights-improvement-08-0.00347.hdf5")
depth_encoder_model.summary()

training_sv_imgs = np.load(prefix + "training_sv_imgs.npy")
training_sv_imgs = training_sv_imgs / 255.0
training_sv_imgs = np.reshape(training_sv_imgs, (-1, 256, 288, 2))

## Rotation and translation between two frames
R = np.load(prefix + "depth estimation R.npy")
R_I = np.load(prefix + "depth estimation R_I.npy")
P = np.load(prefix + "depth estimation P.npy")
P = np.reshape(P, (-1, 3, 1))
P_I = np.load(prefix + "depth estimation P_I.npy")
P_I = np.reshape(P_I, (-1, 3, 1))
## Saturation and Value channel of endoscopic images
sv_imgs_1 = np.load(prefix + "depth estimation sv img_1.npy")
sv_imgs_1 = np.array(sv_imgs_1, dtype='float32')
sv_imgs_1 = sv_imgs_1 / 255.0
sv_imgs_2 = np.load(prefix + "depth estimation sv img_2.npy")  
sv_imgs_2 = np.array(sv_imgs_2, dtype='float32')
sv_imgs_2 = sv_imgs_2 / 255.0 

## Training mask images are binary ones showing where we have a valid depth value
training_mask_imgs_1 = np.load(prefix + "depth estimation mask img_1.npy")
training_mask_imgs_1 = np.reshape(training_mask_imgs_1, (-1, 256, 288, 1))
training_mask_imgs_2 = np.load(prefix + "depth estimation mask img_2.npy")   
training_mask_imgs_2 = np.reshape(training_mask_imgs_2, (-1, 256, 288, 1))

## Training masked depth images have normalized depth value for only valid locations
## All zero for other invalid locations
training_masked_depth_imgs_1 = np.load(prefix + "depth estimation masked depth img_1.npy")
training_masked_depth_imgs_1 = np.reshape(training_masked_depth_imgs_1, (-1, 256, 288, 1))
training_masked_depth_imgs_2 = np.load(prefix + "depth estimation masked depth img_2.npy") 
training_masked_depth_imgs_2 = np.reshape(training_masked_depth_imgs_2, (-1, 256, 288, 1))

## Image mask of endoscope
img_mask = np.load(prefix + "mask.npy")
img_mask = img_mask / img_mask.max()
img_mask = np.array(img_mask, dtype='float32')
img_mask = np.reshape(img_mask, (256, 288, 1))

## Eroding mask to avoid some boundary effects when comparing the difference between two depth images
## The kernel size might be further reduced
kernel = np.ones((30,30),np.uint8)
img_mask_erode = cv2.erode(img_mask, kernel, iterations = 1)
img_mask_erode[:15, :] = 0
img_mask_erode = np.reshape(img_mask_erode, (256, 288, 1))

## Meaningless training groundtruth
allzeros_groundtruth_output = np.zeros((R.shape[0], 1))

# downsampling the image size by 4 x 4 to speed up the computation
downsampling = 4

## Load endoscope intrinsic matrix
## We need to change the intrinsic matrix to allow for downsampling naturally
stream = open(prefix + "endoscope.yaml", 'r')
doc_endoscope = yaml.load(stream)
intrinsic_data = doc_endoscope['camera_matrix']['data']
intrinsic_matrix = np.zeros((3,3))
for j in range(3):
    for i in range(3):
        intrinsic_matrix[j][i] = intrinsic_data[j * 3 + i] / downsampling

## Because we shrinked the mask image, we need to change the principal point correspondingly
intrinsic_matrix[0][2] = 129.2657625
intrinsic_matrix[1][2] = 113.10556459
intrinsic_matrix[2][2] = 1.0


# Input image pairs
input_img_1 = Input(shape=(256, 288, 2))
input_img_2 = Input(shape=(256, 288, 2))
input_mask_img_1 = Input(shape=(256, 288, 1))
input_mask_img_2 = Input(shape=(256, 288, 1))
input_sparse_masked_depth_img_1 = Input(shape=(256, 288, 1))
input_sparse_masked_depth_img_2 = Input(shape=(256, 288, 1))

input_translation_vector = Input(shape=(3, 1))
input_rotation_matrix = Input(shape=(3, 3))

input_translation_vector_inverse = Input(shape=(3, 1))
input_rotation_matrix_inverse = Input(shape=(3, 3))

# Network architecture for depth estimation (This part can use more 
## advanced architecture to improve the depth estimation results)
conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')
maxpool1 = MaxPooling2D((2, 2))
conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')
maxpool2 = MaxPooling2D((2, 2))
conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')
convt1 = Conv2DTranspose(64, (3, 3), padding='same', activation='relu')
upsampl1 = UpSampling2D((2, 2))
convt2 = Conv2DTranspose(32, (3, 3), padding='same', activation='relu')
upsampl2 = UpSampling2D((2, 2))
convt3 = Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid')

conv1_static = Conv2D(32, (3, 3), padding='same', activation='relu', trainable=False)
maxpool1_static = MaxPooling2D((2, 2))
conv2_static = Conv2D(64, (3, 3), padding='same', activation='relu', trainable=False)
maxpool2_static = MaxPooling2D((2, 2))
conv3_static = Conv2D(128, (3, 3), padding='same', activation='relu', trainable=False)
convt1_static = Conv2DTranspose(64, (3, 3), padding='same', activation='relu', trainable=False)
upsampl1_static = UpSampling2D((2, 2))
convt2_static = Conv2DTranspose(32, (3, 3), padding='same', activation='relu', trainable=False)
upsampl2_static = UpSampling2D((2, 2))
convt3_static = Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid', trainable=False)

x = conv1(input_img_1)
x = maxpool1(x)
x = conv2(x)
x = maxpool2(x)
x = conv3(x)
x = convt1(x)
x = upsampl1(x)
x = convt2(x)
x = upsampl2(x)
estimated_depth_map_1 = convt3(x)

x = conv1_static(input_img_2)
x = maxpool1_static(x)
x = conv2_static(x)
x = maxpool2_static(x)
x = conv3_static(x)
x = convt1_static(x)
x = upsampl1_static(x)
x = convt2_static(x)
x = upsampl2_static(x)
estimated_depth_map_2 = convt3_static(x)

## Masking out the depth estimation so that it will only have non-zero value at sparse sampling positions
masked_output_depth_img_1 = multiply([estimated_depth_map_1, input_mask_img_1])
masked_output_depth_img_2 = multiply([estimated_depth_map_2, input_mask_img_2])

## Loss related to sparse depth points
## The weights for that is currently too small compared to what I assign to the depth consistency.
## Need to raise the weights for this loss if we want to take the sparse depth points into greater consideration
sparse_masked_mean_squared_difference_1 = Lambda(sparse_mean_squared_difference, \
                                                 output_shape=sparse_mean_squared_difference_output_shape, arguments={'weight': 0.5})([input_sparse_masked_depth_img_1, masked_output_depth_img_1])
sparse_masked_mean_squared_difference_2 = Lambda(sparse_mean_squared_difference, \
                                                 output_shape=sparse_mean_squared_difference_output_shape, arguments={'weight': 0.5})([input_sparse_masked_depth_img_2, masked_output_depth_img_2])

## This layer is used to try to mask out the specularity regions 
## because the specularity will not move in the same pattern as the other valid regions between frames
## So we need to mask out these contaminated regions
specularity_mask_1 = SpecularityMaskingLayer(threshold = 4.0)(input_img_1)
specularity_mask_2 = SpecularityMaskingLayer(threshold = 4.0)(input_img_2)

## Masking out the region beyond endoscope boundary
masked_estimated_depth_map_1 = ArgumentMaskingLayer(img_mask)(estimated_depth_map_1)
masked_estimated_depth_map_2 = ArgumentMaskingLayer(img_mask)(estimated_depth_map_2)

## Suppose the affine transform we have is 1^T_2
## Un-normalize the depth value to real depth
masked_estimated_unlog_depth_map_1 = Lambda(depth_unlog, output_shape=depth_unlog_output_shape, arguments={'scale_factor': 7.0})(masked_estimated_depth_map_1)
masked_estimated_unlog_depth_map_2 = Lambda(depth_unlog, output_shape=depth_unlog_output_shape, arguments={'scale_factor': 7.0})(masked_estimated_depth_map_2)

estimated_unlog_depth_map_1 = Lambda(depth_unlog, output_shape=depth_unlog_output_shape, arguments={'scale_factor': 7.0})(estimated_depth_map_1)
estimated_unlog_depth_map_2 = Lambda(depth_unlog, output_shape=depth_unlog_output_shape, arguments={'scale_factor': 7.0})(estimated_depth_map_2)

## Preparing the specularity mask in both frames
specularity_mask_warped_1 = DepthWarpingLayer_specularity(intrinsic_matrix)([specularity_mask_2, estimated_unlog_depth_map_1, estimated_unlog_depth_map_2, input_translation_vector, input_rotation_matrix])
specularity_mask_warped_2 = DepthWarpingLayer_specularity(intrinsic_matrix)([specularity_mask_1, estimated_unlog_depth_map_2, estimated_unlog_depth_map_1, input_translation_vector_inverse, input_rotation_matrix_inverse])

union_specularity_mask_1 = UnionMaskingLayer()([specularity_mask_warped_1, specularity_mask_1])
union_specularity_mask_2 = UnionMaskingLayer()([specularity_mask_warped_2, specularity_mask_2])

## Depth warping to get the warped depth image in another camera frame
synthetic_depth_map_1 = DepthWarpingLayer(intrinsic_matrix)([masked_estimated_unlog_depth_map_1, masked_estimated_unlog_depth_map_2, input_translation_vector, input_rotation_matrix])
synthetic_depth_map_2 = DepthWarpingLayer(intrinsic_matrix)([masked_estimated_unlog_depth_map_2, masked_estimated_unlog_depth_map_1, input_translation_vector_inverse, input_rotation_matrix_inverse])

synthetic_depth_map_1 = ArgumentMaskingLayer(img_mask_erode)(synthetic_depth_map_1)
synthetic_depth_map_2 = ArgumentMaskingLayer(img_mask_erode)(synthetic_depth_map_2)

## Clip the warped depth map because there could be minus depth value because of imprecise depth estimation before
## Therefore, we need to reassure that the masking regions are right (masking regions should have zero depth value)
clipped_synthetic_depth_map_1 = Lambda(clip_close_to_zero, output_shape=clip_close_to_zero_output_shape)(synthetic_depth_map_1)
clipped_synthetic_depth_map_2 = Lambda(clip_close_to_zero, output_shape=clip_close_to_zero_output_shape)(synthetic_depth_map_2)

## Masking out the specularity regions
masked_synthetic_depth_map_1 = multiply([clipped_synthetic_depth_map_1, union_specularity_mask_1])
masked_synthetic_depth_map_2 = multiply([clipped_synthetic_depth_map_2, union_specularity_mask_2])

## Masking out the regions beyond the endoscope boundary
masked_estimated_unlog_depth_map_1 = ArgumentMaskingLayer(img_mask_erode)(masked_estimated_unlog_depth_map_1)
masked_estimated_unlog_depth_map_2 = ArgumentMaskingLayer(img_mask_erode)(masked_estimated_unlog_depth_map_2)

## Masking out the specularity regions
masked_estimated_unlog_depth_map_1 = multiply([masked_estimated_unlog_depth_map_1, union_specularity_mask_1])
masked_estimated_unlog_depth_map_2 = multiply([masked_estimated_unlog_depth_map_2, union_specularity_mask_2])

full_masked_estimated_unlog_depth_map_1 = Lambda(mask_invalid_element, output_shape=mask_invalid_element_output_shape)([masked_synthetic_depth_map_1, masked_estimated_unlog_depth_map_1])
full_masked_estimated_unlog_depth_map_2 = Lambda(mask_invalid_element, output_shape=mask_invalid_element_output_shape)([masked_synthetic_depth_map_2, masked_estimated_unlog_depth_map_2])

## Image difference is just for visualizing
difference_image_1 = Lambda(image_difference, output_shape=image_difference_output_shape)([masked_synthetic_depth_map_1, full_masked_estimated_unlog_depth_map_1])
difference_image_2 = Lambda(image_difference, output_shape=image_difference_output_shape)([masked_synthetic_depth_map_2, full_masked_estimated_unlog_depth_map_2])

## Loss related to depth consistency
depth_map_mean_squared_difference_1 = Lambda(mean_squared_difference, output_shape=mean_squared_difference_output_shape, arguments={'weight': 0.1})([full_masked_estimated_unlog_depth_map_1, masked_synthetic_depth_map_1])
depth_map_mean_squared_difference_2 = Lambda(mean_squared_difference, output_shape=mean_squared_difference_output_shape, arguments={'weight': 0.1})([full_masked_estimated_unlog_depth_map_2, masked_synthetic_depth_map_2])

## Adding up all losses together to form the final loss function
mse_loss = add([sparse_masked_mean_squared_difference_1, sparse_masked_mean_squared_difference_2, depth_map_mean_squared_difference_1, depth_map_mean_squared_difference_2])

model = Model([input_sparse_masked_depth_img_1, input_sparse_masked_depth_img_2, input_mask_img_1, input_mask_img_2, \
               input_img_1, input_img_2, input_translation_vector, input_translation_vector_inverse, 
              input_rotation_matrix, input_rotation_matrix_inverse], mse_loss)
model.summary()
sgd = optimizers.SGD(lr=1.0e-5, momentum=0.9, nesterov=True)
model.compile(loss=customized_loss, optimizer=sgd)

## The Alternative updating seems to work if every freeze-one-train-one iteration has more epochs
## Otherwise it will just be a oscillation and the result will be the same as training both branches
alternative_updating = AlternativeUpdating(10)
filepath = prefix_seq + "warped_depth_estimation_network_weights-improvement-{epoch:02d}-{val_loss:.5f}.hdf5"
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, verbose=1, mode='auto', epsilon=0.00001, cooldown=0, min_lr=0)
checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
earlyStopping = EarlyStopping(monitor='val_loss', patience=60, verbose=0, min_delta=0.00001, mode='auto')
##
history = model.fit([training_masked_depth_imgs_1, training_masked_depth_imgs_2, training_mask_imgs_1, training_mask_imgs_2, \
                     sv_imgs_1, sv_imgs_2, P, P_I, R, R_I], allzeros_groundtruth_output, batch_size=5, \
                     epochs=500, verbose=1, callbacks=[earlyStopping, checkpointer, reducelr, alternative_updating], validation_split=0.05, validation_data=None, shuffle=False, class_weight=None, sample_weight=None)
