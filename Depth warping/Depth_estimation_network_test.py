# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 13:35:48 2017

@author: DELL1
"""

import cv2 
import numpy as np
import yaml
from plyfile import PlyData, PlyElement
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

import sys
sys.path.insert(0,"G:/Johns Hopkins University/Projects/Depth estimation/Scripts/Depth Warping Layer")
from DepthWarpingLayer import *

sys.path.insert(0,"G:/Johns Hopkins University/Projects/Depth estimation/Scripts/Depth Warping Layer")
from DepthWarpingLayerWithSpecularityWarping import *

sys.path.insert(0,"G:/Johns Hopkins University/Projects/Depth estimation/Scripts/Argument Masking Layer")
from ArgumentMaskingLayer import *

sys.path.insert(0,"G:/Johns Hopkins University/Projects/Depth estimation/Scripts/Specularity Masking Layer")
from SpecularityMaskingLayer import *

sys.path.insert(0,"G:/Johns Hopkins University/Projects/Depth estimation/Scripts/Union Masking Layer")
from UnionMaskingLayer import *

def visualize_depth_map(depth_map_test, title):
    depth_map_test = np.abs(depth_map_test)
    min_value, max_value, min_loc, max_loc = cv2.minMaxLoc(depth_map_test)
    depth_map_visualize = (depth_map_test - min_value) / (max_value - min_value) * 255
    depth_map_visualize = np.asarray(depth_map_visualize, dtype = 'uint8')
    cv2.imshow(title, depth_map_visualize)
    cv2.waitKey(100)

    
    
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
    
def depth_unlog(x, scale_factor):
    return K.switch(x > 0.01, (K.exp(x * scale_factor) - 4.0) / 4.0 + 0.25, 0)
def depth_unlog_output_shape(input_shape):
    return input_shape
    
def depth_log(x, scale_factor):
    return K.switch(x > 0.01, K.log(4.0 * (point_3D_position_camera[2] - 0.25) + 4.0) / scale_factor, 0)
def depth_log_output_shape(input_shape):
    return input_shape

def mean_squared_difference(x, weight):
    x1, x2 = x
    valid_sum =  K.greater(x1, 0.01).sum()
#    mean_squared = weight * K.log(K.sum(K.square(x1 - x2)) / valid_sum + 1.0)
    mse = weight * K.sqrt(K.sum(K.square(x1 - x2)) / valid_sum)
    return mse
def mean_squared_difference_output_shape(input_shapes):
    input_shape1, input_shape2 = input_shapes
    return (input_shape1[0], 1)
   
def sparse_mean_squared_difference(x, weight):
    x1, x2 = x
    valid_sum = K.greater(x1, 0.01).sum()
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
#    valid_sum = K.greater(y_true, 0.0).sum()
#    return K.sum(K.exp(K.square((y_true - K.sum(y_true) / valid_sum) / 0.2)) * K.square(y_true-y_pred)) / valid_sum
#    return K.sum(K.square(y_true - y_pred)) / K.greater(y_true, 0.0).sum()

prefix = 'G:/Johns Hopkins University/Projects/Sinus Navigation/Data/'
prefix_seq = prefix + 'seq01/'

get_custom_objects().update({"customized_loss":customized_loss})
depth_encoder_model = load_model(prefix_seq +"sv_3layer_sigmoid_log_depth_encoder_weights-improvement-08-0.00347.hdf5")
depth_encoder_model.summary()

training_sv_imgs = np.load(prefix + "training_sv_imgs.npy")
training_sv_imgs = training_sv_imgs / 255.0
training_sv_imgs = np.reshape(training_sv_imgs, (-1, 256, 288, 2))

R = np.load(prefix + "depth estimation R.npy")
R_I = np.load(prefix + "depth estimation R_I.npy")
P = np.load(prefix + "depth estimation P.npy")
P_I = np.load(prefix + "depth estimation P_I.npy")

sv_imgs_1 = np.load(prefix + "depth estimation sv img_1.npy")
sv_imgs_1 = np.array(sv_imgs_1, dtype='float32')
sv_imgs_1 = sv_imgs_1 / 255.0
sv_imgs_2 = np.load(prefix + "depth estimation sv img_2.npy")  
sv_imgs_2 = np.array(sv_imgs_2, dtype='float32')
sv_imgs_2 = sv_imgs_2 / 255.0 

training_mask_imgs_1 = np.load(prefix + "depth estimation mask img_1.npy")
training_mask_imgs_1 = np.reshape(training_mask_imgs_1, (-1, 256, 288, 1))
training_mask_imgs_2 = np.load(prefix + "depth estimation mask img_2.npy")   
training_mask_imgs_2 = np.reshape(training_mask_imgs_2, (-1, 256, 288, 1))

training_masked_depth_imgs_1 = np.load(prefix + "depth estimation masked depth img_1.npy")
training_masked_depth_imgs_1 = np.reshape(training_masked_depth_imgs_1, (-1, 256, 288, 1))
training_masked_depth_imgs_2 = np.load(prefix + "depth estimation masked depth img_2.npy") 
training_masked_depth_imgs_2 = np.reshape(training_masked_depth_imgs_2, (-1, 256, 288, 1))

img_mask = np.load(prefix + "mask.npy")
img_mask = img_mask / img_mask.max()
img_mask = np.array(img_mask, dtype='float32')
img_mask = np.reshape(img_mask, (256, 288, 1))

kernel = np.ones((30,30),np.uint8)
img_mask_erode = cv2.erode(img_mask, kernel, iterations = 1)
img_mask_erode[:15, :] = 0
img_mask_erode = np.reshape(img_mask_erode, (256, 288, 1))

P = np.reshape(P, (-1, 3, 1))
P_I = np.reshape(P_I, (-1, 3, 1))

allzeros_groundtruth_output = np.zeros((R.shape[0], 1))

# downsampling the image size by 4 x 4
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

## Two branches of depth estimation network with shared layers
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

## TODO: Found the problem: minus synchronized depth is not processed properly
masked_output_depth_img_1 = multiply([estimated_depth_map_1, input_mask_img_1])
masked_output_depth_img_2 = multiply([estimated_depth_map_2, input_mask_img_2])
sparse_masked_mean_squared_difference_1 = Lambda(sparse_mean_squared_difference, \
                                                 output_shape=sparse_mean_squared_difference_output_shape, arguments={'weight': 0.5})([input_sparse_masked_depth_img_1, masked_output_depth_img_1])
sparse_masked_mean_squared_difference_2 = Lambda(sparse_mean_squared_difference, \
                                                 output_shape=sparse_mean_squared_difference_output_shape, arguments={'weight': 0.5})([input_sparse_masked_depth_img_2, masked_output_depth_img_2])

specularity_mask_1 = SpecularityMaskingLayer(threshold = 4.0)(input_img_1)
specularity_mask_2 = SpecularityMaskingLayer(threshold = 4.0)(input_img_2)

masked_estimated_depth_map_1 = ArgumentMaskingLayer(img_mask)(estimated_depth_map_1)
masked_estimated_depth_map_2 = ArgumentMaskingLayer(img_mask)(estimated_depth_map_2)

##TODO:  We need a specularity masking layer to mask out these confusing regions
## Generate specularity region mask and use mask_invalid_element to mask out

### Suppose the affine transform we have is 1^T_2
masked_estimated_unlog_depth_map_1 = Lambda(depth_unlog, output_shape=depth_unlog_output_shape, arguments={'scale_factor': 7.0})(masked_estimated_depth_map_1)
masked_estimated_unlog_depth_map_2 = Lambda(depth_unlog, output_shape=depth_unlog_output_shape, arguments={'scale_factor': 7.0})(masked_estimated_depth_map_2)

estimated_unlog_depth_map_1 = Lambda(depth_unlog, output_shape=depth_unlog_output_shape, arguments={'scale_factor': 7.0})(estimated_depth_map_1)
estimated_unlog_depth_map_2 = Lambda(depth_unlog, output_shape=depth_unlog_output_shape, arguments={'scale_factor': 7.0})(estimated_depth_map_2)

specularity_mask_warped_1 = DepthWarpingLayer_specularity(intrinsic_matrix)([specularity_mask_2, estimated_unlog_depth_map_1, estimated_unlog_depth_map_2, input_translation_vector, input_rotation_matrix])
specularity_mask_warped_2 = DepthWarpingLayer_specularity(intrinsic_matrix)([specularity_mask_1, estimated_unlog_depth_map_2, estimated_unlog_depth_map_1, input_translation_vector_inverse, input_rotation_matrix_inverse])

union_specularity_mask_1 = UnionMaskingLayer()([specularity_mask_warped_1, specularity_mask_1])
union_specularity_mask_2 = UnionMaskingLayer()([specularity_mask_warped_2, specularity_mask_2])

synthetic_depth_map_1 = DepthWarpingLayer(intrinsic_matrix)([masked_estimated_unlog_depth_map_1, masked_estimated_unlog_depth_map_2, input_translation_vector, input_rotation_matrix])
synthetic_depth_map_2 = DepthWarpingLayer(intrinsic_matrix)([masked_estimated_unlog_depth_map_2, masked_estimated_unlog_depth_map_1, input_translation_vector_inverse, input_rotation_matrix_inverse])

synthetic_depth_map_1 = ArgumentMaskingLayer(img_mask_erode)(synthetic_depth_map_1)
synthetic_depth_map_2 = ArgumentMaskingLayer(img_mask_erode)(synthetic_depth_map_2)

clipped_synthetic_depth_map_1 = Lambda(clip_close_to_zero, output_shape=clip_close_to_zero_output_shape)(synthetic_depth_map_1)
clipped_synthetic_depth_map_2 = Lambda(clip_close_to_zero, output_shape=clip_close_to_zero_output_shape)(synthetic_depth_map_2)

masked_synthetic_depth_map_1 = multiply([clipped_synthetic_depth_map_1, union_specularity_mask_1])
masked_synthetic_depth_map_2 = multiply([clipped_synthetic_depth_map_2, union_specularity_mask_2])

masked_estimated_unlog_depth_map_1 = ArgumentMaskingLayer(img_mask_erode)(masked_estimated_unlog_depth_map_1)
masked_estimated_unlog_depth_map_2 = ArgumentMaskingLayer(img_mask_erode)(masked_estimated_unlog_depth_map_2)

masked_estimated_unlog_depth_map_1 = multiply([masked_estimated_unlog_depth_map_1, union_specularity_mask_1])
masked_estimated_unlog_depth_map_2 = multiply([masked_estimated_unlog_depth_map_2, union_specularity_mask_2])

full_masked_estimated_unlog_depth_map_1 = Lambda(mask_invalid_element, output_shape=mask_invalid_element_output_shape)([masked_synthetic_depth_map_1, masked_estimated_unlog_depth_map_1])
full_masked_estimated_unlog_depth_map_2 = Lambda(mask_invalid_element, output_shape=mask_invalid_element_output_shape)([masked_synthetic_depth_map_2, masked_estimated_unlog_depth_map_2])

difference_image_1 = Lambda(image_difference, output_shape=image_difference_output_shape)([masked_synthetic_depth_map_1, full_masked_estimated_unlog_depth_map_1])
difference_image_2 = Lambda(image_difference, output_shape=image_difference_output_shape)([masked_synthetic_depth_map_2, full_masked_estimated_unlog_depth_map_2])
###
##
depth_map_mean_squared_difference_1 = Lambda(mean_squared_difference, output_shape=mean_squared_difference_output_shape, arguments={'weight': 0.1})([full_masked_estimated_unlog_depth_map_1, masked_synthetic_depth_map_1])
depth_map_mean_squared_difference_2 = Lambda(mean_squared_difference, output_shape=mean_squared_difference_output_shape, arguments={'weight': 0.1})([full_masked_estimated_unlog_depth_map_2, masked_synthetic_depth_map_2])
###
#
mse_loss = add([sparse_masked_mean_squared_difference_1, sparse_masked_mean_squared_difference_2, depth_map_mean_squared_difference_1, depth_map_mean_squared_difference_2])
#
#model = Model([input_sparse_masked_depth_img_1, input_sparse_masked_depth_img_2, input_mask_img_1, input_mask_img_2, \
#               input_img_1, input_img_2, input_translation_vector, input_translation_vector_inverse, 
#              input_rotation_matrix, input_rotation_matrix_inverse], mse_loss)
#
model = Model([input_sparse_masked_depth_img_1, input_sparse_masked_depth_img_2, input_mask_img_1, input_mask_img_2, \
               input_img_1, input_img_2, input_translation_vector, input_translation_vector_inverse, 
              input_rotation_matrix, input_rotation_matrix_inverse], [estimated_unlog_depth_map_1, estimated_unlog_depth_map_2])

model.summary()
#adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
sgd = optimizers.SGD(lr=1.0e-3, momentum=0.9, nesterov=True) #, 
model.compile(loss=customized_loss, optimizer=sgd)

model.load_weights(prefix_seq + "warped_depth_estimation_network_weights-improvement-00-0.26062.hdf5")

model.layers[3].set_weights(model.layers[2].get_weights())
model.layers[7].set_weights(model.layers[6].get_weights())
model.layers[11].set_weights(model.layers[10].get_weights())
model.layers[13].set_weights(model.layers[12].get_weights()) 
model.layers[17].set_weights(model.layers[16].get_weights())
model.layers[21].set_weights(model.layers[20].get_weights())


count = 40
results = model.predict([training_masked_depth_imgs_1[:count], training_masked_depth_imgs_2[:count], training_mask_imgs_1[:count], training_mask_imgs_2[:count], \
                     sv_imgs_1[:count], sv_imgs_2[:count], P[:count], P_I[:count], R[:count], R_I[:count]], batch_size = 5)

sparse_only_model = Model([input_img_1], estimated_unlog_depth_map_1)
sparse_only_model.layers[1].set_weights(depth_encoder_model.layers[1].get_weights())
sparse_only_model.layers[3].set_weights(depth_encoder_model.layers[3].get_weights())
sparse_only_model.layers[5].set_weights(depth_encoder_model.layers[5].get_weights())
sparse_only_model.layers[6].set_weights(depth_encoder_model.layers[6].get_weights())
sparse_only_model.layers[8].set_weights(depth_encoder_model.layers[8].get_weights())
sparse_only_model.layers[10].set_weights(depth_encoder_model.layers[10].get_weights())

sparse_only_results = sparse_only_model.predict(sv_imgs_1[:count])

## Generate Point cloud
for i in range(count):
    indexes =np.where(img_mask > 130 / 255.0)
    start_h = indexes[0].min()
    end_h = indexes[0].max()
    start_w = indexes[1].min()
    end_w = indexes[1].max()
    
    f_x = intrinsic_data[0] / downsampling
    c_x = intrinsic_data[2] / downsampling
    f_y = intrinsic_data[4] / downsampling
    c_y = intrinsic_data[5] / downsampling
    
    sv_img_1 = sv_imgs_1[i]
    
    depth_data = results[0][i]
    depth_data = np.reshape(depth_data, (256, 288)) 
    
    height, width, channel = sv_img_1.shape
    point_clouds = []
    
    #img_ratio = (sv_img_1[:, :, 1] * 255.0) / (1.0 + sv_img_1[:, :, 0] * 255.0)
    #img_ratio = np.asarray(np.around((img_ratio - img_ratio.min()) / (img_ratio.max() - img_ratio.min()) * 255.0), dtype = np.uint8)
    #threshold, binary_image = cv2.threshold(img_ratio, 3, 255, cv2.THRESH_BINARY)
    #ret, markers = cv2.connectedComponents(binary_image)
    #cv2.imshow("specularity candidates", binary_image)
    #sum_list = []
    #label_list = []
    #for label in range(ret):
    #  if (label > 0):
    #      sum_temp = (markers == label).sum()
    #      label_list.append(label)
    #      sum_list.append(sum_temp)
    #      
    #for j in range(len(label_list)):
    #  if(sum_list[j] > height * width * 0.1):
    #      shit_image = np.asarray(((markers==label_list[j]) * 255), dtype = np.uint8)
    #      binary_image = binary_image - shit_image
    #      
    #cv2.imshow("specularity clean", binary_image)
    #cv2.waitKey(10)
    
    for h in range(height):
        for w in range(width):
            z = depth_data[h, w]
            x = (w + start_w - c_x) / f_x * z
            y = (h + start_h - c_y) / f_y * z
            v = sv_img_1[h, w, 1] * 255
            point_clouds.append((x, y, z, v, v, v))
                    
    vertex = np.array(point_clouds, dtype = [('x', 'f4'), ('y', 'f4'),('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(prefix_seq + 'sv_point_cloud_' + ('%08d')%(i) + '.ply')
    
    visualize_depth_map(depth_data, "depth map")
    
    
    depth_data = sparse_only_results[i]
#    depth_data = (np.exp(depth_data * 7.0) - 4.0) / 4.0 + 0.25

    height, width, channel = sv_img_1.shape
    point_clouds = []
    
    #img_ratio = (sv_img_1[:, :, 1] * 255.0) / (1.0 + sv_img_1[:, :, 0] * 255.0)
    #img_ratio = np.asarray(np.around((img_ratio - img_ratio.min()) / (img_ratio.max() - img_ratio.min()) * 255.0), dtype = np.uint8)
    #threshold, binary_image = cv2.threshold(img_ratio, 3, 255, cv2.THRESH_BINARY)
    #ret, markers = cv2.connectedComponents(binary_image)
    #cv2.imshow("specularity candidates", binary_image)
    #sum_list = []
    #label_list = []
    #for label in range(ret):
    #  if (label > 0):
    #      sum_temp = (markers == label).sum()
    #      label_list.append(label)
    #      sum_list.append(sum_temp)
    #      
    #for j in range(len(label_list)):
    #  if(sum_list[j] > height * width * 0.1):
    #      shit_image = np.asarray(((markers==label_list[j]) * 255), dtype = np.uint8)
    #      binary_image = binary_image - shit_image
    #      
    #cv2.imshow("specularity clean", binary_image)
    #cv2.waitKey(10)
    
    for h in range(height):
        for w in range(width):
            z = depth_data[h, w]
            x = (w + start_w - c_x) / f_x * z
            y = (h + start_h - c_y) / f_y * z
            v = sv_img_1[h, w, 1] * 255
            point_clouds.append((x, y, z, v, v, v))
                    
    vertex = np.array(point_clouds, dtype = [('x', 'f4'), ('y', 'f4'),('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(prefix_seq + 'sparse_only_sv_point_cloud_' + ('%08d')%(i) + '.ply')
    
    visualize_depth_map(depth_data, "depth map sparse only")
    cv2.imshow("v map", sv_img_1[:, :, 1])
    cv2.waitKey()