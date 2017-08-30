# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 13:35:48 2017

@author: DELL1
"""

import cv2 
import numpy as np
import yaml
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Flatten, concatenate, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers import add, merge, multiply
from keras.layers import RepeatVector, Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model, load_model
from keras import optimizers
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.models import model_from_json

from plyfile import PlyData, PlyElement

import sys
sys.path.insert(0,"G:/Johns Hopkins University/Projects/Depth estimation/Scripts/Depth Warping Layer")
from DepthWarpingLayer import *

sys.path.insert(0,"G:/Johns Hopkins University/Projects/Depth estimation/Scripts/Argument Masking Layer")
from ArgumentMaskingLayer import *

def visualize_depth_map(depth_map_test, title):
    min_value, max_value, min_loc, max_loc = cv2.minMaxLoc(depth_map_test)
    depth_map_visualize = (depth_map_test - min_value) / (max_value - min_value) * 255
    depth_map_visualize = np.asarray(depth_map_visualize, dtype = 'uint8')
    cv2.imshow(title, depth_map_visualize)
    cv2.waitKey(100)

    
## Masking out invalid region for warped depth map
def mask_invalid_element(x):
    x1, x2 = x
    return K.switch(x1 > 1.0e-7, x2, 0)
def mask_invalid_element_output_shape(input_shape):
    shape1, shape2 = input_shape
    return shape1 
    
def depth_unlog(x):
    return (K.exp(x * 6.0) - 4.0) / 4.0 + 0.25
def depth_unlog_output_shape(input_shape):
    return input_shape

def mean_squared_difference(x, weight):
    x1, x2 = x
    return weight * K.sum(K.square(x1 - x2)) / K.greater(x1, 1.0e-7).sum()
def mean_squared_difference_output_shape(input_shape):
    return (input_shape[0], 1)    
   
def sparse_mean_squared_difference(x, weight):
    x1, x2 = x
    valid_sum = K.greater(x1, 1.0e-7).sum()
    return weight * K.sum(K.exp(K.square((x1 - K.sum(x1) / valid_sum) / 0.2)) * K.square(x1-x2)) / valid_sum
def sparse_mean_squared_difference_output_shape(input_shape):
    return (input_shape[0], 1) 
    
def depth_log(x):
    return K.switch(x > 1.0e-7, K.log(4.0 * (x - 0.25) + 4.0) / 6.0, 0)  
def depth_log_output_shape(input_shape):
    return input_shape
    
def customized_loss(y_true, y_pred):
    return y_pred ## - y_true  


prefix = 'G:/Johns Hopkins University/Projects/Sinus Navigation/Data/'
prefix_seq = prefix + 'seq01/'

get_custom_objects().update({"customized_loss":customized_loss})
#depth_encoder_model = load_model(prefix_seq +"sv_3layer_sigmoid_shift_log_depth_encoder_weights-improvement-58-0.00583.hdf5")
#
###file:///G:/Johns Hopkins University/Projects/Sinus Navigation/Data/seq01/warped_depth_estimation_network_weights-improvement-29-0.08673.hdf5
#depth_encoder_model.summary()

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
conv2 = Conv2D(64, (5, 5), padding='same', activation='relu')
maxpool2 = MaxPooling2D((2, 2))
conv3 = Conv2D(128, (5, 5), padding='same', activation='relu')
convt1 = Conv2DTranspose(64, (5, 5), padding='same', activation='relu')
upsampl1 = UpSampling2D((2, 2))
convt2 = Conv2DTranspose(32, (5, 5), padding='same', activation='relu')
upsampl2 = UpSampling2D((2, 2))
convt3 = Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid')


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

x = conv1(input_img_2)
x = maxpool1(x)
x = conv2(x)
x = maxpool2(x)
x = conv3(x)
x = convt1(x)
x = upsampl1(x)
x = convt2(x)
x = upsampl2(x)
estimated_depth_map_2 = convt3(x)

masked_output_depth_img_1 = multiply([estimated_depth_map_1, input_mask_img_1])
masked_output_depth_img_2 = multiply([estimated_depth_map_2, input_mask_img_2])

sparse_masked_mean_squared_difference_1 = Lambda(mean_squared_difference, output_shape=mean_squared_difference_output_shape, arguments={'weight': 0.25})([masked_output_depth_img_1, input_sparse_masked_depth_img_1])
sparse_masked_mean_squared_difference_2 = Lambda(mean_squared_difference, output_shape=mean_squared_difference_output_shape, arguments={'weight': 0.25})([masked_output_depth_img_2, input_sparse_masked_depth_img_2])

## Suppose the affine transform we have is 1^T_2
estimated_unlog_depth_map_1 = Lambda(depth_unlog, output_shape=depth_unlog_output_shape)(estimated_depth_map_1)
estimated_unlog_depth_map_2 = Lambda(depth_unlog, output_shape=depth_unlog_output_shape)(estimated_depth_map_2)

masked_estimated_unlog_depth_map_1 = ArgumentMaskingLayer(img_mask)(estimated_unlog_depth_map_1)
masked_estimated_unlog_depth_map_2 = ArgumentMaskingLayer(img_mask)(estimated_unlog_depth_map_2)

synthetic_depth_map_1 = DepthWarpingLayer(intrinsic_matrix, img_mask)([masked_estimated_unlog_depth_map_1, masked_estimated_unlog_depth_map_2, input_translation_vector, input_rotation_matrix])
synthetic_depth_map_2 = DepthWarpingLayer(intrinsic_matrix, img_mask)([masked_estimated_unlog_depth_map_2, masked_estimated_unlog_depth_map_1, input_translation_vector_inverse, input_rotation_matrix_inverse])

#masked_synthetic_depth_map_1 = Lambda(mask_invalid_element_argument, output_shape=mask_invalid_element_argument_output_shape, 
#       arguments={'mask': img_mask})(synthetic_depth_map_1)
#masked_synthetic_depth_map_2 = Lambda(mask_invalid_element_argument, output_shape=mask_invalid_element_argument_output_shape, 
#       arguments={'mask': img_mask})(synthetic_depth_map_2)
masked_synthetic_depth_map_1 = ArgumentMaskingLayer(img_mask)(synthetic_depth_map_1)
masked_synthetic_depth_map_2 = ArgumentMaskingLayer(img_mask)(synthetic_depth_map_2)

true_masked_unlog_estimated_depth_map_1 = Lambda(mask_invalid_element, output_shape=mask_invalid_element_output_shape)([masked_synthetic_depth_map_1, masked_estimated_unlog_depth_map_1])
true_masked_unlog_estimated_depth_map_2 = Lambda(mask_invalid_element, output_shape=mask_invalid_element_output_shape)([masked_synthetic_depth_map_2, masked_estimated_unlog_depth_map_2])

#masked_synthetic_depth_map_1 = BatchNormalization()(masked_synthetic_depth_map_1)
#masked_synthetic_depth_map_2 = BatchNormalization()(masked_synthetic_depth_map_2)
#true_masked_unlog_estimated_depth_map_1 = BatchNormalization()(true_masked_unlog_estimated_depth_map_1)
#true_masked_unlog_estimated_depth_map_2 = BatchNormalization()(true_masked_unlog_estimated_depth_map_2)

masked_log_synthetic_depth_map_1 = Lambda(depth_log, output_shape=depth_log_output_shape)(masked_synthetic_depth_map_1)
masked_log_synthetic_depth_map_2 = Lambda(depth_log, output_shape=depth_log_output_shape)(masked_synthetic_depth_map_2)
masked_log_estimated_depth_map_1 = Lambda(depth_log, output_shape=depth_log_output_shape)(true_masked_unlog_estimated_depth_map_1)
masked_log_estimated_depth_map_2 = Lambda(depth_log, output_shape=depth_log_output_shape)(true_masked_unlog_estimated_depth_map_1)

depth_map_mean_squared_difference_1 = Lambda(mean_squared_difference, output_shape=mean_squared_difference_output_shape, arguments={'weight': 0.25})([masked_log_synthetic_depth_map_1, masked_log_estimated_depth_map_1])
depth_map_mean_squared_difference_2 = Lambda(mean_squared_difference, output_shape=mean_squared_difference_output_shape, arguments={'weight': 0.25})([masked_log_synthetic_depth_map_2, masked_log_estimated_depth_map_2])

mse_loss = add([sparse_masked_mean_squared_difference_1, sparse_masked_mean_squared_difference_2, depth_map_mean_squared_difference_1, depth_map_mean_squared_difference_2])

#model = Model([input_img_1, input_img_2, input_translation_vector, input_translation_vector_inverse, 
#              input_rotation_matrix, input_rotation_matrix_inverse], [synthetic_depth_map_1, synthetic_depth_map_2, 
#true_masked_unlog_estimated_depth_map_1, true_masked_unlog_estimated_depth_map_2, estimated_unlog_depth_map_1, estimated_unlog_depth_map_2, mse_loss])

model = Model([input_sparse_masked_depth_img_1, input_sparse_masked_depth_img_2, input_mask_img_1, input_mask_img_2, \
               input_img_1, input_img_2, input_translation_vector, input_translation_vector_inverse, 
              input_rotation_matrix, input_rotation_matrix_inverse], mse_loss)

#model = Model([input_img_1], masked_estimated_unlog_depth_map_1)
model.summary()

#model.layers[2].set_weights(depth_encoder_model.layers[1].get_weights())
#model.layers[4].set_weights(depth_encoder_model.layers[3].get_weights())
#model.layers[6].set_weights(depth_encoder_model.layers[5].get_weights())
#model.layers[7].set_weights(depth_encoder_model.layers[6].get_weights())
#model.layers[9].set_weights(depth_encoder_model.layers[8].get_weights())
#model.layers[11].set_weights(depth_encoder_model.layers[10].get_weights())

sgd = optimizers.SGD(lr=1.0e-3, momentum=0.9, nesterov=True)
model.compile(loss=customized_loss, optimizer=sgd)
model.load_weights(prefix_seq + "warped_depth_estimation_network_weights-improvement-17-0.00586.hdf5")


depth_model = Model([input_img_1], estimated_unlog_depth_map_1)
depth_model.summary()
depth_model.compile(loss=customized_loss, optimizer=sgd)

depth_model.summary()
depth_model.layers[1].set_weights(model.layers[2].get_weights())
depth_model.layers[3].set_weights(model.layers[4].get_weights())
depth_model.layers[5].set_weights(model.layers[6].get_weights())
depth_model.layers[6].set_weights(model.layers[7].get_weights())
depth_model.layers[8].set_weights(model.layers[9].get_weights())
depth_model.layers[10].set_weights(model.layers[11].get_weights())
estimated_depth_img = depth_model.predict([sv_imgs_1])


depth_encoder_model = load_model(prefix_seq +"sv_3layer_sigmoid_shift_log_depth_encoder_weights-improvement-58-0.00583.hdf5")
depth_model.layers[1].set_weights(depth_encoder_model.layers[1].get_weights())
depth_model.layers[3].set_weights(depth_encoder_model.layers[3].get_weights())
depth_model.layers[5].set_weights(depth_encoder_model.layers[5].get_weights())
depth_model.layers[6].set_weights(depth_encoder_model.layers[6].get_weights())
depth_model.layers[8].set_weights(depth_encoder_model.layers[8].get_weights())
depth_model.layers[10].set_weights(depth_encoder_model.layers[10].get_weights())
estimated_depth_img_original = depth_model.predict([sv_imgs_1])
#img = (np.exp(estimated_depth_img_original[0] * 9.0) - 4.0) / 4.0 + 0.25





## Generate Point cloud
i = 5

visualize_depth_map(estimated_depth_img[i], "1")
visualize_depth_map(estimated_depth_img_original[i], "2")
#m = 1
indexes =np.where(img_mask > 130 / 255.0)
start_h = indexes[0].min()
end_h = indexes[0].max()
start_w = indexes[1].min()
end_w = indexes[1].max()

#prefix_seq = prefix + 'seq' + ('%02d')%(m) + '/'
#img = cv2.imread(prefix_seq + ('frame%04d')%(i) + '.png')
#img = cv2.resize(img, (img.shape[1] / downsampling, img.shape[0] / downsampling), interpolation = cv2.INTER_CUBIC)

f_x = intrinsic_data[0] / downsampling
c_x = intrinsic_data[2] / downsampling
f_y = intrinsic_data[4] / downsampling
c_y = intrinsic_data[5] / downsampling

sv_img_1 = sv_imgs_1[i]

depth_data = estimated_depth_img[i]
depth_data = np.reshape(depth_data, (256, 288)) 

depth_img = (depth_data - depth_data.min()) / (depth_data.max() - depth_data.min()) * 255.0
depth_img = np.array(depth_img, dtype=np.uint8)
plt.imshow(depth_img)
plt.yticks(np.array([]))
plt.xticks(np.array([]))
plt.show()

height, width, channel = sv_img_1.shape
point_clouds = []

img_ratio = (sv_img_1[:, :, 1] * 255.0) / (1.0 + sv_img_1[:, :, 0] * 255.0)
img_ratio = np.asarray(np.around((img_ratio - img_ratio.min()) / (img_ratio.max() - img_ratio.min()) * 255.0), dtype = np.uint8)
threshold, binary_image = cv2.threshold(img_ratio, 2, 255, cv2.THRESH_BINARY)
ret, markers = cv2.connectedComponents(binary_image)
cv2.imshow("specularity candidates", binary_image)
sum_list = []
label_list = []
for label in range(ret):
  if (label > 0):
      sum_temp = (markers == label).sum()
      label_list.append(label)
      sum_list.append(sum_temp)
      
for j in range(len(label_list)):
  if(sum_list[j] > height * width * 0.1):
      shit_image = np.asarray(((markers==label_list[j]) * 255), dtype = np.uint8)
      binary_image = binary_image - shit_image
      
cv2.imshow("specularity clean", binary_image)
cv2.waitKey(10)

for h in range(height):
    for w in range(width):
#        if (img_mask[h, w] > 22 / 25.0 and binary_image[h, w] < 130):
        z = depth_data[h, w]
        x = (w + start_w - c_x) / f_x * z
        y = (h + start_h - c_y) / f_y * z
        v = sv_img_1[h, w, 1]
        point_clouds.append((x, y, z, v, v, v))
                
vertex = np.array(point_clouds, dtype = [('x', 'f4'), ('y', 'f4'),('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
el = PlyElement.describe(vertex, 'vertex')
PlyData([el]).write(prefix_seq + 'sv_point_cloud_' + ('%08d')%(i) + '.ply')