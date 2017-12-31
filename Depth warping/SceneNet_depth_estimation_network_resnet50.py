from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Flatten, concatenate, Dropout, Lambda, add
from keras.models import Model, load_model
from keras.layers.normalization import BatchNormalization
from keras.utils.generic_utils import get_custom_objects
from keras import optimizers
from keras import backend as K
from keras.layers import add, merge, multiply
from keras.layers import RepeatVector, Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import numpy as np
import cv2
import math
import sys

sys.path.insert(0,"/home/xingtong/PycharmProjects/SceneNet/Scripts/Depth Warping Layer")
from DepthWarpingLayer import *

sys.path.insert(0,"/home/xingtong/PycharmProjects/SceneNet/Scripts/Depth Warping Layer")
from DepthWarpingLayerWithSpecularityWarping import *

sys.path.insert(0,"/home/xingtong/PycharmProjects/SceneNet/Scripts/Argument Masking Layer")
from ArgumentMaskingLayer import *

sys.path.insert(0,"/home/xingtong/PycharmProjects/SceneNet/Scripts/Specularity Masking Layer")
from SpecularityMaskingLayer import *

sys.path.insert(0,"/home/xingtong/PycharmProjects/SceneNet/Scripts/Union Masking Layer")
from UnionMaskingLayer import *

sys.path.insert(0,"/home/xingtong/PycharmProjects/SceneNet/Scripts/Callbacks")
from Callback_alternative_updating import *

def visualize_depth_map(depth_map_test, title, border_width):
    depth_map_test = np.abs(depth_map_test)
    depth_map_test = depth_map_test[border_width:-border_width, border_width:-border_width]
    min_value, max_value, min_loc, max_loc = cv2.minMaxLoc(depth_map_test)
    depth_map_visualize = (depth_map_test - min_value) / (max_value - min_value) * 255
    depth_map_visualize = np.asarray(depth_map_visualize, dtype='uint8')
    depth_map_visualize_heatmap = cv2.applyColorMap(depth_map_visualize, cv2.COLORMAP_JET)
    cv2.imshow(title, depth_map_visualize_heatmap)
    cv2.waitKey(100)

def camera_intrinsic_transform(vfov=45,hfov=60,pixel_width=320,pixel_height=240):
    camera_intrinsics = np.zeros((3,3))
    camera_intrinsics[2,2] = 1
    camera_intrinsics[0,0] = (pixel_width/2.0)/math.tan(math.radians(hfov/2.0))
    camera_intrinsics[0,2] = pixel_width/2.0
    camera_intrinsics[1,1] = (pixel_height/2.0)/math.tan(math.radians(vfov/2.0))
    camera_intrinsics[1,2] = pixel_height/2.0
    return camera_intrinsics

def mean_squared_difference(x, weight):
    x1, x2 = x
    valid_sum = 0.5 * (K.greater(x1, 0.01).sum() + K.greater(x2, 0.01).sum() + 2.0)
    mse = weight * K.sqrt(K.sum(K.square(x1 - x2)) / valid_sum)
    return mse

def mean_squared_difference_output_shape(input_shapes):
    input_shape1, input_shape2 = input_shapes
    return (input_shape1[0], 1)

# Masking out invalid region for warped depth map
def mask_invalid_element(x):
    x1, x2 = x
    return K.switch(x1 < 0.001, 0, x2)

def mask_invalid_element_output_shape(input_shapes):
    shape1, shape2 = input_shapes
    return shape1

def customized_loss(y_true, y_pred):
    return y_pred - y_true

get_custom_objects().update({"customized_loss":customized_loss})

prefix = "/home/xingtong/PycharmProjects/SceneNet/data/extracted_training_data/Ten_trajectories/"

# Read training data
training_color_img_1 = np.load(prefix + "color_img_1.npy")
resized_training_color_img_1 = []
for img in training_color_img_1:
    resized_training_color_img_1.append(cv2.resize(img, (320, 256)))
resized_training_color_img_1 = np.array(resized_training_color_img_1, dtype='float32')
resized_training_color_img_1 = resized_training_color_img_1 / 255.0

training_color_img_2 = np.load(prefix + "color_img_2.npy")
resized_training_color_img_2 = []
for img in training_color_img_2:
    resized_training_color_img_2.append(cv2.resize(img, (320, 256)))
resized_training_color_img_2 = np.array(resized_training_color_img_2, dtype='float32')
resized_training_color_img_2 = resized_training_color_img_2 / 255.0


[height, width, channel] = training_color_img_1[0].shape

R = np.load(prefix + "R.npy")
R_I = np.load(prefix + "R_I.npy")
P = np.load(prefix + "P.npy")
P_I = np.load(prefix + "P_I.npy")
P = np.array(P, dtype='float32')
P = np.reshape(P, (-1, 3, 1))
P_I = np.array(P_I, dtype='float32')
P_I = np.reshape(P_I, (-1, 3, 1))
R = np.array(R, dtype='float32')
R_I = np.array(R_I, dtype='float32')

intrinsic_matrix = camera_intrinsic_transform(pixel_height = 256, pixel_width = 320)
allzeros_groundtruth_output = np.zeros((R.shape[0], 1))

# Net architecture
input_img_1 = Input(shape=(256, 320, 3))
input_img_2 = Input(shape=(256, 320, 3))

input_translation_vector = Input(shape=(3, 1))
input_rotation_matrix = Input(shape=(3, 3))

input_translation_vector_inverse = Input(shape=(3, 1))
input_rotation_matrix_inverse = Input(shape=(3, 3))

base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(256, 320, 3), pooling=None)
base_model.layers.pop()
# base_model.layers[-1].outbound_nodes = []
# base_model.outputs = [base_model.layers[-1].output]
# base_model.summary()

resnet_model = Model(inputs = [base_model.input], outputs = [base_model.layers[-1].output])
for layer in resnet_model.layers:
    layer.trainable=False

resnet_model.summary()

convt1 = Conv2DTranspose(1024, (1, 1), padding='same', activation='relu')
batch_norm1 = BatchNormalization()
upsampl1 = UpSampling2D((2, 2))
convt2 = Conv2DTranspose(512, (3, 3), padding='same', activation='relu')
upsampl2 = UpSampling2D((2, 2))
convt3 = Conv2DTranspose(256, (3, 3), padding='same', activation='relu')
upsampl3 = UpSampling2D((2, 2))
convt4 = Conv2DTranspose(128, (3, 3), padding='same', activation='relu')
upsampl4 = UpSampling2D((2, 2))
convt5 = Conv2DTranspose(64, (3, 3), padding='same', activation='relu')
upsampl5 = UpSampling2D((2, 2))
convt6 = Conv2DTranspose(1, (3, 3), padding='same', activation='relu')


x = resnet_model(input_img_1)
x = convt1(x)
x = batch_norm1(x)
x = upsampl1(x)
x = convt2(x)
x = upsampl2(x)
x = convt3(x)
x = upsampl3(x)
x = convt4(x)
x = upsampl4(x)
x = convt5(x)
x = upsampl5(x)
depth_output_1 = convt6(x)

x = resnet_model(input_img_2)
x = convt1(x)
x = batch_norm1(x)
x = upsampl1(x)
x = convt2(x)
x = upsampl2(x)
x = convt3(x)
x = upsampl3(x)
x = convt4(x)
x = upsampl4(x)
x = convt5(x)
x = upsampl5(x)
depth_output_2 = convt6(x)

synthetic_depth_map_1 = DepthWarpingLayer(intrinsic_matrix)([depth_output_1, depth_output_2, input_translation_vector, input_rotation_matrix])
synthetic_depth_map_2 = DepthWarpingLayer(intrinsic_matrix)([depth_output_2, depth_output_1, input_translation_vector_inverse, input_rotation_matrix_inverse])

masked_estimated_depth_map_1 = Lambda(mask_invalid_element, output_shape=mask_invalid_element_output_shape)([synthetic_depth_map_1, depth_output_1])
masked_estimated_depth_map_2 = Lambda(mask_invalid_element, output_shape=mask_invalid_element_output_shape)([synthetic_depth_map_2, depth_output_2])

depth_map_mean_squared_difference_1 = Lambda(mean_squared_difference, output_shape=mean_squared_difference_output_shape, arguments={'weight': 0.5})([synthetic_depth_map_1, masked_estimated_depth_map_1])
depth_map_mean_squared_difference_2 = Lambda(mean_squared_difference, output_shape=mean_squared_difference_output_shape, arguments={'weight': 0.5})([synthetic_depth_map_2, masked_estimated_depth_map_2])

mse_loss = add([depth_map_mean_squared_difference_1, depth_map_mean_squared_difference_2])

model = Model([input_img_1, input_img_2, input_translation_vector, input_translation_vector_inverse, input_rotation_matrix, input_rotation_matrix_inverse], mse_loss)

model.summary()
sgd = optimizers.SGD(lr=1.0e-8, momentum=0.9, nesterov=True)
model.compile(loss=customized_loss, optimizer=sgd)
#
filepath = "/home/xingtong/PycharmProjects/SceneNet/data/trained_network/Ten_trajectories/resnet50/warped_depth_estimation_network_weights-improvement-{epoch:02d}-{val_loss:.5f}.hdf5"
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.00001, cooldown=0, min_lr=0)
checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, min_delta=0.00001, mode='auto')
##
history = model.fit([resized_training_color_img_1, resized_training_color_img_2, P, P_I, R, R_I], allzeros_groundtruth_output, batch_size=20,
                    epochs=500, verbose=1, callbacks=[earlyStopping, checkpointer, reducelr], validation_split=0.05, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)
