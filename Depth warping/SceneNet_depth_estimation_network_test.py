import cv2
import math
import sys
import numpy as np
import yaml
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Flatten, concatenate, Dropout, Lambda, add
from keras.layers.normalization import BatchNormalization
from keras.layers import add, merge, multiply, Concatenate, Activation
from keras.layers import RepeatVector, Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model, load_model
from keras import optimizers
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.models import model_from_json
import theano



# sys.path.insert(0,"/home/xingtong/PycharmProjects/SceneNet/Scripts/Depth Warping Layer")
# from DepthWarpingLayer import *
#
# sys.path.insert(0,"/home/xingtong/PycharmProjects/SceneNet/Scripts/Depth Warping Layer")
# from DepthWarpingMapCalculateLayer import *
#
# sys.path.insert(0,"/home/xingtong/PycharmProjects/SceneNet/Scripts/Argument Masking Layer")
# from ArgumentMaskingLayer import *
#
# sys.path.insert(0,"/home/xingtong/PycharmProjects/SceneNet/Scripts/Specularity Masking Layer")
# from SpecularityMaskingLayer import *
#
# sys.path.insert(0,"/home/xingtong/PycharmProjects/SceneNet/Scripts/Union Masking Layer")
# from UnionMaskingLayer import *
#
# sys.path.insert(0,"/home/xingtong/PycharmProjects/SceneNet/Scripts/Callbacks")
# from Callback_alternative_updating import *
#
# def visualize_depth_map(depth_map_test, title, border_width, min_value, max_value):
#     depth_map_test = np.abs(depth_map_test)
#     depth_map_test = depth_map_test[border_width:-border_width, border_width:-border_width]
#     depth_map_visualize = (depth_map_test - min_value) / (max_value - min_value) * 255
#
#     depth_map_visualize[depth_map_visualize < 0] = 0
#     depth_map_visualize[depth_map_visualize > 255] = 255
#
#     depth_map_visualize = np.asarray(depth_map_visualize, dtype='uint8')
#     depth_map_visualize_heatmap = cv2.applyColorMap(depth_map_visualize, cv2.COLORMAP_JET)
#
#     cv2.imshow(title, depth_map_visualize_heatmap)
#     cv2.waitKey(100)
#
# def depth_unlog(x, scale_factor):
#     return K.exp(x * scale_factor) - 5.0
# def depth_unlog_output_shape(input_shape):
#     return input_shape
#
# def depth_log(x, scale_factor):
#     return K.log(x + 5.0) / scale_factor
# def depth_log_output_shape(input_shape):
#     return input_shape
#
# def camera_intrinsic_transform(vfov=45,hfov=60,pixel_width=320,pixel_height=240):
#     camera_intrinsics = np.zeros((3,3))
#     camera_intrinsics[2,2] = 1
#     camera_intrinsics[0,0] = (pixel_width/2.0)/math.tan(math.radians(hfov/2.0))
#     camera_intrinsics[0,2] = pixel_width/2.0
#     camera_intrinsics[1,1] = (pixel_height/2.0)/math.tan(math.radians(vfov/2.0))
#     camera_intrinsics[1,2] = pixel_height/2.0
#     return camera_intrinsics
#
# def mean_squared_difference(x, weight):
#     x1, x2 = x
#     valid_sum = 0.5 * (K.greater(x1, 0.01).sum() + K.greater(x2, 0.01).sum())
#     #    mean_squared = weight * K.log(K.sum(K.square(x1 - x2)) / valid_sum + 1.0)
#     mse = weight * K.sqrt(K.sum(K.square(x1 - x2)) / valid_sum)
#     return mse
#
# def mean_squared_difference_output_shape(input_shapes):
#     input_shape1, input_shape2 = input_shapes
#     return (input_shape1[0], 1)
#
# def huber_error(x, weight):
#     x1, x2 = x
#     valid_sum = 0.5 * (K.greater(x1, 0.0001).sum() + K.greater(x2, 0.0001).sum())
#     abs_error = K.abs(x1 - x2)
#     c = 0.2 * K.max(abs_error)
#     squared_error = (K.square(x1 - x2) + c * c) / (2 * c)
#     error = K.switch(abs_error <= c, abs_error, squared_error)
#     mean_error = weight * K.sum(error) / (valid_sum + 1.0)
#     return mean_error
#
#
# def huber_error_output_shape(input_shapes):
#     input_shape1, input_shape2 = input_shapes
#     return (input_shape1[0], 1)
#
# ## Masking out invalid region for warped depth map
# def mask_invalid_element(x):
#     x1, x2 = x
#     return K.switch(x1 < 0.001, 0, x2)
#
# def mask_invalid_element_output_shape(input_shapes):
#     shape1, shape2 = input_shapes
#     return shape1
#
# def clip_close_to_zero(x):
#     return K.switch(x < 0.001, 0, x)
# def clip_close_to_zero_output_shape(input_shape):
#     return input_shape
#
# # def sparse_mean_squared_difference(x, weight):
# #     x1, x2 = x
# #     valid_sum = K.greater(x1, 0.01).sum()
# #     return weight * K.sum(K.exp(K.square((x1 - K.sum(x1) / valid_sum) / 0.2)) * K.square(x1 - x2)) / valid_sum
# #
# # def sparse_mean_squared_difference_output_shape(input_shapes):
# #     input_shape1, input_shape2 = input_shapes
# #     return (input_shape1[0], 1)
# #
# # def image_difference(x):
# #     x1, x2 = x
# #     return x1 - x2
# #
# # def image_difference_output_shape(input_shape):
# #     return input_shape[0]
# #
# def customized_loss(y_true, y_pred):
#     return y_pred - y_true
#
# get_custom_objects().update({"customized_loss":customized_loss})
#
# prefix = "/home/xingtong/PycharmProjects/SceneNet/data/extracted_training_data/Ten_trajectories/"
#
# training_color_img_1 = np.load(prefix + "color_img_1.npy")
# [height, width, channel] = training_color_img_1[0].shape
# training_color_img_1 = np.array(training_color_img_1, dtype='float32')
# training_color_img_1 = training_color_img_1 / 255.0
#
# training_color_img_2 = np.load(prefix + "color_img_2.npy")
# training_color_img_2 = np.array(training_color_img_2, dtype='float32')
# training_color_img_2 = training_color_img_2 / 255.0
#
# R = np.load(prefix + "R.npy")
# R_I = np.load(prefix + "R_I.npy")
# P = np.load(prefix + "P.npy")
# P_I = np.load(prefix + "P_I.npy")
#
# training_depth_img_1 = np.load(prefix + "depth_img_1.npy")
# training_depth_img_1 = np.array(training_depth_img_1, dtype='float32')
# training_depth_img_2 = np.load(prefix + "depth_img_2.npy")
# training_depth_img_2 = np.array(training_depth_img_2, dtype='float32')
#
#
# indexes = np.arange(len(R))
# shuffled_indexes = np.random.shuffle(indexes)
# training_color_img_1 = training_color_img_1[shuffled_indexes]
# training_color_img_2 = training_color_img_2[shuffled_indexes]
# R = R[shuffled_indexes]
# R_I = R_I[shuffled_indexes]
# P = P[shuffled_indexes]
# P_I = P_I[shuffled_indexes]
# training_depth_img_1 = training_depth_img_1[shuffled_indexes]
# training_depth_img_2 = training_depth_img_2[shuffled_indexes]
#
# training_color_img_1 = np.reshape(training_color_img_1, (-1, height, width, channel))
# training_color_img_2 = np.reshape(training_color_img_2, (-1, height, width, channel))
# training_depth_img_1 = np.reshape(training_depth_img_1, (-1, height, width, 1))
# training_depth_img_2 = np.reshape(training_depth_img_2, (-1, height, width, 1))
#
# P = np.array(P, dtype='float32')
# P = np.reshape(P, (-1, 3, 1))
# P_I = np.array(P_I, dtype='float32')
# P_I = np.reshape(P_I, (-1, 3, 1))
# R = np.array(R, dtype='float32')
# R = np.reshape(R, (-1, 3, 3))
# R_I = np.array(R_I, dtype='float32')
# R_I = np.reshape(R_I, (-1, 3, 3))
#
# intrinsic_matrix = camera_intrinsic_transform()
#
# allzeros_groundtruth_output = np.zeros((R.shape[0], 1))
#
# input_img_1 = Input(shape=(height, width, channel))
# input_img_2 = Input(shape=(height, width, channel))
#
# input_translation_vector = Input(shape=(3, 1))
# input_rotation_matrix = Input(shape=(3, 3))
#
# input_translation_vector_inverse = Input(shape=(3, 1))
# input_rotation_matrix_inverse = Input(shape=(3, 3))
#
# input_depth_img_1 = Input(shape=(height, width, 1))
# input_depth_img_2 = Input(shape=(height, width, 1))
#
# # conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')
# # maxpool1 = MaxPooling2D((2, 2))
# # conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')
# # maxpool2 = MaxPooling2D((2, 2))
# # conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')
# # maxpool3 = MaxPooling2D((4, 4))
# # conv4 = Conv2D(512, (3, 3), padding='same', activation='relu')
# # convt1 = Conv2DTranspose(128, (3, 3), padding='same', activation='relu')
# # upsampl1 = UpSampling2D((4, 4))
# # convt2 = Conv2DTranspose(64, (3, 3), padding='same', activation='relu')
# # upsampl2 = UpSampling2D((2, 2))
# # convt3 = Conv2DTranspose(32, (3, 3), padding='same', activation='relu')
# # upsampl3 = UpSampling2D((2, 2))
# # convt4 = Conv2DTranspose(1, (3, 3), padding='same', activation='relu')
# #
# # x = conv1(input_img_1)
# # x = maxpool1(x)
# # x = conv2(x)
# # x = maxpool2(x)
# # x = conv3(x)
# # x = convt2(x)
# # x = upsampl2(x)
# # x = convt3(x)
# # x = upsampl3(x)
# # estimated_depth_map_1 = convt4(x)
# #
# # x = conv1(input_img_2)
# # x = maxpool1(x)
# # x = conv2(x)
# # x = maxpool2(x)
# # x = conv3(x)
# # x = convt2(x)
# # x = upsampl2(x)
# # x = convt3(x)
# # x = upsampl3(x)
# # estimated_depth_map_2 = convt4(x)
# #
# # synthetic_depth_map_1 = DepthWarpingLayer(intrinsic_matrix)([estimated_depth_map_1, estimated_depth_map_2, input_translation_vector, input_rotation_matrix])
# # synthetic_depth_map_2 = DepthWarpingLayer(intrinsic_matrix)([estimated_depth_map_2, estimated_depth_map_1, input_translation_vector_inverse, input_rotation_matrix_inverse])
# #
# # clipped_synthetic_depth_map_1 = Lambda(clip_close_to_zero, output_shape=clip_close_to_zero_output_shape)(synthetic_depth_map_1)
# # clipped_synthetic_depth_map_2 = Lambda(clip_close_to_zero, output_shape=clip_close_to_zero_output_shape)(synthetic_depth_map_2)
# #
# # masked_estimated_unlog_depth_map_1 = Lambda(mask_invalid_element, output_shape=mask_invalid_element_output_shape)([clipped_synthetic_depth_map_1, estimated_depth_map_1])
# # masked_estimated_unlog_depth_map_2 = Lambda(mask_invalid_element, output_shape=mask_invalid_element_output_shape)([clipped_synthetic_depth_map_2, estimated_depth_map_2])
# #
# # depth_map_mean_squared_difference_1 = Lambda(huber_error, output_shape=huber_error_output_shape, arguments={'weight': 0.5})([clipped_synthetic_depth_map_1, masked_estimated_unlog_depth_map_1])
# # depth_map_mean_squared_difference_2 = Lambda(huber_error, output_shape=huber_error_output_shape, arguments={'weight': 0.5})([clipped_synthetic_depth_map_2, masked_estimated_unlog_depth_map_2])
# #
# # model = Model([input_img_1, input_img_2, input_translation_vector, input_translation_vector_inverse, input_rotation_matrix, input_rotation_matrix_inverse],
# #               [estimated_depth_map_1, estimated_depth_map_2, clipped_synthetic_depth_map_1, clipped_synthetic_depth_map_2])
# #
# #
#
# synthetic_depth_map_1 = DepthWarpingLayer(intrinsic_matrix)([input_depth_img_1, input_depth_img_2, input_translation_vector, input_rotation_matrix])
# synthetic_depth_map_2 = DepthWarpingLayer(intrinsic_matrix)([input_depth_img_2, input_depth_img_1, input_translation_vector_inverse, input_rotation_matrix_inverse])
#
#
# depth_map_inplace_2 = DepthWarpingMapCalculateLayer(intrinsic_matrix)([input_depth_img_1, input_depth_img_2, input_translation_vector, input_rotation_matrix])
# depth_map_inplace_1 = DepthWarpingMapCalculateLayer(intrinsic_matrix)([input_depth_img_2, input_depth_img_1, input_translation_vector, input_rotation_matrix])
#
#
# masked_estimated_unlog_depth_map_1 = Lambda(mask_invalid_element, output_shape=mask_invalid_element_output_shape)([synthetic_depth_map_1, input_depth_img_1])
# masked_estimated_unlog_depth_map_2 = Lambda(mask_invalid_element, output_shape=mask_invalid_element_output_shape)([synthetic_depth_map_2, input_depth_img_2])
#
# model = Model(inputs = [input_depth_img_1, input_depth_img_2, input_translation_vector, input_translation_vector_inverse, input_rotation_matrix, input_rotation_matrix_inverse],
#               outputs = [synthetic_depth_map_1, synthetic_depth_map_2, masked_estimated_unlog_depth_map_1, masked_estimated_unlog_depth_map_2, depth_map_inplace_1, depth_map_inplace_2])
#
#
# # model.load_weights("/home/xingtong/PycharmProjects/SceneNet/data/trained_network/Ten_trajectories/warped_depth_estimation_network_weights-improvement-00-1.59549.hdf5")
#
# model.summary()
# sgd = optimizers.SGD(lr=1.0e-5, momentum=0.9, nesterov=True)
# model.compile(loss=customized_loss, optimizer=sgd)
#
#
# trajectory_index = 1
# count_per_traj = 40
# results = model.predict([training_depth_img_1[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj],
#                          training_depth_img_2[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj],
#                          P[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj],
#                          P_I[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj],
#                         R[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj],
#                          R_I[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj]], batch_size = 100)
#
# cv2.destroyAllWindows()
# for i in range(0, count_per_traj):
#     print(i)
#
#     border_width = 1
#     depth_img_1 = training_depth_img_1[i + trajectory_index * count_per_traj]
#     depth_img_2 = training_depth_img_2[i + trajectory_index * count_per_traj]
#     color_img_1 = training_color_img_1[i + trajectory_index * count_per_traj]
#     color_img_2 = training_color_img_2[i + trajectory_index * count_per_traj]
#     cv2.imshow("rgb 1", color_img_1[border_width:-border_width, border_width:-border_width])
#     cv2.imshow("rgb 2", color_img_2[border_width:-border_width, border_width:-border_width])
#
#
#     ## I need to deal with the serious occlusion here to make this depth warping stuff really work
#     depth_map = results[0][i]
#     mean_value = np.mean(depth_map)
#     min_value = np.min(depth_map)
#     max_value = 2*(mean_value - min_value) + mean_value
#     #
#     # visualize_depth_map(results[0][i], "depth map 1", border_width, min_value, max_value)
#     # visualize_depth_map(results[1][i], "depth map 2", border_width, min_value, max_value)
#     # visualize_depth_map(results[2][i], "synthetic depth map 1", border_width, min_value, max_value)
#     # visualize_depth_map(results[3][i], "synthetic depth map 2", border_width, min_value, max_value)
#
#     visualize_depth_map(results[0][i], "synthetic depth map 1", border_width, min_value, max_value)
#     visualize_depth_map(results[1][i], "synthetic depth map 2", border_width, min_value, max_value)
#     visualize_depth_map(results[2][i], "original depth map 1", border_width, min_value, max_value)
#     visualize_depth_map(results[3][i], "original depth map 2", border_width, min_value, max_value)
#     visualize_depth_map(results[4][i], "in-place depth map 1", border_width, min_value, max_value)
#     visualize_depth_map(results[5][i], "in-place depth map 2", border_width, min_value, max_value)
#     # visualize_depth_map(depth_img_1, "original depth map 1", border_width, min_value, max_value)
#     # visualize_depth_map(depth_img_2, "original depth map 2", border_width, min_value, max_value)
#
#     visualize_depth_map(np.abs(results[0][i] - results[2][i]), "difference depth map 1", border_width, min_value, max_value)
#     visualize_depth_map(np.abs(results[1][i] - results[3][i]), "difference depth map 2", border_width, min_value, max_value)
#
#     print(np.max(np.abs(results[0][i] - results[2][i])))
#     print(np.max(np.abs(results[1][i] - results[3][i])))
#     # visualize_depth_map(results[0][i], "synthetic depth map 1")
#     # visualize_depth_map(results[1][i], "synthetic depth map 2")
#     # visualize_depth_map(depth_img_1[i], "original depth map 1")
#     # visualize_depth_map(depth_img_2[i], "original depth map 2")
#     # visualize_depth_map(results[0][i] - depth_img_1[i], "difference 1")
#     # visualize_depth_map(results[1][i] - depth_img_2[i], "difference 2")
#
#     cv2.waitKey()
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

def visualize_depth_map(depth_map_test, title, border_width, min_value, max_value):
    depth_map_test = np.abs(depth_map_test)
    depth_map_test = depth_map_test[border_width:-border_width, border_width:-border_width]
    depth_map_visualize = (depth_map_test - min_value) / (max_value - min_value) * 255

    depth_map_visualize[depth_map_visualize < 0] = 0
    depth_map_visualize[depth_map_visualize > 255] = 255

    depth_map_visualize = np.asarray(depth_map_visualize, dtype='uint8')
    depth_map_visualize_heatmap = cv2.applyColorMap(depth_map_visualize, cv2.COLORMAP_JET)

    cv2.imshow(title, depth_map_visualize_heatmap)
    cv2.waitKey(100)

def depth_unlog(x, scale_factor):
    return K.exp(x * scale_factor) - 5.0
def depth_unlog_output_shape(input_shape):
    return input_shape

def depth_log(x, scale_factor):
    return K.log(x + 5.0) / scale_factor
def depth_log_output_shape(input_shape):
    return input_shape

def camera_intrinsic_transform(vfov=45,hfov=60,pixel_width=320,pixel_height=240):
    camera_intrinsics = np.zeros((3,3))
    camera_intrinsics[2,2] = 1
    camera_intrinsics[0,0] = (pixel_width/2.0)/math.tan(math.radians(hfov/2.0))
    camera_intrinsics[0,2] = pixel_width/2.0
    camera_intrinsics[1,1] = (pixel_height/2.0)/math.tan(math.radians(vfov/2.0))
    camera_intrinsics[1,2] = pixel_height/2.0
    return camera_intrinsics

# def mean_squared_difference(x, weight):
#     x1, x2 = x
#     valid_sum = 0.5 * (K.greater(x1, 0.01).sum() + K.greater(x2, 0.01).sum())
#     #    mean_squared = weight * K.log(K.sum(K.square(x1 - x2)) / valid_sum + 1.0)
#     mse = weight * K.sqrt(K.sum(K.square(x1 - x2)) / valid_sum)
#     return mse
#
# def mean_squared_difference_output_shape(input_shapes):
#     input_shape1, input_shape2 = input_shapes
#     return (input_shape1[0], 1)

def huber_error(x, weight):
    x1, x2, x3 = x
    valid_sum = 0.5 * (K.greater(K.abs(x1), 0.0001).sum() + K.greater(K.abs(x2), 0.0001).sum())
    abs_error = x3 * K.abs(x1 - x2)
    c = 0.2 * K.max(abs_error)
    squared_error = x3 * (K.square(x1 - x2) + c * c) / (2 * c)
    error = K.switch(abs_error <= c, abs_error, squared_error)
    mean_error = weight * K.sum(error) / (valid_sum + 1.0)
    return mean_error
def huber_error_output_shape(input_shapes):
    input_shape1, input_shape2, input_shape3 = input_shapes
    return (input_shape1[0], 1)


# def huber_error(x, weight):
#     x1, x2 = x
#     valid_sum = 0.5 * (K.greater(K.abs(x1), 0.0001).sum() + K.greater(K.abs(x2), 0.0001).sum())
#     abs_error = K.abs(x1 - x2)
#     c = 0.2 * K.max(abs_error)
#     squared_error = (K.square(x1 - x2) + c * c) / (2 * c)
#     error = K.switch(abs_error <= c, abs_error, squared_error)
#     mean_error = weight * K.sum(error) / (valid_sum + 1.0)
#     return mean_error
# def huber_error_output_shape(input_shapes):
#     input_shape1, input_shape2 = input_shapes
#     return (input_shape1[0], 1)

## make as few masking as possible
def explainability_mask_loss(x, weight):
    return weight * K.sum(-K.log(x)) / K.greater_equal(x, 0.0).sum()
def explainability_mask_loss_output_shape(input_shapes):
    input_shape1 = input_shapes
    return (input_shape1[0], 1)

## Masking out invalid region for warped depth map
def mask_invalid_element(x):
    x1, x2 = x
    return K.switch(K.abs(x1) < 0.001, 0, x2)
def mask_invalid_element_output_shape(input_shapes):
    shape1, shape2 = input_shapes
    return shape1

# def clip_close_to_zero(x):
#     return K.switch(x < 0.001, 0, x)
# def clip_close_to_zero_output_shape(input_shape):
#     return input_shape

# def custom_activation(x):
#     return 1 / (K.sigmoid(x))
#
# get_custom_objects().update({'custom_activation': Activation(custom_activation)})


# def sparse_mean_squared_difference(x, weight):
#     x1, x2 = x
#     valid_sum = K.greater(x1, 0.01).sum()
#     return weight * K.sum(K.exp(K.square((x1 - K.sum(x1) / valid_sum) / 0.2)) * K.square(x1 - x2)) / valid_sum
# def sparse_mean_squared_difference_output_shape(input_shapes):
#     input_shape1, input_shape2 = input_shapes
#     return (input_shape1[0], 1)

def customized_loss(y_true, y_pred):
    return y_pred - y_true

# theano.config.exception_verbosity = 'high'
# theano.config.optimizer = "None"

get_custom_objects().update({"customized_loss":customized_loss})

prefix = "/home/xingtong/PycharmProjects/SceneNet/data/extracted_training_data/Ten_trajectories/"


training_color_img_1 = np.load(prefix + "color_img_1.npy")
[height, width, channel] = training_color_img_1[0].shape
training_color_img_1 = np.array(training_color_img_1, dtype='float32')
training_color_img_1 = training_color_img_1 / 255.0

training_color_img_2 = np.load(prefix + "color_img_2.npy")
training_color_img_2 = np.array(training_color_img_2, dtype='float32')
training_color_img_2 = training_color_img_2 / 255.0

R = np.load(prefix + "R.npy")
R_I = np.load(prefix + "R_I.npy")
P = np.load(prefix + "P.npy")
P_I = np.load(prefix + "P_I.npy")
#
# depth_img_1 = np.load(prefix + "depth_img_1.npy")
# depth_img_1 = np.array(depth_img_1, dtype='float32')
# depth_img_2 = np.load(prefix + "depth_img_2.npy")
# depth_img_2 = np.array(depth_img_2, dtype='float32')

indexes = np.arange(len(R))
shuffled_indexes = np.random.shuffle(indexes)
training_color_img_1 = training_color_img_1[shuffled_indexes]
training_color_img_2 = training_color_img_2[shuffled_indexes]
R = R[shuffled_indexes]
R_I = R_I[shuffled_indexes]
P = P[shuffled_indexes]
P_I = P_I[shuffled_indexes]
# depth_img_1 = depth_img_1[shuffled_indexes]
# depth_img_2 = depth_img_2[shuffled_indexes]

training_color_img_1 = np.reshape(training_color_img_1, (-1, height, width, channel))
training_color_img_2 = np.reshape(training_color_img_2, (-1, height, width, channel))
# depth_img_1 = np.reshape(depth_img_1, (-1, height, width, 1))
# depth_img_2 = np.reshape(depth_img_2, (-1, height, width, 1))

P = np.array(P, dtype='float32')
P = np.reshape(P, (-1, 3, 1))
P_I = np.array(P_I, dtype='float32')
P_I = np.reshape(P_I, (-1, 3, 1))
R = np.array(R, dtype='float32')
R = np.reshape(R, (-1, 3, 3))
R_I = np.array(R_I, dtype='float32')
R_I = np.reshape(R_I, (-1, 3, 3))

intrinsic_matrix = camera_intrinsic_transform()
print(intrinsic_matrix.shape)

allzeros_groundtruth_output = np.zeros((R.shape[0], 1))

input_img_1 = Input(shape=(height, width, channel))
input_img_2 = Input(shape=(height, width, channel))

input_translation_vector = Input(shape=(3, 1))
input_rotation_matrix = Input(shape=(3, 3))

input_translation_vector_inverse = Input(shape=(3, 1))
input_rotation_matrix_inverse = Input(shape=(3, 3))

input_depth_img_1 = Input(shape=(height, width, 1))
input_depth_img_2 = Input(shape=(height, width, 1))

conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')
batch_norm1 = BatchNormalization()
maxpool1 = MaxPooling2D((2, 2))
conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')
batch_norm2 = BatchNormalization()
maxpool2 = MaxPooling2D((2, 2))
conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')
batch_norm3 = BatchNormalization()
maxpool3 = MaxPooling2D((4, 4))
conv4 = Conv2D(512, (3, 3), padding='same', activation='relu')
batch_norm4 = BatchNormalization()
convt1 = Conv2DTranspose(128, (3, 3), padding='same', activation='relu')
batch_norm5 = BatchNormalization()
upsampl1 = UpSampling2D((4, 4))
convt2 = Conv2DTranspose(64, (3, 3), padding='same', activation='relu')
batch_norm6 = BatchNormalization()
upsampl2 = UpSampling2D((2, 2))
convt3 = Conv2DTranspose(32, (3, 3), padding='same', activation='relu')
batch_norm7 = BatchNormalization()
upsampl3 = UpSampling2D((2, 2))
convt4 = Conv2DTranspose(1, (3, 3), padding='same', activation='relu')

concat1 = Concatenate(axis=-1)
conv1_explainability = Conv2D(32, (3, 3), padding='same', activation='relu')
batch_norm_explainability1 = BatchNormalization()
maxpool1_explainability = MaxPooling2D((2, 2))
conv2_explainability = Conv2D(64, (3, 3), padding='same', activation='relu')
batch_norm_explainability2 = BatchNormalization()
maxpool2_explainability = MaxPooling2D((2, 2))
conv3_explainability = Conv2D(128, (3, 3), padding='same', activation='relu')
batch_norm_explainability3 = BatchNormalization()
maxpool3_explainability = MaxPooling2D((4, 4))
conv4_explainability = Conv2D(512, (3, 3), padding='same', activation='relu')
batch_norm_explainability4 = BatchNormalization()
convt1_explainability = Conv2DTranspose(128, (3, 3), padding='same', activation='relu')
batch_norm_explainability5 = BatchNormalization()
upsampl1_explainability = UpSampling2D((4, 4))
convt2_explainability = Conv2DTranspose(64, (3, 3), padding='same', activation='relu')
batch_norm_explainability6 = BatchNormalization()
upsampl2_explainability = UpSampling2D((2, 2))
convt3_explainability = Conv2DTranspose(32, (3, 3), padding='same', activation='relu')
batch_norm_explainability7 = BatchNormalization()
upsampl3_explainability = UpSampling2D((2, 2))
convt4_explainability = Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid')

x = conv1(input_img_1)
x = batch_norm1(x)
x = maxpool1(x)
x = conv2(x)
x = batch_norm2(x)
x = maxpool2(x)
x = conv3(x)
# x = batch_norm3(x)
# x = maxpool3(x)
# x = conv4(x)
# x = batch_norm4(x)
# x = convt1(x)
# x = batch_norm5(x)
# x = upsampl1(x)
x = convt2(x)
x = batch_norm6(x)
x = upsampl2(x)
x = convt3(x)
x = batch_norm7(x)
x = upsampl3(x)
estimated_depth_map_1 = convt4(x)

x = conv1(input_img_2)
x = batch_norm1(x)
x = maxpool1(x)
x = conv2(x)
x = batch_norm2(x)
x = maxpool2(x)
x = conv3(x)
# x = batch_norm3(x)
# x = maxpool3(x)
# x = conv4(x)
# x = batch_norm4(x)
# x = convt1(x)
# x = batch_norm5(x)
# x = upsampl1(x)
x = convt2(x)
x = batch_norm6(x)
x = upsampl2(x)
x = convt3(x)
x = batch_norm7(x)
x = upsampl3(x)
estimated_depth_map_2 = convt4(x)

concat_input = concat1([input_img_1, input_img_2])
x = conv1_explainability(concat_input)
x = batch_norm_explainability1(x)
x = maxpool1_explainability(x)
x = conv2_explainability(x)
x = batch_norm_explainability2(x)
x = maxpool2_explainability(x)
x = conv3_explainability(x)
# x = batch_norm_explainability3(x)
# x = maxpool3_explainability(x)
# x = conv4_explainability(x)
# x = batch_norm_explainability4(x)
# x = convt1_explainability(x)
# x = batch_norm_explainability5(x)
# x = upsampl1_explainability(x)
x = convt2_explainability(x)
x = batch_norm_explainability6(x)
x = upsampl2_explainability(x)
x = convt3_explainability(x)
x = batch_norm_explainability7(x)
x = upsampl3_explainability(x)
explainability_mask = convt4_explainability(x)

synthetic_depth_map_1 = DepthWarpingLayer(intrinsic_matrix)([estimated_depth_map_1, estimated_depth_map_2, input_translation_vector, input_rotation_matrix])
synthetic_depth_map_2 = DepthWarpingLayer(intrinsic_matrix)([estimated_depth_map_2, estimated_depth_map_1, input_translation_vector_inverse, input_rotation_matrix_inverse])

masked_estimated_unlog_depth_map_1 = Lambda(mask_invalid_element, output_shape=mask_invalid_element_output_shape)([synthetic_depth_map_1, estimated_depth_map_1])
masked_estimated_unlog_depth_map_2 = Lambda(mask_invalid_element, output_shape=mask_invalid_element_output_shape)([synthetic_depth_map_2, estimated_depth_map_2])

depth_map_mean_squared_difference_1 = Lambda(huber_error, output_shape=huber_error_output_shape, arguments={'weight': 0.5})([synthetic_depth_map_1, masked_estimated_unlog_depth_map_1, explainability_mask]) #, explainability_mask
depth_map_mean_squared_difference_2 = Lambda(huber_error, output_shape=huber_error_output_shape, arguments={'weight': 0.5})([synthetic_depth_map_2, masked_estimated_unlog_depth_map_2, explainability_mask]) #, explainability_mask
explainability_masking_loss_ = Lambda(explainability_mask_loss, output_shape=explainability_mask_loss_output_shape, arguments={'weight': 10})(explainability_mask)

model = Model(inputs = [input_img_1, input_img_2, input_translation_vector, input_translation_vector_inverse, input_rotation_matrix, input_rotation_matrix_inverse],
              outputs = [synthetic_depth_map_1, synthetic_depth_map_2, estimated_depth_map_1, estimated_depth_map_2, explainability_mask, masked_estimated_unlog_depth_map_1, masked_estimated_unlog_depth_map_2])

model.load_weights("/home/xingtong/PycharmProjects/SceneNet/data/trained_network/Ten_trajectories/warped_depth_estimation_network_weights-improvement-11-7.69150.hdf5")

# Toiletry 284 training sample pairs succeed /home/xingtong/PycharmProjects/SceneNet/data/trained_network/warped_depth_estimation_network_weights-improvement-39-21.69228.hdf5
# Toiletry 2840 pairs succeed "/home/xingtong/PycharmProjects/SceneNet/data/trained_network/Ten_trajectories/warped_depth_estimation_network_weights-improvement-02-58.72300.hdf5"
# Toiletry 2840 pairs succeed "/home/xingtong/PycharmProjects/SceneNet/data/trained_network/Ten_trajectories/warped_depth_estimation_network_weights-improvement-11-7.69150.hdf5"

model.summary()
sgd = optimizers.SGD(lr=1.0e-5, momentum=0.9, nesterov=True)
model.compile(loss=customized_loss, optimizer=sgd)


trajectory_index = 5
count_per_traj = 40
results = model.predict([training_color_img_1[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj],
                         training_color_img_2[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj],
                         P[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj],
                         P_I[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj],
                        R[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj],
                         R_I[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj]], batch_size = 100)

cv2.destroyAllWindows()
for i in range(0, count_per_traj):
    print(i)

    border_width = 1
    color_img_1 = training_color_img_1[i + trajectory_index * count_per_traj]
    color_img_2 = training_color_img_2[i + trajectory_index * count_per_traj]
    cv2.imshow("rgb 1", color_img_1[border_width:-border_width, border_width:-border_width])
    cv2.imshow("rgb 2", color_img_2[border_width:-border_width, border_width:-border_width])


    ## I need to deal with the serious occlusion here to make this depth warping stuff really work
    depth_map = results[0][i]
    median_value = np.median(depth_map[depth_map > 5])
    min_value = np.min(depth_map[depth_map > 5])
    max_value = (median_value - min_value) + median_value

    print(min_value, max_value)
    #
    # visualize_depth_map(results[0][i], "depth map 1", border_width, min_value, max_value)
    # visualize_depth_map(results[1][i], "depth map 2", border_width, min_value, max_value)
    # visualize_depth_map(results[2][i], "synthetic depth map 1", border_width, min_value, max_value)
    # visualize_depth_map(results[3][i], "synthetic depth map 2", border_width, min_value, max_value)

    visualize_depth_map(results[0][i], "synthetic depth map 1", border_width, min_value, max_value)
    visualize_depth_map(results[1][i], "synthetic depth map 2", border_width, min_value, max_value)
    visualize_depth_map(results[2][i], "original depth map 1", border_width, min_value, max_value)
    visualize_depth_map(results[3][i], "original depth map 2", border_width, min_value, max_value)
    cv2.imshow("mask", results[4][i] * 255)

    # visualize_depth_map(results[4][i], "in-place depth map 1", border_width, min_value, max_value)
    # visualize_depth_map(results[5][i], "in-place depth map 2", border_width, min_value, max_value)
    # # visualize_depth_map(depth_img_1, "original depth map 1", border_width, min_value, max_value)
    # # visualize_depth_map(depth_img_2, "original depth map 2", border_width, min_value, max_value)
    #
    # visualize_depth_map(np.abs(results[0][i] - results[2][i]), "difference depth map 1", border_width, min_value, max_value)
    # visualize_depth_map(np.abs(results[1][i] - results[3][i]), "difference depth map 2", border_width, min_value, max_value)
    #
    # print(np.max(np.abs(results[0][i] - results[2][i])))
    # print(np.max(np.abs(results[1][i] - results[3][i])))
    # visualize_depth_map(results[0][i], "synthetic depth map 1")
    # visualize_depth_map(results[1][i], "synthetic depth map 2")
    # visualize_depth_map(depth_img_1[i], "original depth map 1")
    # visualize_depth_map(depth_img_2[i], "original depth map 2")
    # visualize_depth_map(results[0][i] - depth_img_1[i], "difference 1")
    # visualize_depth_map(results[1][i] - depth_img_2[i], "difference 2")

    cv2.waitKey()