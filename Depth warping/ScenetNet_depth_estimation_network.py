import cv2
import math
import sys
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

def mean_squared_difference(x, weight):
    x1, x2 = x
    valid_sum = 0.5 * (K.greater(x1, 0.01).sum() + K.greater(x2, 0.01).sum())
    #    mean_squared = weight * K.log(K.sum(K.square(x1 - x2)) / valid_sum + 1.0)
    mse = weight * K.sqrt(K.sum(K.square(x1 - x2)) / valid_sum)
    return mse

def mean_squared_difference_output_shape(input_shapes):
    input_shape1, input_shape2 = input_shapes
    return (input_shape1[0], 1)
#
# def sparse_mean_squared_difference(x, weight):
#     x1, x2 = x
#     valid_sum = K.greater(x1, 0.01).sum()
#     return weight * K.sum(K.exp(K.square((x1 - K.sum(x1) / valid_sum) / 0.2)) * K.square(x1 - x2)) / valid_sum
#
# def sparse_mean_squared_difference_output_shape(input_shapes):
#     input_shape1, input_shape2 = input_shapes
#     return (input_shape1[0], 1)
#
# def image_difference(x):
#     x1, x2 = x
#     return x1 - x2
#
# def image_difference_output_shape(input_shape):
#     return input_shape[0]
#
def customized_loss(y_true, y_pred):
    return y_pred - y_true

get_custom_objects().update({"customized_loss":customized_loss})

prefix = "/home/xingtong/PycharmProjects/SceneNet/data/extracted_training_data/Ten_trajectories/"

training_color_img_1 = np.load(prefix + "color_img_1.npy")
training_color_img_1 = np.array(training_color_img_1, dtype='float32')
training_color_img_1 = training_color_img_1 / 255.0

training_color_img_2 = np.load(prefix + "color_img_2.npy")
training_color_img_2 = np.array(training_color_img_2, dtype='float32')
training_color_img_2 = training_color_img_2 / 255.0

[height, width, channel] = training_color_img_1[0].shape

R = np.load(prefix + "R.npy")
R_I = np.load(prefix + "R_I.npy")
P = np.load(prefix + "P.npy")
P_I = np.load(prefix + "P_I.npy")

depth_img_1 = np.load(prefix + "depth_img_1.npy")
depth_img_1 = np.array(depth_img_1, dtype='float32')
depth_img_1 = np.reshape(depth_img_1, (-1, height, width, 1))

depth_img_2 = np.load(prefix + "depth_img_2.npy")
depth_img_2 = np.array(depth_img_2, dtype='float32')
depth_img_2 = np.reshape(depth_img_2, (-1, height, width, 1))

intrinsic_matrix = camera_intrinsic_transform()
print(intrinsic_matrix.shape)

P = np.array(P, dtype='float32')
P = np.reshape(P, (-1, 3, 1))

P_I = np.array(P_I, dtype='float32')
P_I = np.reshape(P_I, (-1, 3, 1))

R = np.array(R, dtype='float32')
R_I = np.array(R_I, dtype='float32')

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
maxpool1 = MaxPooling2D((2, 2))
conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')
maxpool2 = MaxPooling2D((2, 2))
conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')
maxpool3 = MaxPooling2D((4, 4))
conv4 = Conv2D(512, (3, 3), padding='same', activation='relu')
convt1 = Conv2DTranspose(128, (3, 3), padding='same', activation='relu')
upsampl1 = UpSampling2D((4, 4))
convt2 = Conv2DTranspose(64, (3, 3), padding='same', activation='relu')
upsampl2 = UpSampling2D((2, 2))
convt3 = Conv2DTranspose(32, (3, 3), padding='same', activation='relu')
upsampl3 = UpSampling2D((2, 2))
convt4 = Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid')

# conv1_static = Conv2D(32, (3, 3), padding='same', activation='relu', trainable=False)
# maxpool1_static = MaxPooling2D((2, 2))
# conv2_static = Conv2D(64, (3, 3), padding='same', activation='relu', trainable=False)
# maxpool2_static = MaxPooling2D((2, 2))
# conv3_static = Conv2D(128, (3, 3), padding='same', activation='relu', trainable=False)
# convt1_static = Conv2DTranspose(64, (3, 3), padding='same', activation='relu', trainable=False)
# upsampl1_static = UpSampling2D((2, 2))
# convt2_static = Conv2DTranspose(32, (3, 3), padding='same', activation='relu', trainable=False)
# upsampl2_static = UpSampling2D((2, 2))
# convt3_static = Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid', trainable=False)

x = conv1(input_img_1)
x = maxpool1(x)
x = conv2(x)
x = maxpool2(x)
x = conv3(x)
# x = maxpool3(x)
# x = conv4(x)
x = convt2(x)
x = upsampl2(x)
x = convt3(x)
x = upsampl3(x)
estimated_depth_map_1 = convt4(x)

x = conv1(input_img_2)
x = maxpool1(x)
x = conv2(x)
x = maxpool2(x)
x = conv3(x)
# x = maxpool3(x)
# x = conv4(x)
x = convt2(x)
x = upsampl2(x)
x = convt3(x)
x = upsampl3(x)
estimated_depth_map_2 = convt4(x)

# x = conv1(input_img_1)
# x = maxpool1(x)
# x = conv2(x)
# x = maxpool2(x)
# x = conv3(x)
# x = maxpool3(x)
# x = conv4(x)
# x = convt1(x)
# x = upsampl1(x)
# x = convt2(x)
# x = upsampl2(x)
# x = convt3(x)
# x = upsampl3(x)
# estimated_depth_map_1 = convt4(x)
#
# x = conv1(input_img_2)
# x = maxpool1(x)
# x = conv2(x)
# x = maxpool2(x)
# x = conv3(x)
# x = maxpool3(x)
# x = conv4(x)
# x = convt1(x)
# x = upsampl1(x)
# x = convt2(x)
# x = upsampl2(x)
# x = convt3(x)
# x = upsampl3(x)
# estimated_depth_map_2 = convt4(x)

# x = conv1_static(input_img_2)
# x = maxpool1_static(x)
# x = conv2_static(x)
# x = maxpool2_static(x)
# x = conv3_static(x)
# x = convt1_static(x)
# x = upsampl1_static(x)
# x = convt2_static(x)
# x = upsampl2_static(x)
# estimated_depth_map_2 = convt3_static(x)

estimated_unlog_depth_map_1 = Lambda(depth_unlog, output_shape=depth_unlog_output_shape, arguments={'scale_factor': 10.0})(estimated_depth_map_1)
estimated_unlog_depth_map_2 = Lambda(depth_unlog, output_shape=depth_unlog_output_shape, arguments={'scale_factor': 10.0})(estimated_depth_map_2)

synthetic_depth_map_1 = DepthWarpingLayer(intrinsic_matrix)([estimated_unlog_depth_map_1, estimated_unlog_depth_map_2, input_translation_vector, input_rotation_matrix])
synthetic_depth_map_2 = DepthWarpingLayer(intrinsic_matrix)([estimated_unlog_depth_map_2, estimated_unlog_depth_map_1, input_translation_vector_inverse, input_rotation_matrix_inverse])

depth_map_mean_squared_difference_1 = Lambda(mean_squared_difference, output_shape=mean_squared_difference_output_shape, arguments={'weight': 0.5})([synthetic_depth_map_1, estimated_depth_map_1])
depth_map_mean_squared_difference_2 = Lambda(mean_squared_difference, output_shape=mean_squared_difference_output_shape, arguments={'weight': 0.5})([synthetic_depth_map_2, estimated_depth_map_2])

# mse_loss = add([depth_map_mean_squared_difference_1, depth_map_mean_squared_difference_2])
#
# model = Model([input_img_1, input_img_2, input_translation_vector, input_translation_vector_inverse, input_rotation_matrix, input_rotation_matrix_inverse], mse_loss)

## The reason why the depth map provided here is somehow not consistent with our depth warping layer is because of the definition of the depth value.
## They use the length between the point and the principle point of the camera as the depth value

# synthetic_depth_map_1 = DepthWarpingLayer(intrinsic_matrix)([input_depth_img_1, input_depth_img_2, input_translation_vector, input_rotation_matrix])
# synthetic_depth_map_2 = DepthWarpingLayer(intrinsic_matrix)([input_depth_img_2, input_depth_img_1, input_translation_vector_inverse, input_rotation_matrix_inverse])
#
# #
model = Model([input_img_1, input_img_2, input_translation_vector, input_translation_vector_inverse, input_rotation_matrix, input_rotation_matrix_inverse],
              [estimated_unlog_depth_map_1, estimated_unlog_depth_map_2, synthetic_depth_map_1, synthetic_depth_map_2])

model.load_weights("/home/xingtong/PycharmProjects/SceneNet/data/trained_network/Ten_trajectories/warped_depth_estimation_network_weights-improvement-28-3.02104.hdf5")
                  # "warped_depth_estimation_network_weights-improvement-40-0.94606.hdf5")

# Toiletry 284 training sample pairs succeed /home/xingtong/PycharmProjects/SceneNet/data/trained_network/warped_depth_estimation_network_weights-improvement-39-21.69228.hdf5


model.summary()
sgd = optimizers.SGD(lr=1.0e-6, momentum=0.9, nesterov=True)
model.compile(loss=customized_loss, optimizer=sgd)

# # alternative_updating = AlternativeUpdating(1)
# filepath = "/home/xingtong/PycharmProjects/SceneNet/data/trained_network/Ten_trajectories/warped_depth_estimation_network_weights-improvement-{epoch:02d}-{val_loss:.5f}.hdf5"
# reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.00001, cooldown=0, min_lr=0)
# checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
# earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, min_delta=0.00001, mode='auto')
# ##
# history = model.fit([training_color_img_1, training_color_img_2, P, P_I, R, R_I], allzeros_groundtruth_output, batch_size=20,
#                     epochs=500, verbose=1, callbacks=[earlyStopping, checkpointer, reducelr], validation_split=0.05, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)


trajectory_index = 0
count_per_traj = 100
results = model.predict([training_color_img_1[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj],
                         training_color_img_2[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj],
                         P[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj],
                         P_I[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj],
                        R[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj],
                         R_I[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj]], batch_size = 20)

cv2.destroyAllWindows()
for i in range(0, count_per_traj):
    print(i)

    border_width = 20
    color_img_1 = training_color_img_1[i + trajectory_index * count_per_traj]
    color_img_2 = training_color_img_2[i + trajectory_index * count_per_traj]
    cv2.imshow("rgb 1", color_img_1[border_width:-border_width, border_width:-border_width])
    cv2.imshow("rgb 2", color_img_2[border_width:-border_width, border_width:-border_width])
    visualize_depth_map(results[0][i], "depth map 1", border_width)
    visualize_depth_map(results[1][i], "depth map 2", border_width)

    # visualize_depth_map(results[0][i], "synthetic depth map 1")
    # visualize_depth_map(results[1][i], "synthetic depth map 2")
    # visualize_depth_map(depth_img_1[i], "original depth map 1")
    # visualize_depth_map(depth_img_2[i], "original depth map 2")
    # visualize_depth_map(results[0][i] - depth_img_1[i], "difference 1")
    # visualize_depth_map(results[1][i] - depth_img_2[i], "difference 2")

    cv2.waitKey()
