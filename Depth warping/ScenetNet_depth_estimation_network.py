import cv2
import math
import sys
import numpy as np
import yaml
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Flatten, concatenate, Dropout, Lambda, add, Add
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

def mean_squared_difference(x, weight):
    x1, x2 = x
    valid_sum =  K.greater(x1, 0.0001).sum()
    abs_error = K.sum(K.abs(x1 - x2))
    c = 0.2 * K.max(abs_error)
    squared_error = (K.square(x1 - x2) + c * c) / (2 * c)
    error = K.switch(abs_error <= c, abs_error, squared_error)
    return weight * K.sum(error) / (valid_sum + 1.0)
def mean_squared_difference_output_shape(input_shapes):
    input_shape1, input_shape2 = input_shapes
    return (input_shape1[0], 1)

def depth_smoothness_loss(x, weight):
    laplacian_kernel = -np.ones((3, 3, 1, 1), dtype = np.float32)
    laplacian_kernel[1, 1] = 8.0
    laplacian_kernel = theano.shared(laplacian_kernel)
    filtered_x = K.conv2d(x,kernel=laplacian_kernel, padding='valid',data_format='channels_last')
    return weight * K.mean(K.abs(filtered_x))
def depth_smoothness_loss_output_shape(input_shapes):
    input_shape1 = input_shapes
    return (input_shape1[0], 1)

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

mask_img = np.load(prefix + "mask_img.npy")
mask_img = np.array(mask_img, dtype='float32')
mask_img = np.reshape(mask_img, (-1, 240, 320, 1))

## trajectory is in meter unit while the original depth value is in millimeter unit.
masked_depth_img_1 = np.load(prefix + "masked_depth_img_1.npy")
masked_depth_img_1 = np.array(masked_depth_img_1, dtype='float32')
masked_depth_img_1 = masked_depth_img_1 / 1000.0
masked_depth_img_1 = np.reshape(masked_depth_img_1, (-1, 240, 320, 1))

masked_depth_img_2 = np.load(prefix + "masked_depth_img_2.npy")
masked_depth_img_2 = np.array(masked_depth_img_2, dtype='float32')
masked_depth_img_2 = masked_depth_img_2 / 1000.0
masked_depth_img_2 = np.reshape(masked_depth_img_2, (-1, 240, 320, 1))

## Shuffle training data
indexes = np.arange(len(R))
shuffled_indexes = np.random.shuffle(indexes)
training_color_img_1 = training_color_img_1[shuffled_indexes]
training_color_img_2 = training_color_img_2[shuffled_indexes]
R = R[shuffled_indexes]
R_I = R_I[shuffled_indexes]
P = P[shuffled_indexes]
P_I = P_I[shuffled_indexes]

training_color_img_1 = np.reshape(training_color_img_1, (-1, height, width, channel))
training_color_img_2 = np.reshape(training_color_img_2, (-1, height, width, channel))

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

input_color_img_1 = Input(shape=(height, width, channel))
input_color_img_2 = Input(shape=(height, width, channel))

input_translation_vector = Input(shape=(3, 1))
input_rotation_matrix = Input(shape=(3, 3))

input_translation_vector_inverse = Input(shape=(3, 1))
input_rotation_matrix_inverse = Input(shape=(3, 3))

input_depth_img_1 = Input(shape=(height, width, 1))
input_depth_img_2 = Input(shape=(height, width, 1))


## Branch 1 and 2
conv1 = Conv2D(32, (3, 3), padding='same', strides=(2, 2), activation='linear') ##output: 120x160x32
activation1 = Activation(activation='relu')
batch_norm1 = BatchNormalization()

conv2 = Conv2D(64, (3, 3), padding='same', strides=(2, 2), activation='linear') ##output: 60x80x64
activation2 = Activation(activation='relu')
batch_norm2 = BatchNormalization()

conv3 = Conv2D(128, (3, 3), padding='same', strides=(2, 2), activation='linear') ##output: 30x40x128
activation3 = Activation(activation='relu')
batch_norm3 = BatchNormalization()

conv4 = Conv2D(256, (3, 3), padding='same', strides=(2, 2), activation='relu') ##output: 15x20x256
batch_norm4 = BatchNormalization()

convt1 = Conv2DTranspose(256, (3, 3), padding='same', strides=(1, 1), activation='relu') ##output: 15x20x256
batch_norm4_mirror = BatchNormalization()

convt2 = Conv2DTranspose(128, (3, 3), padding='same', strides=(2, 2), activation='linear') ##output: 30x40x128

add3 = Add()
activation3_mirror = Activation(activation='relu')
batch_norm3_mirror = BatchNormalization()

convt3 = Conv2DTranspose(64, (3, 3), padding='same', strides=(2, 2), activation='linear') ##output: 60x80x64

add2 = Add()
activation2_mirror = Activation(activation='relu')
batch_norm2_mirror = BatchNormalization()

convt4 = Conv2DTranspose(32, (3, 3), padding='same', strides=(2, 2), activation='linear') ##output: 120x160x32

add1 = Add()
activation1_mirror = Activation(activation='relu')
batch_norm1_mirror = BatchNormalization()

convt5 = Conv2DTranspose(1, (3, 3), padding='same', strides=(2, 2), activation='relu') ##output: 240x320x1


## Branch 3
concat1_exp = Concatenate(axis=-1)

conv1_exp = Conv2D(32, (3, 3), padding='same', strides=(2, 2), activation='linear', name='conv_1_exp') ##output: 120x160x32
activation1_exp = Activation(activation='relu', name='activation_1_exp')
batch_norm1_exp = BatchNormalization(name='batch_norm1_exp')

conv2_exp = Conv2D(64, (3, 3), padding='same', strides=(2, 2), activation='linear', name='conv_2_exp') ##output: 60x80x64
activation2_exp = Activation(activation='relu', name='activation_2_exp')
batch_norm2_exp = BatchNormalization(name='batch_norm2_exp')

conv3_exp = Conv2D(128, (3, 3), padding='same', strides=(2, 2), activation='linear', name='conv_3_exp') ##output: 30x40x128
activation3_exp = Activation(activation='relu', name='activation_3_exp')
batch_norm3_exp = BatchNormalization(name='batch_norm3_exp')

conv4_exp = Conv2D(256, (3, 3), padding='same', strides=(2, 2), activation='relu', name='conv_4_exp') ##output: 15x20x256
batch_norm4_exp = BatchNormalization(name='batch_norm4_exp')

convt1_exp = Conv2DTranspose(256, (3, 3), padding='same', strides=(1, 1), activation='relu', name='convt_1_exp') ##output: 15x20x256
batch_norm4_mirror_exp = BatchNormalization(name='batch_norm4_mirror_exp')

convt2_exp = Conv2DTranspose(128, (3, 3), padding='same', strides=(2, 2), activation='linear', name='convt_2_exp') ##output: 30x40x128

add3_exp = Add(name='add_3_exp')
activation3_mirror_exp = Activation(activation='relu', name='activation_3_mirror_exp')
batch_norm3_mirror_exp = BatchNormalization(name='batch_norm3_mirror_exp')

convt3_exp = Conv2DTranspose(64, (3, 3), padding='same', strides=(2, 2), activation='linear', name='convt_3_exp') ##output: 60x80x64

add2_exp = Add(name='add_2_exp')
activation2_mirror_exp = Activation(activation='relu', name='activation_2_mirror_exp')
batch_norm2_mirror_exp = BatchNormalization(name='batch_norm2_mirror_exp')

convt4_exp = Conv2DTranspose(32, (3, 3), padding='same', strides=(2, 2), activation='linear', name='convt_4_exp') ##output: 120x160x32

add1_exp = Add(name='add_1_exp')
activation1_mirror_exp = Activation(activation='relu', name='activation_1_mirror_exp')
batch_norm1_mirror_exp = BatchNormalization(name='batch_norm1_mirror_exp')

convt5_exp = Conv2DTranspose(1, (3, 3), padding='same', strides=(2, 2), activation='relu', name='convt_5_exp') ##output: 240x320x1

## Adding skip connection
## Branch 1
conv1_output = conv1(input_color_img_1)

conv2_input = batch_norm1(activation1(conv1_output))
conv2_output = conv2(conv2_input)

conv3_input = batch_norm2(activation2(conv2_output))
conv3_output = conv3(conv3_input)

conv4_input = batch_norm3(activation3(conv3_output))
conv4_output = conv4(conv4_input)

convt1_input = batch_norm4(conv4_output)
convt1_output = convt1(convt1_input)

convt2_input = batch_norm4_mirror(convt1_output)
convt2_output = convt2(convt2_input)

convt3_input = batch_norm3_mirror(activation3_mirror(add3([conv3_output, convt2_output])))
convt3_output = convt3(convt3_input)

convt4_input = batch_norm2_mirror(activation2_mirror(add2([conv2_output, convt3_output])))
convt4_output = convt4(convt4_input)

convt5_input = batch_norm1_mirror(activation1_mirror(add1([conv1_output, convt4_output])))
estimated_depth_map_1 = convt5(convt5_input)

## Branch 2
conv1_output = conv1(input_color_img_2)

conv2_input = batch_norm1(activation1(conv1_output))
conv2_output = conv2(conv2_input)

conv3_input = batch_norm2(activation2(conv2_output))
conv3_output = conv3(conv3_input)

conv4_input = batch_norm3(activation3(conv3_output))
conv4_output = conv4(conv4_input)

convt1_input = batch_norm4(conv4_output)
convt1_output = convt1(convt1_input)

convt2_input = batch_norm4_mirror(convt1_output)
convt2_output = convt2(convt2_input)

convt3_input = batch_norm3_mirror(activation3_mirror(add3([conv3_output, convt2_output])))
convt3_output = convt3(convt3_input)

convt4_input = batch_norm2_mirror(activation2_mirror(add2([conv2_output, convt3_output])))
convt4_output = convt4(convt4_input)

convt5_input = batch_norm1_mirror(activation1_mirror(add1([conv1_output, convt4_output])))
estimated_depth_map_2 = convt5(convt5_input)


## Branch 3
concat_input_exp = concat1_exp([input_color_img_1, input_color_img_2])

conv1_output_exp = conv1_exp(concat_input_exp)

conv2_input_exp = batch_norm1_exp(activation1_exp(conv1_output_exp))
conv2_output_exp = conv2_exp(conv2_input_exp)

conv3_input_exp = batch_norm2_exp(activation2_exp(conv2_output_exp))
conv3_output_exp = conv3_exp(conv3_input_exp)

conv4_input_exp = batch_norm3_exp(activation3_exp(conv3_output_exp))
conv4_output_exp = conv4_exp(conv4_input_exp)

convt1_input_exp = batch_norm4_exp(conv4_output_exp)
convt1_output_exp = convt1_exp(convt1_input_exp)

convt2_input_exp = batch_norm4_mirror_exp(convt1_output_exp)
convt2_output_exp = convt2_exp(convt2_input_exp)

convt3_input_exp = batch_norm3_mirror_exp(activation3_mirror_exp(add3_exp([conv3_output_exp, convt2_output_exp])))
convt3_output_exp = convt3_exp(convt3_input_exp)

convt4_input_exp = batch_norm2_mirror_exp(activation2_mirror_exp(add2_exp([conv2_output_exp, convt3_output_exp])))
convt4_output_exp = convt4_exp(convt4_input_exp)

convt5_input_exp = batch_norm1_mirror_exp(activation1_mirror_exp(add1_exp([conv1_output_exp, convt4_output_exp])))
explainability_mask = convt5_exp(convt5_input_exp)



synthetic_depth_map_1 = DepthWarpingLayer(intrinsic_matrix)([estimated_depth_map_1, estimated_depth_map_2, input_translation_vector, input_rotation_matrix])
synthetic_depth_map_2 = DepthWarpingLayer(intrinsic_matrix)([estimated_depth_map_2, estimated_depth_map_1, input_translation_vector_inverse, input_rotation_matrix_inverse])

masked_estimated_unlog_depth_map_1 = Lambda(mask_invalid_element, output_shape=mask_invalid_element_output_shape)([synthetic_depth_map_1, estimated_depth_map_1])
masked_estimated_unlog_depth_map_2 = Lambda(mask_invalid_element, output_shape=mask_invalid_element_output_shape)([synthetic_depth_map_2, estimated_depth_map_2])

depth_map_mean_squared_difference_1 = Lambda(huber_error, output_shape=huber_error_output_shape, arguments={'weight': 0.5})([synthetic_depth_map_1, masked_estimated_unlog_depth_map_1, explainability_mask])
depth_map_mean_squared_difference_2 = Lambda(huber_error, output_shape=huber_error_output_shape, arguments={'weight': 0.5})([synthetic_depth_map_2, masked_estimated_unlog_depth_map_2, explainability_mask])
explainability_masking_loss_ = Lambda(explainability_mask_loss, output_shape=explainability_mask_loss_output_shape, arguments={'weight': 1.0})(explainability_mask)

mse_loss = add([depth_map_mean_squared_difference_1, depth_map_mean_squared_difference_2, explainability_masking_loss_])
model = Model([input_color_img_1, input_color_img_2, input_translation_vector, input_translation_vector_inverse, input_rotation_matrix, input_rotation_matrix_inverse], mse_loss)

# model.load_weights("/home/xingtong/PycharmProjects/SceneNet/data/trained_network/Ten_trajectories/warped_depth_estimation_network_weights-improvement-11-7.69150.hdf5")

model.summary()
# # sgd = optimizers.SGD(lr=1.0e-4, momentum=0.9, nesterov=True)
# adam = optimizers.adam(lr = 1.0e-4)
# model.compile(loss=customized_loss, optimizer=adam)
#
# filepath = "/home/xingtong/PycharmProjects/SceneNet/data/trained_network/Ten_trajectories/warped_depth_estimation_network_weights-improvement-{epoch:02d}-{val_loss:.5f}.hdf5"
# reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.00001, cooldown=0, min_lr=0)
# checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
# earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, min_delta=0.00001, mode='auto')
# history = model.fit([training_color_img_1, training_color_img_2, P, P_I, R, R_I], allzeros_groundtruth_output, batch_size=20,
#                     epochs=500, verbose=1, callbacks=[earlyStopping, checkpointer, reducelr], validation_split=0.2, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)
#

