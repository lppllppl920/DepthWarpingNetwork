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
explainability_masking_loss_ = Lambda(explainability_mask_loss, output_shape=explainability_mask_loss_output_shape, arguments={'weight': 1.0})(explainability_mask)

mse_loss = add([depth_map_mean_squared_difference_1, depth_map_mean_squared_difference_2, explainability_masking_loss_]) #, explainability_masking_loss
model = Model([input_img_1, input_img_2, input_translation_vector, input_translation_vector_inverse, input_rotation_matrix, input_rotation_matrix_inverse], mse_loss)

model.load_weights("/home/xingtong/PycharmProjects/SceneNet/data/trained_network/Ten_trajectories/warped_depth_estimation_network_weights-improvement-11-7.69150.hdf5")

model.summary()
sgd = optimizers.SGD(lr=2.0e-4, momentum=0.9, nesterov=True)
model.compile(loss=customized_loss, optimizer=sgd)

filepath = "/home/xingtong/PycharmProjects/SceneNet/data/trained_network/Ten_trajectories/warped_depth_estimation_network_weights-improvement-{epoch:02d}-{val_loss:.5f}.hdf5"
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.00001, cooldown=0, min_lr=0)
checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, min_delta=0.00001, mode='auto')
##
history = model.fit([training_color_img_1, training_color_img_2, P, P_I, R, R_I], allzeros_groundtruth_output, batch_size=20,
                    epochs=500, verbose=1, callbacks=[earlyStopping, checkpointer, reducelr], validation_split=0.2, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)


