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
import theano.tensor
from matplotlib import axis

theano.config.exception_verbosity = 'high'

def motion_creating(x):

    x1,x2 = x
    theta, updates = theano.scan(fn=lambda rotation: [theano.tensor.arctan2(rotation[2, 1], rotation[2, 2]),
                                                      theano.tensor.arctan2(-rotation[2, 0], theano.tensor.sqrt(rotation[2, 1] * rotation[2, 1] + rotation[2, 2] * rotation[2, 2])),
                                                      theano.tensor.arctan2(rotation[1, 0], rotation[0, 0])],
                               outputs_info=None,
                               sequences=[x1])

    euler_angle = K.reshape(theta, (-1, 1, 1, 3)) / 3.2
    translation = K.reshape(x2, (-1, 1, 1, 3)) / 0.6

    motion =  K.concatenate([translation, euler_angle], axis=-1)
    tiled_motion = K.tile(motion, (1, 240, 320, 1))
    tiled_motion = K.reshape(tiled_motion, (-1, 240, 320, 6))


    return tiled_motion


def motion_creating_output_shape(input_shapes):
    input_shape1, input_shape2 = input_shapes
    return (input_shape1[0], 240, 320, 6)

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

def abs_error(x, weight):
    x1, x2, x3 = x
    valid_sum = 0.5 * (K.greater(K.abs(x1), 0.0001).sum() + K.greater(K.abs(x2), 0.0001).sum())
    abs_error = K.sum(x3 * K.abs(x1 - x2)) / (valid_sum + 1.0)
    return abs_error
def abs_error_output_shape(input_shapes):
    input_shape1, input_shape2, input_shape3 = input_shapes
    return (input_shape1[0], 1)

def mean_squared_difference(x, weight):
    x1, x2 = x
    valid_sum =  K.greater(x1, 0.0001).sum()
    abs_error = K.sum(K.abs(x1 - x2))
    # c = 0.2 * K.max(abs_error)
    # squared_error = (K.square(x1 - x2) + c * c) / (2 * c)
    # error = K.switch(abs_error <= c, abs_error, squared_error)
    # mean_error = weight * K.sum(error) / (valid_sum + 1.0)
    return abs_error / (valid_sum + 1.0)
def mean_squared_difference_output_shape(input_shapes):
    input_shape1, input_shape2 = input_shapes
    return (input_shape1[0], 1)

## make as few masking as possible
def explainability_mask_loss(x, weight):
    return weight * K.mean(-K.log(x))
def explainability_mask_loss_output_shape(input_shapes):
    input_shape1 = input_shapes
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

## Masking out invalid region for warped depth map
def mask_invalid_element(x):
    x1, x2 = x
    return K.switch(K.abs(x1) < 0.001, 0, x2)
def mask_invalid_element_output_shape(input_shapes):
    shape1, shape2 = input_shapes
    return shape1

def customized_loss(y_true, y_pred):
    return y_pred - y_true


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

## trajectory is in meter unit while the original depth value is in millimeter unit. motherfucker
masked_depth_img_1 = np.load(prefix + "masked_depth_img_1.npy")
masked_depth_img_1 = np.array(masked_depth_img_1, dtype='float32')
masked_depth_img_1 = masked_depth_img_1 / 1000.0
masked_depth_img_1 = np.reshape(masked_depth_img_1, (-1, 240, 320, 1))

masked_depth_img_2 = np.load(prefix + "masked_depth_img_2.npy")
masked_depth_img_2 = np.array(masked_depth_img_2, dtype='float32')
masked_depth_img_2 = masked_depth_img_2 / 1000.0
masked_depth_img_2 = np.reshape(masked_depth_img_2, (-1, 240, 320, 1))

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

print(np.max(np.abs(P)))
print(np.max(np.abs(P_I)))

intrinsic_matrix = camera_intrinsic_transform()


allzeros_groundtruth_output = np.zeros((R.shape[0], 1))

input_img_t = Input(shape=(240, 320, 3))
input_img_t_plus_1 = Input(shape=(240, 320, 3))
input_masked_depth_img_t = Input(shape=(240, 320, 1))
input_masked_depth_img_t_plus_1 = Input(shape=(240, 320, 1))
input_mask_img = Input(shape=(240, 320, 1))
input_translation_vector = Input(shape=(3, 1))
input_rotation_matrix = Input(shape=(3, 3))
input_translation_vector_inverse = Input(shape=(3, 1))
input_rotation_matrix_inverse = Input(shape=(3, 3))


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


## Convert translation and rotation matrix to a 6x1 representation and repeat it to the same size as image as input.
concat_input_img = Concatenate(axis=-1)([input_img_t, input_img_t_plus_1])
concat_input_motion = Lambda(motion_creating, output_shape=motion_creating_output_shape)([input_rotation_matrix, input_translation_vector])
concat_input = Concatenate(axis=-1)([concat_input_img, concat_input_motion])

concat_input_img_inverse = Concatenate(axis=-1)([input_img_t_plus_1, input_img_t])
concat_input_motion_inverse = Lambda(motion_creating, output_shape=motion_creating_output_shape)([input_rotation_matrix_inverse, input_translation_vector_inverse])
concat_input_inverse = Concatenate(axis=-1)([concat_input_img_inverse, concat_input_motion_inverse])

x = conv1(concat_input)
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

x = conv1(concat_input_inverse)
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


x = conv1_explainability(concat_input_img)
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

smoothness_metric_1 = Lambda(depth_smoothness_loss, output_shape=depth_smoothness_loss_output_shape, arguments={'weight': 0.2})(estimated_depth_map_1)
smoothness_metric_2 = Lambda(depth_smoothness_loss, output_shape=depth_smoothness_loss_output_shape, arguments={'weight': 0.2})(estimated_depth_map_2)

masked_output_depth_img_1 = multiply([input_mask_img, estimated_depth_map_1])
masked_output_depth_img_2 = multiply([input_mask_img, estimated_depth_map_2])

sparse_masked_mean_squared_difference_1 = Lambda(mean_squared_difference,
                                                 output_shape=mean_squared_difference_output_shape, arguments={'weight': 0.5})([input_masked_depth_img_t, masked_output_depth_img_1])
sparse_masked_mean_squared_difference_2 = Lambda(mean_squared_difference,
                                                 output_shape=mean_squared_difference_output_shape, arguments={'weight': 0.5})([input_masked_depth_img_t_plus_1, masked_output_depth_img_2])

synthetic_depth_map_1 = DepthWarpingLayer(intrinsic_matrix)([estimated_depth_map_1, estimated_depth_map_2, input_translation_vector, input_rotation_matrix])
synthetic_depth_map_2 = DepthWarpingLayer(intrinsic_matrix)([estimated_depth_map_2, estimated_depth_map_1, input_translation_vector_inverse, input_rotation_matrix_inverse])

masked_estimated_unlog_depth_map_1 = Lambda(mask_invalid_element, output_shape=mask_invalid_element_output_shape)([synthetic_depth_map_1, estimated_depth_map_1])
masked_estimated_unlog_depth_map_2 = Lambda(mask_invalid_element, output_shape=mask_invalid_element_output_shape)([synthetic_depth_map_2, estimated_depth_map_2])

depth_map_mean_squared_difference_1 = Lambda(abs_error, output_shape=abs_error_output_shape, arguments={'weight': 0.5})([synthetic_depth_map_1, masked_estimated_unlog_depth_map_1, explainability_mask])
depth_map_mean_squared_difference_2 = Lambda(abs_error, output_shape=abs_error_output_shape, arguments={'weight': 0.5})([synthetic_depth_map_2, masked_estimated_unlog_depth_map_2, explainability_mask])
explainability_masking_loss_metric = Lambda(explainability_mask_loss, output_shape=explainability_mask_loss_output_shape, arguments={'weight': 0.3})(explainability_mask)

mse_loss = add([smoothness_metric_1, smoothness_metric_2, sparse_masked_mean_squared_difference_1, sparse_masked_mean_squared_difference_2,
                depth_map_mean_squared_difference_1, depth_map_mean_squared_difference_2, explainability_masking_loss_metric])

# model = Model([input_mask_img, input_masked_depth_img_t, input_masked_depth_img_t_plus_1, input_img_t, input_img_t_plus_1,
#                input_translation_vector, input_translation_vector_inverse, input_rotation_matrix, input_rotation_matrix_inverse], mse_loss)
#
# model.summary()
# sgd = optimizers.SGD(lr=1.0e-3, momentum=0.9, nesterov=True, clipnorm=100.)
# model.compile(loss=customized_loss, optimizer=sgd)
#
# filepath = "/home/xingtong/PycharmProjects/SceneNet/data/trained_network/Ten_trajectories/two_frame_depth_estimation_network_weights-improvement-{epoch:02d}-{val_loss:.5f}.hdf5"
# reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.00001, cooldown=0, min_lr=0)
# checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
# earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, min_delta=0.00001, mode='auto')
# ##
#
# history = model.fit([mask_img, masked_depth_img_1, masked_depth_img_2, training_color_img_1, training_color_img_2, P, P_I, R, R_I], allzeros_groundtruth_output, batch_size=40,
#                     epochs=500, verbose=1, callbacks=[earlyStopping, checkpointer, reducelr], validation_split=0.2, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)


model = Model(inputs=[input_mask_img, input_masked_depth_img_t, input_masked_depth_img_t_plus_1, input_img_t, input_img_t_plus_1,
               input_translation_vector, input_translation_vector_inverse, input_rotation_matrix, input_rotation_matrix_inverse], outputs=[estimated_depth_map_1, estimated_depth_map_2, explainability_mask, mse_loss])

model.summary()
sgd = optimizers.SGD(lr=1.0e-4, momentum=0.9, nesterov=True, clipnorm=10.)
model.compile(loss=customized_loss, optimizer=sgd)

model.load_weights("/home/xingtong/PycharmProjects/SceneNet/data/trained_network/Ten_trajectories/two_frame_depth_estimation_network_weights-improvement-10-2.76993.hdf5")


trajectory_index = 3
count_per_traj = 40
results = model.predict([mask_img[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj],
                         masked_depth_img_1[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj],
                         masked_depth_img_2[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj],
                         training_color_img_1[trajectory_index * count_per_traj: (trajectory_index + 1) * count_per_traj],
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
    mean_value = np.mean(depth_map[depth_map > 0.1])
    min_value = np.min(depth_map[depth_map > 0.1])
    max_value = (mean_value - min_value) + mean_value

    print(min_value, max_value)


    visualize_depth_map(results[0][i], "estimated depth map 1", border_width, min_value, max_value)
    visualize_depth_map(results[1][i], "estimated depth map 2", border_width, min_value, max_value)
    cv2.imshow("explainability mask", results[2][i] * 255.0)
    cv2.waitKey()


