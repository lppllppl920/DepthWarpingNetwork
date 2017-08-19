# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 05:07:48 2017

@author: DELL1
"""
from theano.tensor.nlinalg import matrix_inverse
from keras import backend as K
import theano.tensor as T
import numpy as np
import theano
import cv2

def _repeat(x, n_repeats):
    rep = T.ones((n_repeats,), dtype='int32').dimshuffle('x', 0)
    x = K.dot(x.reshape((-1, 1)), rep)
    return x.flatten()


def _interpolate(im, x, y):
    # constants
    num_batch, height, width, channels = im.shape
#    height_f = K.cast(height, 'float32')
#    width_f = K.cast(width, 'float32')
    zero = K.zeros([], dtype='int64')
    max_y = K.cast(im.shape[1] - 1, 'int64')
    max_x = K.cast(im.shape[2] - 1, 'int64')

#    # scale indices from [-1, 1] to [0, width/height]
#    x = (x + 1.0)*(width_f) / 2.0
#    y = (y + 1.0)*(height_f) / 2.0

    # do sampling
    x0 = K.cast(T.floor(x), 'int64')
    x1 = x0 + 1
    y0 = K.cast(T.floor(y), 'int64')
    y1 = y0 + 1

    x0 = T.clip(x0, zero, max_x)
    x1 = T.clip(x1, zero, max_x)
    y0 = T.clip(y0, zero, max_y)
    y1 = T.clip(y1, zero, max_y)
    
    dim2 = width
    dim1 = width*height
    base = _repeat(
        T.arange(num_batch, dtype='int32')*dim1, height*width)
    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels in the flat image and restore channels dim
    im_flat = K.reshape(im, (-1, channels))
    Ia = im_flat[idx_a]
    Ib = im_flat[idx_b]
    Ic = im_flat[idx_c]
    Id = im_flat[idx_d]

    # and finanly calculate interpolated values
    x0_f = K.cast(x0, 'float32')
    x1_f = K.cast(x1, 'float32')
    y0_f = K.cast(y0, 'float32')
    y1_f = K.cast(y1, 'float32')
    wa = ((x1_f-x) * (y1_f-y)).dimshuffle(0, 'x')
    wb = ((x1_f-x) * (y-y0_f)).dimshuffle(0, 'x')
    wc = ((x-x0_f) * (y1_f-y)).dimshuffle(0, 'x')
    wd = ((x-x0_f) * (y-y0_f)).dimshuffle(0, 'x')
    output = K.sum([wa*Ia, wb*Ib, wc*Ic, wd*Id], axis=0)
    return output
    
def visualize_depth_map(depth_map_test, title):
    
    min_value, max_value, min_loc, max_loc = cv2.minMaxLoc(depth_map_test)
    depth_map_visualize = (depth_map_test - min_value) / (max_value - min_value) * 255
    depth_map_visualize = np.asarray(depth_map_visualize, dtype = 'uint8')
    cv2.imshow(title, depth_map_visualize)
    cv2.waitKey(100)
    
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


depth_map_1 = T.tensor4()
depth_map_2 = T.tensor4()
num_batch, height, width, channels = depth_map_1.shape

#print(num_batch.eval({depth_map_1: depth_data}))

## Generate same meshgrid for each depth map to calculate value
x_grid = K.dot(T.ones((height, 1)),
        T.arange(width, dtype='float32').dimshuffle('x', 0))
y_grid = K.dot(T.arange(height, dtype='float32').dimshuffle(0, 'x'),
        T.ones((1, width)))
x_grid = x_grid.reshape((1, height, width, 1))
y_grid = y_grid.reshape((1, height, width, 1))
x_grid = K.repeat_elements(x_grid, num_batch, axis = 0)
y_grid = K.repeat_elements(y_grid, num_batch, axis = 0)


#print(y_grid.shape.eval({depth_map_1: depth_data}))

rotation_matrices = T.tensor3()
rotation_matrices_inverse = rotation_matrices.dimshuffle((0, 2, 1))
intrinsic_matrix = T.matrix()

intrinsic_matrix_inverse = matrix_inverse(intrinsic_matrix)

#print(rotation_matrices_inverse.eval({rotation_matrices: rotation_data}))
#print(rotation_data)

temp_mat, updates = theano.scan(fn=lambda rotation_I, intrinsic_mat: T.dot(intrinsic_mat, rotation_I),
                          outputs_info=None,
                          sequences=[rotation_matrices_inverse],
                          non_sequences=[intrinsic_matrix])

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

#print(temp_mat.shape.eval({rotation_matrices: rotation_data, intrinsic_matrix: P}))
translation_vectors = T.tensor3()
W = T.batched_dot(temp_mat, -translation_vectors)

#print(temp_mat.eval({rotation_matrices: rotation_data, intrinsic_matrix: P}))
#print(W.eval({rotation_matrices: rotation_data, intrinsic_matrix: P, translation_vectors: translation_data}))

M, updates = theano.scan(fn=lambda mat, intrinsic_mat_I: T.dot(mat, intrinsic_mat_I),
                      outputs_info=None,
                      sequences=[temp_mat],
                      non_sequences=[intrinsic_matrix_inverse])

W_2, updates = theano.scan(fn=lambda translation_vector, intrinsic_mat: T.dot(intrinsic_mat, translation_vector),
                      outputs_info=None,
                      sequences=[translation_vectors],
                      non_sequences=[intrinsic_matrix])


temp_mat_2, updates = theano.scan(fn=lambda rotation_mat, intrinsic_mat: T.dot(intrinsic_mat, rotation_mat),
                          outputs_info=None,
                          sequences=[rotation_matrices],
                          non_sequences=[intrinsic_matrix])

M_2, updates = theano.scan(fn=lambda mat, intrinsic_mat_I: T.dot(mat, intrinsic_mat_I),
                          outputs_info=None,
                          sequences=[temp_mat_2],
                          non_sequences=[intrinsic_matrix_inverse])

depth_map_2_calculate, updates = theano.scan(fn=lambda W, M, u, v, z_1, : W[2, 0] + z_1 * (M[2, 0] * u + M[2, 1] * v + M[2, 2]),
                  outputs_info=None,
                  sequences=[W, M, x_grid, y_grid, depth_map_1])
   
u_2, updates = theano.scan(fn=lambda W, M, u, v, z_1, z_2_calculate, : (z_1 * (M[0, 0] * u + M[0, 1] * v + M[0, 2]) + W[0, 0]) / z_2_calculate,
                  outputs_info=None,
                  sequences=[W, M, x_grid, y_grid, depth_map_1, depth_map_2_calculate])

v_2, updates = theano.scan(fn=lambda W, M, u, v, z_1, z_2_calculate, : (z_1 * (M[1, 0] * u + M[1, 1] * v + M[1, 2]) + W[1, 0]) / z_2_calculate,
                  outputs_info=None,
                  sequences=[W, M, x_grid, y_grid, depth_map_1, depth_map_2_calculate])

depth_map_1_calculate, updates = theano.scan(fn=lambda W_2, M_2, u, v, z_2, : W_2[2, 0] + z_2 * (M_2[2, 0] * u + M_2[2, 1] * v + M_2[2, 2]),
                  outputs_info=None,
                  sequences=[W_2, M_2, x_grid, y_grid, depth_map_2])

u_2_flat = u_2.flatten()
v_2_flat = v_2.flatten()

depth_map_1_transformed_flat = _interpolate(depth_map_1_calculate, u_2_flat, v_2_flat)
depth_map_1_transformed = K.reshape(depth_map_1_transformed_flat,
               (num_batch, height, width, channels))

#print(x_grid.shape.eval({depth_map_1: depth_data}))
#print(W.shape.eval({rotation_matrices: rotation_data, intrinsic_matrix: P, translation_vectors: translation_data}))

u_2_eval = u_2.eval({rotation_matrices: rotation_data, intrinsic_matrix: P, \
                          translation_vectors: translation_data, depth_map_1: synthesis_depth_data})
#print(u_2_eval)

W_eval = W.eval({rotation_matrices: rotation_data, intrinsic_matrix: P, translation_vectors: translation_data})
M_eval = M.eval({rotation_matrices: rotation_data, intrinsic_matrix: P})

depth_img_1_calculate_eval = depth_map_1_calculate.eval({rotation_matrices: rotation_data, intrinsic_matrix: P, \
                          translation_vectors: translation_data, depth_map_1: synthesis_depth_data, depth_map_2: depth_data})
visualize_depth_map(depth_img_1_calculate_eval[1], '1')

depth_img_2_calculate_eval = depth_map_2_calculate.eval({rotation_matrices: rotation_data, intrinsic_matrix: P, \
                          translation_vectors: translation_data, depth_map_1: synthesis_depth_data})
visualize_depth_map(depth_img_2_calculate_eval[1], '2')

depth_map_1_transformed_eval = depth_map_1_transformed.eval({rotation_matrices: rotation_data, intrinsic_matrix: P, \
                          translation_vectors: translation_data, depth_map_1: synthesis_depth_data, depth_map_2: depth_data})
visualize_depth_map(depth_map_1_transformed_eval[1], '1 transform')
