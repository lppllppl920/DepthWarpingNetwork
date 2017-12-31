import theano
import theano.tensor as T
from keras import backend as K
from keras.engine.topology import Layer
from theano.tensor.nlinalg import matrix_inverse

class DepthWarpingLayer(Layer):
    
    def __init__(self, intrinsic_matrix, **kwargs):
        super(DepthWarpingLayer, self).__init__(**kwargs)
        self.intrinsic_matrix = intrinsic_matrix
    def compute_output_shape(self, input_shapes):
        return input_shapes[0]

#    def get_config(self):
#        config = {'intrinsic_matrix': self.intrinsic_matrix, 'mask': self.mask}
#        base_config = super(DepthWarpingLayer, self).get_config()
#        return dict(list(base_config.items()) + list(config.items()))
        
    def call(self, x):
        depth_map_1, depth_map_2, translation_vector, rotation_matrix = x
        intrinsic_matrix = theano.shared(self.intrinsic_matrix)
        warped_depth_map = _depth_warping(intrinsic_matrix, depth_map_1, depth_map_2, translation_vector, rotation_matrix)

        return warped_depth_map

def _depth_warping(intrinsic_matrix, depth_map_1, depth_map_2, translation_vectors, rotation_matrices):
        
    ## Generate same meshgrid for each depth map to calculate value    
    num_batch, height, width, channels = depth_map_1.shape


    x_grid = K.dot(T.ones((height, 1), dtype='float32'),
                   K.reshape(K.arange(width, dtype='float32'), (1, -1)))

    y_grid = K.dot(K.reshape(K.arange(height, dtype='float32'), (-1, 1)),
            T.ones((1, width), dtype='float32'))

    x_grid = K.reshape(x_grid, (1, height, width, 1))
    y_grid = K.reshape(y_grid, (1, height, width, 1))

    x_grid = K.repeat_elements(x_grid, num_batch, axis = 0)
    y_grid = K.repeat_elements(y_grid, num_batch, axis = 0)
    
    rotation_matrices_inverse = rotation_matrices.dimshuffle((0, 2, 1))
    
    intrinsic_matrix_inverse = matrix_inverse(intrinsic_matrix)


    ## Calculate parameters for warping
    temp_mat, updates = theano.scan(fn=lambda rotation_I, intrinsic_mat: T.dot(intrinsic_mat, rotation_I),
                              outputs_info=None,
                              sequences=[rotation_matrices_inverse],
                              non_sequences=[intrinsic_matrix])
    
    W = T.batched_dot(temp_mat, -translation_vectors)
    
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
    
    # masked_depth_map_2_calculate, updates = theano.scan(fn=lambda x, y: K.switch(y > 1.0e-7, x, 1.0e20),
    #       outputs_info=None,
    #       sequences=[depth_map_2_calculate, depth_map_1])
        
    u_2, updates = theano.scan(fn=lambda W, M, u, v, z_1, z_2_calculate, : (z_1 * (M[0, 0] * u + M[0, 1] * v + M[0, 2]) + W[0, 0]) / z_2_calculate,
                      outputs_info=None,
                      sequences=[W, M, x_grid, y_grid, depth_map_1, depth_map_2_calculate]) #masked_depth_map_2_calculate
    
    v_2, updates = theano.scan(fn=lambda W, M, u, v, z_1, z_2_calculate, : (z_1 * (M[1, 0] * u + M[1, 1] * v + M[1, 2]) + W[1, 0]) / z_2_calculate,
                      outputs_info=None,
                      sequences=[W, M, x_grid, y_grid, depth_map_1, depth_map_2_calculate]) #masked_depth_map_2_calculate
    
    depth_map_1_calculate, updates = theano.scan(fn=lambda W_2, M_2, u, v, z_2, : W_2[2, 0] + z_2 * (M_2[2, 0] * u + M_2[2, 1] * v + M_2[2, 2]),
                      outputs_info=None,
                      sequences=[W_2, M_2, x_grid, y_grid, depth_map_2])

    
    # masked_depth_map_1_calculate, updates = theano.scan(fn=lambda x, y: K.switch(y > 1.0e-7, x, 0),
    #       outputs_info=None,
    #       sequences=[depth_map_1_calculate, depth_map_2])
    
    u_2_flat = u_2.flatten()
    v_2_flat = v_2.flatten()

    depth_map_1_transformed_flat = _interpolate(depth_map_1_calculate, u_2_flat, v_2_flat)  #masked_depth_map_1_calculate


    return K.reshape(depth_map_1_transformed_flat,
                   (num_batch, height, width, channels))





    ## Experiments














    
    
def _repeat(x, n_repeats):
    rep = T.ones((n_repeats,), dtype='int32').dimshuffle('x', 0)
    x = K.dot(x.reshape((-1, 1)), rep)
    return x.flatten()


def _interpolate(im, x, y):
    # constants
    num_batch, height, width, channels = im.shape
    zero = K.zeros([], dtype='int64')
    max_y = K.cast(im.shape[1] - 1, 'int64')
    max_x = K.cast(im.shape[2] - 1, 'int64')

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

