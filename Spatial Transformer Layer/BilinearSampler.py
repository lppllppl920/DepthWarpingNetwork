import theano.tensor as T
from keras import backend as K
from keras.engine.topology import Layer

class BilinearSamplerLayer(Layer):

    def __init__(self,  **kwargs):
        super(BilinearSamplerLayer, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def call(self, x, mask=None):
        image_input, appearance_flow = x
        output =_sampling(image_input, appearance_flow)
        return output


##########################
#    TRANSFORMER LAYERS  #
##########################


def _repeat(x, n_repeats):
    rep = T.ones((n_repeats,), dtype='int32').dimshuffle('x', 0)
    x = K.dot(x.reshape((-1, 1)), rep)
    return x.flatten()


def _interpolate(im, x, y):
    # constants
    num_batch, height, width, channels = im.shape
    height_f = K.cast(height, 'float32')
    width_f = K.cast(width, 'float32')
    zero = K.zeros([], dtype='int64')
    max_y = K.cast(im.shape[1] - 1, 'int64')
    max_x = K.cast(im.shape[2] - 1, 'int64')

    # scale indices from [-1, 1] to [0, width/height]    
    x = (x + 1.0)*(width_f) / 2.0
    y = (y + 1.0)*(height_f) / 2.0

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


def _linspace(start, stop, num):
    # produces results identical to:
    # np.linspace(start, stop, num)
    start = K.cast(start, 'float32')
    stop = K.cast(stop, 'float32')
    num = K.cast(num, 'float32')
    step = (stop-start)/(num-1)
    return T.arange(num, dtype='float32')*step+start

def _sampling(image_input, appearance_flow):
    num_batch, height, width, channels = appearance_flow.shape
    
    
    appearance_flow = K.reshape(appearance_flow, (-1, channels))
    
    ## Separate X and Y coordinate
    x_delta, y_delta = appearance_flow[:, 0], appearance_flow[:, 1]
    x_delta_flat = x_delta.flatten()
    y_delta_flat = y_delta.flatten()
    
    
    x_base = K.dot(T.ones((height, 1)),
                _linspace(-1.0, 1.0, width).dimshuffle('x', 0))
    y_base = K.dot(_linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
                T.ones((1, width)))

    x_base = x_base.reshape((1, height, width))
    y_base = y_base.reshape((1, height, width))
    
    x_base = K.repeat_elements(x_base, num_batch, axis = 0)
    y_base = K.repeat_elements(y_base, num_batch, axis = 0)
#    x_base = K.repeat(x_base, num_batch)
#    y_base = K.repeat(y_base, num_batch)
#    
#    x_base = x_base.dimshuffle(1, 0, 2)
#    y_base = y_base.dimshuffle(1, 0, 2)
    
    x_base_flat = x_base.flatten()
    y_base_flat = y_base.flatten()
    
    x_s_flat = 0.3 * x_delta_flat + x_base_flat
    y_s_flat = 0.3 * y_delta_flat + y_base_flat
    
    input_transformed = _interpolate(image_input, x_s_flat, y_s_flat)
    
    num_batch, height, width, channels = image_input.shape
    output = K.reshape(input_transformed,
                   (num_batch, height, width, channels))
    return output