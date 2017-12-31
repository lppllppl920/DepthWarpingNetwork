# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 21:58:40 2017

@author: DELL1
"""

import theano
import theano.tensor as T
from keras import backend as K
from keras.engine.topology import Layer
from theano.tensor.nlinalg import matrix_inverse

class SpecularityMaskingLayer(Layer):
    
    def __init__(self, threshold = 2.0, **kwargs):
        super(SpecularityMaskingLayer, self).__init__(**kwargs)
        self.threshold = threshold
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 1)

    def call(self, x):
        v_x = x[:, :,:,1]
        num_batch, height, width = v_x.shape
        v_x = v_x.reshape((num_batch, height, width, 1))
        
        s_x = x[:, :,:,0]
        num_batch, height, width = s_x.shape
        s_x = s_x.reshape((num_batch, height, width, 1))
#        masked_depth_x = K.switch((v_x * 255.0) / (1.0 + s_x * 255.0) > self.threshold, 0, depth_x)
        specularity_mask, updates = theano.scan(fn=lambda v_x_, s_x_: K.switch((v_x_ * 255.0) / (1.0 + s_x_ * 255.0) > self.threshold, 0, 1.0),
                  outputs_info=None,
                  sequences=[v_x, s_x])
        return specularity_mask