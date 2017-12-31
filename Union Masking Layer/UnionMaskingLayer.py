# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 21:26:54 2017

@author: DELL1
"""

import theano
import theano.tensor as T
from keras import backend as K
from keras.engine.topology import Layer
from theano.tensor.nlinalg import matrix_inverse

class UnionMaskingLayer(Layer):
    
    def __init__(self, **kwargs):
        super(UnionMaskingLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        input_shape1, input_shape2 = input_shapes
        return input_shape1

    def call(self, x):
        mask_1, mask_2 = x
        union_mask, updates = theano.scan(fn=lambda x, y: K.switch(y < 0.5, 0, x),
                  outputs_info=None,
                  sequences=[mask_1, mask_2])
        return union_mask