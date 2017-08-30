# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:04:43 2017

@author: DELL1
"""

import theano
import theano.tensor as T
from keras import backend as K
from keras.engine.topology import Layer
from theano.tensor.nlinalg import matrix_inverse

class ArgumentMaskingLayer(Layer):
    
    def __init__(self, mask, **kwargs):
        super(ArgumentMaskingLayer, self).__init__(**kwargs)
        self.mask = mask

    def compute_output_shape(self, input_shapes):
        return input_shapes

#    def get_config(self):
#        config = {'mask': self.mask}
#        base_config = super(ArgumentMaskingLayer, self).get_config()
#        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):
        mask = theano.shared(self.mask)
        masked_x, updates = theano.scan(fn=lambda x, y: K.switch(y < 1.0, 0, x),
                  outputs_info=None,
                  sequences=[x],
                  non_sequences=[mask])
        return masked_x