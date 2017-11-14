# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:50:12 2017

@author: DELL1
"""

import keras

class AlternativeUpdating(keras.callbacks.Callback):
    def __init__(self, epoch_count):
        keras.callbacks.Callback.__init__(self)
        self.epoch_count = epoch_count

    def on_train_begin(self, logs={}):
        return
         
    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        if((epoch + 1) % self.epoch_count == 0):
            self.model.layers[3].set_weights(self.model.layers[2].get_weights())
            self.model.layers[7].set_weights(self.model.layers[6].get_weights())
            self.model.layers[11].set_weights(self.model.layers[10].get_weights())
            self.model.layers[13].set_weights(self.model.layers[12].get_weights()) 
            self.model.layers[17].set_weights(self.model.layers[16].get_weights())
            self.model.layers[21].set_weights(self.model.layers[20].get_weights())
            print("weights updated")
        return
    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
