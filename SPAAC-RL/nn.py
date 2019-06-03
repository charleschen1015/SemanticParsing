# ======================================================================================
# 2019
# Project: Context-Dependent Semantic Parsing
# Author: Charles Chen
# Email: lc971015@ohio.edu
# Paper: Context-Dependent Semantic Parsing over Temporally Structured Data, NAACL 2019.
# ======================================================================================

 
import tensorflow as tf
import tensorflow.contrib.layers as layers

class NN(object):
    def __init__(self, config):
        self.config = config
        self.is_train = True if config.phase == 'train' else False
        self.prepare()

    def prepare(self):
        config = self.config


        self.fc_kernel_initializer = tf.random_uniform_initializer(
            minval = -config.fc_kernel_initializer_scale,
            maxval = config.fc_kernel_initializer_scale)

        if self.is_train and config.fc_kernel_regularizer_scale > 0:
            self.fc_kernel_regularizer = layers.l2_regularizer(
                scale = config.fc_kernel_regularizer_scale)
        else:
            self.fc_kernel_regularizer = None

        if self.is_train and config.fc_activity_regularizer_scale > 0:
            self.fc_activity_regularizer = layers.l1_regularizer(
                scale = config.fc_activity_regularizer_scale)
        else:
            self.fc_activity_regularizer = None

 

    def dense(self,
              inputs,
              units,
              activation = tf.tanh,
              use_bias = True,
              name = None):
        if activation is not None:
            activity_regularizer = self.fc_activity_regularizer
        else:
            activity_regularizer = None
        return tf.layers.dense(
            inputs = inputs,
            units = units,
            activation = activation,
            use_bias = use_bias,
            trainable = self.is_train,
            kernel_initializer = self.fc_kernel_initializer,
            kernel_regularizer = self.fc_kernel_regularizer,
            activity_regularizer = activity_regularizer,
            name = name)

    def dropout(self,
                inputs,
                name = None):
        return tf.layers.dropout(
            inputs = inputs,
            rate = self.config.fc_drop_rate,
            training = self.is_train)

    def batch_norm(self,
                   inputs,
                   name = None):
        return tf.layers.batch_normalization(
            inputs = inputs,
            training = self.train_cnn,
            trainable = self.train_cnn,
            name = name
        )



