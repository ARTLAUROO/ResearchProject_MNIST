from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

SEED = None
NUM_CHANNELS = 1 # TODO get from input/flags
IMAGE_SIZE = 28 # TODO get from input/flags

# We will replicate the model structure for the training subgraph, as well
# as the evaluation subgraphs, while sharing the trainable parameters.
def model(data, N_KERNELS_LAYER_1, N_KERNELS_LAYER_2, N_NODES_FULL_LAYER, NUM_LABELS, train=False):
  """The Model definition."""
  # 2D convolution, with 'SAME' padding (i.e. the output feature map has
  # the same size as the input). Note that {strides} is a 4D array whose
  # shape matches the data layout: [image index, y, x, depth].

  with tf.variable_scope('conv1') as scope:
    # add first conv layer
    initializer = tf.truncated_normal_initializer(stddev=0.1,
                                                  seed=SEED, 
                                                  dtype=tf.float32)
    kernel = tf.get_variable('weights',
                            [5, 5, NUM_CHANNELS, N_KERNELS_LAYER_1],
                            initializer=initializer)
    conv = tf.nn.conv2d(data,
                        kernel,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    initializer = tf.zeros_initializer([N_KERNELS_LAYER_1], dtype=tf.float32) # TODO use datatype()
    biases = tf.get_variable('biases', initializer=initializer) 
    bias = tf.nn.bias_add(conv, biases)
    relu = tf.nn.relu(bias, name=scope.name)
  
  pool = tf.nn.max_pool(relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        name='pool1')

  with tf.variable_scope('conv2') as scope:
    initializer = tf.truncated_normal_initializer(stddev=0.1,
                                                  seed=SEED, 
                                                  dtype=tf.float32)  # TODO use datatype()
    kernel = tf.get_variable('weights',
                            [5, 5, N_KERNELS_LAYER_1, N_KERNELS_LAYER_2],
                            initializer=initializer)
    conv = tf.nn.conv2d(pool,
                        kernel,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    initializer = tf.constant_initializer(0.1, dtype=tf.float32) # TODO use datatype()
    biases = tf.get_variable('biases',
                            shape=[N_KERNELS_LAYER_2],
                            initializer=initializer) 
    bias = tf.nn.bias_add(conv, biases)
    relu = tf.nn.relu(bias, name=scope.name)

  pool = tf.nn.max_pool(relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        name='pool2')

  if not True:
    fc_size = IMAGE_SIZE // 2 * IMAGE_SIZE // 2 * N_KERNELS_LAYER_1
  else:
    fc_size = IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * N_KERNELS_LAYER_2

  with tf.variable_scope('local1') as scope:
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    initializer = tf.truncated_normal_initializer(stddev=0.1,
                                                  seed=SEED,
                                                  dtype=tf.float32) # TODO use datatype()
    weights = tf.get_variable('weights', 
                              [fc_size, N_NODES_FULL_LAYER], 
                              initializer=initializer) 
    initializer = tf.constant_initializer(0.1, dtype=tf.float32) # TODO use datatype()
    biases = tf.get_variable('biases',
                            shape=[N_NODES_FULL_LAYER],
                            initializer=initializer) 
    local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      local1 = tf.nn.dropout(local1, 0.5, seed=SEED)

  with tf.variable_scope('softmax_linear') as scope:
    initializer = tf.truncated_normal_initializer(stddev=0.1,
                                                  seed=SEED,
                                                  dtype=tf.float32) # TODO use datatype()
    weights = tf.get_variable('weights', 
                              shape=[N_NODES_FULL_LAYER, NUM_LABELS],
                              initializer=initializer)
    initializer = tf.constant_initializer(0.1, dtype=tf.float32) # TODO use datatype()
    biases = tf.get_variable('biases',
                              shape=[NUM_LABELS], 
                              initializer=initializer)
    softmax_linear = tf.add(tf.matmul(local1, weights), biases, name=scope.name)

  return softmax_linear