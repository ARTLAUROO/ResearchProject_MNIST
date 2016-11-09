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

# We will replicate the model structure for the training subgraph, as well
# as the evaluation subgraphs, while sharing the trainable parameters.
def model(data, conv_weights, conv_biases, fc_weights, fc_biases, train=False):
  """The Model definition."""
  # 2D convolution, with 'SAME' padding (i.e. the output feature map has
  # the same size as the input). Note that {strides} is a 4D array whose
  # shape matches the data layout: [image index, y, x, depth].

  n_layers = len(conv_weights)
  assert n_layers >= 1 and n_layers == len(conv_biases)
  assert len(fc_weights) == len(fc_biases)

  with tf.variable_scope('conv1') as scope:
    # add first conv layer
    conv = tf.nn.conv2d(data,
                        conv_weights[0],
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases[0]))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

  # add subsequent conv layers
  if n_layers > 1:
    with tf.variable_scope('conv2') as scope:
      conv = tf.nn.conv2d(pool,
                          conv_weights[1],
                          strides=[1, 1, 1, 1],
                          padding='SAME')
      relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases[1]))
      pool = tf.nn.max_pool(relu,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')

  with tf.variable_scope('local1') as scope:
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc_weights[0]) + fc_biases[0])
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)

  with tf.variable_scope('softmax_linear') as scope:
    softmax_linear = tf.matmul(hidden, fc_weights[1]) + fc_biases[1]

  return softmax_linear