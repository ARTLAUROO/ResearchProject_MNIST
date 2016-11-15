from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import gzip
import os

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np
from sklearn.decomposition import PCA

import mnist
import mnist_input as input

# Are overwritten by main args
TWO_LAYERS = None             # If true two conv layers are used, else one
N_KERNELS_LAYER_1 = None
N_KERNELS_LAYER_2 = None
N_NODES_FULL_LAYER = None
SESSION_NAME = None


# TODO multiple channels
def to_array_shaped (var_shaped_w, var_shaped_b):
  """" Constructs array[N_KERNELS, KERNEL_SIZE + bias] from corresponding tf vars """
  height, width, channels, n = var_shaped_w.shape
  flattened_kernel_size = (height * width) + 1
  array = np.ndarray ([channels * n, flattened_kernel_size])

  # fill array with values from kernel
  for kernel_i in xrange(n):
    # fill array with weight values
    for i in xrange (height):
      for j in xrange (width):
        array_idx = (i * width) + j
        array[kernel_i, array_idx] = var_shaped_w[i, j, 0, kernel_i]
    # add to array bias values
    array[kernel_i, -1] = var_shaped_b[kernel_i]

  return array

# TODO multiple channels
def to_var_shaped (array, height, width, channels, n):
  """" Constructs tf vars from array[N_KERNELS, KERNEL_SIZE + bias] """
  variable_w = np.ndarray ([height, width, channels, n])
  variable_b = np.ndarray ([n])

  # fill variable with values from kernel array
  for kernel_i in xrange (n):
    # fill weights
    for i in xrange (height):
      for j in xrange (width):
        array_idx = (i * width) + j
        variable_w[i, j, 0, kernel_i] = array[kernel_i, array_idx]
    # fill biases
    variable_b[kernel_i] = array[kernel_i, -1]

  return variable_w, variable_b


def pca_reduction(weights, biases, n_components):
  pca = PCA(n_components=n_components)
  array = to_array_shaped(weights, biases)

  array = np.transpose(array)
  pca_data = pca.fit_transform(array)

  array = pca.inverse_transform(pca_data)
  array = np.transpose(array)

  # cumsums = np.cumsum(pca.explained_variance_ratio_) * 100

  weights, biases = to_var_shaped(array, 5, 5, 1, N_KERNELS_LAYER_1)

  return weights, biases

def pca():
  with tf.Graph().as_default():
    test_data, test_labels = input.data(False)

    eval_data = tf.placeholder(mnist.data_type(),
                               shape=(mnist.BATCH_SIZE,
                                      mnist.IMAGE_SIZE,
                                      mnist.IMAGE_SIZE,
                                      mnist.NUM_CHANNELS))

    logits = mnist.model(eval_data,
                         N_KERNELS_LAYER_1,
                         N_KERNELS_LAYER_2,
                         N_NODES_FULL_LAYER,
                         mnist.NUM_LABELS,
                         False)  # eval model

    # Predictions for the test and validation, which we'll compute less often.
    eval_prediction = tf.nn.softmax(logits)

    saver = tf.train.Saver(max_to_keep=None)

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
      train_size = mnist.TRAIN_SIZE - mnist.VALIDATION_SIZE
      n_steps = int(mnist.NUM_EPOCHS * train_size) // mnist.BATCH_SIZE
      ckpt_path = mnist.CHECKPOINT_DIR + mnist.CHECKPOINT_FILENAME + '-' + str(n_steps)
      saver.restore(sess, ckpt_path)

      # pca
      with tf.variable_scope('conv1') as scope:
        scope.reuse_variables()

        variable_w = tf.get_variable('weights')
        variable_b = tf.get_variable('biases')

        weights, biases = pca_reduction(variable_w.eval(), variable_b.eval(), 32)

        sess.run(variable_w.assign(weights))
        sess.run(variable_b.assign(biases))

if __name__ == '__main__':
  if len(sys.argv) is not 5:
    print('invalid number of arguments')
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    exit()

  TWO_LAYERS = sys.argv[1] == 'True'
  N_KERNELS_LAYER_1 = int(sys.argv[2])
  N_KERNELS_LAYER_2 = int(sys.argv[3])
  N_NODES_FULL_LAYER = int(sys.argv[4])

  pca()