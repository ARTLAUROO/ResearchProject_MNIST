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
import mnist_eval

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
  _, _, _, n_kernels_layer_1 = weights.shape

  pca = PCA(n_components=n_components)
  array = to_array_shaped(weights, biases)

  array = np.transpose(array)
  pca_data = pca.fit_transform(array)

  array = pca.inverse_transform(pca_data)
  array = np.transpose(array)

  cumsum = np.cumsum(pca.explained_variance_ratio_) * 100
  print('Total of variance kept: %f.2 in %d components' % (cumsum[-1], n_components))

  weights, biases = to_var_shaped(array, 5, 5, 1, n_kernels_layer_1)

  return weights, biases


def generate_pca_plots():
  train_dirs =  [x[0] for x in os.walk(mnist.CHECKPOINT_DIR)]
  train_dirs = train_dirs[1:] # skip base dir

  for train_dir in train_dirs:
    train_ckpts = os.listdir(train_dir)

    settings = train_dir.split('_')
    settings = settings[-1]
    settings = settings.split('-')

    # TODO hack
    if len(settings) is 4: # single layer
      convs = [int(settings[1])]
    else:
      convs = [int(settings[1]), int(settings[2])]
    locals = [int(settings[-1])]

    for train_ckpt in train_ckpts:
      if 'meta' in train_ckpt or 'checkpoint' in train_ckpt:
        continue
      pca(train_dir + '/' + train_ckpt, convs, locals)



def pca(ckpt_path, convs, locals):
  print("PCA for ckpt file: %s" % ckpt_path)

  with tf.Graph().as_default():
    test_data, test_labels = input.data(False)

    # Build up model in order to load/save variables and apply pca
    eval_data = tf.placeholder(,
                               shape=(mnist.BATCmnist.data_type()H_SIZE,
                                      mnist.IMAGE_SIZE,
                                      mnist.IMAGE_SIZE,
                                      mnist.NUM_CHANNELS))
    N_KERNELS_LAYER_1 = convs[0]
    if len(convs) == 1:
      logits = mnist.model(eval_data,
                           convs[0],
                           None,
                           locals[0],
                           mnist.NUM_LABELS,
                           False)  # eval model
    else:
      logits = mnist.model(eval_data,
                           convs[0],
                           convs[1],
                           locals[0],
                           mnist.NUM_LABELS,
                           False)  # eval model


    # Predictions for the test and validation, which we'll compute less often.
    eval_prediction = tf.nn.softmax(logits)

    saver = tf.train.Saver(max_to_keep=None)

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
      saver.restore(sess, ckpt_path)

      # Print test error before pca
      print("PCA:before")
      mnist_eval.eval_once(ckpt_path)

      # pca
      with tf.variable_scope('conv1') as scope:
        scope.reuse_variables()

        variable_w = tf.get_variable('weights')
        variable_b = tf.get_variable('biases')

        original_w = variable_w.eval()
        original_b = variable_b.eval()

        for i in xrange(1):
          weights, biases = pca_reduction(original_w, original_b, i+1)

          sess.run(variable_w.assign(weights))
          sess.run(variable_b.assign(biases))

          print("PCA:%d" % i)
          test_error = mnist_eval.run(sess,
                                      test_data,
                                      test_labels,
                                      eval_data,
                                      eval_prediction)

        # # Save variables
        # ckpt_path = mnist.CHECKPOINT_DIR + 'pca.ckpt'
        # ckpt_path = saver.save(sess, ckpt_path)
        # print('Saved checkpoint file: %s' % ckpt_path)
        #
        # # Print test error after pca
        # mnist_eval.eval(ckpt_path)

if __name__ == '__main__':
  generate_pca_plots()