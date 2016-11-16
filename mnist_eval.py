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
TWO_LAYERS = True             # If true two conv layers are used, else one
N_KERNELS_LAYER_1 = 32
N_KERNELS_LAYER_2 = 64
N_NODES_FULL_LAYER = 512
SESSION_NAME = None


def get_settings_from_name(name):
    """Extracts settings of CNN from a properly formatted name.

    Keyword arguments:
    name -- Example: 15-Nov-2016_16-26-06_K-32-L-256, each number after the K
            and L represent a layer of that size and type. K is for kernel in
            a convolutional layer, L is for local in a dense layer.

    Return:
    Two lists, first filled with numbers representing conv layers, second filled
    with numbers representing dense layers. From the example: [32] [256]
    """

    settings = name.split('_')
    settings = settings[-1]  # drop date and time prefix
    settings = settings.split('-')

    l_idx = settings.index("L")
    convl = [int(setting) for setting in settings[1:l_idx]] # omit K
    dense = [int(setting) for setting in settings[l_idx+1:]] # omit L

    return convl, dense


def eval(ckpt_path):
  with tf.Graph().as_default():
    test_data, test_labels = input.data(False)

    eval_data = tf.placeholder(mnist.data_type(),
                               shape=(mnist.BATCH_SIZE,
                                      mnist.IMAGE_SIZE,
                                      mnist.IMAGE_SIZE,
                                      mnist.NUM_CHANNELS))

    # TODO load settings from file name
    print(ckpt_path)
    dir_name = ckpt_path.split('/')
    print(dir_name)
    dir_name = dir_name[-2] # drop path prefix
    print(dir_name)

    convl, dense = get_settings_from_name(dir_name)
    print(convl)
    print(dense)

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
      saver.restore(sess, ckpt_path)
      # Print test error
      run(sess, test_data, test_labels, eval_data, eval_prediction)

def run(sess, test_data, test_labels, eval_data, eval_prediction):
  predictions = mnist.eval_in_batches(test_data,
                                      sess,
                                      eval_data,
                                      eval_prediction)
  test_error = mnist.error_rate(predictions, test_labels)
  print('Test error: %.2f%%' % test_error)
  return test_error

if __name__ == '__main__':
  ckpt_path = '/tmp/mnist/ckpts/15-Nov-2016_16-21-13_K-32-64-L-4/mnist.ckpt-1718'
  eval(ckpt_path)