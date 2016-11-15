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


def eval(ckpt_path):
  with tf.Graph().as_default():
    test_data, test_labels = input.data(False)

    eval_data = tf.placeholder(mnist.data_type(),
                               shape=(mnist.BATCH_SIZE,
                                      mnist.IMAGE_SIZE,
                                      mnist.IMAGE_SIZE,
                                      mnist.NUM_CHANNELS))

    # TODO load settings from file name
    settings = ckpt_path.split('/')
    settings = settings[-2] # drop path prefix
    settings = settings.split('_')
    settings = settings[-1]  # drop filename prefix
    settings = settings.split('-')

    N_KERNELS_LAYER_1 = int(settings[1])
    if len(settings) is 4:
      N_KERNELS_LAYER_2 = None
    elif len(settings) is 5:
      N_KERNELS_LAYER_2 = int(settings[2])
    else:
      print(settings)
      assert False
    N_NODES_FULL_LAYER = int(settings[-1])

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
  if len(sys.argv) is not 5:
    print('invalid number of arguments')
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    exit()

  TWO_LAYERS = sys.argv[1] == 'True'
  N_KERNELS_LAYER_1 = int(sys.argv[2])
  N_KERNELS_LAYER_2 = int(sys.argv[3])
  N_NODES_FULL_LAYER = int(sys.argv[4])

  # use last ckpt file from training
  n_steps = int(mnist.NUM_EPOCHS * train_size) // mnist.BATCH_SIZE
  ckpt_path = mnist.CHECKPOINT_DIR + mnist.CHECKPOINT_FILENAME + '-' + str(
    n_steps)
  eval(ckpt_path)