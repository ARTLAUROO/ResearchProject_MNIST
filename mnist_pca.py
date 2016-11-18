from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

import sys
import os

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

import tensorflow as tf

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

  return weights, biases, cumsum[-1]


def generate_pca_plots():
  train_dirs =  [x[0] for x in os.walk(mnist.CHECKPOINT_DIR)]
  train_dirs = train_dirs[1:] # skip base dir

  for train_dir in train_dirs:
    train_ckpts = os.listdir(train_dir)

    for train_ckpt in train_ckpts:
      if 'meta' in train_ckpt or 'checkpoint' in train_ckpt:
        continue
      generate_pca_plot(train_dir + '/' + train_ckpt)


def generate_pca_plot(ckpt_path):
  print('Generating plot for %s' % ckpt_path)
  errors, cumsums = eval_pca(ckpt_path)

  settings = ckpt_path.split('/')
  dir_name = settings[-2]
  file_name = settings[-1]

  plt.figure()
  plt.ylabel('Percentage')
  plt.xlabel('N PCA components')
  plt.axis([-0.5, len(errors), -2, 102])
  plt.grid(True)
  error_points, = plt.plot(errors, 'ro', label='Error on test set')
  cumsum_points, = plt.plot(cumsums, 'bx', label='Kernel variance kept')

  plt.legend(bbox_to_anchor=(0.9, 0.85),
             bbox_transform=plt.gcf().transFigure,
             handler_map={error_points: HandlerLine2D(numpoints=1),
                          cumsum_points: HandlerLine2D(numpoints=1)})

  #plt.show()
  if not os.path.exists(mnist.PLOT_DIR + dir_name):
    os.makedirs(mnist.PLOT_DIR + dir_name)
  plt.savefig(mnist.PLOT_DIR  + dir_name + '/' + file_name + '.jpg')


def eval_pca(ckpt_path):
    errors = []
    cumsums = []
    pca_range = 32

    with tf.Graph().as_default():

      # Load settings from dir name
      dir_name = ckpt_path.split('/')
      dir_name = dir_name[-2]  # drop path prefix
      convl_settings, dense_settings = mnist_eval.get_settings_from_name(dir_name)

      # Construct model
      data_batch = tf.placeholder(mnist.data_type(),
                                  shape=(mnist.BATCH_SIZE,
                                         mnist.IMAGE_SIZE,
                                         mnist.IMAGE_SIZE,
                                         mnist.NUM_CHANNELS))
      logits = mnist.model(data_batch,
                           convl_settings,
                           dense_settings,
                           mnist.NUM_LABELS,
                           False)  # eval model

      prediction = tf.nn.softmax(logits)

      saver = tf.train.Saver()

      with tf.Session() as sess:
          saver.restore(sess, ckpt_path)

          # apply PCA
          with tf.variable_scope('conv1') as scope:
              scope.reuse_variables()

              variable_w = tf.get_variable('weights')
              variable_b = tf.get_variable('biases')

              original_w = variable_w.eval()
              original_b = variable_b.eval()

              data, labels = input.data(False)
              for i in xrange(pca_range):
                  n_components = i + 1
                  weights, biases, cumsum = pca_reduction(original_w,
                                                          original_b,
                                                          n_components)
                  sess.run(variable_w.assign(weights))
                  sess.run(variable_b.assign(biases))
                  cumsums.append(cumsum)

                  print("PCA:%d" % i)
                  predictions = mnist_eval.eval_in_batches(data,
                                                           sess,
                                                           data_batch,
                                                           prediction)
                  error = mnist.error_rate(predictions, labels)
                  print('PCA %d error: %.2f%%' % (n_components, error))
                  errors.append(error)
    return errors, cumsums


def pca(ckpt_path, convs, locals):
  print("PCA for ckpt file: %s" % ckpt_path)

  with tf.Graph().as_default():
    test_data, test_labels = input.data(False)

    # Build up model in order to load/save variables and apply pca
    eval_data = tf.placeholder(1,
                               shape=(mnist.BATCmnist.data_type(),
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
  n_args = len(sys.argv)
  if n_args is 2:
    generate_pca_plot(sys.argv[1])
  else:
    generate_pca_plots()