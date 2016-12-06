from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

import sys
import os

import tensorflow as tf

import numpy as np
from sklearn.decomposition import PCA

import model
import eval
import train


# TODO multiple channels
def to_array_shaped(var_shaped_w, var_shaped_b):
  """"
  Returns array[N_KERNELS, KERNEL_SIZE + bias] derived from  the corresponding
  tf vars var_shaped_w and var_shaped_b
  """
  height, width, channels, n = var_shaped_w.shape
  flattened_kernel_size = (height * width) + 1
  array = np.ndarray([channels * n, flattened_kernel_size])

  # fill array with values from kernel
  for kernel_i in xrange(n):
    # fill array with weight values
    for i in xrange(height):
      for j in xrange(width):
        array_idx = (i * width) + j
        array[kernel_i, array_idx] = var_shaped_w[i, j, 0, kernel_i]
    # add to array bias values
    array[kernel_i, -1] = var_shaped_b[kernel_i]
  return array


# TODO multiple channels
def to_var_shaped(array, height, width, channels, n):
  """" Constructs tf vars from array[N_KERNELS, KERNEL_SIZE + bias] """
  variable_w = np.ndarray([height, width, channels, n])
  variable_b = np.ndarray([n])

  # fill variable with values from kernel array
  for kernel_i in xrange(n):
    # fill weights
    for i in xrange(height):
      for j in xrange(width):
        array_idx = (i * width) + j
        variable_w[i, j, 0, kernel_i] = array[kernel_i, array_idx]
    # fill biases
    variable_b[kernel_i] = array[kernel_i, -1]
  return variable_w, variable_b


def pca_reduction(weights, biases, n_components):
  """
  Applies pca on the weights + biases, n_components is the amount of dimensions
  kept.
  :param weights: combined with weights, pca is applied to their combination
  :param biases: combined with biases, pca is applied to their combination
  :param n_components: number of dimensions that are allowed to contain info
  :return: weights, biases on which pca is applied. Plus the total amount of
  variance kept
  """
  _, _, _, n_kernels_layer_1 = weights.shape

  _pca = PCA(n_components=n_components)
  array = to_array_shaped(weights, biases)
  array = np.transpose(array)

  pca_data = _pca.fit_transform(array)
  array = _pca.inverse_transform(pca_data)

  array = np.transpose(array)
  weights, biases = to_var_shaped(array,
                                  train.CONV_KERNEL_SIZE,
                                  train.CONV_KERNEL_SIZE,
                                  model.N_CHANNELS,
                                  n_kernels_layer_1)

  cumsum = np.cumsum(_pca.explained_variance_ratio_) * 100

  return weights, biases, cumsum[-1]


def create_pca_ckpt(ckpt_path):
  """
  Given a checkpoint file it generates
  train.CONV_KERNEL_SIZE * train.CONV_KERNEL_SIZE = n new checkpoint files. Each
  with same values except for the weights and biases in the first convolutional
  layer. To these values pca is applied, with variance in the dimensions:
  1 .. n.  These new checkpoint files are placed in a new dir, which is created
  at the same location as ckpt_path. These new checkpoints file are named after
  the original checkpoint file plus the pca dimensions and the total variance
  that was kept in them.
  :param ckpt_path: path to checkpoint file to which pca must be applied.
  """
  print('Apply PCA to CKPT: ' + ckpt_path)
  with tf.Graph().as_default(), tf.Session() as sess:
    eval.load_model(ckpt_path, sess)

    saver = tf.train.Saver(max_to_keep=None)

    with tf.variable_scope('conv1') as scope:
      scope.reuse_variables()
      variable_w = tf.get_variable('weights')
      variable_b = tf.get_variable('biases')

      original_w = variable_w.eval()
      original_b = variable_b.eval()

    # Dir to save pca ckpts
    pca_ckpt_dir = ckpt_path + '-pca/'
    if not tf.gfile.Exists(pca_ckpt_dir):
      tf.gfile.MakeDirs(pca_ckpt_dir)

    ckpt_path_splitted = ckpt_path.split('.')
    for i in xrange(25):
      n_components = i + 1
      pca_weights, pca_biases, cumsum_variance = pca_reduction(original_w,
                                                               original_b,
                                                               n_components)

      sess.run(variable_w.assign(pca_weights))
      sess.run(variable_b.assign(pca_biases))

      # <..>/<..>.ckpt-xxx to # <..>/<..>.pca-n_components.ckpt-xxx
      pca_ckpt_path = '{}pca-{}_v-{:.2f}_.{}'.format(pca_ckpt_dir, n_components,
                                                     cumsum_variance,
                                                     ckpt_path_splitted[-1])
      pca_ckpt_file_path = saver.save(sess, pca_ckpt_path)
      print('Saved PCA checkpoint file: {}'.format(pca_ckpt_file_path))


def create_pca_ckpts(ckpt_dir_path):
  """
  Applies the function create_pca_ckpts to all ckpts in the dir denoted by
  ckpt_dir_path
  :param ckpt_dir_path: path to dir containing checkpoints
  """
  print('Evaluating DIR: ' + ckpt_dir_path)
  ckpts = []
  for f in os.listdir(ckpt_dir_path):
    if '.ckpt' in f and '.meta' not in f:
      ckpts.append(path + f)

  for ckpt in sorted(ckpts):
    create_pca_ckpt(ckpt)


if __name__ == '__main__':
  if len(sys.argv) < 2:
    msg = 'usage: python pca.py ckpt/path/file.ckpt ; ' \
          'or: python pca.py ckpt/path/dir/'
    print(msg)
    exit()

  path = sys.argv[1]
  if '.ckpt' in path:
    create_pca_ckpt(path)
  else:
    create_pca_ckpts(path)
