from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
import numpy

import train


PLOT_DIR = '/home/s1259008/research_project/tmp/mnist/plots/'

TRAIN_SIZE = 60000            # Size of the training set.
TEST_SIZE = 10000             # Size of the test set.
VALIDATION_SIZE = 5000        # Size of the validation set.
SEED = None
USE_FP16 = False

N_LABELS = 10
IMAGE_SIZE = 28
PIXEL_DEPTH = 255
N_CHANNELS = 1

# Constants describing the training process.
DECAY_STEP_SIZE = 60000  # TODO == TRAIN_SIZE


def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if USE_FP16:
    return tf.float16
  else:
    return tf.float32


# We will replicate the inference structure for the training subgraph, as well
# as the evaluation subgraphs, while sharing the trainable parameters.
def inference(data, conv_settings, full_settings, n_labels, dropout_pl):
  """
  Applies an inference model to data. The model is shape according to the
  settings defined by conv_settings and full_settings. The rate of dropout is
  controlled with the dropout_pl
  :param data: MNIST images op
  :param conv_settings: list with ints denoting the conv sizes
  :param full_settings: list with ints denoting the full layer sizes
  :param n_labels: In case of MNISt 10
  :param dropout_pl: Controls the dropout applied
  :return: logits
  """
  assert len(conv_settings) > 0 and len(full_settings) > 0

  tf.image_summary('input', data, max_images=3, collections=None, name=None)

  # 2D convolution, with 'SAME' padding (i.e. the output feature map has
  # the same size as the input). Note that {strides} is a 4D array whose
  # shape matches the data layout: [image index, y, x, depth].

  # Add first convl layer
  with tf.variable_scope('conv1') as scope:
      initializer = tf.truncated_normal_initializer(stddev=0.1,
                                                    seed=SEED,
                                                    dtype=tf.float32)
      kernel = tf.get_variable('weights',
                               [5, 5, N_CHANNELS, conv_settings[0]],
                               initializer=initializer)
      conv = tf.nn.conv2d(data,
                          kernel,
                          strides=[1, 1, 1, 1],
                          padding='SAME')
      initializer = tf.zeros_initializer([conv_settings[0]], dtype=data_type())
      biases = tf.get_variable('biases', initializer=initializer)
      bias = tf.nn.bias_add(conv, biases)
      relu = tf.nn.relu(bias, name=scope.name)

  pool = tf.nn.max_pool(relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        name='pool1')

  # tensor = tf.split(3, conv_settings[0], pool, name='split')
  # for i in xrange(len(tensor)):
  #     tf.image_summary('conv1_kernel-' + str(i),
  #                      tensor[i],
  #                      max_images=3,
  #                      collections=None,
  #                      name=None)

  # Add second convl layer
  if len(conv_settings) > 1:
      with tf.variable_scope('conv2') as scope:
          initializer = tf.truncated_normal_initializer(stddev=0.1,
                                                        seed=SEED,
                                                        dtype=data_type())
          kernel = tf.get_variable('weights',
                                   [5, 5, conv_settings[0], conv_settings[1]],
                                   initializer=initializer)
          conv = tf.nn.conv2d(pool,
                              kernel,
                              strides=[1, 1, 1, 1],
                              padding='SAME')
          initializer = tf.constant_initializer(0.1, dtype=data_type())
          biases = tf.get_variable('biases',
                                   shape=[conv_settings[1]],
                                   initializer=initializer)
          bias = tf.nn.bias_add(conv, biases)
          relu = tf.nn.relu(bias, name=scope.name)

      pool = tf.nn.max_pool(relu,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool2')

  # Add first dense layer
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
                                                    dtype=data_type())
      # img height/width after pooling, note each convl layer is followed by a
      # single pool layer
      img_height = (IMAGE_SIZE // (2 * len(conv_settings)))
      img_width = (IMAGE_SIZE // (2 * len(conv_settings)))
      img_size = img_width * img_height
      # convl_sizes[-1] images are produced by the last convl layer, each pixel
      # in those images is connected with each node in the dense layer
      fc_size = conv_settings[-1] * img_size
      weights = tf.get_variable('weights',
                                [fc_size, full_settings[0]],
                                initializer=initializer)
      initializer = tf.constant_initializer(0.1, dtype=data_type())
      biases = tf.get_variable('biases',
                               shape=[full_settings[0]],
                               initializer=initializer)
      local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
      # Add a 50% dropout during training only. Dropout also scales
      # activations such that no rescaling is needed at evaluation time.

  with tf.name_scope('dropout'):
    local1 = tf.nn.dropout(local1, dropout_pl, seed=SEED)

  # Add final softmax layer
  with tf.variable_scope('softmax_linear') as scope:
      initializer = tf.truncated_normal_initializer(stddev=0.1,
                                                    seed=SEED,
                                                    dtype=data_type())
      weights = tf.get_variable('weights',
                                shape=[full_settings[0], n_labels],
                                initializer=initializer)
      initializer = tf.constant_initializer(0.1, dtype=data_type())
      biases = tf.get_variable('biases',
                               shape=[n_labels],
                               initializer=initializer)
      softmax_linear = tf.add(tf.matmul(local1, weights),
                              biases,
                              name=scope.name)

  return softmax_linear


def loss(logits, labels):
  loss_value = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))

  tf.get_variable_scope().reuse_variables()
  regularizers = (tf.nn.l2_loss(tf.get_variable('local1/weights'))
                  + tf.nn.l2_loss(tf.get_variable('local1/biases'))
                  + tf.nn.l2_loss(tf.get_variable('softmax_linear/weights'))
                  + tf.nn.l2_loss(tf.get_variable('softmax_linear/biases')))
  # Add the regularization term to the loss_value.
  loss_value += 5e-4 * regularizers
  tf.scalar_summary('loss_value', loss_value)

  return loss_value


def training(loss_value, batch):
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      0.01,                       # Base learning rate.
      batch * train.BATCH_SIZE,         # Current index into the dataset.
      DECAY_STEP_SIZE,                 # Decay step.
      0.95,                       # Decay rate.
      staircase=True)

  tf.scalar_summary('learning rate', learning_rate)

  # --------- TRAINING: ADJUST WEIGHTS ---------

  # Use simple momentum for the optimization.
  train_op = tf.train.MomentumOptimizer(learning_rate,
                                        0.9).minimize(loss_value,
                                                      global_step=batch)
  return train_op


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])


def generate_ckpt_name(conv_sizes, local_sizes):
  ckpt_name = time.strftime("%d-%b-%Y_%H-%M-%S_", time.gmtime())

  # add kernel layer data
  ckpt_name += 'K'
  for size in conv_sizes:
    string = '-%d' % size
    ckpt_name += string

  # add local layer data
  ckpt_name += '-L'
  for size in local_sizes:
    string = '-%d' % size
    ckpt_name += string

  return ckpt_name


def ckpt_name_to_values(ckpt_name):
  conv = []
  local = []

  # strip date and time
  ckpt_name = ckpt_name.split(':')
  ckpt_name = ckpt_name[-1]

  # strip file extension
  ckpt_name = ckpt_name.split('.')
  ckpt_name = ckpt_name[0]

  data = ckpt_name.split('_')
  parsed_kernels = False
  for d in data:
    print('/' + d + '/')
    if d == 'K':
      pass
    elif d == 'L':
      parsed_kernels = True
    elif not parsed_kernels:
      conv.append(int(d))
    else:
      local.append(int(d))

  return conv, local
