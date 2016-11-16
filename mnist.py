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

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = '/tmp/mnist/data'
TENSORBOARD_DIRECTORY = '/tmp/mnist/tensorboard'
CHECKPOINT_DIR = '/tmp/mnist/ckpts/'
CHECKPOINT_FILENAME = 'mnist.ckpt'
PLOT_DIR = '/tmp/mnist/plots/'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
TRAIN_SIZE = 60000            # Size of the training set.
TEST_SIZE = 10000             # Size of the test set.
VALIDATION_SIZE = 5000        # Size of the validation set.
SEED = None                   # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_FREQUENCY = 100        # Number of evaluations for an entire run.
SAVE_FREQUENCY = 10
USE_FP16 = False

# Constants describing the training process.
DECAY_STEP_SIZE = 60000 # TODO == TRAIN_SIZE

def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if USE_FP16:
    return tf.float16
  else:
    return tf.float32

# We will replicate the model structure for the training subgraph, as well
# as the evaluation subgraphs, while sharing the trainable parameters.
def model(data, convl_sizes, dense_sizes, n_labels, train=False):
    """The Model definition. Currently supports 1/2 convl layers and 1 dense
        layer CNN.

    Keyword arguments:
    data -- Input to the CNN.
    convl_sizes -- List in which int specifies the size of the respective
        convolutional layer, must non-empty. Only accepts 1 or 2 layers
        currently.
    dense_sizes -- List in which int specifies the size of the respective
        dense layer, must non-empty. Only accepts 1 layer currently.
    n_labels -- Number of labels.
    train -- True if models is to be used for training.
    """
    assert len(convl_sizes) > 0 and len(dense_sizes) > 0

    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].

    # add first convl layer
    with tf.variable_scope('conv1') as scope:
        initializer = tf.truncated_normal_initializer(stddev=0.1,
                                                      seed=SEED,
                                                      dtype=tf.float32)
        kernel = tf.get_variable('weights',
                                 [5, 5, NUM_CHANNELS, convl_sizes[0]],
                                 initializer=initializer)
        conv = tf.nn.conv2d(data,
                            kernel,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        initializer = tf.zeros_initializer([convl_sizes[0]], dtype=data_type())
        biases = tf.get_variable('biases', initializer=initializer)
        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias, name=scope.name)

    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME',
                          name='pool1')

    # add second convl layer
    if len(convl_sizes) > 1:
        with tf.variable_scope('conv2') as scope:
            initializer = tf.truncated_normal_initializer(stddev=0.1,
                                                          seed=SEED,
                                                          dtype=data_type())
            kernel = tf.get_variable('weights',
                                     [5, 5, convl_sizes[0], convl_sizes[1]],
                                     initializer=initializer)
            conv = tf.nn.conv2d(pool,
                                kernel,
                                strides=[1, 1, 1, 1],
                                padding='SAME')
            initializer = tf.constant_initializer(0.1, dtype=data_type())
            biases = tf.get_variable('biases',
                                     shape=[convl_sizes[1]],
                                     initializer=initializer)
            bias = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(bias, name=scope.name)

        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name='pool2')

        fc_size = IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * convl_sizes[1]
    else:
        fc_size = IMAGE_SIZE // 2 * IMAGE_SIZE // 2 * convl_sizes[0]

    # add first dense layer
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
        weights = tf.get_variable('weights',
                                  [fc_size, dense_sizes[0]],
                                  initializer=initializer)
        initializer = tf.constant_initializer(0.1, dtype=data_type())
        biases = tf.get_variable('biases',
                                 shape=[dense_sizes[0]],
                                 initializer=initializer)
        local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            local1 = tf.nn.dropout(local1, 0.5, seed=SEED)

    # add final softmax layer
    with tf.variable_scope('softmax_linear') as scope:
        initializer = tf.truncated_normal_initializer(stddev=0.1,
                                                      seed=SEED,
                                                      dtype=data_type())
        weights = tf.get_variable('weights',
                                  shape=[dense_sizes[0], n_labels],
                                  initializer=initializer)
        initializer = tf.constant_initializer(0.1, dtype=data_type())
        biases = tf.get_variable('biases',
                                 shape=[n_labels],
                                 initializer=initializer)
        softmax_linear = tf.add(tf.matmul(local1, weights), biases, name=scope.name)

    return softmax_linear

def loss(logits, labels):
  loss = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))

  tf.get_variable_scope().reuse_variables()
  regularizers = (tf.nn.l2_loss(tf.get_variable('local1/weights')) 
                  + tf.nn.l2_loss(tf.get_variable('local1/biases')) 
                  + tf.nn.l2_loss(tf.get_variable('softmax_linear/weights')) 
                  + tf.nn.l2_loss(tf.get_variable('softmax_linear/biases')))
  # Add the regularization term to the loss.
  loss += 5e-4 * regularizers
  tf.scalar_summary('loss', loss)
  
  return loss

def train(loss, batch):
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      0.01,                       # Base learning rate.
      batch * BATCH_SIZE,         # Current index into the dataset.
      DECAY_STEP_SIZE,                 # Decay step.
      0.95,                       # Decay rate.
      staircase=True)

  tf.scalar_summary('learning rate', learning_rate)

  # --------- TRAINING: ADJUST WEIGHTS ---------

  # Use simple momentum for the optimization.
  train_op = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
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
    str = '-%d' % size
    ckpt_name += str

  # add local layer data
  ckpt_name += '-L'
  for size in local_sizes:
    str = '-%d' % size
    ckpt_name += str

  return ckpt_name

def ckpt_name_to_values(ckptname):
  conv = []
  local = []

  # strip date and time
  ckptname = ckptname.split(':')
  ckptname = ckptname[-1]

  # strip file extension
  ckptname = ckptname.split('.')
  ckptname = ckptname[0]

  data = ckptname.split('_')
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
