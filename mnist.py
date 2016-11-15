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
EVAL_BATCH_SIZE = BATCH_SIZE
EVAL_FREQUENCY = 100        # Number of evaluations for an entire run.
SAVE_FREQUENCY = 10

# Constants describing the training process.
DECAY_STEP_SIZE = 60000 # TODO == TRAIN_SIZE

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