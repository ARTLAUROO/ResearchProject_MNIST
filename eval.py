# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
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

import matplotlib as mp
import matplotlib.pyplot as plt

import numpy as np
from sklearn.decomposition import PCA

from PIL import Image
from numpy import linalg as LA
import pylab as pl

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = '/tmp/mnist/data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
SAVE_FREQUENCY = 1000
N_FIRST_LAYER_FILTERS = 32

CKPT_PATH = "/tmp/mnist/ckpts/"
CKPT_PREFIX = "mnist_pca_3-"
CKPT_IDS = [
  "0",
  "1000",
  "2000",
  "3000",
  "4000",
  "5000",
  "6000",
  "7000",
  "8000",
  "9000",
  "9999"
]
VIS_DIR = "visualizations/"

tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            "Use half floats instead of full floats if True.")
FLAGS = tf.app.flags.FLAGS

def to_array_shaped (var_shaped_w, var_shaped_b):
  rect_shaped = np.zeros([N_FIRST_LAYER_FILTERS, (5*5)+1])
  for kernel_i in xrange(N_FIRST_LAYER_FILTERS):
    for i in xrange(5):
      for j in xrange(5):
        rect_shaped[kernel_i, (i*5)+j] = var_shaped_w[i, j, 0, kernel_i]
    rect_shaped[kernel_i, (5*5)] = var_shaped_b[kernel_i]
  return rect_shaped

def to_var_shaped (rect_shaped):
  var_shaped_w = np.zeros([5, 5, 1, N_FIRST_LAYER_FILTERS])
  var_shaped_b = np.zeros([N_FIRST_LAYER_FILTERS])
  for kernel_i in xrange(N_FIRST_LAYER_FILTERS):
    for i in xrange(5):
      for j in xrange(5):
        var_shaped_w[i, j, 0, kernel_i] = rect_shaped[kernel_i, i * 5 + j]
    var_shaped_b[kernel_i] = rect_shaped[kernel_i, (5*5)]
  return var_shaped_w, var_shaped_b

# variable: [HEIGHT, WIDHT, N_IN, N_OUT]
def extract_kernels(variable, biases):
  array = variable.eval()
  biases_array = biases.eval()
  shape = array.shape
  IN_N = shape[2]
  OUT_N = shape[3]
  HEIGHT = shape[0]
  WIDTH = shape[1]

  for in_idx in xrange(IN_N):
    for out_idx in xrange(OUT_N):
      fltr = np.zeros([HEIGHT, WIDTH], dtype=np.float32)
      for i in xrange(HEIGHT):
        for j in xrange(WIDTH):
          fltr[i, j] = array[i, j, in_idx, out_idx]
      # add bias as a full row under the weights
      bias = biases_array[filter_idx]
      bias_row = FILTER_SIZE * [bias]
      weights_and_bias_row = np.concatenate((filter_array, [bias_row]), axis=0)
      yield(fltr, in_idx, out_idx)

def visualize_matrix(values, name, vis_dir):
  plt.figure(1, figsize=(1,1))
  plt.imshow(values, interpolation="none", cmap="gray")
  plt.savefig(vis_dir + name + '.png')
  print("saved: " + name)

def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32


def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.Size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels


def fake_data(num_images):
  """Generate a fake dataset that matches the dimensions of MNIST."""
  data = numpy.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])


def main(argv=None):  # pylint: disable=unused-argument
  if FLAGS.self_test:
    print('Running self-test.')
    train_data, train_labels = fake_data(256)
    validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
    test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
    num_epochs = 1
  else:
    # Get the data.
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = NUM_EPOCHS
  train_size = train_labels.shape[0]

  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(
      data_type(),
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
  eval_data = tf.placeholder(
      data_type(),
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when we call:
  # {tf.initialize_all_variables().run()}
  conv1_weights = tf.Variable(
      tf.truncated_normal([5, 5, NUM_CHANNELS, N_FIRST_LAYER_FILTERS],  # 5x5 filter, depth N_FIRST_LAYER_FILTERS
                          stddev=0.1,
                          seed=SEED, dtype=data_type()))
  conv1_biases = tf.Variable(tf.zeros([N_FIRST_LAYER_FILTERS], dtype=data_type()))
  conv2_weights = tf.Variable(tf.truncated_normal(
      [5, 5, N_FIRST_LAYER_FILTERS, 64], stddev=0.1,
      seed=SEED, dtype=data_type()))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
  fc1_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                          stddev=0.1,
                          seed=SEED,
                          dtype=data_type()))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))
  fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=data_type()))
  fc2_biases = tf.Variable(tf.constant(
      0.1, shape=[NUM_LABELS], dtype=data_type()))

  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  def model(data, train=False):

    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, train_labels_node))

  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
  # Add the regularization term to the loss.
  loss += 5e-4 * regularizers

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0, dtype=data_type())
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      0.01,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)
  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       global_step=batch)

  # Predictions for the current training minibatch.
  train_prediction = tf.nn.softmax(logits)

  # Predictions for the test and validation, which we'll compute less often.
  eval_prediction = tf.nn.softmax(model(eval_data))

  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  saver = tf.train.Saver(max_to_keep=None)

  # plt.figure(1, figsize=[45,7.2])
  # plt_layout = 25 * 10
  # plt_id = 1
  # plt.suptitle('1st layer, n = ' + str(N_FIRST_LAYER_FILTERS))

  random_data = np.random.rand(N_FIRST_LAYER_FILTERS,25)

  def plot_errors(errors, cumsums, ckpt_id,  display=False, save=True):
    name = '1st layer, n_filters: ' + str(N_FIRST_LAYER_FILTERS) + " ckpt_id: " + ckpt_id
    plt.figure()
    plt.title(name)
    plt.ylabel('test error %')
    plt.xlabel('pca components')
    plt.axis([-0.1, len(errors) - 0.9, -1, 100])
    plt.grid(True)
    plt.plot(errors, 'ro')
    plt.plot(cumsums, 'rx')
    if display:
      plt.show()
    if save:
      plt.savefig(name + '.jpg')

  def plot_cumsums(cumsums, ckpt_id):
    plt.figure()
    plt.title('1st layer, n_filters: ' + str(N_FIRST_LAYER_FILTERS) + " ckpt_id: " + ckpt_id)
    plt.ylabel('cumsum ratio explained variance')
    plt.xlabel('pca components')
    plt.axis([-0.1, len(cumsums) - 0.9, -1, 100])
    plt.grid(True)
    plt.plot(cumsums, 'o')
    plt.show()

  for ckpt_id in CKPT_IDS:
    # Create a local session to run the training.
    start_time = time.time()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
      
      saver.restore(sess, CKPT_PATH + CKPT_PREFIX + ckpt_id)
      print('Loaded %s' % ckpt_id)

      w_original = conv1_weights.eval()
      b_original = conv1_biases.eval()
      errors = []
      cumsums = []

      for i in xrange(32):
        pca = PCA(n_components=i+1)
        data = to_array_shaped(w_original, b_original)
        data = np.transpose(data)
        final_data = pca.fit_transform(data)
        new_data = pca.inverse_transform(final_data)
        
        if i+1 == 32:
          cumsums = np.cumsum(pca.explained_variance_ratio_) * 100
        
        # NOISE
        # _, variance_free_shape = new_data[:, i+1:32].shape
        # noise = tf.truncated_normal(
        #   [(5*5)+1, variance_free_shape], 
        #   stddev=0.1, 
        #   seed=SEED, 
        #   dtype=data_type()
        # ).eval()
        # print('noise shape')
        # print(noise.shape)
        # new_data[:, i+1:32] = noise

        new_data = np.transpose(new_data)
        w, b = to_var_shaped(new_data)
        sess.run(conv1_weights.assign(w))
        sess.run(conv1_biases.assign(b))

        # Finally print the result!
        test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
        errors.append(test_error)
        print('%d Test error: %.2f%%' % (i, test_error))
        if FLAGS.self_test:
          print('test_error', test_error)
          assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
              test_error,)
      # plot_cumsums(cumsums, ckpt_id)
      plot_errors(errors, cumsums, ckpt_id)
      

      # PRINT STATS
      # print("data")
      # print(data)

      # print("data shape")
      # print(data.shape)

      # print("final_data shape")
      # print(final_data.shape)

      # print("components")
      # print(pca.components_)

      # print("explained variance")
      # print(pca.explained_variance_)

      # print("explained variance ratio")
      # print(pca.explained_variance_ratio_)

      # print("explained variance ratio cumsum")
      # print(np.cumsum(pca.explained_variance_ratio_))

      # print("weights original")
      # print(conv1_weights.eval())

      # print("weights original shape")
      # print(tf.shape(conv1_weights).evalweights_array_to_rect_shape()


      # print("weights transformed")
      # print(conv1_weights.eval())

      # create plots
      # plt.subplot(plt_layout + plt_id)
      # plt_id = plt_id + 1
      # cumsum = np.cumsum(pca.explained_variance_ratio_)
      # cumsum[24] = 1
      # plt.plot(cumsum)

      # pca.fit(random_data)
      # plt.plot(np.cumsum(pca.explained_variance_ratio_))
      
      

  #     plt.xticks(np.arange(1, 26, 1))
  #     plt.yticks(np.arange(0, 1.1, 0.1))
  #     plt.xlabel('Dimensions')
  #     plt.ylabel('%')
  #     plt.grid(True)
  #     plt.title('error: ' + str(test_error))

  #     plt.savefig('graph.png')
  # plt.show()


if __name__ == '__main__':
  tf.app.run()
