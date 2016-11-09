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

import mnist_model as model

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
NUM_EPOCHS = 3
EVAL_BATCH_SIZE = BATCH_SIZE
EVAL_FREQUENCY = 100        # Number of evaluations for an entire run.
SAVE_FREQUENCY = 10

# Are overwritten by main args
TWO_LAYERS = None             # If true two conv layers are used, else one
N_KERNELS_LAYER_1 = None
N_KERNELS_LAYER_2 = None
N_NODES_FULL_LAYER = None
SESSION_NAME = None

tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            "Use half floats instead of full floats if True.")
FLAGS = tf.app.flags.FLAGS

def to_array_shaped (var_shaped_w, var_shaped_b):
  rect_shaped = np.zeros([N_KERNELS_LAYER_1, (5*5)+1])
  for kernel_i in xrange(N_KERNELS_LAYER_1):
    for i in xrange(5):
      for j in xrange(5):
        rect_shaped[kernel_i, (i*5)+j] = var_shaped_w[i, j, 0, kernel_i]
    rect_shaped[kernel_i, (5*5)] = var_shaped_b[kernel_i]
  return rect_shaped

def to_var_shaped (rect_shaped):
  var_shaped_w = np.zeros([5, 5, 1, N_KERNELS_LAYER_1])
  var_shaped_b = np.zeros([N_KERNELS_LAYER_1])
  for kernel_i in xrange(N_KERNELS_LAYER_1):
    for i in xrange(5):
      for j in xrange(5):
        var_shaped_w[i, j, 0, kernel_i] = rect_shaped[kernel_i, i * 5 + j]
    var_shaped_b[kernel_i] = rect_shaped[kernel_i, (5*5)]
  return var_shaped_w, var_shaped_b

def plot_errors(errors, cumsums, ckpt_id,  display=False, save=True):
  name = '1st layer, n_filters: ' + str(N_KERNELS_LAYER_1) + " ckpt_id: " + ckpt_id
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
    # --------- LOAD DATA INTO LISTS ---------

    # Get the data.
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, TRAIN_SIZE)
    train_labels = extract_labels(train_labels_filename, TRAIN_SIZE)
    test_data = extract_data(test_data_filename, TEST_SIZE)
    test_labels = extract_labels(test_labels_filename, TEST_SIZE)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = NUM_EPOCHS

  train_size = train_labels.shape[0]

  # --------- CREATE INPUT PLACEHOLDERS ---------

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

  # --------- CREATE MODEL VARIABLES ---------

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when we call:
  # {tf.initialize_all_variables().run()}
  conv_weights = []
  conv_biases = []

  conv1_weights = tf.Variable(
      tf.truncated_normal([5, 5, NUM_CHANNELS, N_KERNELS_LAYER_1],  # 5x5 filter, depth N_KERNELS_LAYER_1.
                          stddev=0.1,
                          seed=SEED, 
                          dtype=data_type()))
  conv_weights.append(conv1_weights)
  
  conv1_biases = tf.Variable(tf.zeros([N_KERNELS_LAYER_1], dtype=data_type()))
  conv_biases.append(conv1_biases)

  if TWO_LAYERS:
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, N_KERNELS_LAYER_1, N_KERNELS_LAYER_2], 
                            stddev=0.1,
                            seed=SEED, 
                            dtype=data_type()))
    conv_weights.append(conv2_weights)

    conv2_biases = tf.Variable(tf.constant(0.1, 
                                          shape=[N_KERNELS_LAYER_2], 
                                          dtype=data_type()))
    conv_biases.append(conv2_biases)

  fc_weights = []
  fc_biases = []

  if not TWO_LAYERS:
    fc_size = IMAGE_SIZE // 2 * IMAGE_SIZE // 2 * N_KERNELS_LAYER_1
  else:
    fc_size = IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * N_KERNELS_LAYER_2
  fc1_weights = tf.Variable(  # fully connected, depth N_NODES_FULL_LAYER.
      tf.truncated_normal([fc_size, N_NODES_FULL_LAYER],
                          stddev=0.1,
                          seed=SEED,
                          dtype=data_type()))
  fc_weights.append(fc1_weights)

  fc1_biases = tf.Variable(tf.constant(0.1, shape=[N_NODES_FULL_LAYER], dtype=data_type()))
  fc_biases.append(fc1_biases)

  fc2_weights = tf.Variable(tf.truncated_normal([N_NODES_FULL_LAYER, NUM_LABELS],
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=data_type()))
  fc_weights.append(fc2_weights)

  fc2_biases = tf.Variable(tf.constant(
      0.1, shape=[NUM_LABELS], dtype=data_type()))
  fc_biases.append(fc2_biases)

  # --------- TRAINING: LOSS ---------

  # Training computation: logits + cross-entropy loss.
  logits = model.model(train_data_node, conv_weights, conv_biases, fc_weights, fc_biases, True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, train_labels_node))

  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
  # Add the regularization term to the loss.
  loss += 5e-4 * regularizers

  tf.scalar_summary('loss', loss)

  # --------- TRAINING: LEARNING RATE ---------

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

  tf.scalar_summary('learning rate', learning_rate)

  # --------- TRAINING: ADJUST WEIGHTS ---------

  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       global_step=batch)

  # Predictions for the current training minibatch.
  train_prediction = tf.nn.softmax(logits)

  # Predictions for the test and validation, which we'll compute less often.
  eval_prediction = tf.nn.softmax(model.model(eval_data, conv_weights, conv_biases, fc_weights, fc_biases))

  # Variables used for summaries
  validation_error_variable = tf.Variable(100.0)
  validation_error_summary = tf.scalar_summary('Validation Error Rate', 
                                                validation_error_variable)
  test_error_variable = tf.Variable(100.0)
  test_error_summary = tf.scalar_summary('Test Error Rate', test_error_variable)

  # Add ops to save and restore all the variables.
  saver = tf.train.Saver(max_to_keep=None)

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


  if True: # TRAIN
    # Create a local session to run the training.
    start_time = time.time()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
      merged = tf.merge_all_summaries()
      
      # Save summary of each separate run in a different dir indicated by datetime 
      def gen_summary_path():
        datetime = time.strftime("%d %b %Y %H:%M:%S", time.gmtime())
        summary_path = TENSORBOARD_DIRECTORY + '/' + datetime
        summary_path += ' L1:%d' % N_KERNELS_LAYER_1
        if TWO_LAYERS:
          summary_path += ' L2:%d' % N_KERNELS_LAYER_2
        summary_path += ' FC:%d' % N_NODES_FULL_LAYER
        return summary_path
      summary_writer = tf.train.SummaryWriter(gen_summary_path(), sess.graph)

      # Run all the initializers to prepare the trainable parameters.
      tf.initialize_all_variables().run()
      print('Initialized!')
      summary = sess.run(test_error_summary)
      # Loop through training steps.
      n_steps = int(num_epochs * train_size) // BATCH_SIZE
      save_frequency = n_steps // SAVE_FREQUENCY
      for step in xrange(n_steps):
        # Compute the offset of the current minibatch in the data.
        # Note that we could use better randomization across epochs.
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
        # This dictionary maps the batch data (as a numpy array) to the
        # node in the graph it should be fed to.
        feed_dict = {train_data_node: batch_data,
                     train_labels_node: batch_labels}
        # Run the graph and fetch some of the nodes.
        _, l, lr, predictions, summary = sess.run(
            [optimizer, loss, learning_rate, train_prediction, merged],
            feed_dict=feed_dict)

        if step % EVAL_FREQUENCY == 0 or step + 1 == n_steps:
          summary_writer.add_summary(summary, step)

          elapsed_time = time.time() - start_time
          start_time = time.time()
          print('Step %d (epoch %.2f), %.1f ms' %
                (step, float(step) * BATCH_SIZE / train_size,
                 1000 * elapsed_time / EVAL_FREQUENCY))
          print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
          print('Minibatch error: %.2f%%' % error_rate(predictions, batch_labels))
          # assign validation error rate to  its corresponding variable to add it
          # to the summary
          validation_error_rate = error_rate(eval_in_batches(validation_data, sess), 
                                              validation_labels)
          print('Validation error: %.2f%%' % validation_error_rate)
          # elaborous way to store the validation summary
          sess.run(validation_error_variable.assign(validation_error_rate))
          validation_summary = sess.run(validation_error_summary)
          summary_writer.add_summary(validation_summary, step)

          sys.stdout.flush()

          summary_writer.add_summary(summary, step)

        if step % save_frequency == 0 or step + 1 == n_steps:
          # Save the variables to disk.
          save_path = saver.save(sess,  
                            CHECKPOINT_DIR + SESSION_NAME, 
                            global_step=step)
          print("Model saved in file: %s" % save_path)

      test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
      # Finally print the result!
      print('Test error: %.2f%%' % test_error)
      # elaborous way to store the test summary
      sess.run(test_error_variable.assign(test_error))
      summary = sess.run(test_error_summary)
      summary_writer.add_summary(summary, n_steps)
      
      if FLAGS.self_test:
        print('test_error', test_error)
        assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
            test_error,)

      summary_writer.close()

  if False: # PCA
    ckpts_ids = ['0',
                '257',
                '514',
                '771',
                '1028',
                '1285',
                '1542',
                '1799',
                '2056',
                '2313',
                '2570',
                '2577']

    for ckpt_id in ckpts_ids:
      with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        load_path = saver.restore(sess, CHECKPOINT_DIR + '08-Nov-2016-23:45:22_L1:32-FC:32-'+ ckpt_id)
        print('loaded ckpt: %s' % load_path)

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

          new_data = np.transpose(new_data)
          w, b = to_var_shaped(new_data)
          sess.run(conv1_weights.assign(w))
          sess.run(conv1_biases.assign(b))

          test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
          errors.append(test_error)
          print('%d Test error: %.2f%%' % (i, test_error))
        
        plot_errors(errors, cumsums, ckpt_id)



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

  def gen_sess_name():
    date_time = time.strftime("%d-%b-%Y-%H:%M:%S", time.gmtime())
    
    model_layout = 'L1:%d' % N_KERNELS_LAYER_1
    if TWO_LAYERS:
      model_layout += '-L2:%d' % N_KERNELS_LAYER_2
    model_layout += '-FC:%d' % N_NODES_FULL_LAYER

    return date_time + "_" + model_layout

  SESSION_NAME = gen_sess_name()
  print('Started session: %s' % SESSION_NAME)
  tf.app.run()
