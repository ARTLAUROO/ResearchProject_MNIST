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
N_KERNELS_LAYER_1 = None
N_KERNELS_LAYER_2 = None
N_NODES_FULL_LAYER = None
SESSION_NAME = None

tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            "Use half floats instead of full floats if True.")
FLAGS = tf.app.flags.FLAGS


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


def generate_train_dir(conv_sizes, local_sizes):
  dir_name = time.strftime("%d-%b-%Y_%H-%M-%S_", time.gmtime())

  # add kernel layer data
  dir_name += 'K'
  for size in conv_sizes:
    str = '-%d' % size
    dir_name += str

  # add local layer data
  dir_name += '-L'
  for size in local_sizes:
    str = '-%d' % size
    dir_name += str

  dir_path = mnist.CHECKPOINT_DIR + dir_name
  if not tf.gfile.Exists(dir_path):
    tf.gfile.MakeDirs(dir_path)

  return dir_path


def train():
  with tf.Graph().as_default():
    global_step = tf.Variable(0, dtype=mnist.data_type(), trainable=False)

    # Get the data.
    train_data, train_labels, validation_data, validation_labels = input.data(True)
    train_size = train_labels.shape[0]


    train_data_node = tf.placeholder(mnist.data_type(),
                                     shape=(mnist.BATCH_SIZE,
                                            mnist.IMAGE_SIZE,
                                            mnist.IMAGE_SIZE,
                                            mnist.NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.int64, shape=(mnist.BATCH_SIZE,))

    eval_data = tf.placeholder(mnist.data_type(),
                               shape=(mnist.BATCH_SIZE,
                                      mnist.IMAGE_SIZE,
                                      mnist.IMAGE_SIZE,
                                      mnist.NUM_CHANNELS))

    logits = mnist.model(train_data_node,
                           N_KERNELS_LAYER_1,
                           N_KERNELS_LAYER_2,
                           N_NODES_FULL_LAYER,
                           mnist.NUM_LABELS,
                           True)  # training model

    loss = mnist.loss(logits, train_labels_node)

    train_op = mnist.train(loss, global_step)

    train_prediction = tf.nn.softmax(logits)

    # Predictions for the test and validation, which we'll compute less often.
    eval_prediction = tf.nn.softmax(mnist.model(eval_data,
                                                N_KERNELS_LAYER_1,
                                                N_KERNELS_LAYER_2,
                                                N_NODES_FULL_LAYER,
                                                mnist.NUM_LABELS,
                                                False))

    # Setup saver and related vars
    saver = tf.train.Saver(max_to_keep=None)
    # create dir for ckpts
    if N_KERNELS_LAYER_2 is None:
      conv_sizes = [N_KERNELS_LAYER_1]
    else:
      conv_sizes = [N_KERNELS_LAYER_1, N_KERNELS_LAYER_2]
    local_sizes = [N_NODES_FULL_LAYER]
    train_dir = generate_train_dir(conv_sizes, local_sizes)
    ckpt_path = train_dir + '/mnist.ckpt'

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
      tf.initialize_all_variables().run()

      n_steps = int(mnist.NUM_EPOCHS * train_size) // mnist.BATCH_SIZE
      for step in xrange(n_steps):
        # Compute the offset of the current minibatch in the data.
        # Note that we could use better randomization across epochs.
        offset = (step * mnist.BATCH_SIZE) % (train_size - mnist.BATCH_SIZE)
        batch_data = train_data[offset:(offset + mnist.BATCH_SIZE), ...]
        batch_labels = train_labels[offset:(offset + mnist.BATCH_SIZE)]
        # This dictionary maps the batch data (as a numpy array) to the
        # node in the graph it should be fed to.
        feed_dict = {train_data_node: batch_data,
                     train_labels_node: batch_labels}

        # train
        _, l, predictions = sess.run([train_op, loss, train_prediction],
                                     feed_dict=feed_dict)

        # Print validation error
        if step % mnist.EVAL_FREQUENCY == 0:
          predictions = mnist.eval_in_batches(validation_data,
                                              sess,
                                              eval_data,
                                              eval_prediction)
          validation_error = mnist.error_rate(predictions, validation_labels)
          print('Validation error: %.1f%%' % validation_error)

        # Save variables
        if step % (n_steps // mnist.SAVE_FREQUENCY) == 0:
          saver.save(sess, ckpt_path, global_step=step)
          print('Saved checkpoint file: %s' % ckpt_path)

      # Save variables
      saver.save(sess, ckpt_path, global_step=n_steps)
      print('Saved checkpoint file: %s' % ckpt_path)

      # Print test error
      #TODO



def main(argv=None):  # pylint: disable=unused-argument
  train()
  if FLAGS.self_test:
    print('Running self-test.')
    train_data, train_labels = mnist.fake_data(256)
    validation_data, validation_labels = mnist.fake_data(EVAL_BATCH_SIZE)
    test_data, test_labels = mnist.fake_data(mnist.EVAL_BATCH_SIZE)
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
      mnist.data_type(),
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
  eval_data = tf.placeholder(
      mnist.data_type(),
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  # --------- TRAINING: LOSS ---------

  # Training computation: logits + cross-entropy loss.
  logits = model.model(train_data_node, N_KERNELS_LAYER_1, N_KERNELS_LAYER_2, N_NODES_FULL_LAYER, NUM_LABELS, True)

  loss = model.loss(logits, train_labels_node)
  
  # --------- TRAINING: LEARNING RATE ---------

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  global_step = tf.Variable(0, dtype=mnist.data_type(), trainable=False)
  
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  optimizer = model.train(loss, global_step)
  
  # Predictions for the current training minibatch.
  train_prediction = tf.nn.softmax(logits)

  # Predictions for the test and validation, which we'll compute less often.
  eval_prediction = tf.nn.softmax(model.model(eval_data, N_KERNELS_LAYER_1, N_KERNELS_LAYER_2, N_NODES_FULL_LAYER, NUM_LABELS))

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
        _, l, predictions, summary = sess.run(
            [optimizer, loss, train_prediction, merged],
            feed_dict=feed_dict)

        if step % EVAL_FREQUENCY == 0 or step + 1 == n_steps:
          summary_writer.add_summary(summary, step)

          elapsed_time = time.time() - start_time
          start_time = time.time()
          print('Step %d (epoch %.2f), %.1f ms' %
                (step, float(step) * BATCH_SIZE / train_size,
                 1000 * elapsed_time / EVAL_FREQUENCY))
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
  if len(sys.argv) is 4:
    N_KERNELS_LAYER_1 = int(sys.argv[1])
    N_KERNELS_LAYER_2 = int(sys.argv[2])
    N_NODES_FULL_LAYER = int(sys.argv[3])
  elif len(sys.argv) is 3:
    N_KERNELS_LAYER_1 = int(sys.argv[1])
    N_KERNELS_LAYER_2 = None
    N_NODES_FULL_LAYER = int(sys.argv[2])
  else:
    print('invalid number of arguments')
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    exit()

  train()