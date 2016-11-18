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

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import mnist
import mnist_input as input
import mnist_eval




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


def train(convl_settings, dense_settings):
  print("Training model convl:" + str(convl_settings) + ' dense:' + str(dense_settings))


  with tf.Graph().as_default():
    global_step = tf.Variable(0, dtype=mnist.data_type(), trainable=False)

    # train inference
    train_images_input = tf.placeholder(mnist.data_type(),
                                        shape=(mnist.BATCH_SIZE,
                                               mnist.IMAGE_SIZE,
                                               mnist.IMAGE_SIZE,
                                               mnist.NUM_CHANNELS))
    logits = mnist.model(train_images_input,
                         convl_settings,
                         dense_settings,
                         mnist.NUM_LABELS,
                         True)  # training model
    train_prediction = tf.nn.softmax(logits)

    # train loss
    train_labels_input = tf.placeholder(tf.int64, shape=(mnist.BATCH_SIZE,))
    loss = mnist.loss(logits, train_labels_input)

    # train apply gradients
    train_op = mnist.train(loss, global_step)

    # validation inference
    validation_images_input = tf.placeholder(mnist.data_type(),
                                             shape=(mnist.BATCH_SIZE,
                                                    mnist.IMAGE_SIZE,
                                                    mnist.IMAGE_SIZE,
                                                    mnist.NUM_CHANNELS))
    validation_prediction = tf.nn.softmax(mnist.model(validation_images_input,
                                                      convl_settings,
                                                      dense_settings,
                                                      mnist.NUM_LABELS,
                                                      False))

    # Setup saver and related vars
    saver = tf.train.Saver(max_to_keep=None)
    # create dir for ckpts
    train_dir = generate_train_dir(convl_settings, dense_settings)
    ckpt_path = train_dir + '/mnist.ckpt'

    # Setup summary writer for Tensorboard
    merged = tf.merge_all_summaries()

    # Get the data.
    train_images, train_labels, validation_images, validation_labels = input.data(True)
    train_size = train_labels.shape[0]

    #  train
    with tf.Session() as sess:
      tf.initialize_all_variables().run()
      summary_writer = tf.train.SummaryWriter(
        mnist.TENSORBOARD_DIRECTORY + '/train',
        sess.graph)

      n_steps = int(mnist.NUM_EPOCHS * train_size) // mnist.BATCH_SIZE
      for step in xrange(n_steps):
        # Compute the offset of the current minibatch in the data.
        # Note that we could use better randomization across epochs.
        offset = (step * mnist.BATCH_SIZE) % (train_size - mnist.BATCH_SIZE)
        batch_data = train_images[offset:(offset + mnist.BATCH_SIZE), ...]
        batch_labels = train_labels[offset:(offset + mnist.BATCH_SIZE)]
        # This dictionary maps the batch data (as a numpy array) to the
        # node in the graph it should be fed to.
        feed_dict = {train_images_input: batch_data,
                     train_labels_input: batch_labels}

        # train
        _, l, predictions, summary = sess.run([train_op, loss, train_prediction, merged],
                                     feed_dict=feed_dict)

        # Print validation error and save summaries
        if step % mnist.EVAL_FREQUENCY == 0:
          predictions = mnist_eval.eval_in_batches(validation_images,
                                                   sess,
                                                   validation_images_input,
                                                   validation_prediction)
          validation_error = mnist.error_rate(predictions, validation_labels)
          print('Validation error: %.2f%% after %d/%d steps' %
                (validation_error, step, n_steps))

        summary_writer.add_summary(summary, step)

        # Save variables
        if step % (n_steps // mnist.SAVE_FREQUENCY) == 0 or step + 1 is n_steps:
          ckpt_file_path = saver.save(sess, ckpt_path, global_step=step)
          print('Saved checkpoint file: %s' % ckpt_file_path)

      # Print test error
      mnist_eval.eval_once(ckpt_file_path)


if __name__ == '__main__':
  n_args = len(sys.argv)
  if n_args is 3:
    convl = [int(sys.argv[1])]
    dense = [int(sys.argv[2])]
    train(convl, dense)
  elif n_args is 4:
    convl = [int(sys.argv[1]), int(sys.argv[2])]
    dense = [int(sys.argv[3])]
    train(convl, dense)
  else:
    print('invalid number of arguments')
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    print('Ways of calling:')
    print('python mnist_train n_convl1 n_dense1')
    print('python mnist_train n_convl1 n_convl2 n_dense1')
    exit()