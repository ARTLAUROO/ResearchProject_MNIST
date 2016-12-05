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

# Settings
CKPT_DIR = '/home/s1259008/research_project/tmp/mnist/ckpts/'

SUMMARY_FREQ = 10
EVAL_FREQ = 1000
SAVE_FREQ = 1000

N_EPOCHS = 10
BATCH_SIZE = 64
DROPOUT_RATE = 0.5

NUM_LABELS = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1


def generate_experiment_id(conv_settings, full_settings):
  date_time = time.strftime("%d-%b-%Y_%H-%M-%S", time.gmtime())

  # Add conv layer settings
  layer_settings = 'C'
  for setting in conv_settings:
    layer_settings += '-{:d}'.format(setting)

  # Add full layer settings
  layer_settings += '-F'
  for setting in full_settings:
    layer_settings += '-{:d}'.format(setting)

  return layer_settings + '_' + date_time


def create_experiment_dir(path):
  if not tf.gfile.Exists(path):
    tf.gfile.MakeDirs(path)


def calc_total_params():
  """From: http://stackoverflow.com/a/38161314/2351350"""
  total_parameters = 0
  for variable in tf.trainable_variables():
    # Shape is an array of tf.Dimension
    shape = variable.get_shape()
    print('v shape %s' % shape)
    variable_parametes = 1
    for dim in shape:
      variable_parametes *= dim.value
    print('v params %d' % variable_parametes)
    total_parameters += variable_parametes
  print('total params %d' % total_parameters)


def train(conv_settings, full_settings):
  print("Training model convl:" + str(conv_settings) + ' dense:' + str(full_settings))

  with tf.Graph().as_default():
    global_step = tf.Variable(0, dtype=mnist.data_type(), trainable=False)

    # Inputs
    images_input = tf.placeholder(mnist.data_type(),
                                  shape=(BATCH_SIZE,
                                         IMAGE_SIZE,
                                         IMAGE_SIZE,
                                         NUM_CHANNELS))
    labels_input = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    dropout_pl = tf.placeholder(tf.float32)

    # Inference
    logits = mnist.model(images_input, conv_settings, full_settings, NUM_LABELS, dropout_pl)
    prediction = tf.nn.softmax(logits)

    # Loss
    loss = mnist.loss(logits, labels_input)

    # Train node
    train_op = mnist.train(loss, global_step)

    # Setup saver and related vars
    saver = tf.train.Saver(max_to_keep=None)
    # Create dir for ckpts
    experiment_id = generate_experiment_id(conv_settings, full_settings)
    experiment_dir = CKPT_DIR + experiment_id + '/'
    create_experiment_dir(experiment_dir)
    ckpt_path = experiment_dir + 'mnist.ckpt'

    # Setup summary writer for Tensorboard
    merged = tf.merge_all_summaries()

    # Get data.
    train_images, train_labels, validation_images, validation_labels = input.data(True)
    train_size = train_labels.shape[0]

    #  Train
    with tf.Session() as sess:
      tf.initialize_all_variables().run()
      summary_writer = tf.train.SummaryWriter(CKPT_DIR + 'train', sess.graph)

      n_steps = int(N_EPOCHS * train_size) // BATCH_SIZE
      for step in xrange(n_steps):
        # Train
        # Generate batch
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_images[offset:(offset + BATCH_SIZE), ...]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
        # Update net
        feed_dict = {images_input: batch_data, labels_input: batch_labels, dropout_pl: DROPOUT_RATE}
        _, l, predictions, summary = sess.run([train_op, loss, prediction, merged], feed_dict=feed_dict)

        # Validate
        if step % EVAL_FREQ == 0:
          predictions = mnist_eval.eval_in_batches(validation_images,
                                                   sess,
                                                   images_input,
                                                   prediction,
                                                   dropout_pl)
          validation_error = mnist.error_rate(predictions, validation_labels)
          print('Validation error: %.2f%% after %d/%d steps' % (validation_error, step, n_steps))

        # Summaries
        if step % SUMMARY_FREQ == 0:
          summary_writer.add_summary(summary, step)

        # Save
        if step % SAVE_FREQ == 0 or step + 1 == n_steps:
          ckpt_file_path = saver.save(sess, ckpt_path, global_step=step)
          print('Saved checkpoint file: %s' % ckpt_file_path)

      # Evaluate
      eval_images, eval_labels = input.data(False)
      predictions = mnist_eval.eval_in_batches(eval_images, sess, images_input, prediction, dropout_pl)
      validation_error = mnist.error_rate(predictions, validation_labels)
      print('Validation error: %.2f%% after %d/%d steps' % (validation_error, step, n_steps))


if __name__ == '__main__':
  """
  Usage, one of the following:
  python train.py n_conv_1 n_conv_2 n_full_1
  python train.py n_conv_1 n_full_1
  """

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