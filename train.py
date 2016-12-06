from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf

import model
import input
import eval

from experiment_id import ExperimentID

# Settings
BASE_DIR = '/home/s1259008/research_project/experiments/'
CKPT_DIR = BASE_DIR + 'mnist/ckpts/'
LOGS_DIR = BASE_DIR + 'mnist/logs/'
CKPT_FILENAME = 'mnist.ckpt'

SUMMARY_FREQ = 100
EVAL_FREQ = 1000
SAVE_FREQ = 1000

N_EPOCHS = 1
BATCH_SIZE = 100
DROPOUT_RATE = 0.5

CONV_KERNEL_SIZE = 5

def create_experiment_dirs(experiment_id):
  """
  Creates the directorries needed to run an experiment. Currently only creates
  the dir to save the checkpoint files.
  :param experiment_id: id belonging to the experiment.
  """
  ckpt_path = CKPT_DIR + str(experiment_id)
  if not tf.gfile.Exists(ckpt_path):
    tf.gfile.MakeDirs(ckpt_path)


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
  """
  Sets up a network according to the settings specified and trains it with the
  settings specified by the global parameters.
  :param conv_settings: list with ints signifying the size of each conv layer,
    can be empty if conv layers are to be omitted
  :param full_settings: list with ints signifying the size of each full layer,
    can be empty if full layers are to be omitted
  """
  experiment_id = ExperimentID()
  experiment_id.init_settings(conv_settings,
                              full_settings,
                              BATCH_SIZE,
                              N_EPOCHS)
  print('Experiment id {}'.format(experiment_id))

  with tf.Graph().as_default():
    global_step = tf.Variable(0, dtype=model.data_type(), trainable=False)

    # Inputs
    images_pl = tf.placeholder(model.data_type(),
                               shape=(BATCH_SIZE,
                                      model.IMAGE_SIZE,
                                      model.IMAGE_SIZE,
                                      model.N_CHANNELS))
    labels_pl = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    dropout_pl = tf.placeholder(tf.float32)

    # Inference
    logits = model.inference(images_pl,
                             conv_settings,
                             full_settings,
                             model.N_LABELS,
                             dropout_pl)
    prediction = tf.nn.softmax(logits)

    # Loss
    loss = model.loss(logits, labels_pl)

    # Train node
    train_op = model.training(loss, global_step)

    # Setup saver and related vars
    saver = tf.train.Saver(max_to_keep=None)
    # Create dir for ckpts
    create_experiment_dirs(experiment_id)
    ckpt_path = '{}{}/{}'.format(CKPT_DIR, experiment_id, CKPT_FILENAME)

    # Setup summary writer for Tensorboard
    merged = tf.merge_all_summaries()

    # Get data.
    train_images, train_labels, valid_images, valid_labels = input.data(True)
    train_size = train_labels.shape[0]

    #  Train
    with tf.Session() as sess:
      tf.initialize_all_variables().run()
      summary_writer = tf.train.SummaryWriter(LOGS_DIR + 'training', sess.graph)

      step = 0
      if N_EPOCHS >= 0:
        n_steps = int(N_EPOCHS * train_size) // BATCH_SIZE
      else:
        n_steps = None
      while step < n_steps or N_EPOCHS == -1:
        # Train
        # Generate batch
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_images[offset:(offset + BATCH_SIZE), ...]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
        # Update net
        feed_dict = {images_pl: batch_data,
                     labels_pl: batch_labels,
                     dropout_pl: DROPOUT_RATE}
        ops = [train_op, loss, prediction, merged]
        _, l, train_predictions, summary = sess.run(ops, feed_dict=feed_dict)

        # Validate
        if step != 0 and step % EVAL_FREQ == 0:
          validation_predictions = eval.eval_in_batches(valid_images,
                                                        sess,
                                                        images_pl,
                                                        prediction,
                                                        dropout_pl)
          validation_error = model.error_rate(validation_predictions,
                                              valid_labels)
          msg = 'Validation error: {:.2f}% after {}/{} steps'
          print(msg.format(validation_error, step, n_steps))

        # Save
        if step != 0 and step % SAVE_FREQ == 0 or step + 1 == n_steps:
          ckpt_file_path = saver.save(sess, ckpt_path, global_step=step)
          print('Saved checkpoint file: {}'.format(ckpt_file_path))

        # Summaries
        if step % SUMMARY_FREQ == 0:
          train_error = model.error_rate(train_predictions, batch_labels)
          msg = 'Training error: {:.2f}% after {}/{} steps, loss {:.2f}'
          print(msg.format(train_error, step, n_steps, l))
          summary_writer.add_summary(summary, step)

        step += 1

      # Evaluate
      test_images, test_labels = input.data(False)
      test_predictions = eval.eval_in_batches(test_images,
                                              sess,
                                              images_pl,
                                              prediction,
                                              dropout_pl)
      test_error = model.error_rate(test_predictions, test_labels)
      print('Test error: {:.2f}%'.format(test_error))


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
