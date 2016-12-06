from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import numpy as np
import tensorflow as tf

import model
import input
import train
from experiment_id import ExperimentID


def get_settings_from_experiment_id(_experiment_id):
    """Extracts settings of CNN from a properly formatted name.

    Keyword arguments:
    name -- Example: 1C-32-64-F-1024_B-100-E-None_05-Dec-2016_11-40-20, each
    number after the K
            and L represent a layer of that size and type. K is for kernel in
            a convolutional layer, L is for local in a full layer.

    Return:
    Two lists, first filled with numbers representing conv layers, second filled
    with numbers representing full layers. From the example: [32] [256]
    """

    settings = _experiment_id.split('_')

    layer_settings = settings[0].split('-')

    full_idx = layer_settings.index("F")
    conv = [int(setting) for setting in layer_settings[1:full_idx]]  # omit K
    full = [int(setting) for setting in layer_settings[full_idx + 1:]]  # omit L

    return conv, full


def load_model(ckpt_path, _experiment_id, sess):
  print('Loading: {}'.format(ckpt_path))

  conv_settings = _experiment_id.conv
  full_settings = _experiment_id.full

  # Inputs
  images_pl = tf.placeholder(model.data_type(),
                             shape=(train.BATCH_SIZE,
                                    model.IMAGE_SIZE,
                                    model.IMAGE_SIZE,
                                    model.N_CHANNELS))
  dropout_pl = tf.placeholder(tf.float32)

  # Define model
  logits = model.inference(images_pl,
                           conv_settings,
                           full_settings,
                           model.N_LABELS,
                           dropout_pl)
  prediction = tf.nn.softmax(logits)

  # Restore model
  saver = tf.train.Saver()
  saver.restore(sess, ckpt_path)

  # TODO don't return images_pl and prediction
  return images_pl, dropout_pl, prediction


def eval_in_batches(data, sess, eval_data, eval_prediction, dropout_pl):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < train.BATCH_SIZE:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, model.N_LABELS), dtype=np.float32)
    for begin in xrange(0, size, train.BATCH_SIZE):
        end = begin + train.BATCH_SIZE
        if end <= size:
            # use no dropout
            predictions[begin:end, :] = sess.run(eval_prediction,
                                                 feed_dict={eval_data: data[begin:end, ...], dropout_pl: 1})
        else:
            # use no dropout
            batch_predictions = sess.run(eval_prediction,
                                         feed_dict={eval_data: data[-train.BATCH_SIZE:, ...], dropout_pl: 1})
            predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions


def eval_ckpt(path, experiment_id):
  print('Evaluating CKPT: ' + path)
  test_data, test_labels = input.data(False)
  with tf.Graph().as_default(), tf.Session() as sess:
    images_pl, dropout_pl, prediction = load_model(path, experiment_id, sess)
    predictions = eval_in_batches(test_data,
                                  sess,
                                  images_pl,
                                  prediction,
                                  dropout_pl)
    test_error = model.error_rate(predictions, test_labels)
    print('Test error: {:.2f}%'.format(test_error))
  return test_error


def eval_dir(dir_path):
  print('Evaluating DIR: ' + dir_path)
  ckpts = []
  for file in os.listdir(dir_path):
    if '.ckpt' in file and not '.meta' in file:
      ckpts.append(dir_path + file)

  dir_name = dir_path.split('/')
  dir_name = dir_name[-2]

  for ckpt in sorted(ckpts):
    # Load settings from dir name
    ckpt_path_splitted = ckpt.split('/')
    _experiment_id = ExperimentID()
    _experiment_id.init_string(dir_name)

    eval_ckpt(ckpt, _experiment_id)

if __name__ == '__main__':
    if (len(sys.argv) < 2):
      msg = 'usage: python eval.py ckpt/path/file.ckpt ; ' \
            'or: python eval.py ckpt/path/dir/'
      print(msg)
      exit()

    path = sys.argv[1]
    if '.ckpt' in path:
      # Obtain experiment id from path (parent dir)
      path_splitted = path.split('/')
      ckpt_dir = path_splitted[-2]
      experiment_id = ExperimentID()
      experiment_id.init_string(ckpt_dir)

      eval_ckpt(path, experiment_id)
    else:
      if path[-1] is not '/':
        path += '/'
      eval_dir(path)

    # TODO remove
    # dir_name = ckpt_path.split('/')
    # dir_name = dir_name[-2]  # drop path prefix
    # convl_settings, dense_settings = get_settings_from_experiment_id(dir_name)
    #
    # with open("hyperparam_search.txt", "a") as result_file:
    #   s = ''
    #   for i in convl_settings:
    #     s += str(i) + ' '
    #   for i in dense_settings:
    #     s += str(i) + ' '
    #   s += str(error) + '\n'
    #   result_file.write(s)



