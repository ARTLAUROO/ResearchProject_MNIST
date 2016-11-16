from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import mnist
import mnist_input as input


def get_settings_from_name(name):
    """Extracts settings of CNN from a properly formatted name.

    Keyword arguments:
    name -- Example: 15-Nov-2016_16-26-06_K-32-L-256, each number after the K
            and L represent a layer of that size and type. K is for kernel in
            a convolutional layer, L is for local in a dense layer.

    Return:
    Two lists, first filled with numbers representing conv layers, second filled
    with numbers representing dense layers. From the example: [32] [256]
    """

    settings = name.split('_')
    settings = settings[-1]  # drop date and time prefix
    settings = settings.split('-')

    l_idx = settings.index("L")
    convl = [int(setting) for setting in settings[1:l_idx]]  # omit K
    dense = [int(setting) for setting in settings[l_idx + 1:]]  # omit L

    return convl, dense


def load_model(ckpt_path):
    print('Loading: %s' % ckpt_path)

    with tf.Graph().as_default():
        # Load settings from dir name
        dir_name = ckpt_path.split('/')
        dir_name = dir_name[-2]  # drop path prefix
        convl_settings, dense_settings = get_settings_from_name(dir_name)

        # Construct model
        data_batch = tf.placeholder(mnist.data_type(),
                                    shape=(mnist.BATCH_SIZE,
                                           mnist.IMAGE_SIZE,
                                           mnist.IMAGE_SIZE,
                                           mnist.NUM_CHANNELS))
        logits = mnist.model(data_batch,
                             convl_settings,
                             dense_settings,
                             mnist.NUM_LABELS,
                             False)  # eval model

        prediction = tf.nn.softmax(logits)

        saver = tf.train.Saver()

        sess = tf.Session()
        saver.restore(sess, ckpt_path)

        # TODO don't return data_batch and prediction
        return sess, data_batch, prediction


def eval_in_batches(data, sess, eval_data, eval_prediction):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < mnist.BATCH_SIZE:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, mnist.NUM_LABELS),
                             dtype=np.float32)
    for begin in xrange(0, size, mnist.BATCH_SIZE):
        end = begin + mnist.BATCH_SIZE
        if end <= size:
            predictions[begin:end, :] = sess.run(
                eval_prediction,
                feed_dict={eval_data: data[begin:end, ...]})
        else:
            batch_predictions = sess.run(
                eval_prediction,
                feed_dict={eval_data: data[-mnist.BATCH_SIZE:, ...]})
            predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions


def eval_once(ckpt_path):
    print('Evaluate: %s' % ckpt_path)
    sess, data_batch, prediction = load_model(ckpt_path)
    test_data, test_labels = input.data(False)
    predictions = eval_in_batches(test_data, sess, data_batch, prediction)
    test_error = mnist.error_rate(predictions, test_labels)
    print('Test error: %.2f%%' % test_error)
    return test_error


if __name__ == '__main__':
    # Test
    eval_once(
        '/tmp/mnist/ckpts/15-Nov-2016_16-21-13_K-32-64-L-4/mnist.ckpt-1718')
    eval_once(
        '/tmp/mnist/ckpts/15-Nov-2016_16-26-06_K-32-L-256/mnist.ckpt-1718')
