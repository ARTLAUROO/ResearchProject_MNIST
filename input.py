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
# Code (adapted, by Arthur van Rooijen) from:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/input_data.py

"""Functions for downloading and reading MNIST data."""
import gzip
import os

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
import numpy

import model


# Settings
DATA_DIR = '/tmp/mnist/data/'
MNIST_URL = 'http://yann.lecun.com/exdb/mnist/'


def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""

  if not tf.gfile.Exists(DATA_DIR):
    tf.gfile.MakeDirs(DATA_DIR)
  file_path = os.path.join(DATA_DIR, filename)
  if not tf.gfile.Exists(file_path):
    file_path, _ = urllib.request.urlretrieve(MNIST_URL + filename, file_path)
    with tf.gfile.GFile(file_path) as f:
      size = f.Size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return file_path


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(model.IMAGE_SIZE
                          * model.IMAGE_SIZE
                          * num_images
                          * model.N_CHANNELS)
    read_data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    read_data = (read_data - (model.PIXEL_DEPTH / 2.0)) / model.PIXEL_DEPTH
    read_data = read_data.reshape(num_images,
                                  model.IMAGE_SIZE,
                                  model.IMAGE_SIZE,
                                  model.N_CHANNELS)
    return read_data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels


def data(train):
  """
  Returns the MNIST data sets of LeCunn.
  :param train: If true train & validation sets are returned. Else the test sets
  are returned
  :return: If train was true then:
  training images, training labels, validation images, validation labels
  Else:
  test images, test labels
  """
  if train:
    images_filename = 'train-images-idx3-ubyte.gz'
    labels_filename = 'train-labels-idx1-ubyte.gz'
    size = 60000  # inference.TRAIN_SIZE
  else:
    images_filename = 't10k-images-idx3-ubyte.gz'
    labels_filename = 't10k-labels-idx1-ubyte.gz'
    size = 10000  # inference.TEST_SIZE

  images_file_path = maybe_download(images_filename)
  labels_file_path = maybe_download(labels_filename)

  images = extract_data(images_file_path, size)
  labels = extract_labels(labels_file_path, size)

  if train:
    # Generate a validation set.
    validation_images = images[:model.VALIDATION_SIZE, ...]
    validation_labels = labels[:model.VALIDATION_SIZE]

    train_images = images[model.VALIDATION_SIZE:, ...]
    train_labels = labels[model.VALIDATION_SIZE:]

    return train_images, train_labels, validation_images, validation_labels
  else:
    return images, labels
