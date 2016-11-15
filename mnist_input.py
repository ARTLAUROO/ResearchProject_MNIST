import gzip
import os

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
import numpy

import mnist

def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(mnist.WORK_DIRECTORY):
    tf.gfile.MakeDirs(mnist.WORK_DIRECTORY)
  filepath = os.path.join(mnist.WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(mnist.SOURCE_URL + filename, filepath)
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
    buf = bytestream.read(mnist.IMAGE_SIZE * mnist.IMAGE_SIZE * num_images * mnist.NUM_CHANNELS)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (mnist.PIXEL_DEPTH / 2.0)) / mnist.PIXEL_DEPTH
    data = data.reshape(num_images, mnist.IMAGE_SIZE, mnist.IMAGE_SIZE, mnist.NUM_CHANNELS)
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
      shape=(num_images, mnist.IMAGE_SIZE, mnist.IMAGE_SIZE, mnist.NUM_CHANNELS),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels

def data(train):
  if train:
    images_filename = 'train-images-idx3-ubyte.gz'
    labels_filename = 'train-labels-idx1-ubyte.gz'
    size = mnist.TRAIN_SIZE
  else:
    images_filename = 't10k-images-idx3-ubyte.gz'
    labels_filename = 't10k-labels-idx1-ubyte.gz'
    size = mnist.TEST_SIZE
  # Get the data.

  images_filepath = maybe_download(images_filename)
  labels_filepath = maybe_download(labels_filename)

  # Extract it into numpy arrays.
  images = extract_data(images_filepath, size)
  labels = extract_labels(labels_filepath, size)

  if train:
    # Generate a validation set.
    validation_images = images[:mnist.VALIDATION_SIZE, ...]
    validation_labels = images[:mnist.VALIDATION_SIZE]

    train_images = images[mnist.VALIDATION_SIZE:, ...]
    train_labels = labels[mnist.VALIDATION_SIZE:]

    return train_images, train_labels, validation_images, validation_labels
  else:
    return images, labels