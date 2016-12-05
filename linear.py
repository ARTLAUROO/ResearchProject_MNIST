import os
import gzip

import tensorflow as tf
import numpy as np
from six.moves import urllib
import math

DIR_ROOT = 'training/'
DIR_TENSORBOARD = DIR_ROOT + 'tensorboard/'
DIR_CHECKPOINTS = DIR_ROOT + 'checkpoints/'
CHECKPOINT_FILENAME = 'mnist'
CHECKPOINT_BASE_PATH = DIR_CHECKPOINTS + CHECKPOINT_FILENAME
DIR_PLOTS = 'plots/'
DIR_DATA = '/home/s1259008/research_project/tmp/mnist_data'
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

KAGGLE_DATA = False # When false lecunn data is used, else kaggle data
BATCH_SIZE = 100

N_EPOCHS = None
SHUFFLE = True
EVALUATION_FREQ = 500
KERNEL_SIZE = 5
DROPOUT = 1.0
DECAY_STEP_SIZE = 60000 # TODO == TRAIN_SIZE
MAX_CHECKPOINTS = None
LEARNING_RATE = 0.01
IMAGE_SIZE = 28
N_CHANNELS = 1
PIXEL_DEPTH = 255
N_LABELS = 10

# TEST_SIZE = 10000             # Size of the test set.
# VALIDATION_SIZE = 4200        # Size of the validation set.


# LECUNN DATA -------------------------------------------------------------------------------------------------------

LECUNN_N_TRAIN = 50000
LECUNN_N_TEST = 10000

def lecunn_maybe_download(file_name, dir_path):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(dir_path):
    tf.gfile.MakeDirs(dir_path)
  filepath = os.path.join(dir_path, file_name)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + file_name, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.Size()
    print('Successfully downloaded', file_name, size, 'bytes.')
  return filepath


def lecunn_extract_images(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * N_CHANNELS)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS)
    return data


def lecunn_extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels


def lecunn_get_images_and_labels():
  train_images_filename = 'training-images-idx3-ubyte.gz'
  train_labels_filename = 'training-labels-idx1-ubyte.gz'
  test_images_filename = 't10k-images-idx3-ubyte.gz'
  test_labels_filename = 't10k-labels-idx1-ubyte.gz'

  # Get the data.
  train_images_filepath = lecunn_maybe_download(train_images_filename, DIR_DATA)
  train_labels_filepath = lecunn_maybe_download(train_labels_filename, DIR_DATA)
  test_images_filepath = lecunn_maybe_download(test_images_filename, DIR_DATA)
  test_labels_filepath = lecunn_maybe_download(test_labels_filename, DIR_DATA)

  # Extract it into numpy arrays.
  train_images = lecunn_extract_images(train_images_filepath, LECUNN_N_TRAIN)
  train_labels = lecunn_extract_labels(train_labels_filepath, LECUNN_N_TRAIN)
  test_images = lecunn_extract_images(test_images_filepath, LECUNN_N_TEST)
  test_labels = lecunn_extract_labels(test_labels_filepath, LECUNN_N_TEST)

  return train_images, train_labels, test_images, test_labels


# KAGGLE DATA -------------------------------------------------------------------------------------------------------

# read features
def kaggle_extract_images(filename, usecols=range(1, 785)):
  images = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=usecols, dtype=np.float32)
  images = np.divide(images, 255.0) # scale 0..255 to 0..1
  # TODO scale images between -0.5 and 0.5
  # reshape features to 2d shape
  images = images.reshape((-1, 28, 28, 1))
  return images

# read labels and convert them to 1-hot vectors
def kaggle_extract_labels(filename):
  labels = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=0, dtype=np.int)
  return labels

# generate batches
# def generate_batch(images, labels, batch_size):
#   batch_indexes = np.random.random_integers(0, len(images) - 1, batch_size)
#   batch_features = images[batch_indexes]
#   batch_labels = labels[batch_indexes]
#   return (batch_features, batch_labels)

def kaggle_get_images_and_labels():
  labels = kaggle_extract_labels('/tmp/kaggle/mnist/data/training.csv')
  images = kaggle_extract_images('/tmp/kaggle/mnist/data/training.csv')
  # TODO divide in training and test set
  return images, labels


# INPUT BATCHING -------------------------------------------------------------------------------------------------------

def create_data_vars(images, labels):
  image_init = tf.placeholder(dtype=images.dtype, shape=images.shape)
  label_init = tf.placeholder(dtype=labels.dtype, shape=labels.shape)

  image_var = tf.Variable(image_init, trainable=False, collections=[])
  label_var = tf.Variable(label_init, trainable=False, collections=[])
  return image_var, image_init, label_var, label_init


def input_pipeline(image_var, label_var, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, shuffle=SHUFFLE):
  image, label = tf.train.slice_input_producer([image_var, label_var],
                                               num_epochs=n_epochs,
                                               shuffle=shuffle,
                                               seed=None,
                                               capacity=100,
                                               shared_name=None,
                                               name=None)
  label = tf.cast(label, tf.int64)
  label = tf.one_hot(label, 10)
  image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size)
  return image_batch, label_batch


# MODEL ------------------------------------------------------------------------------------------------------------

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def inference(prev_layer, conv_sizes, full_sizes, keep_prob):
  # with tf.name_scope('input_img_summary'):
  #   tf.image_summary('input', images, max_images=3, collections=None, name=None)
  for full_i in xrange(len(conv_sizes)):
    with tf.variable_scope('conv' + str(full_i)):
      if full_i is 0:
        W_conv1 = weight_variable([5, 5, 1, conv_sizes[full_i]])
      else:
        W_conv1 = weight_variable([5, 5, conv_sizes[full_i-1], conv_sizes[full_i]])
      b_conv1 = bias_variable([conv_sizes[full_i]])
      h_conv1 = tf.nn.relu(conv2d(prev_layer, W_conv1) + b_conv1)

    with tf.name_scope('pool' + str(full_i)):
      prev_layer = max_pool_2x2(h_conv1)

  # with tf.name_scope('conv1_img_summary'):
  #   tensor = tf.split(3, conv1_size, pool, name='split')
  #   for i in xrange(len(tensor)):
  #       tf.image_summary('conv1_kernel-' + str(i), tensor[i], max_images=3, collections=None, name=None)

  if len(conv_sizes) is 0:
    img_size = IMAGE_SIZE
    n_prev_layer = N_CHANNELS
  else:
    img_size = IMAGE_SIZE / (2 * len(conv_sizes))
    n_prev_layer = conv_sizes[-1]
  prev_layer = tf.reshape(prev_layer, [-1, img_size * img_size * n_prev_layer])  # flatten

  for full_i in xrange(len(full_sizes)):
    with tf.variable_scope('full' + str(full_i)):
      if full_i is 0:
        W_fc1 = weight_variable([img_size * img_size * n_prev_layer, full_sizes[full_i]])
      else:
        W_fc1 = weight_variable([full_sizes[full_i-1], full_sizes[full_i]])

      b_fc1 = bias_variable([full_sizes[full_i]])
      prev_layer = tf.nn.relu(tf.matmul(prev_layer, W_fc1) + b_fc1)

  with tf.variable_scope('dropout1'):
    h_fc1_drop = tf.nn.dropout(prev_layer, keep_prob)

  if len(full_sizes) != 0:
    softmax_size = full_sizes[-1]
  else:
    softmax_size = img_size * img_size * n_prev_layer

  # add final softmax layer
  with tf.variable_scope('softmax_linear'):
    W_fc2 = weight_variable([softmax_size, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  return y_conv


def loss(logits, labels):
  """Calculates the loss from the logits and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
  return cross_entropy


def training(cross_entropy, learning_rate):
  """Sets up the training Ops.
  Creates a summarizer to track the loss over time in TensorBoard.
  Creates an optimizer and applies the gradients to all trainable variables.
  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the inference to training.
  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
  Returns:
    train_op: The Op for training.
  """
  train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  return train_op


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier inference, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # Return the number of true entries.
  return accuracy


# TRAINING ---------------------------------------------------------------------------------------------------------

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the inference building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the training or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS))
  labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 10))
  return images_placeholder, labels_placeholder


def fill_feed_dict(images_feed, labels_feed, images_pl, labels_pl, keep_prob, train):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  if train:
    prob = DROPOUT
  else:
    prob = 1.0

  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
      keep_prob: prob
  }
  return feed_dict


# Generate placeholders for the images and labels.
keep_prob = tf.placeholder(tf.float32)
images_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)

# Build a Graph that computes predictions from the inference inference.
logits = inference(images_placeholder, [], [], keep_prob)

# Add to the Graph the Ops for loss calculation.
loss = loss(logits, labels_placeholder)

# Add to the Graph the Ops that calculate and apply gradients.
train_op = training(loss, LEARNING_RATE)

# Add the Op to compare the logits to the labels during evaluation.
eval_correct = evaluation(logits, labels_placeholder)

summary = tf.merge_all_summaries()

saver = tf.train.Saver()

if KAGGLE_DATA:
  print('Loading Kaggle data')
  train_images, train_labels = kaggle_get_images_and_labels()
  # TODO
  test_images = train_images
  test_labels = train_labels
else:
  print('Loading Lecunn data')
  train_images, train_labels, test_images, test_labels = lecunn_get_images_and_labels()

n_train = len(train_labels)
n_test = len(test_labels)

n_train_batches_per_epoch = n_train // BATCH_SIZE
n_test_batches_per_epoch = n_test // BATCH_SIZE

with tf.variable_scope('input'):
  train_image_var, train_image_init, train_label_var, train_label_init = create_data_vars(train_images, train_labels)
  test_image_var, test_image_init, test_label_var, test_label_init = create_data_vars(test_images, test_labels)

  train_images_batch, train_labels_batch = input_pipeline(train_images, train_labels)
  test_images_batch, test_labels_batch = input_pipeline(test_images, test_labels)

init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

with tf.Session() as sess:
  sess.run(init_op)
  sess.run(train_image_var.initializer, feed_dict={train_image_init: train_images})
  sess.run(train_label_var.initializer, feed_dict={train_label_init: train_labels})

  summary_writer = tf.train.SummaryWriter(DIR_TENSORBOARD, sess.graph)

  sess.run(test_image_var.initializer, feed_dict={test_image_init: test_images})
  sess.run(test_label_var.initializer, feed_dict={test_label_init: test_labels})

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  try:
    step = 0
    print('Training..')
    while not coord.should_stop():
      images_feed, labels_feed = sess.run([train_images_batch, train_labels_batch])
      feed_dict = fill_feed_dict(images_feed, labels_feed, images_placeholder, labels_placeholder, keep_prob, train=True)
      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

      if step == n_train_batches_per_epoch * 20:
      # if step % EVALUATION_FREQ == 0:
        print('Evaluating..')
        summed_acc = 0
        for i in xrange(n_test_batches_per_epoch):
          images_feed, labels_feed = sess.run([test_images_batch, test_labels_batch])
          feed_dict = fill_feed_dict(images_feed, labels_feed, images_placeholder, labels_placeholder, keep_prob, train=True)
          acc = sess.run(eval_correct, feed_dict=feed_dict)
          summed_acc += acc
        avg_acc = summed_acc / n_test_batches_per_epoch
        print('  Num examples: %d Avg Accuracy @ 1: %f' % (n_test, avg_acc))
        break
        print('Training..')


      summary = False
      if summary:
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      save = False
      if save:
        checkpoint_file = os.path.join(DIR_CHECKPOINTS, 'inference.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)

      step += 1
  except tf.errors.OutOfRangeError:
    pass
  finally:
    coord.request_stop()

  coord.join(threads)