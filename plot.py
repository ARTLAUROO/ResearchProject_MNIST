from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

import sys
import os

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

import tensorflow as tf

import numpy as np
from sklearn.decomposition import PCA

import model
import input as input
import eval


PLOT_DIR = '/home/s1259008/research_project/experiments/mnist/plots/'


def get_errors_and_cumsums(pca_dir):
  print('Get errors and cumsums: ' + pca_dir)
  errors = []
  cumsums = []

  # Load settings from dir name
  pca_dir_splitted = pca_dir.split('/')
  experiment_id = None
  for string in pca_dir_splitted:
    if 'C-' in string:
      experiment_id = string
      break
  assert experiment_id != None

  errors = [0]*25
  cumsums = [0]*25

  for file in os.listdir(pca_dir):
    if '.ckpt' in file and not '.meta' in file:

      error = eval.eval_ckpt(pca_dir, experiment_id)

      file_splitted = file.split('_')
      cumsum = float(file_splitted[1][2:]) # omit prefix: v-

      idx = int(file_splitted[0][4:]) - 1
      errors[idx] = error
      cumsums[idx] = cumsum

  return errors, cumsums, experiment_id

def create_plot(errors, cumsums, experiment_id):
  print('Creating plot: ' + experiment_id)
  plt.figure()
  plt.title(experiment_id)
  plt.ylabel('Percentage')
  plt.xlabel('N PCA components')
  plt.axis([-0.5, len(errors), -2, 102])
  plt.grid(True)
  error_points, = plt.plot(errors, 'ro', label='Error on test set')
  cumsum_points, = plt.plot(cumsums, 'bx', label='Kernel variance kept')

  plt.legend(bbox_to_anchor=(0.9, 0.85),
             bbox_transform=plt.gcf().transFigure,
             handler_map={error_points: HandlerLine2D(numpoints=1),
                          cumsum_points: HandlerLine2D(numpoints=1)})
  plt.savefig(PLOT_DIR + experiment_id + '.jpg')
  # plt.show()


def create_plots(path):
  print('Creating plots')
  # pass experiment dir path
  # example: ../experiments/mnist/ckpts/C-32-64-F-1024_B-100-E-10_05-Dec-2016_12-04-52/
  for file in os.listdir(path):
    if '-pca' in file:
      errors, cumsums, experiment_id = get_errors_and_cumsums(path + file)
      create_plot(errors, cumsums, experiment_id)


if __name__ == '__main__':
  if (len(sys.argv) < 2):
    print('usage: python plot.py ckpt/path/file.pca.ckpt ; or: python plot.py pca-ckpts/path/dir/')
    exit()

  path = sys.argv[1]
  if '-pca' in path:
    errors, cumsums, experiment_id = get_errors_and_cumsums(path)
    create_plot(errors, cumsums, experiment_id)
  else:
    create_plots(path)