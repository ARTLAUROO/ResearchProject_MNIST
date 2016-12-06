from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

import eval
from experiment_id import ExperimentID


PLOT_DIR = '/home/s1259008/research_project/experiments/mnist/plots/'


def get_errors_and_cumsums(ckpt_dir_path):
  """
  Calculates the errors and cumsums belonging to a checkpoint.
  :param ckpt_dir_path: Checkpoint to calculate pca errors and cumsums from.
  :return: errors, cumsums and the experiment id
  """
  print('Getting errors and cumsums for checkpoint: ' + ckpt_dir_path)

  # Load settings from dir name
  ckpts_dir_path_splitted = ckpt_dir_path.split('/')
  dir_name = ckpts_dir_path_splitted[-3]
  _experiment_id = ExperimentID()
  _experiment_id.init_string(dir_name)

  _errors = [0] * 25
  _cumsums = [0] * 25

  for f in os.listdir(ckpt_dir_path):
    if '.ckpt' in f and '.meta' not in f:
      error = eval.eval_ckpt(ckpt_dir_path + f, _experiment_id)

      file_splitted = f.split('_')
      cumsum = float(file_splitted[1][2:])  # Omit prefix: v-

      idx = int(file_splitted[0][4:]) - 1  # Get applied pca dim as idx
      _errors[idx] = error
      _cumsums[idx] = cumsum

  return _errors, _cumsums, _experiment_id


def create_plot(_errors, _cumsums, _experiment_id):
  """
  Given errors, cumsums and an experiment id a neat plot is generated. It is
  stored in the plots dir and named after the experiment id.
  :param _errors: list
  :param _cumsums: list
  :param _experiment_id:
  """
  print('Creating plot: {}'.format(_experiment_id))
  plt.figure()
  plt.ylabel('Percentage')
  plt.xlabel('N PCA components')
  plt.axis([-0.5, len(_errors), -2, 102])
  plt.grid(True)
  error_points, = plt.plot(_errors, 'ro', label='Error on test set')
  cumsum_points, = plt.plot(_cumsums, 'bx', label='Kernel variance kept')

  plt.legend(bbox_to_anchor=(0.9, 0.85),
             bbox_transform=plt.gcf().transFigure,
             handler_map={error_points: HandlerLine2D(numpoints=1),
                          cumsum_points: HandlerLine2D(numpoints=1)})
  plt.savefig(PLOT_DIR + str(_experiment_id) + '.jpg')
  # plt.show()


def create_plots(dir_path):
  """
  Create plots for all pca dirs in the dir denoted by dir_path.
  :param dir_path: example:
  ../experiments/mnist/ckpts/C-32-64-F-1024_B-100-E-10_05-Dec-2016_12-04-52/
  """
  print('Creating plots for dir: {}'.format(dir_path))
  for f in os.listdir(dir_path):
    if '-pca' in f:
      ckpt_dir = dir_path + f + '/'
      _errors, _cumsums, _experiment_id = get_errors_and_cumsums(ckpt_dir)
      create_plot(_errors, _cumsums, _experiment_id)


if __name__ == '__main__':
  if len(sys.argv) < 2:
    msg = 'usage: python plot.py ckpt/path/file.pca.ckpt ; ' \
          'or: python plot.py pca-ckpts/path/dir/'
    print(msg)
    exit()

  path = sys.argv[1]
  if '-pca' in path:
    if path[-1] is not '/':
      path += '/'
    # Specific checkpoint to which pca must be applied
    errors, cumsums, experiment_id = get_errors_and_cumsums(path)
    create_plot(errors, cumsums, experiment_id)
  else:
    # Dir full of checkpoints to which pca must be applied
    create_plots(path)
