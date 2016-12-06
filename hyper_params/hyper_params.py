import numpy as np

def read_hyper_params():
  smallest = 9999999

  array = np.zeros([792, 4])
  with open("hyperparam_search.txt", "r") as f:
    i = 0
    for line in f:
      numbers = line.strip('\n').split(' ')
      array[i][0] = int(numbers[0])
      if len(numbers) is 4:
        array[i][1] = int(numbers[1])
      array[i][2] = int(numbers[-2])
      array[i][3] = float(numbers[-1])
      i += 1

  return array

def to_csv():
  array = read_hyper_params()
  with open("hyperparams.csv", "w") as f:
    for i in xrange(len(array)):
      f.write(str(int(array[i][0]))+','+str(int(array[i][1]))+','+str(int(array[i][2]))+','+str(array[i][3])+'\n')

to_csv()