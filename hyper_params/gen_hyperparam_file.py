import os
import numpy as np

ckpt_dirs = [x[0] for x in os.walk('/home/s1259008/research_project/tmp/mnist/ckpts/')]
ckpt_dirs = ckpt_dirs[1:]  # skip base dir

ckpt_paths = []
for ckpt_dir in ckpt_dirs:
  ckpts = os.listdir(ckpt_dir)

  for ckpt in ckpts:
    if 'meta' in ckpt or 'checkpoint' in ckpt:
      continue
    if '8000' in ckpt:
      ckpt_paths.append(ckpt_dir + '/' + ckpt)

ckpt_paths = sorted(ckpt_paths)
for ckpt_path in ckpt_paths:
  os.system("python eval.py " + ckpt_path)