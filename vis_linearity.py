#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/01 

from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

from src.features import feats
from src.runner import load_pretrained_env
from utils import *


def run(args):
  model, runner = load_pretrained_env(args)

  df = pd.read_csv(args.test_fp)
  X, _ = model.split_df(df, feats)
  for n, x in enumerate(X):
    if n == args.index: break
  x_n = rescale_norm(x, 0, np.pi)
  D = x.shape[-1]

  plt.clf()
  for d in range(D):
    x_v = tensor.unsqueeze(x_n.clone(), 0)
    x_v = tensor.tile(x_v, [args.n_lerp, 1])
    x_v[:, d] = np.linspace(0, np.pi, args.n_lerp)    # vary value at dim-d in valid vrng [0, pi]
    x_v = l2_norm(x_v)
    o = model.forward(x_v)
    plt.plot(o[:, 0], label=str(d))
  plt.legend()
  plt.show()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-L', '--logdir', default='out/hea_amp', type=Path, help='logdir to pretrained ckpt')
  parser.add_argument('-I', '--index', default=0, type=int, help='sample index')
  parser.add_argument('-N', '--n_lerp', default=100, type=int, help='lerp count')
  parser.add_argument('--test_fp', default=TEST_FILE, type=Path)
  args, _ = parser.parse_known_args()

  run(args)
