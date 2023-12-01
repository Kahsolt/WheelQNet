#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/01 

# PGD attack over pretrained QNNs

from pathlib import Path
from functools import partial
from argparse import ArgumentParser

import numpy as np
from scipy.optimize import approx_fprime
from tqdm import tqdm

from src.runner import load_pretrained_env
from utils import *

pi = np.pi


def run(args):
  model, runner = load_pretrained_env(args)
  Y = get_truth()

  def loss_fn_wrap(y_np:ndarray, x_np:ndarray) -> float:
    nonlocal model
    x = tensor.unsqueeze(QTensor(x_np), 0).astype(kcomplex64)
    y = tensor.unsqueeze(QTensor(y_np), 0).astype(kint64)
    o = model(x)
    l = model.loss(o, y).item()
    return l

  tot, acc, atk, pcr = 0, 0, 0, 0
  X, _ = model.reprocess(pd.read_csv(args.test_fp))
  for x, y in tqdm(zip(X, Y)):
    # original
    pred_raw = model.inference(x).item()

    # attack
    x_np = x.numpy()
    x_adv = x_np + np.random.uniform(size=x_np.shape, low=-1, high=1) * args.eps
    for _ in range(args.step):
      g = approx_fprime(x_np, partial(loss_fn_wrap, y), epsilon=1e-5)
      x_adv += np.sign(g) * args.alpha
      delta = np.clip(x_adv - x_np, -args.eps, args.eps)
      x_adv = x_np + delta

    # check success
    x_adv = QTensor(x_adv).astype(kcomplex64)
    pred_adv = model.inference(x_adv).item()

    tot += 1
    acc += pred_adv == y
    atk += pred_adv != y
    pcr += pred_adv != pred_raw

  print(f'acc: {acc / tot:%}')
  print(f'asr: {atk / tot:%}')
  print(f'pcr: {pcr / tot:%}')


if __name__ == '__main__':
  eval_with_env = lambda x: eval(x, globals(), globals())

  parser = ArgumentParser()
  parser.add_argument('-L', '--logdir', default='out/hea_amp', type=Path, help='logdir to pretrained ckpt')
  parser.add_argument('--test_fp', default=TEST_FILE, type=Path)
  parser.add_argument('--step',    default=10,        type=int)
  parser.add_argument('--eps',     default=pi/100,    type=eval_with_env)
  parser.add_argument('--alpha',   default=pi/1000,   type=eval_with_env)
  args, _ = parser.parse_known_args()

  run(args)
