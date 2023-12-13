#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/29

from argparse import ArgumentParser, Namespace

from src.runner import load_pretrained_env
from src.utils import *


def eval(fp:str) -> ndarray:      # NOTE: export API for the contest judger
  args = Namespace()
  args.logdir = BASE_PATH / 'out' / 'knnq'
  args.test_fp = Path(fp)
  return run_eval(args)


def run_eval(args) -> ndarray:
  model, runner = load_pretrained_env(args)
  X, _ = model.reprocess(pd.read_csv(args.test_fp))
  return runner.infer(X).numpy()


def get_eval_args():
  parser = ArgumentParser()
  parser.add_argument('-L', '--logdir', type=Path, help='logdir to pretrained ckpt')
  parser.add_argument('--test_fp', default=TEST_FILE, type=Path)
  args, _ = parser.parse_known_args()
  return args


if __name__ == '__main__':
  args = get_eval_args()
  if args.logdir:
    preds = run_eval(args)
  else:
    preds = eval(TEST_FILE)
  print(preds)
