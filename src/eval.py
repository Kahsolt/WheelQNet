#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/29

from argparse import ArgumentParser, Namespace
from importlib import import_module

from src.models import Model
from src.runner import Runner
from src.utils import *


def eval(fp:str) -> ndarray:
  args = Namespace()
  args.logdir = BASE_PATH / 'out' / 'submit'
  args.test_fp = Path(fp)
  return run(args)


def run(args) -> ndarray:
  log_dp: Path = args.logdir
  assert log_dp.is_dir(), f'pretrained log_dp no exists: {log_dp}'

  data = load_json(log_dp / 'log.json')
  for k, v in data['args'].items():
    setattr(args, k, v)

  seed_everything(args.seed)
  mod = import_module(f'src.models.{args.model}')
  model: Model = getattr(mod, 'get_model')(args)
  X, _ = model.reprocess(pd.read_csv(args.test_fp))
  runner = Runner(args, model)
  runner.load_ckpt(log_dp / 'model.ckpt')
  return runner.infer(X).numpy()


def get_cmd_args():
  parser = ArgumentParser()
  parser.add_argument('-L', '--logdir', type=Path, help='logdir to pretrained ckpt')
  parser.add_argument('--test_fp', default=TEST_FILE, type=Path)
  args, _ = parser.parse_known_args()
  return args


if __name__ == '__main__':
  run(get_cmd_args())
