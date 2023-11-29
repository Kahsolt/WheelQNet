#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/28 

from pathlib import Path
from pprint import pprint as pp
from argparse import ArgumentParser
from code import interact

from src.runner import load_parameters, StateDict


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('fp', type=Path, help='path to model.ckpt file')
  args = parser.parse_args()

  if not Path(args.fp).is_file():
    args.fp = args.fp / 'model.ckpt'

  param_dict: StateDict = load_parameters(args.fp)
  pp(param_dict)

  interact(local=globals())
