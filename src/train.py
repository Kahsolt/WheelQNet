#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/27

from argparse import ArgumentParser
from src.utils import *


def train(args):
  pass


def get_cmd_args():
  parser = ArgumentParser()
  parser.add_argument('-M')
  args, _ = parser.parse_known_args()
  return args
