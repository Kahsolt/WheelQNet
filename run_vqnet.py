#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/30

from src.train import run_train, get_train_args
from src.eval import run_eval, get_eval_args
from utils import get_truth


args = get_eval_args()
if args.logdir:
  pred = run_eval(args)
  truth = get_truth()
  acc = (pred == truth).sum().item() / len(truth)
  print(f'Acc: {acc:%}')
else:
  args = get_train_args()
  run_train(args)
