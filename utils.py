#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/28

from src.utils import *

TRUTH_FILE = PROCESSED_PATH / 'contest' / 'truth.txt'

PROVIDER_FILES.update({
  'kaggle': {
    'train': 'train.csv',
    'test':  'test.csv',
  },
})


def get_truth() -> ndarray:
  with open(TRUTH_FILE, 'r', encoding='utf-8') as fh:
    data = [int(x) for x in fh.read().strip().split('\n')]
  return np.asarray(data, dtype=np.int64)
