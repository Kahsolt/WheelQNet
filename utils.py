#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/28

import json
import pickle as pkl
from pathlib import Path
from traceback import print_exc
from typing import *

import numpy as np
from numpy import ndarray

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'data'
PROCESSED_PATH = BASE_PATH / 'processed'
PROCESSED_PATH.mkdir(exist_ok=True)

TRUTH_FILE = PROCESSED_PATH / 'contest' / 'truth.txt'
PROVIDER_FILES = {
  'contest': {
    'train': 'train.csv',
    'test':  'test_without_label.csv',
  },
  'kaggle': {
    'train': 'train.csv',
    'test':  'test.csv',
  },
}

SEED = 114514

mean = lambda x: sum(x) / len(x) if len(x) else 0.0


def seed_everything(seed:int):
  import random
  import numpy
  import pyvqnet
  random.seed(seed)
  numpy.random.seed(seed)
  pyvqnet.utils.set_random_seed(seed)


def get_data_fp(split:str='train', provider:str='contest') -> Path:
  return DATA_PATH / provider / PROVIDER_FILES[provider][split]

def get_processed_fp(split:str='train', provider:str='contest') -> Path:
  return PROCESSED_PATH / provider / PROVIDER_FILES[provider][split]

def get_truth() -> ndarray:
  with open(TRUTH_FILE, 'r', encoding='utf-8') as fh:
    return np.asarray([int(x) for x in fh.read().strip().split('\n')], dtype=np.int64)


def save_pkl(data:Any, fp:Path):
  print(f'>> save pkl {fp}')
  with open(fp, 'wb') as fh:
    pkl.dump(data, fh)

def load_pkl(fp:Path) -> Any:
  print(f'>> load pkl {fp}')
  with open(fp, 'rb') as fh:
    return pkl.load(fh)

def save_json(data:Any, fp:Path):
  print(f'>> save json {fp}')
  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump(data, fh, indent=2, ensure_ascii=False)

def load_json(fp:Path) -> Any:
  print(f'>> load json {fp}')
  with open(fp, 'r', encoding='utf-8') as fh:
    return json.load(fh)
