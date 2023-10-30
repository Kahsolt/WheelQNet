#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/29

import warnings ; warnings.filterwarnings('ignore')

import random
import json
import pickle as pkl
from pathlib import Path
from traceback import print_exc
from typing import *

import numpy as np
import pandas as pd
from numpy import ndarray
import pyvqnet as vq

from src.vqi import *

print('>> pyvqnet', vq.__version__)
print('>> numpy', np.__version__)

BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / 'data'
PROCESSED_PATH = BASE_PATH / 'processed'

PROVIDER_FILES = {
  'contest': {
    'train': 'train.csv',
    'test':  'test_without_label.csv',
  },
}

SEED = 114514

mean = lambda x: sum(x) / len(x) if len(x) else 0.0


def seed_everything(seed:int):
  random.seed(seed)
  np.random.seed(seed)
  vq.utils.set_random_seed(seed)


def get_rdata(split:str='train', provider:str='contest') -> DataFrame:
  fp = DATA_PATH / provider / PROVIDER_FILES[provider][split]
  return pd.read_csv(fp)

def get_data(split:str='train', provider:str='contest') -> DataFrame:
  fp = PROCESSED_PATH / provider / PROVIDER_FILES[provider][split]
  return pd.read_csv(fp)


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
