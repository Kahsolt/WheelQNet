#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/29

import warnings ; warnings.filterwarnings('ignore')

import random
import json
import pickle as pkl
from datetime import datetime
from pathlib import Path
from traceback import print_exc
from typing import *

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
import pyvqnet as vq

from src.vqi import *

BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / 'data'
PROCESSED_PATH = BASE_PATH / 'processed'
LOG_PATH = BASE_PATH / 'log'
SRC_PATH = BASE_PATH / 'src'
MODEL_PATH = SRC_PATH / 'models'

PROVIDER_FILES = {
  'contest': {
    'train': 'train.csv',
    'test':  'test_without_label.csv',
  },
}
TEST_FILE = PROCESSED_PATH / 'contest' / PROVIDER_FILES['contest']['test']

MODELS = [fp.stem for fp in MODEL_PATH.iterdir() if not fp.stem.startswith('__')]

SEED = 114514

mean = lambda x: sum(x) / len(x) if len(x) else 0.0
p_zeros = lambda n=1: QTensor([0.0]*n)
dtype_str = get_readable_dtype_str

def data_generator_qtensor(X:QTensor, Y:QTensor, batch_size:int=32, shuffle:bool=True):
  for X, Y in data_generator(X.numpy(), Y.numpy(), batch_size, shuffle):
    yield QTensor(X), QTensor(Y)

def get_param_cnt(model:Module) -> int:
  return sum([p.numel() for p in model.parameters() if p.requires_grad])


def seed_everything(seed:int):
  random.seed(seed)
  np.random.seed(seed)
  vq.utils.set_random_seed(seed)

def ts_path() -> str:
  return str(datetime.now()).replace(' ', '_').replace(':', '')


def minmax_norm(X:QTensor, ret_stats:bool=False) -> Union[QTensor, Tuple[QTensor, Tuple[QTensor, QTensor]]]:
    Xmax = tensor.max(X, 0)
    Xmin = tensor.min(X, 0)
    Xn = (X - Xmin) / (Xmax - Xmin)
    return (Xn, (Xmin, Xmax)) if ret_stats else Xn

def rescale_norm(X:QTensor, low:float=0, high:float=np.pi):
  assert low < high
  vrng = high - low
  X = minmax_norm(X)
  return X * vrng + low

def l2_norm(X:QTensor) -> QTensor:
  return X / tensor.sqrt(tensor.sums(X**2, -1, keepdims=True))


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
