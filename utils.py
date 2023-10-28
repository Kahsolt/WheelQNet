#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/28

import json
import pickle as pkl
from pathlib import Path
from traceback import print_exc
from typing import *

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

mean = lambda x: sum(x) / len(x) if len(x) else 0.0


def get_data_fp(split:str='train', provider:str='contest') -> Path:
  return DATA_PATH / provider / PROVIDER_FILES[provider][split]

def get_processed_fp(split:str='train', provider:str='contest') -> Path:
  return PROCESSED_PATH / provider / PROVIDER_FILES[provider][split]


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
