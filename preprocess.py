#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/27

from argparse import ArgumentParser
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

from features import *
from utils import *


def walk_apply(dp:Path, processor:Callable, reverse:bool=False):
  fps = sorted(dp.iterdir())
  for fp in (reversed(fps) if reverse else fps):
    if fp.is_dir():
      walk_apply(fp, processor, reverse)
    elif fp.is_file() and fp.suffix == '.csv':
      try: processor(fp)
      except: print_exc()


def make_truth():
  fp_o = TRUTH_FILE
  if fp_o.exists() and not args.overwrite:
    print(f'>> ignore truth due to file exists')
    return

  df_i = pd.read_csv(get_data_fp('test'))
  df_r = pd.read_csv(get_data_fp('train', 'kaggle'))
  truth = [
    df_r[df_r['PassengerId'] == id]['Survived'].item()
      for id in df_i['PassengerId']
  ]
  fp_o.parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(fp_o, truth, fmt='%d')


def make_namelists(fp:Path):
  fp_out = PROCESSED_PATH / fp.relative_to(DATA_PATH)
  dp_out = fp_out.with_name('namelists')
  if dp_out.exists() and not args.overwrite:
    print(f'>> ignore {fp} due to folder exists')
    return

  print(f'<< anaylizing {fp}')
  dp_out.mkdir(parents=True, exist_ok=True)
  extract_namelists(pd.read_csv(fp), dp_out)
  print(f'>> saving namelist to {dp_out}')


def make_processed(fp:Path):
  fp_out = PROCESSED_PATH / fp.relative_to(DATA_PATH)
  if fp_out.exists() and not args.overwrite:
    print(f'>> ignore {fp} due to file exists')
    return

  print(f'<< processing {fp}')
  fp_out.parent.mkdir(parents=True, exist_ok=True)
  dp_namelists = fp_out.with_name('namelists')
  assert dp_namelists.exists(), '>> namelists not found :('
  namelists = load_namelists(dp_namelists)
  fp_stats = fp_out.with_name('stats.pkl')
  df = pd.read_csv(fp)
  if fp.name.startswith('train'):
    df, stats = extract_features(df, namelists)
    save_pkl(stats, fp_stats)
  else:   # test
    assert fp_stats.exists(), '>> stats.pkl is not found :('
    stats = load_pkl(fp_stats)
    df, _ = extract_features(df, namelists, stats)
  df.to_csv(fp_out, index=False)
  print(f'>> saving processed to {fp_out}')


def make_plots(fp:Path):
  dp_out = fp.parent / (fp.stem + '.plots')
  if dp_out.exists() and not args.overwrite:
    print(f'>> ignore {fp} due to folder exists')
    return

  print(f'<< plotting {fp}')
  dp_out.mkdir(parents=True, exist_ok=True)
  extract_hists(pd.read_csv(fp), dp_out)
  print(f'>> saving plots to {dp_out}')


def make_report(fp:Path):
  fp_out = fp.with_suffix('.html')
  if fp_out.exists() and not args.overwrite:
    print(f'>> ignore {fp} due to file exists')
    return

  print(f'<< analyzing {fp}')
  fp_out.parent.mkdir(parents=True, exist_ok=True)
  ProfileReport(pd.read_csv(fp), explorative=True).to_file(fp_out)
  print(f'>> saving report to {fp_out}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-f', '--overwrite', action='store_true', help='force overwrite')
  args = parser.parse_args()

  # the truth label for contest testset
  make_truth()

  # generated global stats, mirror DATA_PATH to PROCESSED_PATH
  walk_apply(DATA_PATH, make_namelists)

  # make features, mirror DATA_PATH to PROCESSED_PATH
  walk_apply(DATA_PATH, make_processed, reverse=True)   # assure process 'train' before 'test'

  # generated bi-variate histograms plots
  walk_apply(PROCESSED_PATH, make_plots)

  # generated data profiling reports
  walk_apply(DATA_PATH, make_report)
  walk_apply(PROCESSED_PATH, make_report)
