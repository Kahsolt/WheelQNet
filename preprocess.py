#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/27

from argparse import ArgumentParser
from pathlib import Path

import pandas
from ydata_profiling import ProfileReport

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'data'


def make_report(dp:Path):
  for fp in dp.iterdir():
    if fp.is_dir():
      make_report(fp)
    elif fp.suffix == '.csv':
      fp_out = fp.with_suffix('.html')
      if fp_out.exists() and not args.overwrite:
        print(f'>> ignore {fp} due to file exists')
        continue

      print(f'<< analyzing {fp}')
      ProfileReport(pandas.read_csv(fp), explorative=True).to_file(fp_out)
      print(f'>> saving report to {fp_out}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--overwrite', action='store_true')
  args = parser.parse_args()

  make_report(DATA_PATH)
