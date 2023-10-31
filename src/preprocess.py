#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/27

from argparse import ArgumentParser
from ydata_profiling import ProfileReport

from src.features import *

Processor = Callable[[Path, bool], None]


def walk_apply(dp:Path, processor:Processor, overwrite:bool=False, reverse:bool=False):
  fps = sorted(dp.iterdir())
  for fp in (reversed(fps) if reverse else fps):
    if fp.is_dir():
      walk_apply(fp, processor, overwrite, reverse)
    elif fp.is_file() and fp.suffix == '.csv':
      try: processor(fp, overwrite)
      except: print_exc()


def make_namelists(fp:Path, overwrite:bool=False):
  fp_out = PROCESSED_PATH / fp.relative_to(DATA_PATH)
  dp_out = fp_out.with_name('namelists')

  print(f'<< gathering {fp}')
  dp_out.mkdir(parents=True, exist_ok=True)
  extract_namelists(pd.read_csv(fp), dp_out)
  print(f'>> saving namelists to {dp_out}')


def make_processed(fp:Path, overwrite:bool=False):
  fp_out = PROCESSED_PATH / fp.relative_to(DATA_PATH)
  if fp_out.exists() and not overwrite:
    print(f'>> ignore process {fp} due to file exists')
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


def make_plots(fp:Path, overwrite:bool=False):
  dp_out = fp.parent / (fp.stem + '.plots')
  if dp_out.exists() and not overwrite:
    print(f'>> ignore plots {fp} due to folder exists')
    return

  print(f'<< plotting {fp}')
  dp_out.mkdir(parents=True, exist_ok=True)
  extract_hists(pd.read_csv(fp), dp_out)
  print(f'>> saving plots to {dp_out}')


def make_report(fp:Path, overwrite:bool=False):
  fp_out = fp.with_suffix('.html')
  if fp_out.exists() and not overwrite:
    print(f'>> ignore report {fp} due to file exists')
    return

  print(f'<< profiling {fp}')
  fp_out.parent.mkdir(parents=True, exist_ok=True)
  ProfileReport(pd.read_csv(fp), explorative=True).to_file(fp_out)
  print(f'>> saving report to {fp_out}')


def get_preprocess_args():
  parser = ArgumentParser()
  parser.add_argument('-f', '--overwrite', action='store_true', help='force overwrite')
  args, _ = parser.parse_known_args()
  return args


def run_preprocess(args):
  PROCESSED_PATH.mkdir(exist_ok=True)

  # generated global stats, mirror DATA_PATH to PROCESSED_PATH
  walk_apply(DATA_PATH, make_namelists, args.overwrite)

  # make features, mirror DATA_PATH to PROCESSED_PATH
  walk_apply(DATA_PATH, make_processed, args.overwrite, reverse=True)   # assure process 'train' before 'test'

  # generated bi-variate histograms plots
  walk_apply(PROCESSED_PATH, make_plots, args.overwrite)

  # generated data profiling reports
  walk_apply(DATA_PATH,      make_report, args.overwrite)
  walk_apply(PROCESSED_PATH, make_report, args.overwrite)


if __name__ == '__main__':
  run_preprocess(get_preprocess_args())
