#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/29

from src.preprocess import *
from utils import TRUTH_FILE


def make_truth(overwrite:bool=False):
  fp_o = TRUTH_FILE
  if fp_o.exists() and not overwrite:
    print(f'>> ignore truth due to file exists')
    return

  df_i = get_rdata('test')
  df_r = get_rdata('train', 'kaggle')
  truth = [
    df_r[df_r[DATAID] == id][TARGET].item()
      for id in df_i[DATAID]
  ]
  fp_o.parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(fp_o, truth, fmt='%d')


if __name__ == '__main__':
  # the truth label for contest testset
  make_truth()

  # proxy preprocessing
  args = gat_cmd_args()
  preprocess(args)
