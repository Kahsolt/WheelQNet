#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/31

import matplotlib.pyplot as plt

from src.features.feature import TARGET
from src.features.pca import *
from src.models.hea_amp import feats
from utils import get_data


def run():
  args = get_pca_args()
  assert args.dim in [2, 3], '--dim must in [2, 3]'

  df = get_data('train')
  X = df[feats]
  Y = df[TARGET]

  Z, (scaler, reducer) = run_pca(args, X, ret_ops=True)
  if isinstance(reducer, (PCA, TruncatedSVD)):
    print('explained_variance:', reducer.explained_variance_)
    print('explained_variance_ratio:', reducer.explained_variance_ratio_)
    if isinstance(reducer, TruncatedSVD):
      print('singular_values:', reducer.singular_values_)
  elif isinstance(reducer, TSNE):
    print('kl_divergence:', reducer.kl_divergence_)
  elif isinstance(reducer, MDS):
    print('stress:', reducer.stress_)
  
  pos = (Y == 1).to_numpy()
  Z_pos, Z_neg = Z[pos], Z[~pos]

  plt.clf()
  if args.dim == 3:
    ax = plt.axes(projection='3d')
    ax.scatter3D(Z_pos[:, 0], Z_pos[:, 1], Z_pos[:, 2], s=5, c='b', label='Survived')
    ax.scatter3D(Z_neg[:, 0], Z_neg[:, 1], Z_neg[:, 2], s=5, c='r', label='Dead')
  elif args.dim == 2:
    ax = plt.axes()
    ax.scatter(Z_pos[:, 0], Z_pos[:, 1], s=5, c='b', label='Survived')
    ax.scatter(Z_neg[:, 0], Z_neg[:, 1], s=5, c='r', label='Dead')
  plt.legend()
  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  run()
