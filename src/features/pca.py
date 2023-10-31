#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/31

from argparse import ArgumentParser

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, Normalizer
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, FastICA
from sklearn.manifold import TSNE, MDS, Isomap

from src.utils import *

Scaler = Union[MinMaxScaler, MaxAbsScaler, StandardScaler, Normalizer]
Reducer = Union[PCA, KernelPCA, TruncatedSVD, FastICA, TSNE, MDS, Isomap]

SCALERS = {
  'minmax': lambda: MinMaxScaler(),
  'maxabs': lambda: MaxAbsScaler(),
  'std':    lambda: StandardScaler(),
  'norm':   lambda: Normalizer(),
}

REDUCERS = {
  'pca':    lambda args: PCA         (n_components=args.dim, random_state=SEED),
  'kpca':   lambda args: KernelPCA   (n_components=args.dim, kernel=args.kernel, random_state=SEED),
  'tsvd':   lambda args: TruncatedSVD(n_components=args.dim, random_state=SEED),
  'fica':   lambda args: FastICA     (n_components=args.dim, random_state=SEED),
  'tsne':   lambda args: TSNE        (n_components=args.dim, random_state=SEED),
  'mds':    lambda args: MDS         (n_components=args.dim, random_state=SEED),
  'isomap': lambda args: Isomap      (n_components=args.dim),
}


def run_pca(args, X:Union[DataFrame, ndarray], ret_ops:bool=False) -> Union[ndarray, Tuple[ndarray, Tuple[Scaler, Reducer]]]:
  if args.scaler != 'none':
    scaler: Scaler = SCALERS[args.scaler]()
    X = scaler.fit_transform(X)
  else:
    scaler = None

  reducer: Reducer = REDUCERS[args.method](args)
  Z = reducer.fit_transform(X)
  return (Z, (scaler, reducer)) if ret_ops else Z


def get_pca_args():
  parser = ArgumentParser()
  parser.add_argument('-M', '--method', default='pca',    choices=['pca', 'kpca', 'tsvd', 'fica', 'tsne', 'mds', 'isomap'])
  parser.add_argument('-K', '--kernel', default='linear', choices=['linear', 'poly', 'rbf', 'sigmoid', 'cosine'], help='kernel for KernelPCA')
  parser.add_argument('-S', '--scaler', default='maxabs', choices=['none', 'minmax', 'maxabs', 'std', 'norm'])
  parser.add_argument('-D', '--dim',    default=8, type=int)
  args, _ = parser.parse_known_args()
  return args
