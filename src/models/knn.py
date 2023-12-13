#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/14 

from src.models import Model
from src.features import feats_all as feats
from src.features.pca import *
from src.utils import *

N_SAMPLES = 790
N_FEAT_PCA = 4

def mode(x:QTensor) -> int:
  x = x.numpy()
  val, cnt = Counter(x).most_common()[0]
  return val


class kNN(Model):

  def __init__(self, args, k:int=5, n_qubits:int=0):
    super().__init__(args, n_qubits)

    self.k = k

    self.ref_x = Parameter(shape=[N_SAMPLES, N_FEAT_PCA], dtype=kfloat32)  # 直接存储训练集
    self.ref_x.requires_grad = False
    self.ref_y = Parameter(shape=[N_SAMPLES], dtype=kint64)    # 直接存储训练集
    self.ref_y.requires_grad = False

  def forward(self, x:QTensor) -> QTensor:
    self.ref_x.data = x.data

  def loss(self, o:QTensor, y:QTensor) -> QTensor:
    self.ref_y.data = y.data

  def inference(self, x:QTensor) -> QTensor:
    ref_x = QTensor(self.ref_x.data)
    ref_y = QTensor(self.ref_y.data)
    offset = int(x.shape == ref_x.shape and tensor.allclose(x, ref_x, 1e-8, 1e-8, False))

    x_ex = tensor.unsqueeze(x, axis=1)          # [B, 1, D]
    ref_x_ex = tensor.unsqueeze(ref_x, axis=0)  # [1, N, D]
    if 'L1': diff = tensor.abs(x_ex - ref_x_ex)
    else:    diff = (x_ex - ref_x_ex) ** 2
    dist = tensor.sums(diff, axis=2)            # [B, N]
    idx_topk = tensor.argsort(dist, axis=1, descending=False)[:, offset:offset+self.k]
    votes = ref_y[idx_topk]
    return QTensor([mode(vote) for vote in votes], dtype=kint64)

  def reprocess(self, df:DataFrame) -> Tuple[QTensor, QTensor]:
    X, Y = self.split_df(df, feats)
    if 'pca':
      fp_pca = Path(self.args.log_dp) / 'pca_stats.pkl'
      if Y is not None:   # train
        args = get_pca_args()
        args.dim = N_FEAT_PCA
        args.scaler = 'std'
        X_np = X.numpy()
        X_np, (scaler, reducer) = run_pca(args, X_np, ret_ops=True)
        save_pca_pickle(fp_pca, scaler, reducer)
        X = QTensor(X_np)
      else:               # infer
        scaler, reducer = load_pca_pickle(fp_pca)
        X_np = X.numpy()
        X_np = apply_pca(X_np, scaler, reducer)
        X = QTensor(X_np)

      if 'plot' and N_FEAT_PCA == 3:
        import matplotlib.pyplot as plt
        plt.clf()
        ax = plt.axes(projection='3d')
        if Y is not None:
          mask = (Y == 1).numpy()
          Z_pos = X_np[ mask]
          Z_neg = X_np[~mask]
          ax.scatter3D(Z_pos[:, 0], Z_pos[:, 1], Z_pos[:, 2], s=5, c='b', label='Survived')
          ax.scatter3D(Z_neg[:, 0], Z_neg[:, 1], Z_neg[:, 2], s=5, c='r', label='Dead')
          plt.legend()
        else:
          ax.scatter3D(X_np[:, 0], X_np[:, 1], X_np[:, 2], s=5, c='g')
        plt.tight_layout()
        plt.show()
    return X, Y


def get_model(args) -> kNN:
  assert args.n_qubits is None, f'n_qubits is fixed for {args.model}: 0'
  assert args.depth is None, f'no depth for {args.model}'
  # kNN doesn't need train, but we save the trainset in .forward()/.loss() as a single batch
  args.epochs = 1
  args.batch_size = 100000
  return kNN(args, args.knn)
