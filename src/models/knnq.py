#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/13 

# https://vqnet20-tutorial.readthedocs.io/en/latest/rst/qml_demo.html#quantum-kmeans
# https://en.wikipedia.org/wiki/Swap_test

from src.models.knn import *

from tqdm import tqdm


class kNNq(kNN):

  def __init__(self, args, k:int=5, n_qubits:int=1+N_FEAT_PCA*2):
    super().__init__(args, k, n_qubits)

    self.n_enc = (n_qubits - 1) // 2

    self.enc_r = lambda x, vqm: VQC_AngleEmbedding(x, wires=[1, 3, 5, 7], q_machine=vqm, rotation='Y')
    self.enc_x = lambda x, vqm: VQC_AngleEmbedding(x, wires=[2, 4, 6, 8], q_machine=vqm, rotation='Y')
    self.cswap = VQC_CSWAPcircuit
    self.H     = Hadamard(wires=0)
    self.meas  = Probability(wires=0)

  def inference(self, x:QTensor) -> QTensor:
    ref_x = QTensor(self.ref_x.data)
    ref_y = QTensor(self.ref_y.data)
    offset = int(ref_x.shape == x.shape and tensor.allclose(x, ref_x, 1e-8, 1e-8, False))

    preds = []
    for x_i in tqdm(x, total=x.shape[0]):     # 逐样本处理
      vqm = self.vqm_reset(ref_x)
      self.enc_r(ref_x, vqm)      # 对比所有参考样本
      self.enc_x(x_i, vqm)
      self.H(q_machine=vqm)
      for q in range(self.n_enc):
        self.cswap([0, 2*q+1, 2*(q+1)], vqm)
      self.H(q_machine=vqm)
      pdist = self.meas(vqm)[:, 1]   # Z-basis meansure / prob of |1> as pseudo-distance
      idx_topk = tensor.argsort(pdist, axis=0, descending=False)[offset:offset+self.k]
      votes = ref_y[idx_topk]
      preds.append(mode(votes))
    return QTensor(preds, dtype=kint64)

  def reprocess(self, df:DataFrame) -> Tuple[QTensor, QTensor]:
    X, Y = super().reprocess(df)
    X = rescale_norm(X, 0, np.pi)
    return X, Y


def get_model(args) -> kNNq:
  n_qubits = 1 + N_FEAT_PCA * 2
  assert args.n_qubits is None, f'n_qubits is fixed for {args.model}: {n_qubits}'
  assert args.depth is None, f'no depth for {args.model}'
  # kNNq doesn't need train, but we save the trainset in .forward()/.loss() as a single batch
  args.epochs = 1
  args.batch_size = 100000
  return kNNq(args, args.knn, n_qubits)
