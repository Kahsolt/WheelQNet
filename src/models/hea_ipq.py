#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/30

from src.models import ModelCE
from src.models.hea_amp import feats
from src.utils import *


class HEA_IPQ(ModelCE):

  def __init__(self, args, n_qubits:int=8, depth:int=2, rots:List[str]=['RX', 'RY'], entgl:str='CNOT', entgl_rule:str='linear'):
    super().__init__(args, n_qubits)

    self.encoder = VQC_IQPEmbedding     # n features => n qubits
    self.ansatz1 = VQC_HardwareEfficientAnsatz(n_qubits, rots, entgl, entgl_rule, depth//2, initial=p_zeros())
    self.ansatz2 = VQC_HardwareEfficientAnsatz(n_qubits, rots, entgl, entgl_rule, depth - depth//2, initial=p_zeros())
    self.out     = Hadamard(wires=0)
    self.measure = Probability(wires=0)

  def forward(self, x:QTensor):
    vqm = self.vqm_reset(x)
    self.encoder(x[:, :self.n_qubits], vqm)
    self.ansatz1(vqm)
    self.encoder(x[:, self.n_qubits:], vqm)
    self.ansatz2(vqm)
    self.out(q_machine=vqm)
    return self.measure(vqm)

  def reprocess(self, df:DataFrame) -> Tuple[QTensor, QTensor]:
    X, Y = self.split_df(df, feats)
    X = rescale_norm(X, 0, np.pi)
    return X, Y


def get_model(args) -> HEA_IPQ:
  n_qubits = 8
  assert args.n_qubits is None, f'n_qubits is fixed for {args.model}: {n_qubits}'
  depth = args.depth or 2
  rots = (args.hea_rots or 'RX,RY').split(',')
  entgl = args.hea_entgl or 'CNOT'
  entgl_rule = args.hea_entgl_rule or 'linear'
  return HEA_IPQ(args, n_qubits, depth, rots, entgl, entgl_rule)
