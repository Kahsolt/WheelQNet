#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/30

from src.models import ModelCE
from src.features import feats
from src.utils import *


class HEA_Amp(ModelCE):

  def __init__(self, args, n_qubits:int=4, depth:int=3, rots:List[str]=['RX', 'RY'], entgl:str='CNOT', entgl_rule:str='linear'):
    super().__init__(args, n_qubits)

    self.encoder = VQC_AmplitudeEmbedding     # 2^n unit features => n qubits
    self.ansatz  = VQC_HardwareEfficientAnsatz(n_qubits, rots, entgl, entgl_rule, depth, initial=p_zeros())
    self.out     = Hadamard(wires=0)
    self.measure = Probability(wires=0)

  def forward(self, x:QTensor):
    vqm = self.vqm_reset(x)
    self.encoder(x, vqm)
    self.ansatz(vqm)
    self.out(q_machine=vqm)
    return self.measure(vqm)

  def reprocess(self, df:DataFrame) -> Tuple[QTensor, QTensor]:
    X, Y = self.split_df(df, feats)
    X = rescale_norm(X, 0, np.pi)
    X = l2_norm(X)
    return X, Y


def get_model(args) -> HEA_Amp:
  n_qubits = 4
  assert args.n_qubits is None, f'n_qubits is fixed for {args.model}: {n_qubits}'
  depth = args.depth or 3
  rots = (args.hea_rots or 'RX,RY').split(',')
  entgl = args.hea_entgl or 'CNOT'
  entgl_rule = args.hea_entgl_rule or 'linear'
  return HEA_Amp(args, n_qubits, depth, rots, entgl, entgl_rule)
