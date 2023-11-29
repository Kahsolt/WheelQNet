#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/30

from src.models import ModelCE
from src.models.hea_amp import feats
from src.utils import *


class HEA_Angle(ModelCE):

  def __init__(self, args, n_qubits:int=8, depth:int=2, enc_rots:List[str]=['Y', 'Z'], rots:List[str]=['RX', 'RY'], entgl:str='CNOT', entgl_rule:str='linear'):
    super().__init__(args, n_qubits)

    self.encoder1 = lambda x, vqm: VQC_AngleEmbedding(x, wires=list(range(n_qubits)), q_machine=vqm, rotation=enc_rots[0])     # n features => n qubits
    self.encoder2 = lambda x, vqm: VQC_AngleEmbedding(x, wires=list(range(n_qubits)), q_machine=vqm, rotation=enc_rots[1])     # n features => n qubits
    self.ansatz  = VQC_HardwareEfficientAnsatz(n_qubits, rots, entgl, entgl_rule, depth, initial=p_zeros())
    self.out     = Hadamard(wires=0)
    self.measure = Probability(wires=0)

  def forward(self, x:QTensor):
    vqm = self.vqm_reset(x)
    self.encoder1(x[:, :self.n_qubits], vqm)
    self.encoder2(x[:, self.n_qubits:], vqm)
    self.ansatz(vqm)
    self.out(q_machine=vqm)
    return self.measure(vqm)

  def reprocess(self, df:DataFrame) -> Tuple[QTensor, QTensor]:
    X, Y = self.split_df(df, feats)
    X = rescale_norm(X, 0, np.pi)
    return X, Y


def get_model(args) -> HEA_Angle:
  n_qubits = 8
  assert args.n_qubits is None, f'n_qubits is fixed for {args.model}: {n_qubits}'
  depth = args.depth or 2
  enc_rots = (args.amp_enc_rots or 'Y,Z').split(',')
  rots = (args.hea_rots or 'RX,RY').split(',')
  entgl = args.hea_entgl or 'CNOT'
  entgl_rule = args.hea_entgl_rule or 'linear'
  return HEA_Angle(args, n_qubits, depth, enc_rots, rots, entgl, entgl_rule)
