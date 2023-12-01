#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/31

from src.models import ModelMSE
from src.features import feats
from src.utils import *

# implements the ansatz in "Circuit-centric quantum classifiers": https://arxiv.org/abs/1804.00633


class CU3(Module):
  
  def __init__(self, control_qubit:int, rot_qubit:int):
    super().__init__()

    self.control_qubit = control_qubit
    self.rot_qubit = rot_qubit
    self.params = Parameter(shape=[3], initializer=zeros, dtype=kcomplex64)

  def forward(self, vqm:QMachine):
    VQC_CRotCircuit(self.params, self.control_qubit, self.rot_qubit, vqm)


class Bais(Module):
  
  def __init__(self):
    super().__init__()

    self.params = Parameter(shape=[1], initializer=zeros, dtype=kfloat32)

  def forward(self, x:QTensor):
    return x - self.params


class CCQC(ModelMSE):

  def __init__(self, args, n_qubits:int=4, depth:int=2):
    super().__init__(args, n_qubits)

    assert depth == 2, 'depth for CCQC ansatz is exactly 2 :('

    self.encoder  = VQC_AmplitudeEmbedding     # flow the essay, 2^n unit features => n qubits
    self.rot1     = ModuleList([U3(wires=i, init_params=p_zeros(3)) for i in range(n_qubits)])
    self.entgl1   = ModuleList([CU3((i+1)%n_qubits, i) for i in reversed(range(0, n_qubits))])
    self.rot2     = ModuleList([U3(wires=i, init_params=p_zeros(3)) for i in range(n_qubits)])
    entgl2, ctrl_bit = [], 0
    for _ in range(n_qubits):
      rot_bit = (ctrl_bit - 3 + n_qubits) % n_qubits
      entgl2.append(CU3(ctrl_bit, rot_bit))
      ctrl_bit = (ctrl_bit - 3 + n_qubits) % n_qubits
    self.entgl2   = ModuleList(entgl2)
    self.rot3     = U3(wires=0, init_params=p_zeros(3))
    self.get_prob = Probability(wires=0)
    self.get_exp  = MeasureAll({'Z0': 1.0})
    self.bias     = Bais()      # NOTE: this must be a Module, otherwise weights will not be saved

  def forward_no_bias(self, x:QTensor) -> QTensor:
    vqm = self.vqm_reset(x)
    self.encoder(x, vqm)
    for rot   in self.rot1:   rot(q_machine=vqm)
    for entgl in self.entgl1: entgl.forward(vqm)    # NOTE: unknown TypeError, directly call `.forward()` instead
    for rot   in self.rot2:   rot(q_machine=vqm)
    for entgl in self.entgl2: entgl.forward(vqm)
    self.rot3(q_machine=vqm)
    return self.measure()    # [B], vrng [0, 1]

  def forward(self, x:QTensor) -> QTensor:
    o = self.forward_no_bias(x)   # [B], quantum
    return self.bias(o)           # [B], classical

  def measure(self) -> QTensor:
    vqm = self.vqm
    if 'computaional basis project':
      prob = self.get_prob(vqm)
      out = prob[:, 1]
    else:   # pauli-z expectation
      expval = self.get_exp(vqm)
      out = (expval + 1) / 2
    return out

  def reprocess(self, df:DataFrame) -> Tuple[QTensor, QTensor]:
    X, Y = self.split_df(df, feats)
    X = rescale_norm(X, 0, np.pi)
    X = l2_norm(X)
    return X, Y


def get_model(args) -> CCQC:
  n_qubits = 4
  assert args.n_qubits is None, f'n_qubits is fixed for {args.model}: {n_qubits}'
  depth = 2
  assert args.depth is None, f'depth is fixed for {args.model}: {depth}'
  return CCQC(args, n_qubits, depth)
