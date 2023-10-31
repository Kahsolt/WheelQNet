#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/31

from src.models import ModelMSE
from src.utils import *

# implments the ansatz in "Circuit-centric quantum classifiers": https://arxiv.org/abs/1804.00633

feats = [
  'Title',            # [0, 4]
  'cnt(Name_lst)',    # [1, 6]
  'cnt(Name_fst)',    # [1, 13]
  'Name_ex',          # [0, 1]
  'bin(Age)',         # [0, 4]
  'cat(Sex)',         # [0, 1]
  'Parch',            # [0, 6]
  'Family_add',       # [0, 10]
  'Family_sub-min',   # [0, 11]
  'Family_sub_abs',   # [0, 6]
  #'Family_mul',       # [0, 16]
  'Pclass-1',         # [0, 2]
  'bin(Fare)',        # [0, 5]
  'log(Fare)',        # [1.3894144, 4.849553]
  'bin(log(Fare))',   # [0, 4]
  'cat(Cabin_pf)',    # [0, 8]
  'cat(Ticket_pf)',   # [0, 11]
]


class CU3(Module):
  
  def __init__(self, control_qubit:int, rot_qubit:int):
    super().__init__()

    self.control_qubit = control_qubit
    self.rot_qubit = rot_qubit
    self.params = Parameter(shape=[3], initializer=zeros, dtype=kcomplex64)

  def forward(self, vqm:QMachine):
    VQC_CRotCircuit(self.params, self.control_qubit, self.rot_qubit, vqm)


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
    self.b        = Parameter(shape=[1], initializer=zeros, dtype=kfloat32)

  def forward(self, x:QTensor):
    vqm = self.vqm_reset(x)
    self.encoder(x, vqm)
    for rot   in self.rot1:   rot(q_machine=vqm)
    for entgl in self.entgl1: entgl.forward(vqm)    # NOTE: unknown TypeError, directly call `.forward()` instead
    for rot   in self.rot2:   rot(q_machine=vqm)
    for entgl in self.entgl2: entgl.forward(vqm)
    self.rot3(q_machine=vqm)
    if 'computaional basis project':
      prob = self.get_prob(vqm)
      out = prob[:, 1]
    else:   # pauli-z expectation
      expval = self.get_exp(vqm)
      out = (expval + 1) / 2
    return out - self.b

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
