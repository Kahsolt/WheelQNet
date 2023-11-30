#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/30

from src.models import ModelMSE
from src.models.ccqc import CU3
from src.utils import *

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


class WheelQNet(ModelMSE):

  def __init__(self, args, n_qubits:int=4, depth:int=1):
    super().__init__(args, n_qubits)

    assert depth == 1, 'depth for WheelQNet ansatz is exactly 1 :('

    self.encoder  = VQC_AmplitudeEmbedding      # 2^n unit features => n qubits
    self.rot      = ModuleList([RY(wires=i, init_params=p_zeros()) for i in range(n_qubits)])
    self.entgl    = ModuleList([CU3((i+1)%n_qubits, i) for i in reversed(range(0, n_qubits))])
    self.meas_all = Probability(wires=list(range(self.n_qubits)))
    self.enc_ry   = lambda x, vqm: VQC_AngleEmbedding(x, wires=[0], q_machine=vqm, rotation='Y')
    self.bias     = ModuleList([RY(wires=0, init_params=p_zeros()) for _ in range(n_qubits**2)])
    self.meas_0   = Probability(wires=[0])

  def forward(self, x:QTensor) -> QTensor:
    ''' feature pass '''
    vqm = self.vqm_reset(x)
    self.encoder(x, vqm)
    for rot   in self.rot:   rot(q_machine=vqm)
    for entgl in self.entgl: entgl.forward(vqm)    # NOTE: unknown TypeError, directly call `.forward()` instead
    o = self.meas_all(vqm)      # [B, N^2], vrng [0, 1]
    ''' wheel pass '''
    z = (o - 0.5) * np.pi       # vrng [-pi/2, pi/2]
    vqm = self.vqm_reset(z)
    for i, bias in enumerate(self.bias):
      self.enc_ry(z[:, i:i+1], vqm)
      bias(q_machine=vqm)
    o = self.meas_0(vqm)
    return o[:, 0]

  def reprocess(self, df:DataFrame) -> Tuple[QTensor, QTensor]:
    X, Y = self.split_df(df, feats)
    X = rescale_norm(X, 0, np.pi)
    X = l2_norm(X)
    return X, Y


def get_model(args) -> WheelQNet:
  n_qubits = 4
  assert args.n_qubits is None, f'n_qubits is fixed for {args.model}: {n_qubits}'
  depth = 1
  assert args.depth is None, f'depth is fixed for {args.model}: {depth}'
  return WheelQNet(args, n_qubits, depth)
