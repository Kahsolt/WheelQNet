#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/29

from src.models.ccqc import CCQC
from src.utils import *


class CCQCQ(CCQC):

  ''' pure-quantum CQCC, replacing the classical bias to a RY gate '''

  def __init__(self, args, n_qubits:int=4, depth:int=2):
    super().__init__(args, n_qubits, depth)

    self.enc_ry = lambda x, vqm: VQC_AngleEmbedding(x, wires=[0], q_machine=vqm, rotation='Y')

    del self.bias
    self.bias = RY(wires=0, init_params=p_zeros())

  def forward(self, x:QTensor):
    o = self.forward_no_bias(x)   # [B], vrng [0, 1]
    z = (o - 0.5) * np.pi         # vrng, [-pi/2, pi/2]
    z = tensor.unsqueeze(z, -1)   # [B, D=1]

    vqm = self.vqm_reset(z)
    self.enc_ry(z, vqm)
    self.bias(q_machine=vqm)
    return self.measure()         # [B], vrng [0, 1]


def get_model(args) -> CCQCQ:
  n_qubits = 4
  assert args.n_qubits is None, f'n_qubits is fixed for {args.model}: {n_qubits}'
  depth = 2
  assert args.depth is None, f'depth is fixed for {args.model}: {depth}'
  return CCQCQ(args, n_qubits, depth)
