#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/31

from src.models.ccqc import CCQC
from src.runner import Runner
from src.train import get_train_args
from src.utils import *

# implements 8-3 encoder & 3-8 decoder using CCQC ansatz

def get_XY(n:int=8):
  X, Y = [], []
  k = int(np.ceil(np.log2(n)))
  for i in range(n):
    x = np.zeros(n) ; x[i] = 1
    X.append(x)
    y = np.asarray([int(e) for e in bin(i)[2:].rjust(k, '0')])
    Y.append(y)
  return QTensor(X, kint64), QTensor(Y, kint64)


class CCQC83Encoder(CCQC):

  def __init__(self, args, depth:int=2):
    super().__init__(args, 8, depth)

    #self.encoder = VQC_BasisEmbedding         # n bits => n qubits; NOTE: fuck it does not support batching :( 
    self.encoder = VQC_AmplitudeEmbedding     # 2^n unit features => n qubits
    self.rot3    = ModuleList([U3(wires=i, init_params=p_zeros(3)) for i in range(3)])
    self.get_exp = ModuleList([MeasureAll({f'Z{i}': 1.0})          for i in range(3)])
    self.bias    = Parameter(shape=[3], initializer=zeros, dtype=kfloat32)
    del self.get_prob, self.b

  def forward(self, x:QTensor):
    vqm = self.vqm_reset(x)
    self.encoder(x, vqm)
    for rot   in self.rot1:   rot(q_machine=vqm)
    for entgl in self.entgl1: entgl.forward(vqm)
    for rot   in self.rot2:   rot(q_machine=vqm)
    for entgl in self.entgl2: entgl.forward(vqm)
    for rot   in self.rot3:   rot(q_machine=vqm)
    expval = tensor.concatenate([self.get_exp[i](vqm) for i in range(3)], axis=1)
    out = (expval + 1) / 2
    return out - self.bias


class CCQC38Decoder(CCQC):

  def __init__(self, args, depth:int=2):
    super().__init__(args, 8, depth)

    self.encoder = VQC_AmplitudeEmbedding     # n bits => n qubits
    self.rot3    = ModuleList([U3(wires=i, init_params=p_zeros(3)) for i in range(8)])
    self.get_exp = ModuleList([MeasureAll({f'Z{i}': 1.0})          for i in range(8)])
    self.bias    = Parameter(shape=[8], initializer=zeros, dtype=kfloat32)
    del self.get_prob, self.b

  def forward(self, x:QTensor):
    vqm = self.vqm_reset(x)
    self.encoder(x, vqm)
    for rot   in self.rot1:   rot(q_machine=vqm)
    for entgl in self.entgl1: entgl.forward(vqm)
    for rot   in self.rot2:   rot(q_machine=vqm)
    for entgl in self.entgl2: entgl.forward(vqm)
    for rot   in self.rot3:   rot(q_machine=vqm)
    expval = tensor.concatenate([self.get_exp[i](vqm) for i in range(8)], axis=1)
    out = (expval + 1) / 2
    return out - self.bias


def run(args):
  X, Y = get_XY()
  print('X:') ; print(X)
  print('Y:') ; print(Y)

  if 'run encoder':
    model = CCQC83Encoder(args, args.depth)
    print('param_cnt:', get_param_cnt(model))
    runner = Runner(args, model)
    runner.train(X, Y, run_eval=False)
    print(model.inference(X).astype(kint64))

  if 'run decoder':
    model = CCQC38Decoder(args, args.depth)
    print('param_cnt:', get_param_cnt(model))
    runner = Runner(args, model)
    runner.train(Y, X, run_eval=False)
    print(model.inference(Y).astype(kint64))


if __name__ == '__main__':
  args = get_train_args()
  args.batch_size = 8
  args.epochs = 100
  args.model = ''
  args.depth = 2

  run(args)
