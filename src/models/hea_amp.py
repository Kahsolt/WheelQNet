#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/30

from src.models import Model
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


class MyModel(Model):

  def __init__(self, args, n_qubits:int=4, depth:int=3, rots:List[str]=['RX', 'RY'], entgl:str='CNOT', entgl_rule:str='linear'):
    super().__init__(n_qubits)

    self.args = args

    self.encoder   = VQC_AmplitudeEmbedding     # 2^n unit features => n qubits
    self.ansatz    = VQC_HardwareEfficientAnsatz(n_qubits, rots, entgl, entgl_rule, depth)
    self.out       = Hadamard(wires=0)
    self.measure   = Probability(wires=0)
    self.criterion = CrossEntropyLoss()

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

  def get_loss(self, o:QTensor, y:QTensor):
    return self.criterion(y, o)


def get_model(args) -> Model:
  n_qubits = 4
  assert args.n_qubits is None, f'n_qubits is fixed for {args.model}: {n_qubits}'
  depth = args.depth or 3
  rots = (args.hea_rots or 'RX,RY').split(',')
  entgl = args.hea_entgl or 'CNOT'
  entgl_rule = args.hea_entgl_rule or 'linear'
  return MyModel(args, n_qubits, depth, rots, entgl, entgl_rule)
