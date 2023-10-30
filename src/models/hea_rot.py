#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/30

from src.models import Model
from src.models.hea_amp import feats
from src.utils import *


class MyModel(Model):

  def __init__(self, args, n_qubits:int=8, depth:int=2, enc_rots:List[str]=['RY', 'RZ'], rots:List[str]=['RX', 'RY'], entgl:str='CNOT', entgl_rule:str='linear'):
    super().__init__(n_qubits)

    self.args = args

    #self.encoder = VQC_AngleEmbedding     # n features => n qubits
    encoders = []
    for i in range(n_qubits):
      for j in range(len(enc_rots)):
        i_feat = i * len(enc_rots) + j
        if i_feat >= len(feats): break
        wire = i_feat % n_qubits
        rot = globals()[enc_rots[j]]
        encoders.append(rot(wires=wire, has_params=True, trainable=True, init_params=p_zeros()))
    self.encoder   = ModuleList(encoders)
    self.ansatz    = VQC_HardwareEfficientAnsatz(n_qubits, rots, entgl, entgl_rule, depth)
    self.out       = Hadamard(wires=0)
    self.measure   = Probability(wires=0)
    self.criterion = CrossEntropyLoss()

  def forward(self, x:QTensor):
    vqm = self.vqm_reset(x)
    for i, enc in enumerate(self.encoder):
      enc(x[:, i], vqm)
    self.ansatz(vqm)
    self.out(q_machine=vqm)
    return self.measure(vqm)

  def reprocess(self, df:DataFrame) -> Tuple[QTensor, QTensor]:
    X, Y = self.split_df(df, feats)
    X = rescale_norm(X, 0, np.pi)
    return X, Y

  def get_loss(self, o:QTensor, y:QTensor):
    return self.criterion(y, o)


def get_model(args) -> Model:
  enc_rots = (args.amp_enc_rots or 'RY,RZ').split(',')
  n_qubits = int(np.ceil(len(feats) / len(enc_rots)))
  assert args.n_qubits is None, f'n_qubits is auto-computed for {args.model}: {n_qubits}'
  depth = args.depth or 2
  rots = (args.hea_rots or 'RX,RY').split(',')
  entgl = args.hea_entgl or 'CNOT'
  entgl_rule = args.hea_entgl_rule or 'linear'
  return MyModel(args, n_qubits, depth, enc_rots, rots, entgl, entgl_rule)
