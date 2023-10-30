#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/30

from src.features import TARGET
from src.utils import *


class Model(Module):

  def __init__(self, n_qubits:int=4, dtype:int=kcomplex64):
    super().__init__()

    self.n_qubits = n_qubits
    self.vqm = QMachine(n_qubits, dtype)

  def vqm_reset(self, x:QTensor) -> QMachine:
    self.vqm.reset_states(x.shape[0])   # bs
    return self.vqm

  def forward(self, x:QTensor):
    raise NotImplementedError

  def split_df(self, df:DataFrame, feats:List[str]) -> Tuple[QTensor, QTensor]:
    Y = QTensor(df[TARGET], kint64) if TARGET in df.columns else None
    X = QTensor(df[feats], kfloat32)
    return X, Y

  def reprocess(self, df:DataFrame) -> Tuple[QTensor, QTensor]:
    raise NotImplementedError

  def get_loss(self, o:QTensor, y:QTensor):
    raise NotImplementedError
