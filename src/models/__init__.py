#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/30

from src.features import TARGET
from src.utils import *


class Model(Module):

  def __init__(self, args, n_qubits:int=4, dtype:int=kcomplex64):
    super().__init__()

    self.args = args
    self.n_qubits = n_qubits
    self.vqm = QMachine(n_qubits, dtype)

  def vqm_reset(self, x:QTensor) -> QMachine:
    self.vqm.reset_states(x.shape[0])   # bs
    return self.vqm

  def split_df(self, df:DataFrame, feats:List[str]) -> Tuple[QTensor, QTensor]:
    Y = QTensor(df[TARGET], kint64) if TARGET in df.columns else None
    X = QTensor(df[feats], kfloat32)
    return X, Y

  def forward(self, x:QTensor) -> QTensor:
    raise NotImplementedError

  def inference(self, x:QTensor) -> QTensor:
    raise NotImplementedError

  def loss(self, o:QTensor, y:QTensor) -> QTensor:
    raise NotImplementedError

  def reprocess(self, df:DataFrame) -> Tuple[QTensor, QTensor]:
    raise NotImplementedError


class ModelCE(Model):

  def __init__(self, args, n_qubits:int=4, dtype:int=kcomplex64):
    super().__init__(args, n_qubits, dtype)

    self.criterion = CrossEntropyLoss()

  def inference(self, x:QTensor) -> QTensor:
    return self(x).argmax(-1, keepdims=False)

  def loss(self, o:QTensor, y:QTensor) -> QTensor:
    return self.criterion(y, o)


class ModelMSE(Model):

  def __init__(self, args, n_qubits:int=4, dtype:int=kcomplex64):
    super().__init__(args, n_qubits, dtype)

    self.criterion = MeanSquaredError()

  def inference(self, x:QTensor) -> QTensor:
    return self(x) > 0.5

  def loss(self, o:QTensor, y:QTensor) -> QTensor:
    return self.criterion(y.astype(kfloat32), o)
