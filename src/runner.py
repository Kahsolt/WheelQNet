#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/30

from src.models import Model
from src.utils import *

StateDict = Dict[str, QTensor]


class Runner:
  
  def __init__(self, args, model:Model):
    self.args = args
    self.model = model

  def load_ckpt(self, fp:Path):
    param_dict: StateDict = load_parameters(fp)
    #self.model.load_state_dict(param_dict)   # NOTE: unknown bug crashes at lower level
    state_dict = self.model.state_dict()
    for k in param_dict:
      if k not in state_dict: continue
      state_dict[k] = param_dict[k]

  def save_ckpt(self, fp:Path):
    state_dict: StateDict = self.model.state_dict()
    param_dict = {k: v for k, v in state_dict.items() if k.endswith('.params')}
    save_parameters(param_dict, fp)

  def train(self, X:QTensor, Y:QTensor) -> Any:
    hp, model = self.args, self.model
    optim_cls = getattr(vq.optim, hp.optim)
    if hp.optim == 'SGD':
      optim = optim_cls(model.parameters(), lr=hp.lr, momentum=hp.sgd_momentum, nesterov=hp.sgd_nesterov)
    else:
      optim = optim_cls(model.parameters(), lr=hp.lr)

    steps = 0
    losses, accs = [], []
    for epoch in range(hp.epochs):
      ''' train '''
      model.train()
      for X_bs, Y_bs in data_generator_qtensor(X, Y, hp.batch_size, shuffle=True):
        out = model(X_bs)
        optim.zero_grad()
        loss = model.loss(out, Y_bs)
        loss.backward()
        optim._step()

        if steps % 10 == 0:
          losses.append(loss.item())
          print(f'>> step {steps}: {losses[-1]}')

        steps += 1

      ''' eval '''
      accs.append(self.eval(X, Y))
      print(f'[Epoch {epoch + 1}/{hp.epochs}] {accs[-1]:%}')

    return {
      'loss': losses,
      'acc': accs,
    }

  def infer(self, X:QTensor) -> QTensor:
    self.model.eval()
    pred = self.model.inference(X)
    return pred.astype(kint64)

  def eval(self, X:QTensor, Y:QTensor) -> float:
    pred = self.infer(X)
    return tensor.sums(pred == Y).item() / Y.shape[0]
