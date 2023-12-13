#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/30

from importlib import import_module

from src.models import Model
from src.utils import *

StateDict = Dict[str, QTensor]


class Runner:
  
  def __init__(self, args, model:Model):
    self.args = args
    self.model = model

  def load_ckpt(self, fp:Path):
    ckpt_dict: StateDict = load_parameters(fp)
    state_dict = self.model.state_dict()
    if 'check keys':
      missing_keys = state_dict.keys() - ckpt_dict.keys()
      if missing_keys: print('>> missing_keys:', missing_keys)
      redundant_keys = ckpt_dict.keys() - state_dict.keys()
      if redundant_keys: print('>> redundant_keys:', redundant_keys)
    for k in ckpt_dict: state_dict[k] = ckpt_dict[k]    # override with ckpt
    self.model.load_state_dict(state_dict)

  def save_ckpt(self, fp:Path):
    state_dict: StateDict = self.model.state_dict()
    param_dict = {k: v for k, v in state_dict.items() if k.endswith('.params') or k in ['ref_x', 'ref_y']}
    save_parameters(param_dict, fp)

  def train(self, X:QTensor, Y:QTensor, run_eval:bool=True) -> Dict[str, List[float]]:
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
        if loss is None: break      # kNNq compatible
        loss.backward()
        optim._step()

        steps += 1

        if steps % 10 == 0:
          losses.append(loss.item())

      ''' eval '''
      if run_eval:
        accs.append(self.eval(X, Y))
        print(f'[Epoch {epoch + 1}/{hp.epochs}] acc: {accs[-1]:%}, loss: {mean(losses[-10:])}')

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


def load_pretrained_env(args) -> Tuple[Model, Runner]:
  log_dp: Path = args.logdir
  assert log_dp.is_dir(), f'pretrained log_dp no exists: {log_dp}'

  data = load_json(log_dp / 'log.json')
  for k, v in data['args'].items():
    setattr(args, k, v)

  seed_everything(args.seed)
  mod = import_module(f'src.models.{args.model}')
  model: Model = getattr(mod, 'get_model')(args)
  model.eval()
  runner = Runner(args, model)
  runner.load_ckpt(log_dp / 'model.ckpt')

  return model, runner
