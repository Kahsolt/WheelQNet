#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/27

import sys
from argparse import ArgumentParser
from importlib import import_module

import matplotlib as mpl ; mpl.use('agg')
import matplotlib.pyplot as plt

from src.models import Model
from src.runner import Runner
from src.utils import *


def plot_metrics(metrics:dict, dp:Path):
  if not metrics: return
  plt.clf()
  plt.subplot(211) ; plt.title('loss')     ; plt.plot(metrics['loss'], 'b')
  plt.subplot(212) ; plt.title('accuracy') ; plt.plot(metrics['acc'],  'r')
  plt.tight_layout()
  plt.savefig(dp / 'loss_acc.png', dpi=600)


def run_train(args):
  log_dp = LOG_PATH / (args.name or ts_path())
  if log_dp.exists() and not args.overwrite:
    print(f'>> ignore due to logdir exists: {log_dp}')
    return
  log_dp.mkdir(parents=True, exist_ok=True)
  args.log_dp = str(log_dp.relative_to(BASE_PATH))

  is_knnq = args.model == 'knnq'
  try:
    seed_everything(args.seed)
    mod = import_module(f'src.models.{args.model}')
    model: Model = getattr(mod, 'get_model')(args)
    param_cnt = get_param_cnt(model)
    print('model:', args.model)
    print('  param_cnt:', param_cnt)
    X, Y = model.reprocess(get_data('train'))
    print('dataset:')
    print('  X.shape:', X.shape)
    print('  Y.shape:', Y.shape)
    runner = Runner(args, model)
    metrics: dict = runner.train(X, Y, run_eval=not is_knnq)
    runner.save_ckpt(log_dp / 'model.ckpt')
    plot_metrics(metrics, log_dp)
  except KeyboardInterrupt:
    print('Exit by Ctrl+C')
  except:
    interrupted = True
    print_exc()
  finally:
    env = locals()
    data = {
      'argv': ' '.join(sys.argv),
      'args': vars(args),
      'interrupted': env.get('interrupted', False),
      'data': {
        'X.shape': env['X'].shape if 'X' in env else None,
        'Y.shape': env['Y'].shape if 'Y' in env else None,
      },
      'model': {
        'param_cnt': env.get('param_cnt'),
      },
      'metrics': env.get('metrics'),
      'libs': {
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'pyvqnet': vq.__version__,
      }
    }
    save_json(data, log_dp / 'log.json')


def get_train_args():
  str_to_bool = lambda x: bool(x.lower() in ['true', '1', 'y', 'yes'])

  parser = ArgumentParser()
  # data
  parser.add_argument('-B', '--batch_size', default=64, type=int)
  parser.add_argument('-E', '--epochs',     default=20, type=int)
  # model
  parser.add_argument('-M', '--model', default='hea_amp', choices=MODELS)
  parser.add_argument('-W', '--wires', type=int, help='aka. n_qubits')
  parser.add_argument('-D', '--depth', type=int)
  # encoder: amplitude
  parser.add_argument('--amp_enc_rots', help='comma seperated X/Y/Z')
  # ansatz: HardwareEfficientAnsatz
  parser.add_argument('--hea_rots', help='comma seperated RX/RY/RZ')
  parser.add_argument('--hea_entgl', choices=['CNOT', 'CZ'])
  parser.add_argument('--hea_entgl_rule', choices=['linear', 'all'])
  # knn
  parser.add_argument('--knn', default=5, type=int, help='neighbor count for kNNq')
  # optim
  parser.add_argument('-O', '--optim', default='SGD', choices=['SGD', 'Adagrad', 'Adadelta', 'RMSProp', 'Adam', 'Adamax'])
  parser.add_argument('--lr', default=0.1, type=float)
  parser.add_argument('--sgd_momentum', default=0.9, type=float)
  parser.add_argument('--sgd_nesterov', default=False, type=str_to_bool)
  # misc
  parser.add_argument('--seed', default=SEED, type=int)
  parser.add_argument('--name', help='log folder name, overwrite the default')
  parser.add_argument('--overwrite', action='store_true')
  args, _ = parser.parse_known_args()
  # alias
  args.n_qubits = args.wires
  return args


if __name__ == '__main__':
  run_train(get_train_args())
