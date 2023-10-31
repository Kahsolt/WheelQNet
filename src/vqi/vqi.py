#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/29

''' export all APIs from pyqpanda & pyvqnet - binary classification related '''

# qpanda: https://pyqpanda-toturial.readthedocs.io/zh/latest/index.html
# vqnet: https://vqnet20-tutorial.readthedocs.io/en/latest/index.html

# dtype
from pyvqnet.dtype import kbool, kint8, kuint8, kint16, kint32, kint64, kfloat32, kfloat64, kcomplex64, kcomplex128

# tensor
from pyvqnet import tensor
from pyvqnet.tensor import QTensor

# data
from pyvqnet.data import data_generator, QuantumDataset

# nn: non-parametrical module
from pyvqnet.nn.pooling import MaxPool1D, AvgPool1D
from pyvqnet.nn.batch_norm import BatchNorm1d
from pyvqnet.nn.layer_norm import LayerNorm1d, LayerNormNd
from pyvqnet.nn.spectral_norm import Spectral_Norm
from pyvqnet.nn.dropout import Dropout
# nn: general parameter
from pyvqnet.nn.parameter import Parameter
# nn: general module
from pyvqnet.nn.module import Module, ModuleList
# nn: activation
from pyvqnet.nn.activation import Sigmoid, ReLu, LeakyReLu, Softmax, Softplus, Softsign, HardSigmoid, ELU, Tanh
# nn: loss
from pyvqnet.nn.loss import MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy, SoftmaxCrossEntropy, NLL_Loss, CrossEntropyLoss

# qnn: init qstate
from pyvqnet.qnn.template import BasisState, Random_Init_Quantum_State
# qnn: data encoding circuit
from pyvqnet.qnn.template import BasicEmbeddingCircuit, AngleEmbeddingCircuit, AmplitudeEmbeddingCircuit, IQPEmbeddingCircuits
# qnn: data transforming circuit
from pyvqnet.qnn.template import RotCircuit, CRotCircuit, CSWAPcircuit, RandomTemplate, BasicEntanglerTemplate, StronglyEntanglingTemplate, ComplexEntangelingTemplate, SimplifiedTwoDesignTemplate
from pyvqnet.qnn.template import QuantumPoolingCircuit
from pyvqnet.qnn.ansatz import HardwareEfficientAnsatz
# qnn: uccsd-specified components
from pyvqnet.qnn.template import MultiRZ_Gate, FermionicSingleExcitation, FermionicDoubleExcitation, UCCSD
# qnn: composed unitary (advanced gate)
from pyvqnet.qnn.template import CCZ, Controlled_Hadamard, FermionicSimulationGate
# qnn: ansatz building helper
from pyvqnet.qnn.qbroadcast import broadcast, subset, wires_all_to_all, wires_pyramid, wires_ring

# qnn: layers & applications
from pyvqnet.qnn.qembed import Quantum_Embedding
from pyvqnet.qnn.qlinear import QLinear
from pyvqnet.qnn.qcnn import QConv, Quanvolution
from pyvqnet.qnn.qae import QAElayer
from pyvqnet.qnn.qdrl import vmodel
from pyvqnet.qnn.qp.quantum_perceptron import QuantumNeuron, CombinationMap
# ref: https://vqnet20-tutorial.readthedocs.io/en/main/rst/qnn.html#svm
from pyvqnet.qnn.qsvm import QSVM, qsvm_load_data, qsvm_data_process
# ref: https://vqnet20-tutorial.readthedocs.io/en/main/rst/qnn.html#qgan
from pyvqnet.qnn.qgan.qgan_utils import QGANLayer
try:
  from pyvqnet.qnn.qregressor.qregressor import QRegressor                   # pip install qiskit
  from pyvqnet.qnn.qaoa import QAOA, Hamiltonian_MaxCut, write_excel_xls     # pip install xlwt
  from pyvqnet.qnn.tensornetwork_simulation import DMCircuit, MPSCircuit     # pip install opt_einsum
except ImportError: pass
# qnn: legacy/deprecated vqc toolbox
from pyvqnet.qnn.pqc import PQCLayer
from pyvqnet.qnn.qvc import Qvc
from pyvqnet.qnn.quantumlayer import QuantumLayer, grad
# qnn: new vqc toolbox, v2.0.8 (2023-09-26) major updates :)
from pyvqnet.qnn import vqc

# qnn: loss
from pyvqnet.qnn.measure import ProbsMeasure, QuantumMeasure, expval
# qnn: noise post-processing
from pyvqnet.qnn.mitigating import zne_with_poly_extrapolate
# qnn: analysis
from pyvqnet.qnn.quantum_expressibility.quantum_express import fidelity_harr_sample, fidelity_of_cir
# qnn: misc internal/unknown
from pyvqnet.qnn.qcircuitlayer import Quantum_Circuit       # qkmeans/qgnn/qrnn/qtransfer stuff
from pyvqnet.qnn.adaptive_optim import AdaptiveOptimizer
import pyvqnet.qnn.classical_shadow
import pyvqnet.qnn.pq_op_wrapper
import pyvqnet.qnn.opt

# optim
# SGD     (params, lr=0.01, momentum=0, nesterov=False)
# Adagrad (params, lr=0.01, epsilon=1e-08)
# Adadelta(params, lr=0.01, beta=0.99, epsilon=1e-08)
# RMSProp (params, lr=0.01, beta=0.99, epsilon=1e-08)
# Adam    (params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08, amsgrad:bool=False)
# Adamax  (params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08)
# Rotosolve(max_iter=50)
from pyvqnet.optim import SGD, Adagrad, Adadelta, RMSProp, Adam, Adamax, Rotosolve

# helpers
from pyvqnet.utils.initializer import ones, zeros, normal, uniform, quantum_uniform, he_normal, he_uniform, xavier_normal, xavier_uniform
from pyvqnet.utils.metrics import precision_recall_f1_2_score, auc_calculate
from pyvqnet.utils.storage import save_parameters, load_parameters
