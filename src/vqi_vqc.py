#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/29

''' export all APIs from pyvqnet.qnn.vqc '''

# vqc module: https://vqnet20-tutorial.readthedocs.io/en/main/rst/qnn.html#vqc

from typing import Any, Union, List, Dict

Wires = Union[int, List[int]]
Indices = List[int]
'''
Pauli operator: 
  {'X0': 0.23}
or 
  {
    'wires': [0, 1], 
    'observables': ['x', 'i'], 
    'coefficient': [0.23, -3.5],
  }
Paulisum: List[Obs]
'''
Obs = Union[Dict[str, Any], List['Obs']]


from pyvqnet.qnn.vqc.qmachine import QMachine

QMachine.__init__         # (num_wires, dtype=kcomplex64)
QMachine.forward          # (x:QTensor, *args, **kwargs)
QMachine.state_dict       # Dict[str, QTensor]
QMachine.set_states       # (states:QTensor)
QMachine.reset_states     # (bsz:int)
QMachine.get_run_status   # ()
QMachine.set_run_status   # (flag)
#vqm = QMachine()
#vqm.num_wires    # int
#vqm.dtype        # int
#vqm.batch_size   # int
#vqm.state        # QTensor
#vqm.states       # QTensor

from pyvqnet.qnn.vqc import INV_SQRT2   # 1/sqrt(2)

from pyvqnet.qnn.vqc import (
  F_DTYPE,  # kfloat32
  D_DTYPE,  # kfloat64
  C_DTYPE,  # kfloat64
  Z_DTYPE,  # kfloat128

  get_readable_dtype_str,   # (dtype_int:int)
)

from pyvqnet.qnn.vqc.qcircuit import (
  # functional: directly evolves the QMachine's state
  VQC_BasisState,             # (basis_state:List[int], wires:Wires, q_machine:QMachine)
  VQC_CCZ,                    # (wires:Wires, q_machine:QMachine)
  VQC_Controlled_Hadamard,    # (wires:Wires, q_machine:QMachine)

  VQC_BasisEmbedding,         # (basis_state:List[int], q_machine:QMachine)
  VQC_AmplitudeEmbedding,     # (input_feature:QTensor, q_machine:QMachine)
  VQC_AngleEmbedding,         # (input_feat:QTensor, wires:Wires, q_machine:QMachine, rotation:int='X')
  VQC_IQPEmbedding,           # (input_feat:QTensor, q_machine:QMachine, rep:int=1)

  VQC_RotCircuit,             # (params:QTensor, q_machine:QMachine, wire:Wires)
  VQC_CRotCircuit,            # (para:QTensor, control_wire:Wires, rot_wire:Wires, q_machine:QMachine)
  VQC_CSWAPcircuit,           # (wires:Wires, q_machine:QMachine)
  VQC_QuantumPoolingCircuit,  # (ignored_wires:Wires, sinks_wires:Wires, params:QTensor, q_machine:QMachine)

  VQC_FermionicSingleExcitation,  # (weight:QTensor, wires:Wires, q_machine:QMachine)
  VQC_FermionicDoubleExcitation,  # (weight:QTensor, wires1:Wires, wires2:Wires, q_machine:QMachine)
  VQC_UCCSD,                      # (weights, wires, s_wires, d_wires, init_state, q_machine)

  # objective: forward(q_machine:QMachine)
  VQC_QuantumEmbedding,             # (num_repetitions_input:int, depth_input:int, num_unitary_layers:int, num_repetitions:int, dtype:int=None, initial:QTensor=None, name:str='')
  VQC_BasicEntanglerTemplate,       # (num_layers:int=1, num_qubits:int=1, rotation:str='RX', initial=None, dtype:int=None)
  VQC_StronglyEntanglingTemplate,   # (num_layers:int=1, num_qubits:int=1, ranges=None, initial=None, dtype:int=None)
  VQC_HardwareEfficientAnsatz,      # (n_qubits:int, single_rot_gate_list:List['RX'|'RY'|'RZ'], entangle_gate:str='CNOT'|'CZ', entangle_rules:str='linear'|'all', depth=1, initial=None, dtype:int=None)
)

# functional: retreive the QMachine's measurement outcome
from pyvqnet.qnn.vqc.qmeasure import (
  VQC_VarMeasure,               # (q_machine:QMachine, obs:Obs)
  VQC_Mutal_Info,               # (q_machine:QMachine, indices0:int, indices1:int, base=None)
  VQC_VN_Entropy,               # (q_machine:QMachine, indices:Indices, base=None)
  VQC_DensityMatrix,            # (q_machine:QMachine, indices:Indices)
  VQC_DensityMatrixFromQstate,  # (state:QMachine.state, indices:Indices)
  VQC_Purity,                   # (state:QMachine.state, qubits_idx:int, num_wires:int)

  Probability,      # (wires:Wires)
  MeasureAll,       # (obs:Obs)
)

# functional: directly evolves the QMachine's state
#   __call__(q_machine:QMachine, wires:Wires, params:ndarray=None, num_wires:int=None, use_dagger:bool=False)
from pyvqnet.qnn.vqc.qcircuit import (
  hadamard,
  i,
  x,
  paulix,
  pauliy,
  pauliz,
  x1,
  y1,
  z1,
  rx,
  rxx,
  ry,
  ryy,
  rz,
  rzx,
  rzz,
  s,
  t,
  p,
  u1,
  u2,
  u3,
  cnot,
  cr,
  cz,
  swap,
  iswap,
  toffoli,
)

# objective:
#   __init__(has_params:bool=False, trainable:bool=False, init_params:ndarray=None, num_wires:int=None, wires:Wires=None, dtype:int=kcomplex64, use_dagger:bool=False)
#   forward(params:ndarray=None, q_machine:QMachine=None, wires:Wires=None)
from pyvqnet.qnn.vqc.qcircuit import (
  Hadamard,
  I,
  PauliX,
  PauliY,
  PauliZ,
  X1,
  Y1,
  Z1,
  RX,
  RXX,
  RY,
  RYY,
  RZ,
  RZX,
  RZZ,
  S,
  T,
  U1,
  U2,
  U3,
  CNOT,
  CR,
  CZ,
  SWAP,
  iSWAP,
  Toffoli,
)

# ↓↓↓ internal misc ↓↓↓

from pyvqnet.qnn.vqc import op_name_dict, quantum_gate_op
from pyvqnet.qnn.vqc.qop import Observable, Operator, Operation, DiagonalOperation, StateEncoder

from pyvqnet.qnn.vqc.qmatrix import (
  u1_matrix,
  u2_matrix,
  u3_matrix,
  cu1_matrix,
  rx_matrix,
  rxx_matrix,
  ry_matrix,
  ryy_matrix,
  rz_matrix,
  rzx_matrix,
  rzz_matrix,
  iswap_matrix,

  mat_dict,
  double_mat_dict,
)
