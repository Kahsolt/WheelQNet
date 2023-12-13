#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/14

from src.utils import *

pi = 3.14159265358979

vqm = QMachine(3, kcomplex64)

hadamard(wires=0, q_machine=vqm)
ry(q_machine=vqm, wires=1, params=[0])
ry(q_machine=vqm, wires=2, params=[pi])
VQC_CSWAPcircuit([0, 1, 2], vqm)
hadamard(wires=0, q_machine=vqm)
prob = Probability(wires=1)(vqm)    # 测量|1>

# <x|y>=1 正交: 测量|1>概率 0.5
# <x|y>=0 重叠: 测量|1>概率 0
print(prob)
