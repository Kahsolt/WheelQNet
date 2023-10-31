#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/29

from src.utils import *


class Model(Module):

  def __init__(self, n_qubits:int=4):
    super().__init__()

    self.enc_RY = ModuleList([RY(wires=i) for i in range(n_qubits)])
    self.enc_RZ = ModuleList([RZ(wires=i) for i in range(n_qubits)])
    self.ansatz = VQC_HardwareEfficientAnsatz(n_qubits, ['rx', 'ry'], entangle_gate='cnot', entangle_rules='linear', depth=8)
    self.measure = Probability(wires=[0, 2])
    self.vqm = QMachine(n_qubits)

  def forward(self, x:QTensor):
    vqm = self.vqm
    vqm.reset_states(x.shape[0])
    for i, encoder in enumerate(self.enc_RY):
      encoder(params=x[:, [2*i]], q_machine=vqm)
    for i, encoder in enumerate(self.enc_RZ):
      encoder(params=x[:, [2*i+1]], q_machine=vqm)
    self.ansatz(q_machine=vqm)
    return self.measure(q_machine=vqm)

N = 8
B = 4
iters = 500
lr = 0.1

X = tensor.arange(1, B * N + 1).reshape([B, N])
X.requires_grad = True
y = tensor.arange(0, B, dtype=kint64).reshape([B])

model = Model()
optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
criterion = CrossEntropyLoss()

model.train()
for i in range(iters):
  optimizer.zero_grad()
  out = model(X)
  loss = criterion(y, out)
  loss.backward()
  optimizer._step()

  if i % 10 == 0: print(f'[{i}/{iters}] loss: {loss.item()}', )

print('truth:')
print(y)
print('pred:')
pred = model(X)
print(pred)
