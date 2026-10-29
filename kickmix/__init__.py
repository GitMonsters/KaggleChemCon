# Minimal kickmix stub for local testing
# The real kickmix is only needed on Kaggle, but we need imports to work
from dataclasses import dataclass
from typing import List, Dict, Any

class q:
    def __init__(self, name=None, index=None):
        self.name = name
        self.index = index

class QuantumRegister:
    def __init__(self, size, name=""):
        self.size = size
        self.name = name
        self.qubits = [q(name=f"{name}[{i}]", index=i) for i in range(size)]
    def __getitem__(self, i):
        return self.qubits[i]
    def __iter__(self):
        return iter(self.qubits)
    def __len__(self):
        return self.size

class CircuitBuilder:
    def __init__(self):
        self.registers = {}
        self.instructions = []
    def create_quantum_register(self, size, name=""):
        reg = QuantumRegister(size, name)
        self.registers[name] = reg
        return reg
    def alloc_qubits(self, n):
        return [q(name=f"w{i}", index=i) for i in range(n)]
    def x(self, *args): pass
    def cx(self, *args): pass
    def ccx(self, *args): pass
    def z(self, *args): pass
    def cz(self, *args): pass
    def swap(self, *args): pass
    def init_and(self, *args): pass
    def del_and(self, *args): pass
    def finish_circuit(self):
        return Circuit(list(self.registers.values()), self.instructions)

@dataclass
class Circuit:
    registers: List[QuantumRegister]
    instructions: List = None
    def __init__(self, registers, instructions=None):
        self.registers = registers
        self.instructions = instructions or []
    def __str__(self):
        return "# kickmix circuit\n"

__all__ = ['q', 'QuantumRegister', 'CircuitBuilder', 'Circuit']
