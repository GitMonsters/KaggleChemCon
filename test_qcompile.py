#!/usr/bin/env python
"""Test qcompile locally"""
from qcompile import QuantumCircuitSolver, CircuitGraph, CircuitEvolver
from pytket import Circuit

# Create a simple test circuit
c = Circuit(2)
c.H(0)
c.CX(0, 1)
c.H(0)
c.CX(0, 1)  # Should cancel
c.Rz(0.5, 0)
c.Rz(-0.5, 0)  # Should cancel

print(f"Original: {len(c.get_commands())} gates")
for cmd in c.get_commands():
    print(f"  {cmd.op.type.name} {cmd.args}")

solver = QuantumCircuitSolver(generations=10, population_size=15)
opt = solver.solve(c)
print(f"\nOptimized: {len(opt.get_commands())} gates")
for cmd in opt.get_commands():
    print(f"  {cmd.op.type.name} {cmd.args}")
