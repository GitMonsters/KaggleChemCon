"""qcompile - Quantum circuit compilation using gitmonsters-style topological evolution.

Adapts octotetrahedral_spatial_rsi patterns to quantum circuits:
- SpatialTopologyEngine -> CircuitGraph (DAG analysis)
- RecursiveSelfImprovementEngine -> CircuitEvolver (GA over rewrite sequences)
"""

from .circuit_graph import CircuitGraph, GateNode
from .evolve import CircuitEvolver, RewritePolicy
from .evaluator import CircuitEvaluator, FuzzTester
from .rewrites import RewriteLibrary
from .solver import solve_circuit, QuantumCircuitSolver
from .prefix_compiler import compile_prefix_family

__all__ = [
    "CircuitGraph",
    "GateNode",
    "CircuitEvolver",
    "RewritePolicy",
    "CircuitEvaluator",
    "FuzzTester",
    "RewriteLibrary",
    "solve_circuit",
    "QuantumCircuitSolver",
    "compile_prefix_family",
]
