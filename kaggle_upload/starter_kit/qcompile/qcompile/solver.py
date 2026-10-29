"""Main solver for the competition"""

from __future__ import annotations

import os
from typing import Any, List

from .circuit_graph import CircuitGraph
from .evolve import CircuitEvolver
from .evaluator import CircuitEvaluator
from .rewrites import RewriteLibrary


class QuantumCircuitSolver:
    """Main solver using gitmonsters-style evolution"""

    def __init__(self, generations: int = 60, population_size: int = 50):
        self.graph = CircuitGraph()
        self.evolver = CircuitEvolver(population_size=population_size)
        self.evaluator = CircuitEvaluator()
        self.rewrites = RewriteLibrary()
        self.generations = generations

    def solve(self, circuit) -> Any:
        """Solve a single circuit"""
        print("[qcompile] Analyzing circuit...")
        try:
            dag = self.graph.parse_circuit(circuit)
            stats = self.graph.get_stats(dag)
            print(
                f"[qcompile] Gates: {stats['num_gates']}, Qubits: {stats['num_qubits']}, 2Q gates: {stats['two_qubit_gates']}"
            )
        except Exception as e:
            print(f"[qcompile] Graph parsing failed: {e}")

        print(f"[qcompile] Evolving rewrite sequence ({self.generations} generations)...")
        best_policy = self.evolver.evolve(circuit, self.evaluator, generations=self.generations)

        print(
            f"[qcompile] Best policy fitness: {best_policy.fitness_score:.2f}, gates: {best_policy.gate_count}"
        )
        print(f"[qcompile] Rewrite sequence: {best_policy.rewrite_sequence}")

        opt_circuit = self.evolver.apply_best(circuit, best_policy)
        # Final pass
        try:
            opt_circuit = self.rewrites.apply_all(opt_circuit)
        except Exception:
            pass
        return opt_circuit

    def solve_batch(self, circuits: List[Any]) -> List[Any]:
        """Solve multiple circuits"""
        results = []
        for i, circuit in enumerate(circuits):
            print(f"\n[{'='*60}]")
            print(f"[qcompile] Circuit {i+1}/{len(circuits)}")
            print(f"[{'='*60}]")
            try:
                opt = self.solve(circuit)
                results.append(opt)
            except Exception as e:
                print(f"[qcompile] Error: {e}")
                results.append(circuit)
        return results


def solve_circuit(circuit, generations: int = 60):
    """Convenience function"""
    solver = QuantumCircuitSolver(generations=generations)
    return solver.solve(circuit)


def main():
    """Main entry point for competition"""
    print("=" * 60)
    print("QCOMPILE - GitMonsters Quantum Circuit Compiler")
    print("Using SpatialTopologyEngine + RecursiveSelfImprovementEngine")
    print("=" * 60)

    data_dirs = ["quantum_comp", "data", "input"]
    for d in data_dirs:
        if os.path.exists(d):
            print(f"[qcompile] Found data dir: {d}")
            break

    print("\n[qcompile] Ready. Use solve_circuit() or QuantumCircuitSolver()")


if __name__ == "__main__":
    main()
