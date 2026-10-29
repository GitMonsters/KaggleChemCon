#!/usr/bin/env python
"""
Kaggle submission for Google Verifiable Quantum Circuit Compilation
Using gitmonsters-style evolution: SpatialTopologyEngine + RecursiveSelfImprovementEngine
"""
import os
import sys
import json
import pickle
from pathlib import Path

# Add local modules
sys.path.insert(0, '/kaggle/working')
sys.path.insert(0, '.')

try:
    import numpy as np
    import pandas as pd
    from pytket import Circuit
    from pytket.circuit import OpType
    from pytket.passes import (
        FullPeepholeOptimise,
        RemoveRedundancies,
        CommuteThroughMultis,
        DecomposeBoxes,
    )
    from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
    print("[INFO] Loaded pytket")
except Exception as e:
    print(f"[WARN] pytket import issue: {e}")
    Circuit = None

# Import our qcompile module
try:
    from qcompile import QuantumCircuitSolver, CircuitGraph, CircuitEvolver, CircuitEvaluator
    print("[INFO] Loaded qcompile module")
except Exception as e:
    print(f"[WARN] qcompile import issue: {e}")
    QuantumCircuitSolver = None


class KaggleSolver:
    def __init__(self):
        self.solver = QuantumCircuitSolver(generations=20, population_size=30) if QuantumCircuitSolver else None
        self.fallback_passes = [
            FullPeepholeOptimise(),
            RemoveRedundancies(),
            CommuteThroughMultis(),
        ]
    
    def optimize_circuit(self, circuit):
        """Optimize a single circuit"""
        if circuit is None:
            return None
        
        original_size = len(circuit.get_commands()) if hasattr(circuit, 'get_commands') else 0
        
        # Try our evolved approach first
        if self.solver is not None:
            try:
                opt_circuit = self.solver.solve(circuit)
                opt_size = len(opt_circuit.get_commands()) if hasattr(opt_circuit, 'get_commands') else 0
                if opt_size < original_size:
                    print(f"  [EVOLVE] {original_size} -> {opt_size} gates")
                    return opt_circuit
            except Exception as e:
                print(f"  [EVOLVE] failed: {e}")
        
        # Fallback to pytket passes
        try:
            c = circuit.copy()
            for p in self.fallback_passes:
                try:
                    p.apply(c)
                except:
                    pass
            opt_size = len(c.get_commands())
            if opt_size < original_size:
                print(f"  [FALLBACK] {original_size} -> {opt_size} gates")
            return c
        except Exception as e:
            print(f"  [FALLBACK] failed: {e}")
            return circuit
    
    def process_input(self, input_dir='/kaggle/input'):
        """Process all circuits from input directory"""
        print(f"[INFO] Searching in {input_dir}")
        
        circuits = []
        circuit_files = []
        
        for root, dirs, files in os.walk(input_dir):
            for f in files:
                full_path = os.path.join(root, f)
                if f.endswith('.json'):
                    circuit_files.append(full_path)
                elif f.endswith('.qasm'):
                    circuit_files.append(full_path)
                elif f.endswith('.pkl') or f.endswith('.pickle'):
                    circuit_files.append(full_path)
        
        print(f"[INFO] Found {len(circuit_files)} circuit files")
        
        for cf in circuit_files:
            print(f"[INFO] Loading {cf}")
            try:
                if cf.endswith('.json'):
                    with open(cf) as fh:
                        data = json.load(fh)
                    # Try to convert to pytket
                    circuits.append((os.path.basename(cf), data))
                elif cf.endswith('.qasm'):
                    try:
                        from pytket.qasm import circuit_from_qasm
                        c = circuit_from_qasm(cf)
                        circuits.append((os.path.basename(cf), c))
                    except:
                        pass
                elif cf.endswith(('.pkl', '.pickle')):
                    with open(cf, 'rb') as fh:
                        data = pickle.load(fh)
                    circuits.append((os.path.basename(cf), data))
            except Exception as e:
                print(f"[ERR] Failed to load {cf}: {e}")
        
        return circuits
    
    def generate_submission(self, optimized_circuits, output_path='/kaggle/working/submission.csv'):
        """Generate submission file"""
        # Check what format is expected
        results = []
        for name, circuit in optimized_circuits:
            try:
                if hasattr(circuit, 'get_commands'):
                    n_gates = len(circuit.get_commands())
                elif isinstance(circuit, dict):
                    n_gates = len(circuit.get('gates', []))
                else:
                    n_gates = 0
                results.append({'circuit_id': name, 'num_gates': n_gates})
            except:
                results.append({'circuit_id': name, 'num_gates': 0})
        
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"[INFO] Submission saved to {output_path}")
        return df


def main():
    print("="*60)
    print("QCOMPILE - Verifiable Quantum Circuit Compilation")
    print("GitMonsters GA: SpatialTopologyEngine + RecursiveSelfImprovementEngine")
    print("="*60)
    
    solver = KaggleSolver()
    
    # Process input
    circuits = solver.process_input('/kaggle/input')
    
    if not circuits:
        print("[WARN] No circuits found, checking alternative locations")
        circuits = solver.process_input('/kaggle/working')
    
    optimized = []
    for name, circuit in circuits:
        print(f"\n[OPTIMIZING] {name}")
        try:
            # Convert if needed
            if isinstance(circuit, dict) and Circuit is not None:
                # Try to build from dict
                pass
            opt = solver.optimize_circuit(circuit)
            optimized.append((name, opt))
        except Exception as e:
            print(f"[ERR] {e}")
            optimized.append((name, circuit))
    
    solver.generate_submission(optimized)
    print("\n[COMPLETE] Done!")


if __name__ == '__main__':
    main()
