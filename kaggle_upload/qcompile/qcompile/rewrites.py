"""Rewrite rules for quantum circuits - local transformations."""

from __future__ import annotations

from typing import Any
import copy


class RewriteLibrary:
    """Library of local circuit rewrites"""

    def __init__(self):
        self.rewrites = [
            ("cancel_inverses", self.cancel_inverses),
            ("commute_gates", self.commute_gates),
            ("fuse_rotations", self.fuse_rotations),
            ("remove_identity", self.remove_identity),
            ("peephole_opt", self.peephole_opt),
            ("synthesise_tket", self.synthesise_tket),
            ("kak_decompose", self.kak_decompose),
            ("full_opt", self.full_opt),
        ]

    def apply(self, circuit, rewrite_name: str) -> Any:
        """Apply a specific rewrite"""
        for name, func in self.rewrites:
            if name == rewrite_name:
                try:
                    return func(circuit)
                except Exception:
                    return circuit
        return circuit

    def apply_all(self, circuit) -> Any:
        """Apply all rewrites until fixed point"""
        changed = True
        iterations = 0
        while changed and iterations < 15:
            changed = False
            try:
                prev = self._circuit_size(circuit)
            except Exception:
                break
            try:
                circuit = self.cancel_inverses(circuit)
            except Exception:
                pass
            try:
                circuit = self.commute_gates(circuit)
            except Exception:
                pass
            try:
                circuit = self.fuse_rotations(circuit)
            except Exception:
                pass
            try:
                circuit = self.remove_identity(circuit)
            except Exception:
                pass
            try:
                circuit = self.peephole_opt(circuit)
            except Exception:
                pass
            try:
                curr = self._circuit_size(circuit)
                if curr < prev:
                    changed = True
            except Exception:
                break
            iterations += 1
        return circuit

    def _circuit_size(self, circuit) -> int:
        try:
            if hasattr(circuit, "get_commands"):
                return len(circuit.get_commands())
            elif hasattr(circuit, "operations"):
                return len(list(circuit.operations))
            elif hasattr(circuit, "__len__"):
                return len(list(circuit))
        except Exception:
            pass
        return 0

    def cancel_inverses(self, circuit):
        try:
            from pytket import Circuit

            if isinstance(circuit, Circuit):
                circuit = circuit.copy()
                try:
                    from pytket.passes import FullPeepholeOptimise

                    FullPeepholeOptimise().apply(circuit)
                    return circuit
                except Exception:
                    pass
                try:
                    from pytket.passes import RemoveRedundancies

                    RemoveRedundancies().apply(circuit)
                except Exception:
                    pass
                return circuit
        except Exception:
            pass
        return circuit

    def commute_gates(self, circuit):
        try:
            from pytket import Circuit

            if isinstance(circuit, Circuit):
                circuit = circuit.copy()
                try:
                    from pytket.passes import CommuteThroughMultis

                    CommuteThroughMultis().apply(circuit)
                except Exception:
                    pass
                return circuit
        except Exception:
            pass
        return circuit

    def fuse_rotations(self, circuit):
        try:
            from pytket import Circuit

            if isinstance(circuit, Circuit):
                circuit = circuit.copy()
                try:
                    from pytket.passes import DecomposeBoxes

                    DecomposeBoxes().apply(circuit)
                except Exception:
                    pass
                try:
                    from pytket.passes import RemoveRedundancies

                    RemoveRedundancies().apply(circuit)
                except Exception:
                    pass
                return circuit
        except Exception:
            pass
        return circuit

    def remove_identity(self, circuit):
        try:
            from pytket import Circuit

            if isinstance(circuit, Circuit):
                circuit = circuit.copy()
                try:
                    from pytket.passes import RemoveRedundancies

                    RemoveRedundancies().apply(circuit)
                    return circuit
                except Exception:
                    pass
                return circuit
        except Exception:
            pass
        return circuit

    def peephole_opt(self, circuit):
        try:
            from pytket import Circuit

            if isinstance(circuit, Circuit):
                circuit = circuit.copy()
                try:
                    from pytket.passes import FullPeepholeOptimise

                    FullPeepholeOptimise().apply(circuit)
                    return circuit
                except Exception:
                    pass
        except Exception:
            pass
        return circuit

    def synthesise_tket(self, circuit):
        try:
            from pytket import Circuit

            if isinstance(circuit, Circuit):
                circuit = circuit.copy()
                try:
                    from pytket.passes import KAKDecomposition

                    KAKDecomposition().apply(circuit)
                    return circuit
                except Exception:
                    pass
                try:
                    from pytket.passes import FullPeepholeOptimise

                    FullPeepholeOptimise().apply(circuit)
                    return circuit
                except Exception:
                    pass
        except Exception:
            pass
        return circuit

    def kak_decompose(self, circuit):
        try:
            from pytket import Circuit

            if isinstance(circuit, Circuit):
                circuit = circuit.copy()
                try:
                    from pytket.passes import KAKDecomposition

                    KAKDecomposition(allow_swaps=False).apply(circuit)
                    return circuit
                except Exception:
                    try:
                        from pytket.passes import KAKDecomposition

                        KAKDecomposition().apply(circuit)
                        return circuit
                    except Exception:
                        pass
        except Exception:
            pass
        return circuit

    def full_opt(self, circuit):
        try:
            from pytket import Circuit

            if isinstance(circuit, Circuit):
                circuit = circuit.copy()
                try:
                    from pytket.passes import FullPeepholeOptimise

                    FullPeepholeOptimise().apply(circuit)
                except Exception:
                    pass
                try:
                    from pytket.passes import KAKDecomposition

                    KAKDecomposition().apply(circuit)
                except Exception:
                    pass
                try:
                    from pytket.passes import RemoveRedundancies

                    RemoveRedundancies().apply(circuit)
                except Exception:
                    pass
                try:
                    from pytket.passes import CommuteThroughMultis

                    CommuteThroughMultis().apply(circuit)
                except Exception:
                    pass
                return circuit
        except Exception:
            pass
        return circuit

    def decompose_to_target(self, circuit, target_gates=None):
        if target_gates is None:
            target_gates = {"Rx", "Rz", "CX"}  # Common basis
        try:
            from pytket import Circuit

            if isinstance(circuit, Circuit):
                circuit = circuit.copy()
                try:
                    from pytket.passes import auto_rebase_pass
                    from pytket.circuit import OpType

                    # Try to rebase if possible
                    pass
                except Exception:
                    pass
                return circuit
        except Exception:
            pass
        return circuit
