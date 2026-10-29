"""Measure exact prefix-circuit costs without the native kickmix simulator."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter_kit"))


class Circuit:
    def __init__(self, num_qubits, instructions):
        self.num_qubits = num_qubits
        self.instructions = instructions

    def operation_statistics(self):
        count = sum(instruction[0] == "CCX" for instruction in self.instructions)
        return SimpleNamespace(num_toffoli_like=count)


class CircuitBuilder:
    def __init__(self):
        self.num_qubits = 0
        self.instructions = []

    def create_quantum_register(self, size, _name):
        register = list(range(self.num_qubits, self.num_qubits + size))
        self.num_qubits += size
        return register

    def alloc_qubits(self, size):
        return self.create_quantum_register(size, "workspace")

    def _emit(self, name, *qubits):
        self.instructions.append((name, *qubits))

    def x(self, *qubits):
        self._emit("X", *qubits)

    def cx(self, *qubits):
        self._emit("CX", *qubits)

    def ccx(self, *qubits):
        self._emit("CCX", *qubits)

    def finish_circuit(self):
        return Circuit(self.num_qubits, self.instructions)


fake_kickmix = ModuleType("kickmix")
fake_kickmix.Circuit = Circuit
fake_kickmix.CircuitBuilder = CircuitBuilder
sys.modules["kickmix"] = fake_kickmix

from qcompile.prefix_compiler import build, delta_batch, symbolic_check  # noqa: E402
from verifier.score import heuristic_costs  # noqa: E402


def main() -> None:
    source = ROOT / "competition_sample/sample_submission/dfthcsos/submission.npz"
    with np.load(source, allow_pickle=False) as archive:
        words = np.array(archive["angle_words"], dtype=np.int64)
    results = []
    for batch_size, clean_cap in ((9, 8), (8, 20), (7, 20)):
        circuits = []
        reports = []
        for k in range((words.shape[1] + batch_size - 1) // batch_size):
            delta = delta_batch(words, batch_size, k)
            circuit, trace, report = build(delta, 12, clean_cap)
            report["symbolic_oracle"] = symbolic_check(trace, delta, 12, circuit.num_qubits)
            reports.append(report)
            circuits.append(circuit)
            print(
                f"J={batch_size} cap={clean_cap} k={k} "
                f"T={report['toffoli_count']} Q={report['num_qubits']} "
                f"gates={report['gates']}",
                flush=True,
            )
        selection = (len(words) - 1).bit_length()
        angle_qubits = max(circuit.num_qubits for circuit in circuits)
        angle_toffoli = 2 * sum(report["toffoli_count"] for report in reports)
        heuristic = heuristic_costs(76, 65, 60, 60, 12)
        persistent = heuristic["persistent_qubits"] + batch_size * 12
        temporary = max(
            heuristic["temporary_qubits"],
            angle_qubits - selection - batch_size * 12,
        )
        total_toffoli = angle_toffoli + heuristic["toffoli"]
        total_qubits = persistent + temporary
        lambda_eff = 49.30136925057438
        score = lambda_eff * total_toffoli * total_qubits
        result = {
            "batch_size": batch_size,
            "clean_cap": clean_cap,
            "angle_toffoli": angle_toffoli,
            "angle_qubits": angle_qubits,
            "total_toffoli": total_toffoli,
            "total_qubits": total_qubits,
            "score": score,
            "circuits": reports,
        }
        results.append(result)
        print("CANDIDATE", json.dumps({k: v for k, v in result.items() if k != "circuits"}), flush=True)
    output = ROOT / "prefix_cost_analysis.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
