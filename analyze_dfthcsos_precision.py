"""Measure DFTHC-SOS verifier metrics across angle precisions."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT / "starter_kit"))

from starter.task1_factorization.solution_io import (  # noqa: E402
    BASELINES_DIRECTORY,
    read_factorization,
)
from starter.task2_angles.encode import TREES, encode_inputs  # noqa: E402
from verifier.reference import load_systems  # noqa: E402
from verifier.score import (  # noqa: E402
    evaluate_inputs,
    heuristic_costs,
    load_inputs,
    passes_accuracy,
)


def main() -> None:
    systems = load_systems()
    factorization = BASELINES_DIRECTORY / systems["FeMoCo76o113e"]["baseline"]
    params, _ = read_factorization(factorization)
    inputs = load_inputs("FeMoCo76o113e", params)
    tree_edges = np.asarray(TREES["log_depth"](inputs.num_orbitals), dtype=np.int64)

    results = []
    for brot in range(8, 17):
        words = encode_inputs(inputs, tree_edges, brot, "nearest")
        metrics = evaluate_inputs(inputs, words, tree_edges, brot)
        heuristic = heuristic_costs(76, 65, 60, 60, brot)
        result = {
            "brot": brot,
            "passes_accuracy": passes_accuracy(metrics),
            "metrics": {
                key: bool(value) if isinstance(value, (bool, np.bool_)) else float(value)
                for key, value in metrics.items()
            },
            "heuristic_toffoli": heuristic["toffoli"],
            "heuristic_persistent_qubits": heuristic["persistent_qubits"],
            "heuristic_temporary_qubits": heuristic["temporary_qubits"],
        }
        results.append(result)
        print(
            f"brot={brot:2d} pass={result['passes_accuracy']!s:5s} "
            f"error_1={1e3 * metrics['error_1']:9.4f} mHa "
            f"error_2={1e3 * metrics['error_2']:9.4f} mHa "
            f"E_gap={metrics['E_gap']:.9f} "
            f"lambda_eff={metrics['lambda_eff']:.9f}"
        )

    output = REPOSITORY_ROOT / "dfthcsos_precision_analysis.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
