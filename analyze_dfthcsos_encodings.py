"""Compare DFTHC-SOS trees and rounding rules with the real verifier metrics."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter_kit"))

from starter.task1_factorization.solution_io import (  # noqa: E402
    BASELINES_DIRECTORY,
    read_factorization,
)
from starter.task2_angles.encode import ROUNDING, TREES, encode_inputs  # noqa: E402
from verifier.reference import load_systems  # noqa: E402
from verifier.score import evaluate_inputs, load_inputs, passes_accuracy  # noqa: E402


def main() -> None:
    systems = load_systems()
    path = BASELINES_DIRECTORY / systems["FeMoCo76o113e"]["baseline"]
    params, _ = read_factorization(path)
    inputs = load_inputs("FeMoCo76o113e", params)
    results = []
    for tree_name, tree_builder in TREES.items():
        edges = np.asarray(tree_builder(inputs.num_orbitals), dtype=np.int64)
        for rounding in ROUNDING:
            for brot in (11, 12, 13):
                words = encode_inputs(inputs, edges, brot, rounding)
                metrics = evaluate_inputs(inputs, words, edges, brot)
                result = {
                    "tree": tree_name,
                    "rounding": rounding,
                    "brot": brot,
                    "passes_accuracy": passes_accuracy(metrics),
                    "metrics": {
                        key: bool(value) if isinstance(value, (bool, np.bool_)) else float(value)
                        for key, value in metrics.items()
                    },
                }
                results.append(result)
                print(
                    f"{tree_name:9s} {rounding:7s} b={brot} "
                    f"pass={result['passes_accuracy']!s:5s} "
                    f"err1={1e3 * metrics['error_1']:8.4f} "
                    f"err2={1e3 * metrics['error_2']:8.4f} "
                    f"gap={metrics['E_gap']:.9f} "
                    f"lambda={metrics['lambda_eff']:.9f}"
                )
    output = ROOT / "dfthcsos_encoding_analysis.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    passing = [result for result in results if result["passes_accuracy"]]
    best = min(passing, key=lambda result: (result["brot"], result["metrics"]["lambda_eff"]))
    print("Best passing encoding:", json.dumps(best, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
