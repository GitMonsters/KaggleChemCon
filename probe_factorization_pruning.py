"""Screen verifier-safe rank/basis pruning for the DFTHC-SOS baseline."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter_kit"))

from starter.task2_angles.encode import TREES, encode_inputs  # noqa: E402
from verifier.score import (  # noqa: E402
    evaluate_inputs,
    heuristic_costs,
    load_inputs,
    passes_accuracy,
)

FACTORIZATION_KEYS = ("U", "W", "H2_I", "H1_I", "Hsym_lower")


def ordinary_qrom_cost(inputs, metrics):
    """Exact ordinary-QROM resource formula, optimized over allowed J."""
    rank, bases, copies = inputs.w.shape
    rows = 76 + rank * bases
    selection = (rows - 1).bit_length()
    heuristic = heuristic_costs(76, rank, bases, copies, 12)
    best = None
    for batch_size in range(1, 76):
        angle_qubits = 2 * selection - 1 + 12 * batch_size
        if angle_qubits > 128:
            break
        circuits = (75 + batch_size - 1) // batch_size
        angle_toffoli = 2 * circuits * (rows - 2)
        toffoli = angle_toffoli + heuristic["toffoli"]
        qubits = (
            heuristic["persistent_qubits"]
            + 12 * batch_size
            + max(heuristic["temporary_qubits"], selection - 1)
        )
        score = metrics["lambda_eff"] * toffoli * qubits
        candidate = {
            "batch_size": batch_size,
            "rows": rows,
            "angle_qubits": angle_qubits,
            "angle_toffoli": angle_toffoli,
            "toffoli": toffoli,
            "qubits": qubits,
            "score": float(score),
        }
        if best is None or candidate["score"] < best["score"]:
            best = candidate
    return best


def prune_ranks(params, count):
    norms = np.linalg.norm(params["W"].reshape(params["W"].shape[0], -1), axis=1)
    keep = np.sort(np.argsort(norms)[count:])
    result = dict(params)
    for key in ("U", "W", "H2_I"):
        result[key] = np.array(params[key][keep])
    return result, {"kind": "rank", "dropped": count, "keep": keep.tolist()}


def prune_basis(params, count):
    order = np.argsort(np.linalg.norm(params["W"], axis=2), axis=1)
    keep = np.sort(order[:, count:], axis=1)
    drop = np.sort(order[:, :count], axis=1)
    result = dict(params)
    result["U"] = np.stack([params["U"][r, keep[r]] for r in range(len(keep))])
    result["W"] = np.stack([params["W"][r, keep[r]] for r in range(len(keep))])
    # Preserve the native W offsets of every retained (rho, c) term.
    result["H2_I"] = params["H2_I"] - np.stack(
        [params["W"][r, drop[r]].sum(axis=0) for r in range(len(drop))]
    )
    return result, {"kind": "basis", "dropped_per_rank": count}


def plain_metrics(metrics):
    return {
        key: bool(value) if isinstance(value, (bool, np.bool_)) else float(value)
        for key, value in metrics.items()
    }


def main() -> None:
    with np.load(ROOT / "competition_sample/sample_submission/dfthcsos/submission.npz") as data:
        baseline = {key: np.array(data[key]) for key in FACTORIZATION_KEYS}
    candidates = [("baseline", baseline, {"kind": "baseline"})]
    for count in (1, 2, 4, 8, 12, 16, 24, 32):
        params, description = prune_ranks(baseline, count)
        candidates.append((f"rank_minus_{count}", params, description))
    for count in (1, 2, 4, 8, 12, 16, 24, 30):
        params, description = prune_basis(baseline, count)
        candidates.append((f"basis_minus_{count}", params, description))

    reports = []
    best = None
    for name, params, description in candidates:
        try:
            inputs = load_inputs("FeMoCo76o113e", params)
            continuous = evaluate_inputs(inputs)
            report = {
                "name": name,
                "description": description,
                "shape_RBC": list(inputs.w.shape),
                "continuous_pass": passes_accuracy(continuous),
                "continuous_metrics": plain_metrics(continuous),
            }
            print(
                f"{name:18s} RBC={inputs.w.shape} continuous="
                f"{report['continuous_pass']} err2={1e3 * continuous['error_2']:.4f} "
                f"gap={continuous['E_gap']:.9f}",
                flush=True,
            )
            if not report["continuous_pass"]:
                reports.append(report)
                continue
            edges = np.asarray(TREES["log_depth"](76), dtype=np.int64)
            words = encode_inputs(inputs, edges, 12, "nearest")
            metrics = evaluate_inputs(inputs, words, edges, 12)
            report["brot12_pass"] = passes_accuracy(metrics)
            report["brot12_metrics"] = plain_metrics(metrics)
            if report["brot12_pass"]:
                report["cost"] = ordinary_qrom_cost(inputs, metrics)
                print(
                    f"  b12 PASS err2={1e3 * metrics['error_2']:.4f} "
                    f"gap={metrics['E_gap']:.9f} score={report['cost']['score']:.6g} "
                    f"J={report['cost']['batch_size']}",
                    flush=True,
                )
                if best is None or report["cost"]["score"] < best[0]["cost"]["score"]:
                    best = (report, params)
            else:
                print(
                    f"  b12 FAIL err1={1e3 * metrics['error_1']:.4f} "
                    f"err2={1e3 * metrics['error_2']:.4f} gap={metrics['E_gap']:.9f}",
                    flush=True,
                )
            reports.append(report)
        except Exception as exc:
            reports.append({"name": name, "description": description, "error": repr(exc)})
            print(name, "ERROR", repr(exc), flush=True)

    output = ROOT / "factorization_pruning_analysis.json"
    output.write_text(json.dumps(reports, indent=2) + "\n")
    if best is not None:
        report, params = best
        np.savez(ROOT / "pruned_factorization.npz", **params)
        print("BEST", json.dumps(report, indent=2))
        print("Wrote pruned_factorization.npz")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
