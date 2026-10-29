"""Part 3 (circuits): from Part 2 angle words to kickmix ANGLE_k circuits.

Pipeline (baseline, ordinary QROMs):

1. ``encode_inputs`` (Part 2) gives the ``(N + R*B, N - 1)`` angle words for a
   factorization on a Givens tree at ``brot`` bits.
2. ``build_angle_ks`` splits the ``N - 1`` words of each row into
   ``K = ceil((N - 1) / J)`` batches of ``J`` words and builds one Qualtran QROM
   per batch for the Boolean function ``f_k(a) = delta[k, a]``
   (the XOR difference of consecutive batches).
3. ``angle_ks_to_circuits`` replays each QROM into a ``kickmix.CircuitBuilder``
   (``angle_k_kickmix.py``), with a ``selection`` register and one
   ``brot``-qubit ``word_j`` register per word (the verifier's register
   contract, ``verifier/circuits.py``).
4. ``verifier.score.evaluate_circuits`` simulates the family on every address,
   checks the contract, recovers the words it loads (which must equal the
   encoded words) and computes the score ``lambda_eff * T_total * Q_total``.

From the repository root::

    python starter/task3_circuits/build_circuits.py Fe2S2-20o30e \\
        --tree log_depth --brot 16 --batch-size 5

This writes, under ``starter/task3_circuits/circuits/<system>/<run>/``:

* ``angle_k_0.kmx``, ``angle_k_1.kmx``, ...: the circuits, applied in order;
* ``submission.npz``: ``tree_edges`` (needed by the verifier), the
  ``angle_words`` the circuits encode (for reference) and the factorization
  arrays ``U``, ``W``, ``H2_I``, ``H1_I``, ``Hsym_lower`` (the verifier's
  factorization input);
* ``summary.json``: settings, per-circuit qubits / ``max_magic`` /
  reaction depth, the score resources, the verifier metrics of the recovered
  words and the score.

Existing files are never overwritten unless ``--overwrite`` is given. Score
such a directory (yours or ours) with ``verifier/verify.py``::

    python verifier/verify.py SYSTEM RUN_DIR/submission.npz RUN_DIR
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import kickmix as km
import numpy as np
import numpy.typing as npt

# Development shim: allow running this file directly without installing; the
# repository root holds dfthc/, verifier/ and starter/.
_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from dfthc.build_angle_k import build_angle_ks  # noqa: E402
from starter.task1_factorization.solution_io import read_factorization  # noqa: E402
from starter.task2_angles.encode import ROUNDING, TREES, encode_inputs  # noqa: E402
from starter.task3_circuits.angle_k_kickmix import (  # noqa: E402
    angle_ks_to_circuits,
    read_kmx_family,
    write_kmx_family,
)
from verifier.circuits import SELECTION_REGISTER, circuit_costs  # noqa: E402
from verifier.reference import FACTORIZATION_KEYS  # noqa: E402
from verifier.score import KEEP_BITS, evaluate_circuits, load_inputs, rejection_reason  # noqa: E402

CIRCUITS_DIRECTORY = Path(__file__).resolve().parent / "circuits"


def build_circuits(
    angle_words: npt.ArrayLike, brot: int, batch_size: int
) -> tuple[km.Circuit, ...]:
    """Baseline ANGLE_k family (ordinary QROMs) as kickmix circuits."""
    words = np.asarray(angle_words)
    if words.ndim != 2 or not np.issubdtype(words.dtype, np.integer):
        raise ValueError("angle_words must be a 2-d integer array")
    if not 1 <= batch_size <= words.shape[1]:
        raise ValueError("batch_size must be between 1 and N - 1")
    return angle_ks_to_circuits(
        build_angle_ks(words.astype(np.int64), brot=brot, batch_size=batch_size)
    )


def default_run_directory(
    system: str,
    factorization: str | Path,
    tree: str,
    brot: int,
    batch_size: int,
    rounding: str,
) -> Path:
    """``circuits/<system>/<factorization stem>_<tree>_b<brot>_J<J>_<rounding>``."""
    name = f"{Path(factorization).stem}_{tree}_b{brot}_J{batch_size}_{rounding}"
    return CIRCUITS_DIRECTORY / system / name


def _plain_metrics(metrics: dict[str, Any]) -> dict[str, float | bool]:
    """Verifier metrics as plain ``float`` / ``bool`` (JSON-serializable)."""
    return {
        name: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
        for name, v in metrics.items()
    }


def _validate_evaluation(
    result: dict[str, Any], words: np.ndarray, brot: int, batch_size: int
) -> None:
    """Reject wrong words, register layouts and verifier-failing submissions."""
    if (result["brot"], result["batch_size"]) != (brot, batch_size):
        raise ValueError("the circuits' word registers do not match brot and batch_size")
    if not np.array_equal(result["angle_words"], words):
        raise ValueError("the simulated circuits do not load the encoded angle words")
    reason = rejection_reason(result["metrics"], result["resources"])
    if reason is not None:
        raise ValueError(f"the verifier rejected the submission: {reason}")


def build_submission(
    system: str,
    factorization: str | Path,
    *,
    tree: str = "log_depth",
    brot: int = 16,
    batch_size: int = 5,
    rounding: str = "nearest",
    output_directory: str | Path | None = None,
    overwrite: bool = False,
    keep_bits: int = KEEP_BITS,
    compiler: str = "qrom",
    clean_cap: int = 8,
) -> dict[str, Any]:
    """Encode, build, verify, score and write the baseline circuits.

    ``factorization`` is the verifier's ``.npz`` format; its
    arrays are copied into ``submission.npz``. ``keep_bits`` is ``chi`` of the
    heuristic state-preparation cost model (see
    ``verifier.score.heuristic_costs``). Returns the summary that is also
    written to ``summary.json``.
    """
    if tree not in TREES:
        raise ValueError(f"tree must be one of {sorted(TREES)}")
    if rounding not in ROUNDING:
        raise ValueError(f"rounding must be one of {ROUNDING}")
    if compiler not in {"qrom", "prefix"}:
        raise ValueError("compiler must be 'qrom' or 'prefix'")
    directory = Path(
        output_directory
        or default_run_directory(system, factorization, tree, brot, batch_size, rounding)
    )
    outputs = [directory / "submission.npz", directory / "summary.json"]
    if not overwrite and (any(p.exists() for p in outputs) or (directory / "angle_k_0.kmx").exists()):
        raise FileExistsError(f"{directory} already holds a submission (use overwrite=True)")

    params, _ = read_factorization(factorization)
    inputs = load_inputs(system, params)
    num_orbitals = inputs.num_orbitals
    tree_edges = np.asarray(TREES[tree](num_orbitals), dtype=np.int64)
    words = encode_inputs(inputs, tree_edges, brot, rounding)
    prefix_reports = []
    if compiler == "prefix":
        from qcompile.prefix_compiler import compile_prefix_family

        circuits, prefix_reports = compile_prefix_family(
            words, brot=brot, batch_size=batch_size, clean_cap=clean_cap
        )
    else:
        circuits = build_circuits(words, brot, batch_size)

    # Verify the baseline before trying an optional optimizer. A candidate is
    # used only when it preserves every word and has a strictly better score.
    result = evaluate_circuits(inputs, circuits, tree_edges, keep_bits)
    _validate_evaluation(result, words, brot, batch_size)
    baseline_score = float(result["score"])
    qcompile_enabled = os.environ.get("QCOMPILE_ENABLE", "0") == "1"
    if qcompile_enabled:
        try:
            from qcompile import QuantumCircuitSolver

            gen = int(os.environ.get("QCOMPILE_GENERATIONS", "60"))
            pop = int(os.environ.get("QCOMPILE_POP_SIZE", "50"))
            solver = QuantumCircuitSolver(generations=gen, population_size=pop)
            candidate = tuple(solver.solve_batch(list(circuits)))
            if len(candidate) != len(circuits):
                raise ValueError("qcompile returned the wrong number of circuits")
            candidate_result = evaluate_circuits(inputs, candidate, tree_edges, keep_bits)
            _validate_evaluation(candidate_result, words, brot, batch_size)
            if float(candidate_result["score"]) < baseline_score:
                circuits = candidate
                result = candidate_result
                print(
                    f"[qcompile] Accepted verified improvement: "
                    f"{baseline_score:.6g} -> {float(result['score']):.6g}"
                )
            else:
                print("[qcompile] Candidate was valid but did not improve the score; using baseline")
        except Exception as e:
            print(f"[qcompile] Candidate rejected; using verified baseline: {e}")

    header = [
        f"system={system} factorization={Path(factorization).name}",
        f"tree={tree} brot={brot} J={batch_size} rounding={rounding}",
        f"num_vectors=N+R*B={words.shape[0]} num_angles=N-1={words.shape[1]}",
        f"\"{SELECTION_REGISTER}\": address, little endian",
        f"\"word_0\" .. \"word_{batch_size - 1}\": one {brot}-qubit word each, little endian",
        "other qubits: workspace, start and end in |0>",
    ]
    paths = write_kmx_family(circuits, directory, header=header, overwrite=overwrite)
    if read_kmx_family(directory) != circuits:
        raise ValueError("the written .kmx files do not parse back to the same circuits")
    np.savez(
        directory / "submission.npz",
        tree_edges=tree_edges,
        angle_words=words,
        **{key: inputs.params[key] for key in FACTORIZATION_KEYS},
    )
    summary = {
        "system": system,
        "factorization": str(factorization),
        "tree": tree,
        "brot": brot,
        "batch_size": batch_size,
        "rounding": rounding,
        "compiler": compiler,
        "clean_cap": clean_cap if compiler == "prefix" else None,
        "prefix_reports": prefix_reports,
        "num_vectors": int(words.shape[0]),
        "num_angles": int(words.shape[1]),
        "circuits": [path.name for path in paths],
        "costs": circuit_costs(circuits),
        "resources": result["resources"],
        "metrics": _plain_metrics(result["metrics"]),
        "score": float(result["score"]),
        "baseline_score": baseline_score,
        "qcompile_enabled": qcompile_enabled,
        "directory": str(directory),
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    """Human-readable per-circuit costs, score resources and verifier metrics."""
    lines = [f"{'k':>3} {'qubits':>7} {'max_magic':>10} {'reaction_depth':>15} {'instructions':>13}"]
    for c in summary["costs"]:
        lines.append(
            f"{c['k']:>3} {c['num_qubits']:>7} {c['max_magic']:>10} "
            f"{c['reaction_depth']:>15} {c['num_instructions']:>13}"
        )
    m, r = summary["metrics"], summary["resources"]
    lines.append(
        f"Toffoli = {r['toffoli']} (2 * ANGLES {r['angles_toffoli']} + outer prep "
        f"{r['outer_state_prep_toffoli']} + inner prep {r['inner_state_prep_toffoli']} "
        f"+ Givens {r['givens_rotations_toffoli']})"
    )
    lines.append(
        f"qubits  = {r['qubits']} (persistent {r['persistent_qubits']} + "
        f"shared temp {r['temporary_qubits']}; ANGLES width {r['angles_qubits']}); "
        f"ANGLES <= 128: {'ok' if r['angles_within_cap'] else 'FAILS (score inf)'}"
    )
    lines.append(
        f"recovered words: error_1 = {1e3 * m['error_1']:.4f} mHa, "
        f"error_2 = {1e3 * m['error_2']:.4f} mHa, lambda_eff = {m['lambda_eff']:.6f}"
    )
    lines.append(
        f"score = lambda_eff * Toffoli * qubits = {summary['score']:.6g}"
    )
    return "\n".join(lines)


__all__ = [
    "CIRCUITS_DIRECTORY",
    "build_circuits",
    "build_submission",
    "default_run_directory",
    "format_summary",
]


if __name__ == "__main__":
    import argparse

    from starter.task1_factorization.solution_io import BASELINES_DIRECTORY
    from verifier.reference import load_systems

    systems = load_systems()
    parser = argparse.ArgumentParser(
        description="Build, verify and write baseline ANGLE_k kickmix circuits."
    )
    parser.add_argument("system", choices=sorted(systems))
    parser.add_argument(
        "factorization", nargs="?", type=Path,
        help="factorization .npz (default: the system's baseline)",
    )
    parser.add_argument("--tree", choices=sorted(TREES), default="log_depth")
    parser.add_argument("--brot", type=int, default=16)
    parser.add_argument("--batch-size", "-J", type=int, default=5)
    parser.add_argument("--rounding", choices=ROUNDING, default="nearest")
    parser.add_argument("--compiler", choices=("qrom", "prefix"), default="qrom")
    parser.add_argument("--clean-cap", type=int, default=8)
    parser.add_argument("--output", type=Path, help="output directory (default: circuits/<system>/<run>)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-bits", type=int, default=KEEP_BITS,
        help="chi of the heuristic state-preparation cost model",
    )
    args = parser.parse_args()
    path = args.factorization or BASELINES_DIRECTORY / systems[args.system]["baseline"]
    result = build_submission(
        args.system, path, tree=args.tree, brot=args.brot, batch_size=args.batch_size,
        rounding=args.rounding, output_directory=args.output, overwrite=args.overwrite,
        keep_bits=args.keep_bits, compiler=args.compiler, clean_cap=args.clean_cap,
    )
    directory = Path(result["directory"])
    print(f"{args.system}: wrote {len(result['circuits'])} circuits to {directory}")
    print(format_summary(result))
    print(
        f"verify with: python verifier/verify.py {args.system} "
        f"{directory / 'submission.npz'} {directory}"
    )
