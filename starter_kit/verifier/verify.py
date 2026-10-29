"""Score a Task 3 submission (kickmix ``.kmb`` or ``.kmx`` circuits): ``verify(...) -> dict``.

Usage::

    from verifier.verify import verify

    verify("Fe2S2-20o30e", "run_dir/")
    verify("Fe2S2-20o30e", "run_dir/", "factorization.npz")
    verify("Fe2S2-20o30e", ["a.kmx", "b.kmx"], "factorization.npz", tree_edges=edges)

    python verifier/verify.py Fe2S2-20o30e run_dir/ [--salt STR] [--seed N]
    python verifier/verify.py Fe2S2-20o30e factorization.npz run_dir/
    python verifier/verify.py Fe2S2-20o30e factorization.npz a.kmx b.kmx --tree-edges edges.npy

Inputs
------
* ``circuits`` (Task 3): a directory with ``angle_k_0.kmx``, ``angle_k_1.kmx``,
  ... (or ``.kmb``), or the circuit files in order. They must follow the
  register rules in ``verifier/circuits.py``; ``brot`` and ``J`` are read from
  the circuits.
* ``factorization`` (Task 1): an ``.npz`` with the trainer's ``U``, ``W``,
  ``H2_I``, ``H1_I`` and ``Hsym_lower`` (``verifier/reference.py``). It
  supplies the weights; the circuits only load the angle words. Optional when
  ``circuits`` is a directory containing ``submission.npz``. Pickles are
  not accepted: save the arrays with ``numpy.savez``. The starter trainer
  writes NPZ directly, and the
  ``submission.npz`` written by its ``build_circuits.py`` also holds them.
* ``tree_edges`` (Task 2): the ``(N - 1, 2)`` Givens tree, as an array or a
  ``.npy``/``.npz`` file. Optional if the circuits directory contains
  ``submission.npz`` with ``tree_edges``.
* ``data_dir``: the directory with ``systems.toml`` and ``integrals/`` (the
  reference FCIDUMPs). Default: ``$DFTHCSOS_DATA_DIR``, else
  ``<repository>/data``.

Measurement outcomes (``HMR`` and ``R``) are seeded with
``initial_seed(submission_sha256(...), salt)``: the SHA-256 of the whole
submission (circuits, factorization and tree), merged with ``--salt`` when
given, or overridden by ``--seed``.

Returns a JSON-serializable summary dict with ``status`` (``"PASS"``) and
``score = lambda_eff * T_total * Q_total`` (formula and rejection rules in
``verifier/score.py``). Raises ``InvalidCircuits`` or ``AssertionError`` if
the submission is rejected; the command line then prints
``{"status": "FAIL", "error": ...}`` and exits 1.

For Tasks 1 and 2 (no circuits) use ``verifier.score.score``.

The ``verifier`` package is self-contained: it imports nothing from ``dfthc``
and needs only numpy, scipy and kickmix.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import numpy.typing as npt

# Development shim: allow `python verifier/verify.py` without installing.
_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[1])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from verifier.circuits import FORMATS, SHOTS_PER_ADDRESS, InvalidCircuits, load_circuits  # noqa: E402
from verifier.reference import DATA_DIR_ENVIRONMENT_VARIABLE, load_systems  # noqa: E402
from verifier.score import (  # noqa: E402
    KEEP_BITS,
    evaluate_circuits,
    load_inputs,
    rejection_reason,
)

SUBMISSION_FILE = "submission.npz"


def load_tree_edges(
    tree_edges: npt.ArrayLike | str | Path | None, circuits: Any
) -> np.ndarray:
    """Resolve ``tree_edges``: an array, a ``.npy``/``.npz`` path, or ``None``.

    With ``None`` the edges are read from ``submission.npz`` next to the
    circuits, which requires ``circuits`` to be a directory.
    """
    if tree_edges is None:
        if not (isinstance(circuits, (str, Path)) and Path(circuits).is_dir()):
            raise ValueError(
                "tree_edges is required unless circuits is a directory with "
                f"{SUBMISSION_FILE}"
            )
        tree_edges = Path(circuits) / SUBMISSION_FILE
        if not tree_edges.is_file():
            raise FileNotFoundError(
                f"no {SUBMISSION_FILE} in {circuits}; pass tree_edges explicitly"
            )
    if isinstance(tree_edges, (str, Path)):
        loaded = np.load(tree_edges, allow_pickle=False)
        if isinstance(loaded, np.ndarray):
            tree_edges = loaded
        else:
            with loaded as data:
                tree_edges = np.array(data["tree_edges"])
    edges = np.asarray(tree_edges)
    if edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError(f"tree_edges must be an (N - 1, 2) integer array, got {edges.shape}")
    return edges.astype(np.int64)


def verify(
    system: str,
    circuits: Any,
    factorization: str | Path | Mapping[str, npt.ArrayLike] | None = None,
    tree_edges: npt.ArrayLike | str | Path | None = None,
    *,
    format: str = "auto",
    shots_per_address: int = SHOTS_PER_ADDRESS,
    seed: int | None = None,
    salt: str = "",
    keep_bits: int = KEEP_BITS,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Verify and score a Task 3 submission; return a summary dict or raise if rejected.

    Args:
        system: system name from ``systems.toml``, e.g. ``"Fe2S2-20o30e"``.
        circuits: your Task 3 output: a directory with ``angle_k_*.kmx`` (or
            ``.kmb``), or the circuit files (or ``kickmix.Circuit`` objects)
            in order.
        factorization: your Task 1 output as an ``.npz`` with ``U``, ``W``,
            ``H2_I``, ``H1_I`` and ``Hsym_lower`` (or a mapping of those arrays).
            Optional if the circuits directory has ``submission.npz``.
            Pickles are not accepted.
        tree_edges: your Task 2 Givens tree (array or ``.npy``/``.npz``).
            Optional if the circuits directory has ``submission.npz``.
        format: ``'auto'`` (default), ``'kmb'`` or ``'kmx'``.
        shots_per_address: shots simulated per address (default 64).
        seed: explicit simulator seed, overriding ``submission_sha256`` and ``salt``.
        salt: merged with ``submission_sha256`` to derive the simulator seed.
        keep_bits: ``chi`` of the state-preparation cost model (default 10).
        data_dir: directory with ``systems.toml`` and ``integrals/`` (default:
            ``$DFTHCSOS_DATA_DIR``, else ``<repository>/data``).

    Returns:
        A JSON-serializable dict: ``status`` (``"PASS"``), ``score``, ``system``,
        ``lambda_eff``, ``toffoli_count``, ``num_qubits``, ``brot``,
        ``batch_size``, ``num_circuits``, ``reaction_depth``, ``num_ops``,
        ``verified_shots``, ``submission_sha256``, and the ``resources`` and
        ``metrics`` breakdowns.

    Raises:
        InvalidCircuits: the circuits break the register contract or do not
            load consistent words.
        AssertionError: an accuracy, gap or 128-qubit gate fails.
    """
    if factorization is None:
        if not (isinstance(circuits, (str, Path)) and Path(circuits).is_dir()):
            raise ValueError(f"factorization is required unless circuits is a directory with {SUBMISSION_FILE}")
        factorization = Path(circuits) / SUBMISSION_FILE
    edges = load_tree_edges(tree_edges, circuits)
    inputs = load_inputs(system, factorization, data_dir=data_dir)
    family = load_circuits(circuits, format=format)
    result = evaluate_circuits(
        inputs, family, edges, keep_bits,
        shots_per_address=shots_per_address, seed=seed, salt=salt,
    )
    reason = rejection_reason(result["metrics"], result["resources"])
    if reason is not None:
        raise AssertionError(reason)
    return {
        "status": "PASS",
        "score": float(result["score"]),
        "system": system,
        "lambda_eff": result["metrics"]["lambda_eff"],
        "toffoli_count": result["resources"]["toffoli"],
        "num_qubits": result["resources"]["qubits"],
        "brot": result["brot"],
        "batch_size": result["batch_size"],
        "num_circuits": len(family),
        "reaction_depth": sum(c.reaction_depth() for c in family),
        "num_ops": sum(len(c) for c in family),
        "verified_shots": result["verified_shots"],
        "submission_sha256": result["submission_sha256"],
        "resources": result["resources"],
        "metrics": result["metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify kickmix ANGLE_k circuits and print the competition score as JSON."
    )
    parser.add_argument("system", help="system name from systems.toml")
    parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="a submission directory (submission.npz and angle_k_*), or a factorization .npz "
        "followed by a directory with angle_k_*.kmb/.kmx or the circuit files in order",
    )
    parser.add_argument(
        "--tree-edges", type=Path,
        help=f".npy or .npz with the Givens tree (default: {SUBMISSION_FILE} in the directory)",
    )
    parser.add_argument("--format", choices=FORMATS, default="auto",
                        help="circuit file format (default: auto-detect .kmb or .kmx)")
    parser.add_argument("--shots-per-address", type=int, default=SHOTS_PER_ADDRESS,
                        help=f"shots simulated per address (default: {SHOTS_PER_ADDRESS})")
    parser.add_argument("--salt", type=str, default="",
                        help="optional salt merged with submission_sha256 to derive the seed (default: empty)")
    parser.add_argument("--seed", type=int, default=None,
                        help="explicit simulator seed overriding submission_sha256 and --salt")
    parser.add_argument(
        "--keep-bits", type=int, default=KEEP_BITS,
        help="chi of the heuristic state-preparation cost model",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        help="directory with systems.toml and integrals/ "
        f"(default: ${DATA_DIR_ENVIRONMENT_VARIABLE}, else <repository>/data)",
    )
    args = parser.parse_args()
    known = load_systems(args.data_dir)
    if args.system not in known:
        parser.error(f"unknown system {args.system!r}; choose from {', '.join(sorted(known))}")
    if len(args.inputs) == 1:
        factorization, circuits = None, args.inputs[0]
    else:
        factorization, *circuits = args.inputs
        circuits = circuits[0] if len(circuits) == 1 else circuits
    try:
        summary = verify(
            args.system, circuits, factorization, args.tree_edges, format=args.format,
            shots_per_address=args.shots_per_address, seed=args.seed, salt=args.salt,
            keep_bits=args.keep_bits, data_dir=args.data_dir,
        )
    except (AssertionError, ValueError, RuntimeError, OSError, KeyError) as ex:
        print(json.dumps({"status": "FAIL", "error": str(ex)}, indent=2))
        sys.exit(1)
    print(json.dumps(summary, indent=2))


__all__ = ["SUBMISSION_FILE", "InvalidCircuits", "load_tree_edges", "main", "verify"]


if __name__ == "__main__":
    main()
