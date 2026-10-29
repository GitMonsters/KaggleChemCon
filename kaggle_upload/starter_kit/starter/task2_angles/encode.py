"""Part 2 (angles): encode a factorization as brot-bit angle words on a tree.

The default pipeline, for a chosen Givens tree, bit width ``brot`` and rounding
rule:

1. Two-body rows: for each normalized ``u_b^(rho)`` (``R*B`` rows), compute the
   tree angles ``theta / pi`` (``trees.tree_angles``) and round them to
   ``brot``-bit words.
2. Decode those words to the vectors the circuit actually prepares and build
   ``X^(rho c)`` from them, so the one-body target already includes the
   two-body rounding error.
3. One-body rows: the ``N`` eigenvectors of ``h1_prime(X)`` (Eq. (38)),
   encoded on the same tree with the same rounding.
4. ``angle_words`` is the ``(N + R*B, N - 1)`` stack: one-body rows first.

A word ``m`` means ``theta / pi = m / 2**(brot - 1)`` in ``[0, 2)``, the
decoding used by ``verifier.dfthc_math.evaluate_dfthc_angle_words``.

Rounding rules (``ROUNDING``):

* ``"nearest"``: ``floor(x * 2**(brot-1) + 1/2) mod 2**brot``.
* ``"floor"``: ``floor(x * 2**(brot-1)) mod 2**brot`` (truncation).

``brot_sweep`` evaluates the pipeline for several bit widths so you can see
where rounding starts to cost accuracy or ``lambda_eff``. Factorizations are
the verifier's ``.npz`` format. From the repository root:

    python starter/task2_angles/encode.py Fe2S2-20o30e --tree log_depth
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import sys
from typing import Any

import numpy as np
import numpy.typing as npt

# Development shim: allow `python starter/task2_angles/encode.py` without
# installing; the repository root holds dfthc/, verifier/ and starter/.
_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from starter.task1_factorization.solution_io import read_factorization  # noqa: E402
from verifier.dfthc_math import reconstruct_u_vectors, reconstruct_x  # noqa: E402
from verifier.score import (  # noqa: E402
    SubmissionInputs,
    evaluate_inputs,
    load_inputs,
    score_from_metrics,
    symmetric_lower,
)
from starter.task2_angles.trees import (  # noqa: E402
    log_depth_tree_edges,
    path_tree_edges,
    tree_angles,
)

ROUNDING = ("nearest", "floor")
TREES = {"path": path_tree_edges, "log_depth": log_depth_tree_edges}


def angles_to_words(
    theta_over_pi: npt.ArrayLike, brot: int, rounding: str = "nearest"
) -> npt.NDArray[np.int64]:
    """Round ``theta / pi`` (any real value, taken mod 2) to brot-bit words."""
    if rounding not in ROUNDING:
        raise ValueError(f"rounding must be one of {ROUNDING}, got {rounding!r}")
    if not 1 <= int(brot) <= 63:
        raise ValueError("brot must be an integer between 1 and 63")
    scaled = np.mod(np.asarray(theta_over_pi, dtype=np.float64), 2.0) * (
        1 << (brot - 1)
    )
    if rounding == "nearest":
        scaled = scaled + 0.5
    return np.floor(scaled).astype(np.int64) % (1 << brot)


def words_to_vectors(
    angle_words: npt.ArrayLike, tree_edges: npt.ArrayLike, brot: int
) -> npt.NDArray[np.float64]:
    """Decode words to the unit vectors the tree prepares (the verifier's map)."""
    words = np.asarray(angle_words)
    vectors = reconstruct_u_vectors(words / (1 << (brot - 1)), tree_edges)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def encode_inputs(
    inputs: SubmissionInputs,
    tree_edges: npt.ArrayLike,
    brot: int,
    rounding: str = "nearest",
) -> npt.NDArray[np.int64]:
    """``encode_angle_words`` for already loaded inputs."""
    edges = np.asarray(tree_edges, dtype=np.int64)
    raw_u = np.asarray(inputs.params["U"], dtype=np.float64)
    rank_r, bases_b, num_orbitals = raw_u.shape
    if edges.shape != (num_orbitals - 1, 2):
        raise ValueError("tree_edges must have shape (N - 1, 2)")

    # Steps 1-2: two-body words, then the vectors they actually encode.
    two_body_words = angles_to_words(
        tree_angles(raw_u.reshape(-1, num_orbitals), edges), brot, rounding
    )
    two_body_vectors = words_to_vectors(two_body_words, edges, brot).reshape(
        rank_r, bases_b, num_orbitals
    )
    X = symmetric_lower(reconstruct_x(two_body_vectors, inputs.w))

    # Step 3: one-body rows are the eigenvectors of h1_prime of the decoded X.
    _, eigenvector_columns = np.linalg.eigh(inputs.h1_prime(X))
    one_body_words = angles_to_words(
        tree_angles(eigenvector_columns.T, edges), brot, rounding
    )
    return np.vstack([one_body_words, two_body_words])


def encode_angle_words(
    system: str,
    factorization: str | Path,
    tree_edges: npt.ArrayLike,
    brot: int,
    rounding: str = "nearest",
) -> npt.NDArray[np.int64]:
    """Return the ``(N + R*B, N - 1)`` angle-word table for a factorization.

    ``factorization`` is the verifier's ``.npz`` format. Pass the
    result, ``tree_edges`` and ``brot`` to ``verifier.score.score``.
    """
    params, _ = read_factorization(factorization)
    return encode_inputs(load_inputs(system, params), tree_edges, brot, rounding)


def brot_sweep(
    system: str,
    factorization: str | Path,
    tree_edges: npt.ArrayLike,
    brots: Iterable[int] = (8, 10, 12, 14, 16, 20, 24, 32),
    rounding: str = "nearest",
) -> list[dict[str, Any]]:
    """Encode and evaluate at each bit width.

    Returns one row per ``brot`` plus a first row with ``brot=None`` for the
    continuous (unrounded) vectors. Each row has the verifier metrics and
    ``score``: the ``lambda_eff`` factor of the score, or ``inf`` if
    ``error_1`` or ``error_2`` exceeds 20 mHa or the gap is out of domain.
    The Toffoli and qubit factors need circuits (Part 3).
    """
    params, _ = read_factorization(factorization)
    inputs = load_inputs(system, params)
    edges = np.asarray(tree_edges, dtype=np.int64)
    exact = evaluate_inputs(inputs)
    rows = [{"brot": None, **exact, "score": score_from_metrics(exact)}]
    for brot in brots:
        words = encode_inputs(inputs, edges, brot, rounding)
        metrics = evaluate_inputs(inputs, words, edges, brot)
        rows.append({"brot": int(brot), **metrics, "score": score_from_metrics(metrics)})
    return rows


def format_sweep(rows: list[dict[str, Any]]) -> str:
    """Render ``brot_sweep`` rows as a plain-text table."""
    lines = [
        f"{'brot':>6} {'error_1 (Ha)':>13} {'error_2 (mHa)':>14} "
        f"{'Lambda':>10} {'E_gap':>9} {'lambda_eff':>11} {'score':>10}"
    ]
    for row in rows:
        brot = "exact" if row["brot"] is None else str(row["brot"])
        lines.append(
            f"{brot:>6} {row['error_1']:>13.3e} {1e3 * row['error_2']:>14.4f} "
            f"{row['Lambda']:>10.4f} {row['E_gap']:>9.5f} "
            f"{row['lambda_eff']:>11.5f} {row['score']:>10.5f}"
        )
    return "\n".join(lines)


__all__ = [
    "ROUNDING",
    "TREES",
    "angles_to_words",
    "brot_sweep",
    "encode_angle_words",
    "encode_inputs",
    "format_sweep",
    "words_to_vectors",
]


if __name__ == "__main__":
    import argparse

    from starter.task1_factorization.solution_io import BASELINES_DIRECTORY
    from verifier.reference import load_systems

    systems = load_systems()
    parser = argparse.ArgumentParser(
        description="Sweep brot for a factorization on a Givens tree."
    )
    parser.add_argument("system", choices=sorted(systems))
    parser.add_argument(
        "factorization", nargs="?", type=Path,
        help="factorization .npz (default: the system's baseline)",
    )
    parser.add_argument("--tree", choices=sorted(TREES), default="log_depth")
    parser.add_argument("--rounding", choices=ROUNDING, default="nearest")
    parser.add_argument(
        "--brots", type=int, nargs="+", default=[8, 10, 12, 14, 16, 20, 24, 32]
    )
    args = parser.parse_args()
    path = args.factorization or BASELINES_DIRECTORY / systems[args.system]["baseline"]
    edges = TREES[args.tree](systems[args.system]["orbitals"])
    print(f"{args.system}: {path.name}, tree={args.tree}, rounding={args.rounding}")
    print(format_sweep(brot_sweep(args.system, path, edges, args.brots, args.rounding)))
