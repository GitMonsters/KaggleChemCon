"""Givens trees for Part 2 (angles): tree edges and vector -> angle conversion.

A Part 2 submission states the ordered tree it uses: ``tree_edges`` is an
``(N - 1, 2)`` integer array of ``(parent, child)`` pairs, rooted at orbital 0,
with every parent reached before its children. The verifier
(``dfthc.verify_angle_k.reconstruct_u_vectors``) rebuilds each unit vector by
starting from ``e_0`` and applying, for edge ``t = (p, q)``,

    u[p], u[q] <- cos(theta_t) u[p] - sin(theta_t) u[q],
                  sin(theta_t) u[p] + cos(theta_t) u[q].

``tree_angles`` is the inverse of that map for any valid tree.

Two trees are provided:

* ``path_tree_edges(N)``: ``(0, 1), (1, 2), ..., (N - 2, N - 1)``. This is the
  sequential-pair tree used by the baseline pipeline
  (``dfthc.dfthc_u_to_tei.collect_qrom_vectors``); its rotations form
  ``N - 1`` sequential layers.
* ``log_depth_tree_edges(N)``: the binary tree of arXiv:2605.30455. The root
  at depth 0 is labeled 0; at depth ``l`` the node labeled ``j`` has left
  child ``j`` and right child ``j + 2**l``. The edges are listed in
  breadth-first order, excluding self-loops and labels ``>= N``; the edges
  added at depth ``l`` act on disjoint orbitals, giving ``ceil(log2(N))``
  layers of parallel rotations.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def path_tree_edges(num_orbitals: int) -> npt.NDArray[np.int64]:
    """Sequential pairs (0, 1), (1, 2), ..., (N - 2, N - 1)."""
    if num_orbitals < 1:
        raise ValueError("num_orbitals must be positive")
    return np.column_stack(
        (np.arange(num_orbitals - 1), np.arange(1, num_orbitals))
    ).astype(np.int64)


def log_depth_tree_edges(num_orbitals: int) -> npt.NDArray[np.int64]:
    """Breadth-first edges of the log-depth binary tree of arXiv:2605.30455.

    For N = 8 this is (0, 1), (0, 2), (1, 3), (0, 4), (2, 6), (1, 5), (3, 7).
    """
    if num_orbitals < 1:
        raise ValueError("num_orbitals must be positive")
    edges: list[tuple[int, int]] = []
    level = [0]  # node labels at the current depth, in breadth-first order
    depth = 0
    while len(edges) < num_orbitals - 1:
        next_level = []
        for label in level:
            # Left child keeps the label (a self-loop, not an edge).
            next_level.append(label)
            right = label + (1 << depth)
            if right < num_orbitals:
                edges.append((label, right))
                next_level.append(right)
        level = next_level
        depth += 1
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2)


def tree_angles(
    vectors: npt.ArrayLike,
    tree_edges: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Return theta / pi in [0, 2) such that the tree maps e_0 to each row.

    ``vectors`` has shape ``(M, N)``; rows are normalized first. The result has
    shape ``(M, N - 1)`` with column ``t`` belonging to ``tree_edges[t]``, and
    ``reconstruct_u_vectors(result, tree_edges)`` returns the normalized rows.
    Encoding theta / pi as a ``brot``-bit word is left to the caller.
    """
    u = np.array(vectors, dtype=np.float64, ndmin=2)
    edges = np.asarray(tree_edges, dtype=np.int64)
    num_vectors, num_orbitals = u.shape
    if edges.shape != (num_orbitals - 1, 2):
        raise ValueError("tree_edges must have shape (N - 1, 2)")
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    if np.any(norms == 0.0) or not np.all(np.isfinite(norms)):
        raise ValueError("vectors must have finite, nonzero norms")
    u /= norms

    # Undo the rotations last-to-first. When edge (p, q) is undone, every later
    # edge has been undone, so (u[p], u[q]) = a (cos theta, sin theta) with a
    # the amplitude on p before the rotation; take a >= 0.
    theta = np.zeros((num_vectors, num_orbitals - 1), dtype=np.float64)
    for index in range(num_orbitals - 2, -1, -1):
        parent, child = edges[index]
        theta[:, index] = np.arctan2(u[:, child], u[:, parent])
        u[:, parent] = np.hypot(u[:, parent], u[:, child])
        u[:, child] = 0.0
    return np.mod(theta / np.pi, 2.0)


__all__ = ["path_tree_edges", "log_depth_tree_edges", "tree_angles"]
