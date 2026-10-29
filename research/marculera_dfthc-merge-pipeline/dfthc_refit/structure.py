"""Exact structural moves on a DF-THC factorization (numpy only).

A factorization holds `U` (R, B, N) basis vectors, weights `W` (R, B, C) and identity offsets
`H2_I` (R, C): group r owns B vectors shared by its C copies, and the two-body tensor is
`sum_rc X^(rc) (x) X^(rc)` with `X^(rc) = sum_b W_rbc u_rb u_rb^T` (u = U normalized).
"""
from __future__ import annotations

import numpy as np


def importance(U: np.ndarray, w: np.ndarray) -> np.ndarray:
    """sum_c ||X^{rc}||_F^2 per group: X^{rc} = sum_b w_brc u_b u_b^T with unit u_b."""
    u = U / np.linalg.norm(U, axis=2, keepdims=True)
    gram = np.einsum('rbp,rdp->rbd', u, u) ** 2  # (u_b . u_d)^2 = <u_b u_b^T, u_d u_d^T>
    return np.einsum('rbc,rbd,rdc->r', w, gram, w)


def grow_bases(p: dict[str, np.ndarray], B: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Give every group B bases: random new vectors with 1e-4 random weights (the only inexact part)."""
    U, w = p['U'], p['W']
    off = p['H2_I'] - w.sum(axis=1)
    R, B0, C = w.shape
    rng = np.random.default_rng(seed)
    Un = np.concatenate([U, rng.normal(size=(R, B - B0, U.shape[2]))], axis=1)
    wn = np.concatenate([w, 1e-4 * rng.normal(size=(R, B - B0, C))], axis=1)
    return {**p, 'U': Un, 'W': wn, 'H2_I': off + wn.sum(axis=1)}


def merge(p: dict[str, np.ndarray], k: int, C: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Merge groups k at a time, then pad to C copies per merged group.

    The merge is exact: a merged group's bases and copies are the members' unions and its weights
    are block-diagonal (a copy of one member puts zero weight on the others' bases), so every
    term of the two-body tensor survives unchanged. Groups of similar importance are merged
    together. Extra copies (C > k * C0) start as 1e-4 random weights over all bases, and a short
    last group gets random unused bases: those two cases are the only inexact ones.
    """
    U, w = p['U'], p['W']
    off = p['H2_I'] - w.sum(axis=1)
    R0, B0, C0 = w.shape
    order = np.argsort(-importance(U, w))
    R1 = -(-R0 // k)
    rng = np.random.default_rng(seed)
    Un = np.zeros((R1, k * B0, U.shape[2]))
    wn = np.zeros((R1, k * B0, max(C, k * C0)))
    offn = np.zeros((R1, max(C, k * C0)))
    for g in range(R1):
        members = order[g * k:(g + 1) * k]
        for j, r in enumerate(members):
            Un[g, j * B0:(j + 1) * B0] = U[r]
            wn[g, j * B0:(j + 1) * B0, j * C0:(j + 1) * C0] = w[r]
            offn[g, j * C0:(j + 1) * C0] = off[r]
        empty = slice(len(members) * B0, k * B0)
        Un[g, empty] = rng.normal(size=Un[g, empty].shape)
        used = len(members) * C0
        wn[g, :, used:] = 1e-4 * rng.normal(size=wn[g, :, used:].shape)
    return {'U': Un, 'W': wn, 'H2_I': offn + wn.sum(axis=1),
            'H1_I': p['H1_I'], 'Hsym_lower': p['Hsym_lower']}


def two_body(U: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The represented two-electron tensor sum_rc vec X^(rc) vec X^(rc)^T, as an (N^2, N^2) matrix."""
    u = U / np.linalg.norm(U, axis=2, keepdims=True)
    X = np.einsum('rbp,rbq,rbc->rcpq', u, u, w).reshape(-1, U.shape[2] ** 2)
    return X.T @ X
