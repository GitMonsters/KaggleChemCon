"""Exact word polish of the one-body rows: lower error_1 at a given word width.

The verifier pairs each one-body row v_j (decoded from its words) with an eigenpair (lambda_j, e_j)
of h1' and reports error_1 = ||sum_j lambda_j v_j v_j^T - h1'||_F. Rounding the eigenvectors to
8-bit words leaves error_1 far above the 20 mHa gate; moving single words by +-1 fixes most of it.
With M the current residual, replacing v_j by v' changes error_1^2 exactly by

    2 lambda_j (v'^T M v' - v_j^T M v_j) + lambda_j^2 ||v' v'^T - v_j v_j^T||^2,

so all moves of a row are scored at once; the best improving move is applied and M updated.
(mrmeows' `refresh_one_body8.py` fills this role in the public 8-bit pipeline; this is our own.)
"""
from __future__ import annotations

import numpy as np

import encode


def polish(words: np.ndarray, h1p: np.ndarray, edges: np.ndarray, brot: int, sweeps: int = 20,
           steps=(-1, 1)) -> np.ndarray:
    """Signed-gauge one-body words (N, N-1) with row j meant for the j-th eigenvector of h1' (ascending)."""
    lam, _ = np.linalg.eigh(h1p)
    mod, lo, hi = 1 << brot, -(1 << (brot - 2)), (1 << (brot - 2)) - 1
    k = np.where(words > hi, words - mod, words).astype(np.int64)
    V = encode.decode(k % mod, edges, brot)                       # (N, N) rows v_j
    M = (V.T * lam) @ V - h1p
    for _ in range(sweeps):
        moved = 0
        for j in range(len(k)):
            cands, moves = [], []
            for t in range(k.shape[1]):
                for s in steps:
                    if lo <= k[j, t] + s <= hi:
                        row = k[j].copy()
                        row[t] += s
                        cands.append(row)
                        moves.append((t, s))
            U = encode.decode(np.array(cands) % mod, edges, brot)   # (n, N)
            v = V[j]
            quad = np.einsum('np,pq,nq->n', U, M, U) - v @ M @ v
            ov = U @ v
            dd = 2 - 2 * ov ** 2                                     # ||u u^T - v v^T||^2 for unit vectors
            gain = 2 * lam[j] * quad + lam[j] ** 2 * dd
            i = int(np.argmin(gain))
            if gain[i] < -1e-15:
                t, s = moves[i]
                k[j, t] += s
                u = U[i]
                M += lam[j] * (np.outer(u, u) - np.outer(v, v))
                V[j] = u
                moved += 1
        if not moved:
            break
    return k % mod


def signed_gauge(words: np.ndarray, edges: np.ndarray, brot: int) -> np.ndarray:
    """The same unit vectors up to sign, with every angle in [-pi/2, pi/2) (exact, integer only).

    Walk the tree backwards (the decoder applies rotation 0 first): a node whose sign was flipped
    negates the angles that mix it in, and an angle outside the gauge moves by pi, which negates
    its first node. The decoded row changes at most by an overall sign.
    """
    mod = 1 << brot
    half, quarter = mod >> 1, mod >> 2
    out = np.asarray(words, np.int64).copy() % mod
    sign = np.ones((len(out), edges.shape[0] + 1), np.int8)
    for k in reversed(range(edges.shape[0])):
        p, q = edges[k]
        a = np.where(sign[:, p] != sign[:, q], (-out[:, k]) % mod, out[:, k])
        flip = (a >= quarter) & (a < 3 * quarter)
        out[:, k] = np.where(flip, (a + half) % mod, a)
        sign[flip, p] *= -1
        sign[:, q] = 1
    return out


def error(words: np.ndarray, h1p: np.ndarray, edges: np.ndarray, brot: int) -> float:
    """error_1 with row j assigned to the j-th eigenvalue of h1' (ascending)."""
    lam = np.linalg.eigvalsh(h1p)
    V = encode.decode(words % (1 << brot), edges, brot)
    return float(np.linalg.norm((V.T * lam) @ V - h1p))


def refresh(h1p: np.ndarray, edges: np.ndarray, brot: int, start: np.ndarray | None = None, kicks: int = 40,
            seed: int = 0) -> np.ndarray:
    """One-body words for h1': the better of rounded eigenvectors and `start`, polished, then improved
    by random kicks (a few words of a few rows moved by +-1, re-polished, kept if better)."""
    _, vecs = np.linalg.eigh(h1p)
    cands = [encode.signed_words(vecs.T, edges, brot)] + ([start] if start is not None else [])
    k = min((polish(c, h1p, edges, brot, steps=(-2, -1, 1, 2)) for c in cands),
            key=lambda c: error(c, h1p, edges, brot))
    best = error(k, h1p, edges, brot)
    rng = np.random.default_rng(seed)
    N = len(k)
    for _ in range(kicks):
        trial = k.copy()
        for r in rng.choice(N, 4, replace=False):
            cols = rng.choice(trial.shape[1], 6, replace=False)
            trial[r, cols] += rng.choice([-1, 1], 6)
        trial = polish(trial % (1 << brot), h1p, edges, brot)
        e = error(trial, h1p, edges, brot)
        if e < best:
            best, k = e, trial
    return signed_gauge(k, edges, brot)
