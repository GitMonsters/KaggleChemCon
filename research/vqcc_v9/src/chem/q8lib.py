"""GPU helpers for 8-bit DF-THC research: target, Givens tree, word decoding, two-body error.

The two-body error is the official error_2 (Frobenius norm of the shifted two-body residual),
evaluated in the orthonormal symmetric pair basis (2926 coordinates for N = 76).
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import torch

N = 76
COMP = pathlib.Path('/home/omen/vqcwork/comp')
TARGET_CACHE = pathlib.Path('/home/omen/vqcwork/chem8/target_ts64.npy')
DEV = 'cuda'


def build_target64() -> np.ndarray:
    if TARGET_CACHE.exists():
        return np.load(TARGET_CACHE)
    sys.path.insert(0, str(COMP / 'verifiers'))
    from dfthcsos.scoreinfra.dfthc_math import symmetric_from_lower
    from dfthcsos.scoreinfra.reference import load_fixed_shifts, load_reference
    base = COMP / 'verifiers' / 'dfthcsos'
    ref = load_reference(base / 'FeMoCo76o113e.fcidump')
    shifts = load_fixed_shifts(base / 'FeMoCo76o113e.fixed_shifts.npz')
    s = np.asarray(symmetric_from_lower(shifts['Hsym_lower'], N), dtype=np.float64)
    t = np.asarray(ref['tei'], dtype=np.float64).copy()
    d = np.arange(N)
    t[:, :, d, d] += 0.5 * s[:, :, None]
    t[d, d, :, :] += 0.5 * s[None, :, :]
    rows, cols = np.tril_indices(N)
    scale = np.where(rows == cols, 1.0, math.sqrt(2.0))
    pairs = rows * N + cols
    ts = t.reshape(N * N, N * N)[np.ix_(pairs, pairs)] * np.outer(scale, scale)
    np.save(TARGET_CACHE, ts)
    return ts


class Pair:
    """Map unit vectors u (..., N) to symmetric pair coordinates of u u^T (..., 2926)."""

    def __init__(self, dev=DEV, dtype=torch.float64):
        rows, cols = np.tril_indices(N)
        self.ri = torch.tensor(rows, device=dev)
        self.ci = torch.tensor(cols, device=dev)
        self.scale = torch.tensor(np.where(rows == cols, 1.0, math.sqrt(2.0)), device=dev, dtype=dtype)

    def __call__(self, u):
        return u[..., self.ri] * u[..., self.ci] * self.scale


class Tree:
    """Level-parallel Givens tree (official parent-before-child order): angles (units of pi) -> unit vectors."""

    def __init__(self, edges, dev=DEV, dtype=torch.float64):
        edges = [(int(p), int(q)) for p, q in edges]
        self.edges = edges
        levels, cur, used = [], [], set()
        for t, (p, q) in enumerate(edges):
            if p in used or q in used:
                levels.append(cur)
                cur, used = [], set()
            cur.append(t)
            used.update((p, q))
        levels.append(cur)
        self.plan = []
        for level in levels:
            colmap = np.full(N, N - 1, dtype=np.int64)
            sgn = np.zeros(N)
            perm = np.arange(N)
            for t in level:
                p, q = edges[t]
                colmap[p] = colmap[q] = t
                sgn[p], sgn[q] = -1.0, 1.0
                perm[p], perm[q] = q, p
            self.plan.append((torch.tensor(colmap, device=dev), torch.tensor(sgn, device=dev, dtype=dtype),
                              torch.tensor(perm, device=dev)))
        self.dev, self.dtype = dev, dtype

    def __call__(self, theta):
        """theta: (..., N-1) in units of pi."""
        shp = theta.shape[:-1]
        th = theta.reshape(-1, N - 1)
        k = th.shape[0]
        ang = math.pi * th
        one = torch.ones(k, 1, device=th.device, dtype=th.dtype)
        c = torch.cat([torch.cos(ang), one], dim=1)
        s = torch.cat([torch.sin(ang), 0 * one], dim=1)
        u = torch.zeros(k, N, device=th.device, dtype=th.dtype)
        u[:, 0] = 1.0
        for colmap, sgn, perm in self.plan:
            u = u * c[:, colmap] + u[:, perm] * (s[:, colmap] * sgn.to(th.dtype))
        return u.reshape(*shp, N)


def signed_words(words, brot):
    half = 1 << (brot - 1)
    return ((np.asarray(words, dtype=np.int64) + half) % (1 << brot)) - half


def theta_of_words(words, brot):
    """Official convention: theta / pi = word / 2**(brot - 1) (mod 2)."""
    return signed_words(words, brot).astype(np.float64) / (1 << (brot - 1))


def gauge_angles(u, edges):
    """Continuous angles (units of pi) of unit vectors in the signed half-open gauge, reproducing +-u."""
    a = np.array(u, dtype=np.float64, copy=True)
    ang = np.zeros((a.shape[0], len(edges)))
    for j in reversed(range(len(edges))):
        p, q = edges[j]
        norm = np.hypot(a[:, p], a[:, q])
        sign = np.where(np.abs(a[:, p]) < 1e-12, np.where(a[:, q] > 0, -1.0, 1.0), np.where(a[:, p] < 0, -1.0, 1.0))
        ang[:, j] = np.where(norm < 1e-12, 0.0, np.arctan2(a[:, q] * sign, a[:, p] * sign))
        a[:, p] = np.where(norm < 1e-12, 0.0, norm * sign)
        a[:, q] = 0.0
    return ang / np.pi


class TwoBody:
    """error_2 = || sum_{r,c} x_rc x_rc^T - T ||_F with x_rc = sum_b w_rbc p(u_rb)."""

    def __init__(self, dev=DEV, dtype=torch.float64):
        self.T = torch.tensor(build_target64(), device=dev, dtype=dtype)
        self.pair = Pair(dev, dtype)
        self.dev, self.dtype = dev, dtype

    def X(self, U, w):
        v = self.pair(U)                                  # (R, B, P)
        return torch.einsum('rbm,rbc->rcm', v, w).reshape(-1, v.shape[-1])   # (R*C, P)

    def resid(self, U, w):
        a = self.X(U, w)
        return a.T @ a - self.T

    def err(self, U, w):
        e = self.resid(U, w)
        return torch.sqrt((e * e).sum())


def load_submission(path):
    z = np.load(path)
    out = {k: np.array(z[k]) for k in z.files}
    return out


def dpen(w):
    a = w.abs().sum(dim=1)
    s = w.sum(dim=1)
    return 0.5 * (a * a - s * s).sum()
